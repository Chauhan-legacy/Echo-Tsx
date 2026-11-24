/**
 * Remote Photoplethysmography (rPPG) Pulse Extraction from Facial Video RGB Signals.
 *
 * Implements two gold-standard physiological optics methods:
 * 1. CHROM (Chrominance-based method - de Haan & Jeanne 2013)
 *    Projects normalized RGB into two orthogonal chrominance signals:
 *      X_s = 3*R - 2*G
 *      Y_s = 1.5*R + G - 1.5*B
 *    Pulse signal: S = X_s - alpha * Y_s, where alpha = std(X_s) / std(Y_s).
 *
 * 2. POS (Plane-Orthogonal-to-Skin - Wang et al. 2017)
 *    Projects temporally normalized RGB signals onto the plane orthogonal to skin tone vector:
 *      S_1 = G - B
 *      S_2 = -2*R + G + B
 *    Adaptive combination: h = S_1 + (std(S_1) / std(S_2)) * S_2.
 *
 * Includes Butterworth bandpass filter (0.7 Hz to 3.5 Hz = 42 to 210 BPM) and Pulse SNR.
 */

export interface RPPGResult {
  pulseSignal: Float64Array;
  heartRateBpm: number;
  snrDb: number;
  confidence: number;
  spectrum: Float64Array;
  frequenciesHz: Float64Array;
}

/**
 * 2nd-order IIR Biquad Bandpass Filter (Butterworth approximation)
 */
function bandpassFilter(
  signal: Float64Array,
  sampleRate: number,
  fLow: number = 0.7,
  fHigh: number = 3.5
): Float64Array {
  const n = signal.length;
  const out = new Float64Array(n);

  // Pre-warp frequencies
  const w0 = (2.0 * Math.PI * Math.sqrt(fLow * fHigh)) / sampleRate;
  const bw = (fHigh - fLow) / Math.sqrt(fLow * fHigh);
  const alpha = Math.sin(w0) * Math.sinh((Math.log(2) / 2) * bw * (w0 / Math.sin(w0)));

  const b0 = alpha;
  const b1 = 0.0;
  const b2 = -alpha;
  const a0 = 1.0 + alpha;
  const a1 = -2.0 * Math.cos(w0);
  const a2 = 1.0 - alpha;

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < n; i++) {
    const x0 = signal[i]!;
    const y0 = (b0 / a0) * x0 + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
    out[i] = y0;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
  }
  return out;
}

/**
 * Standard deviation of Float64Array
 */
function arrayStd(arr: Float64Array): number {
  const n = arr.length;
  if (n < 2) return 1.0;
  let sum = 0.0;
  for (let i = 0; i < n; i++) sum += arr[i]!;
  const mean = sum / n;
  let sumSq = 0.0;
  for (let i = 0; i < n; i++) {
    const d = arr[i]! - mean;
    sumSq += d * d;
  }
  return Math.sqrt(Math.max(1e-12, sumSq / (n - 1)));
}

/**
 * Extract rPPG pulse using CHROM or POS method.
 * Inputs: r, g, b channel traces averaged over the facial region across frames.
 */
export function extractRPPG(
  r: Float64Array,
  g: Float64Array,
  b: Float64Array,
  sampleRateFps: number = 30.0,
  method: 'CHROM' | 'POS' = 'POS'
): RPPGResult {
  const n = Math.min(r.length, g.length, b.length);
  if (n < 30) {
    throw new Error('Signal too short for rPPG analysis (minimum 30 frames)');
  }

  // 1. Temporal normalization: divide each channel by its mean
  let rMean = 0, gMean = 0, bMean = 0;
  for (let i = 0; i < n; i++) {
    rMean += r[i]!;
    gMean += g[i]!;
    bMean += b[i]!;
  }
  rMean /= n; gMean /= n; bMean /= n;

  const rNorm = new Float64Array(n);
  const gNorm = new Float64Array(n);
  const bNorm = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    rNorm[i] = r[i]! / Math.max(1e-6, rMean) - 1.0;
    gNorm[i] = g[i]! / Math.max(1e-6, gMean) - 1.0;
    bNorm[i] = b[i]! / Math.max(1e-6, bMean) - 1.0;
  }

  // 2. Optical projection
  const rawPulse = new Float64Array(n);
  if (method === 'CHROM') {
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = 3.0 * rNorm[i]! - 2.0 * gNorm[i]!;
      ys[i] = 1.5 * rNorm[i]! + gNorm[i]! - 1.5 * bNorm[i]!;
    }
    const stdX = arrayStd(xs);
    const stdY = arrayStd(ys);
    const alpha = stdX / Math.max(1e-6, stdY);
    for (let i = 0; i < n; i++) {
      rawPulse[i] = xs[i]! - alpha * ys[i]!;
    }
  } else {
    // POS (Plane-Orthogonal-to-Skin)
    const s1 = new Float64Array(n);
    const s2 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      s1[i] = gNorm[i]! - bNorm[i]!;
      s2[i] = -2.0 * rNorm[i]! + gNorm[i]! + bNorm[i]!;
    }
    const std1 = arrayStd(s1);
    const std2 = arrayStd(s2);
    const alpha = std1 / Math.max(1e-6, std2);
    for (let i = 0; i < n; i++) {
      rawPulse[i] = s1[i]! + alpha * s2[i]!;
    }
  }

  // 3. Bandpass filter 0.7 - 3.5 Hz
  const filteredPulse = bandpassFilter(rawPulse, sampleRateFps, 0.7, 3.5);

  // 4. Spectral analysis for Heart Rate and SNR
  // Discrete Fourier Transform across physiological range 0.7 - 3.5 Hz
  const numBins = 128;
  const freqs = new Float64Array(numBins);
  const spectrum = new Float64Array(numBins);
  const fMin = 0.7;
  const fMax = 3.5;
  const step = (fMax - fMin) / (numBins - 1);

  let peakPower = 0.0;
  let peakFreq = 1.2; // 72 bpm default

  for (let k = 0; k < numBins; k++) {
    const f = fMin + k * step;
    freqs[k] = f;

    // Goertzel-like single frequency evaluation
    let real = 0.0;
    let imag = 0.0;
    const omega = (2.0 * Math.PI * f) / sampleRateFps;
    for (let i = 0; i < n; i++) {
      const angle = omega * i;
      real += filteredPulse[i]! * Math.cos(angle);
      imag -= filteredPulse[i]! * Math.sin(angle);
    }
    const pwr = (real * real + imag * imag) / n;
    spectrum[k] = pwr;

    if (pwr > peakPower) {
      peakPower = pwr;
      peakFreq = f;
    }
  }

  // Compute SNR: ratio of power in peak band (peakFreq +/- 0.15 Hz) to non-peak power
  let signalPower = 0.0;
  let noisePower = 0.0;
  const peakBand = 0.15;

  for (let k = 0; k < numBins; k++) {
    const f = freqs[k]!;
    const pwr = spectrum[k]!;
    if (Math.abs(f - peakFreq) <= peakBand) {
      signalPower += pwr;
    } else {
      noisePower += pwr;
    }
  }

  const snr = noisePower > 1e-12 ? 10.0 * Math.log10(signalPower / noisePower) : 0.0;
  const confidence = Math.min(1.0, Math.max(0.0, (snr + 5.0) / 15.0));
  const heartRateBpm = peakFreq * 60.0;

  return {
    pulseSignal: filteredPulse,
    heartRateBpm,
    snrDb: snr,
    confidence,
    spectrum,
    frequenciesHz: freqs,
  };
}
