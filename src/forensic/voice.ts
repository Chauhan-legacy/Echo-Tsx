/**
 * Forensic Voice Analysis:
 * 1. Levinson-Durbin Recursion & LPC (Linear Predictive Coding)
 * 2. Formant Frequency and Bandwidth Estimation
 * 3. Cepstral Analysis (LPCC)
 * 4. YIN Fundamental Frequency (F0) Estimator (de Cheveigné & Kawahara 2002)
 */

export interface Formant {
  frequencyHz: number;
  bandwidthHz: number;
}

export interface LPCResult {
  coefficients: Float64Array; // a[0]=1, a[1], ..., a[p]
  reflectionCoefficients: Float64Array; // PARCOR k[0] ... k[p-1]
  predictionErrorVariance: number;
  formants: Formant[];
  cepstrum: Float64Array;
}

export interface YINResult {
  pitchHz: number;
  periodSamples: number;
  periodicity: number; // 1 - d'(tau_min), range [0, 1]
  harmonicityDb: number;
}

/**
 * Autocorrelation of a signal vector for lags 0 to p.
 */
export function autocorrelation(signal: Float64Array, maxLag: number): Float64Array {
  const n = signal.length;
  const p = Math.min(n - 1, maxLag);
  const r = new Float64Array(p + 1);

  for (let lag = 0; lag <= p; lag++) {
    let sum = 0.0;
    for (let i = 0; i < n - lag; i++) {
      sum += signal[i]! * signal[i + lag]!;
    }
    r[lag] = sum;
  }
  return r;
}

/**
 * Levinson-Durbin recursion: solves Toeplitz system R * a = [sigma^2, 0, ..., 0]^T
 * Returns all-pole prediction coefficients a = [1, a_1, ..., a_p]
 */
export function levinsonDurbin(r: Float64Array, order: number): {
  a: Float64Array;
  k: Float64Array;
  errVar: number;
} {
  const p = order;
  const a = new Float64Array(p + 1);
  const k = new Float64Array(p);

  a[0] = 1.0;
  let e = r[0]!;

  if (e <= 1e-12) {
    return { a, k, errVar: 1e-12 };
  }

  const prevA = new Float64Array(p + 1);
  prevA[0] = 1.0;

  for (let m = 1; m <= p; m++) {
    let lambda = 0.0;
    for (let j = 0; j < m; j++) {
      lambda += prevA[j]! * r[m - j]!;
    }

    const reflection = -lambda / e;
    k[m - 1] = reflection;
    a[m] = reflection;

    for (let j = 1; j < m; j++) {
      a[j] = prevA[j]! + reflection * prevA[m - j]!;
    }

    e *= (1.0 - reflection * reflection);
    if (e <= 1e-15) break;

    for (let j = 0; j <= m; j++) {
      prevA[j] = a[j]!;
    }
  }

  return { a, k, errVar: Math.max(1e-12, e) };
}

/**
 * Convert LPC coefficients to Cepstral coefficients (LPCC):
 * c_0 = ln(sigma^2)
 * c_m = - a_m - \sum_{k=1}^{m-1} \frac{k}{m} c_k a_{m-k}
 */
export function lpcToCepstrum(a: Float64Array, errVar: number, numCepstra: number = 13): Float64Array {
  const c = new Float64Array(numCepstra);
  c[0] = Math.log(Math.max(1e-12, errVar));
  const p = a.length - 1;

  for (let m = 1; m < numCepstra; m++) {
    let sum = 0.0;
    const limit = Math.min(m, p);
    for (let k = 1; k < limit; k++) {
      sum += (k / m) * c[k]! * a[m - k]!;
    }
    const am = m <= p ? a[m]! : 0.0;
    c[m] = -am - sum;
  }

  return c;
}

/**
 * Estimate acoustic formants by finding spectral peaks of the LPC filter H(z) = 1 / A(z).
 */
export function estimateFormantsFromLPC(
  a: Float64Array,
  sampleRate: number,
  maxFormants: number = 4
): Formant[] {
  // Evaluate frequency response |1 / A(e^{j omega})|^2 over 0 to sampleRate/2
  const numGrid = 512;
  const p = a.length - 1;
  const logMag = new Float64Array(numGrid);
  const freqs = new Float64Array(numGrid);

  for (let i = 0; i < numGrid; i++) {
    const f = (i * (sampleRate / 2)) / (numGrid - 1);
    freqs[i] = f;
    const omega = (2.0 * Math.PI * f) / sampleRate;

    let real = 1.0;
    let imag = 0.0;
    for (let k = 1; k <= p; k++) {
      const angle = -k * omega;
      real += a[k]! * Math.cos(angle);
      imag += a[k]! * Math.sin(angle);
    }
    const magSq = real * real + imag * imag;
    logMag[i] = -10.0 * Math.log10(Math.max(1e-12, magSq));
  }

  // Find local peaks in LPC spectrum
  const formants: Formant[] = [];
  for (let i = 1; i < numGrid - 1; i++) {
    if (logMag[i]! > logMag[i - 1]! && logMag[i]! > logMag[i + 1]!) {
      const fPeak = freqs[i]!;
      // Simple 3dB bandwidth estimate
      const peakVal = logMag[i]!;
      let bw = 100.0;
      for (let left = i - 1; left >= 0; left--) {
        if (peakVal - logMag[left]! >= 3.0) {
          bw = 2.0 * (fPeak - freqs[left]!);
          break;
        }
      }
      formants.push({
        frequencyHz: Math.round(fPeak),
        bandwidthHz: Math.round(bw),
      });
      if (formants.length >= maxFormants) break;
    }
  }

  return formants;
}

/**
 * Full LPC analysis pipeline.
 */
export function analyzeLPC(
  windowedSignal: Float64Array,
  order: number = 12,
  sampleRate: number = 16000
): LPCResult {
  const r = autocorrelation(windowedSignal, order);
  const { a, k, errVar } = levinsonDurbin(r, order);
  const cepstrum = lpcToCepstrum(a, errVar, 13);
  const formants = estimateFormantsFromLPC(a, sampleRate, 4);

  return {
    coefficients: a,
    reflectionCoefficients: k,
    predictionErrorVariance: errVar,
    formants,
    cepstrum,
  };
}

/**
 * YIN Pitch Estimator (de Cheveigné & Kawahara 2002)
 * High-accuracy sub-sample pitch detection for voice forensics and tracker beacon identification.
 */
export function estimateYINPitch(
  frame: Float64Array,
  sampleRate: number,
  minF0: number = 50.0,
  maxF0: number = 800.0,
  threshold: number = 0.15
): YINResult {
  const n = frame.length;
  const minLag = Math.max(2, Math.floor(sampleRate / maxF0));
  const maxLag = Math.min(n - 1, Math.floor(sampleRate / minF0));
  const half = Math.floor(n / 2);

  // 1. Difference function d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2
  const d = new Float64Array(maxLag + 1);
  for (let tau = minLag; tau <= maxLag; tau++) {
    let sum = 0.0;
    const limit = Math.min(half, n - tau);
    for (let j = 0; j < limit; j++) {
      const diff = frame[j]! - frame[j + tau]!;
      sum += diff * diff;
    }
    d[tau] = sum;
  }

  // 2. Cumulative mean normalized difference d'(tau)
  const dPrime = new Float64Array(maxLag + 1);
  dPrime[0] = 1.0;
  let runningSum = 0.0;
  for (let tau = 1; tau <= maxLag; tau++) {
    runningSum += d[tau]!;
    if (runningSum > 1e-12) {
      dPrime[tau] = (d[tau]! * tau) / runningSum;
    } else {
      dPrime[tau] = 1.0;
    }
  }

  // 3. Absolute thresholding & local minimum search
  let bestTau = -1;
  for (let tau = minLag; tau <= maxLag; tau++) {
    if (dPrime[tau]! < threshold) {
      while (tau + 1 <= maxLag && dPrime[tau + 1]! < dPrime[tau]!) {
        tau++;
      }
      bestTau = tau;
      break;
    }
  }

  // If no point fell below threshold, take global minimum
  if (bestTau === -1) {
    let minVal = 1e9;
    for (let tau = minLag; tau <= maxLag; tau++) {
      if (dPrime[tau]! < minVal) {
        minVal = dPrime[tau]!;
        bestTau = tau;
      }
    }
  }

  // 4. Parabolic sub-sample interpolation around bestTau
  let exactTau = bestTau;
  if (bestTau > minLag && bestTau < maxLag) {
    const alpha = dPrime[bestTau - 1]!;
    const beta = dPrime[bestTau]!;
    const gamma = dPrime[bestTau + 1]!;
    const denom = 2.0 * (alpha - 2.0 * beta + gamma);
    if (Math.abs(denom) > 1e-12) {
      const delta = (alpha - gamma) / denom;
      if (Math.abs(delta) < 1.0) {
        exactTau = bestTau + delta;
      }
    }
  }

  const pitchHz = exactTau > 0 ? sampleRate / exactTau : 0;
  const periodicity = Math.max(0.0, Math.min(1.0, 1.0 - (dPrime[bestTau] ?? 1.0)));
  const harmonicityDb = periodicity > 0.01 ? 10.0 * Math.log10(periodicity / (1.0 - periodicity + 1e-6)) : -20.0;

  return {
    pitchHz,
    periodSamples: exactTau,
    periodicity,
    harmonicityDb,
  };
}
