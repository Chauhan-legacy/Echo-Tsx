/**
 * High-Precision Window Functions & Spectral Correction Factors
 *
 * Implements:
 * 1. Hann Window (Bell shape, 31.5 dB sidelobe attenuation)
 * 2. Hamming Window (43 dB sidelobe attenuation)
 * 3. Blackman-Harris 4-Term (-92 dB sidelobe attenuation, ideal for tracking tones)
 * 4. Kaiser Window (parameterized beta via modified Bessel I0 series)
 * 5. Flat-Top Window (scalloping loss < 0.01 dB, ideal for exact amplitude calibration)
 *
 * Each window exports its exact Coherent Gain (CG) and Equivalent Noise Bandwidth (ENBW).
 */

export interface WindowMetrics {
  window: Float64Array;
  coherentGain: number; // CG = sum(w) / N
  equivalentNoiseBandwidth: number; // ENBW = N * sum(w^2) / (sum(w))^2
}

/**
 * 0th order modified Bessel function of the first kind: I_0(x)
 */
function besselI0(x: number): number {
  let sum = 1.0;
  let term = 1.0;
  const xHalf = x * 0.5;
  for (let k = 1; k <= 30; k++) {
    const f = xHalf / k;
    term *= f * f;
    sum += term;
    if (term < 1e-16 * sum) break;
  }
  return sum;
}

export type WindowType = 'hann' | 'hamming' | 'blackman-harris' | 'kaiser' | 'flat-top';

export function computeWindow(
  type: WindowType,
  size: number,
  options?: { beta?: number }
): WindowMetrics {
  const w = new Float64Array(size);
  const nMinus1 = Math.max(1, size - 1);

  if (type === 'hann') {
    for (let i = 0; i < size; i++) {
      w[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / nMinus1));
    }
  } else if (type === 'hamming') {
    for (let i = 0; i < size; i++) {
      w[i] = 0.54 - 0.46 * Math.cos((2.0 * Math.PI * i) / nMinus1);
    }
  } else if (type === 'blackman-harris') {
    // Exact 4-term Blackman-Harris coefficients
    const a0 = 0.35875;
    const a1 = 0.48829;
    const a2 = 0.14128;
    const a3 = 0.01168;
    for (let i = 0; i < size; i++) {
      const theta = (2.0 * Math.PI * i) / nMinus1;
      w[i] = a0 - a1 * Math.cos(theta) + a2 * Math.cos(2 * theta) - a3 * Math.cos(3 * theta);
    }
  } else if (type === 'kaiser') {
    const beta = options?.beta ?? 8.6; // ~90 dB stopband
    const den = besselI0(beta);
    for (let i = 0; i < size; i++) {
      const k = (2.0 * i) / nMinus1 - 1.0;
      const num = besselI0(beta * Math.sqrt(Math.max(0.0, 1.0 - k * k)));
      w[i] = num / den;
    }
  } else if (type === 'flat-top') {
    // Flat-top window for zero amplitude scalloping
    const a0 = 0.21557895;
    const a1 = 0.41663158;
    const a2 = 0.277263158;
    const a3 = 0.083578947;
    const a4 = 0.006947368;
    for (let i = 0; i < size; i++) {
      const theta = (2.0 * Math.PI * i) / nMinus1;
      w[i] =
        a0 -
        a1 * Math.cos(theta) +
        a2 * Math.cos(2 * theta) -
        a3 * Math.cos(3 * theta) +
        a4 * Math.cos(4 * theta);
    }
  }

  let sum = 0.0;
  let sumSq = 0.0;
  for (let i = 0; i < size; i++) {
    const val = w[i]!;
    sum += val;
    sumSq += val * val;
  }

  const coherentGain = sum / size;
  const equivalentNoiseBandwidth = (size * sumSq) / (sum * sum + 1e-20);

  return {
    window: w,
    coherentGain,
    equivalentNoiseBandwidth,
  };
}

export function generateWindow(type: WindowType, size: number): {
  window: Float64Array;
  coherentGain: number;
  enbw: number;
} {
  const res = computeWindow(type, size);
  return {
    window: res.window,
    coherentGain: res.coherentGain,
    enbw: res.equivalentNoiseBandwidth,
  };
}

