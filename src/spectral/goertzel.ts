/**
 * Second-Order Recursive Goertzel Filter
 *
 * Computes single discrete frequency Fourier coefficients with O(N) operations
 * and O(1) memory, avoiding full FFT computations when evaluating specific
 * grid hum harmonics (50, 100, 150, 200 Hz) or ultrasonic tones.
 */

export interface GoertzelToneResult {
  frequencyHz: number;
  magnitude: number;
  powerDb: number;
  phaseRad: number;
  real: number;
  imag: number;
}

export function goertzelTone(
  signal: ArrayLike<number>,
  sampleRate: number,
  targetFrequencyHz: number
): GoertzelToneResult {
  const N = signal.length;
  if (N === 0) {
    return { frequencyHz: targetFrequencyHz, magnitude: 0, powerDb: -Infinity, phaseRad: 0, real: 0, imag: 0 };
  }

  const k = (N * targetFrequencyHz) / sampleRate;
  const omega = (2.0 * Math.PI * k) / N;
  const cosOmega = Math.cos(omega);
  const sinOmega = Math.sin(omega);
  const coeff = 2.0 * cosOmega;

  let s0 = 0.0;
  let s1 = 0.0;
  let s2 = 0.0;

  for (let i = 0; i < N; i++) {
    s0 = signal[i]! + coeff * s1 - s2;
    s2 = s1;
    s1 = s0;
  }

  // Final complex coefficient
  const real = s1 - s2 * cosOmega;
  const imag = s2 * sinOmega;
  const magSq = (real * real + imag * imag) / (N * N);
  const magnitude = Math.sqrt(magSq);
  const powerDb = 10.0 * Math.log10(Math.max(1e-15, magSq));
  const phaseRad = Math.atan2(imag, real);

  return {
    frequencyHz: targetFrequencyHz,
    magnitude,
    powerDb,
    phaseRad,
    real,
    imag,
  };
}

export { goertzelTone as evaluateGoertzelBin };

