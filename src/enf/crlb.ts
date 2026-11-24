/**
 * Cramér-Rao Lower Bound (CRLB) for Frequency Estimation
 *
 * For a discrete sinusoidal tone in additive white Gaussian noise (AWGN) of length N,
 * sample rate fs, and Signal-to-Noise Ratio (SNR):
 *
 *   Var(f_hat) >= (12 * fs^2) / ( (2*pi)^2 * SNR * N * (N^2 - 1) )
 *   Std(f_hat) >= sqrt(Var(f_hat))
 *
 * This provides the fundamental physical limit of estimation precision.
 */

export function computeFrequencyCRLB(
  numSamples: number,
  sampleRate: number,
  snrLinear: number
): { varianceHz2: number; stdDevHz: number; crlbHz: number } {
  const N = Math.max(3, numSamples);
  const snr = Math.max(1e-6, snrLinear);
  const factor = 12.0 * sampleRate * sampleRate;
  const den = (4.0 * Math.PI * Math.PI) * snr * N * (N * N - 1.0);

  const varianceHz2 = factor / Math.max(1e-20, den);
  const stdDevHz = Math.sqrt(varianceHz2);

  return {
    varianceHz2,
    stdDevHz,
    crlbHz: stdDevHz,
  };
}
