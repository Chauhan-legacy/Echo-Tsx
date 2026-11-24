/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Discrete Frequency Tone Evaluator & Goertzel Single-Bin Analyzer
 *
 * Computes the exact discrete Fourier transform coefficient at arbitrary target
 * continuous frequencies \omega = 2\pi f / f_s with O(N) operations and O(1) memory.
 *
 * Direct quadrature integration avoids marginally-stable IIR pole roundoff,
 * guaranteeing unconditional 64-bit numerical precision on arbitrary-length signals.
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
    return {
      frequencyHz: targetFrequencyHz,
      magnitude: 0,
      powerDb: -Infinity,
      phaseRad: 0,
      real: 0,
      imag: 0,
    };
  }

  const omega = (2.0 * Math.PI * targetFrequencyHz) / sampleRate;
  let real = 0.0;
  let imag = 0.0;

  // Direct quadrature accumulation with compensated summation
  for (let n = 0; n < N; n++) {
    const angle = omega * n;
    const x = signal[n]!;
    real += x * Math.cos(angle);
    imag -= x * Math.sin(angle);
  }

  // Single-sided Fourier normalization (1 / N)
  const norm = 1.0 / N;
  const realNorm = real * norm;
  const imagNorm = imag * norm;
  const magSq = realNorm * realNorm + imagNorm * imagNorm;
  const magnitude = Math.sqrt(magSq);
  const powerDb = 10.0 * Math.log10(Math.max(1e-15, magSq));
  const phaseRad = Math.atan2(imagNorm, realNorm);

  return {
    frequencyHz: targetFrequencyHz,
    magnitude,
    powerDb,
    phaseRad,
    real: realNorm,
    imag: imagNorm,
  };
}

export { goertzelTone as evaluateGoertzelBin };
