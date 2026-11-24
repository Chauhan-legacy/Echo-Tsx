/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
import { describe, it, expect } from 'vitest';
import { Xoshiro128p } from '../src/numerics/prng';
import { kahanSum, kahanDot } from '../src/numerics/kahan';
import { ArbitraryLengthFFT, naiveDFT } from '../src/numerics/fft';
import { gelu, geluDerivative, sigmoid, stableLog1p, stableExpm1 } from '../src/numerics/activations';

describe('Numerics & Precision', () => {
  it('Xoshiro128p PRNG is deterministic and produces valid Gaussian distributions', () => {
    const rng1 = new Xoshiro128p(42);
    const rng2 = new Xoshiro128p(42);

    const s1 = [rng1.nextFloat(), rng1.nextFloat(), rng1.nextFloat()];
    const s2 = [rng2.nextFloat(), rng2.nextFloat(), rng2.nextFloat()];
    expect(s1).toEqual(s2);

    // Test Gaussian statistics
    const N = 5000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < N; i++) {
      const g = rng1.nextGaussian(0, 1.0);
      sum += g;
      sumSq += g * g;
    }
    const mean = sum / N;
    const variance = sumSq / N - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.08);
    expect(Math.abs(variance - 1.0)).toBeLessThan(0.08);
  });

  it('Kahan compensated summation avoids floating-point precision loss', () => {
    const N = 10000;
    const values = new Float64Array(N + 1);
    values[0] = 1e8;
    for (let i = 1; i <= N; i++) {
      values[i] = 1e-8;
    }

    // Naive sum loses precision
    let naiveSum = 0;
    for (let i = 0; i <= N; i++) naiveSum += values[i]!;

    // Kahan sum preserves precision
    const preciseSum = kahanSum(values);
    const expected = 1e8 + N * 1e-8;

    expect(Math.abs(preciseSum - expected)).toBeLessThan(1e-12);
  });

  it('Arbitrary-length FFT (Power-of-2 and Non-Power-of-2 Bluestein) matches Naive DFT (< 1e-10 error)', () => {
    const testSizes = [16, 17, 33, 64, 100];

    for (const N of testSizes) {
      const fft = new ArbitraryLengthFFT(N);
      const realIn = new Float64Array(N);
      const imagIn = new Float64Array(N);

      // Populate deterministic sinusoidal signal
      for (let i = 0; i < N; i++) {
        realIn[i] = Math.sin((2 * Math.PI * 3 * i) / N) + 0.5 * Math.cos((2 * Math.PI * 7 * i) / N);
        imagIn[i] = 0.0;
      }

      // Arbitrary-length FFT
      const { realOut: fReal, imagOut: fImag } = fft.forward(realIn, imagIn);
      // Naive DFT reference
      const { realOut: dReal, imagOut: dImag } = naiveDFT(realIn, imagIn);

      let maxDiff = 0.0;
      for (let i = 0; i < N; i++) {
        const diffR = Math.abs(fReal[i]! - dReal[i]!);
        const diffI = Math.abs(fImag[i]! - dImag[i]!);
        if (diffR > maxDiff) maxDiff = diffR;
        if (diffI > maxDiff) maxDiff = diffI;
      }

      expect(maxDiff).toBeLessThan(1e-10);

      // Inverse transform roundtrip test
      const { realOut: invReal } = fft.inverse(fReal, fImag);
      let invMaxDiff = 0.0;
      for (let i = 0; i < N; i++) {
        const diff = Math.abs(invReal[i]! - realIn[i]!);
        if (diff > invMaxDiff) invMaxDiff = diff;
      }
      expect(invMaxDiff).toBeLessThan(1e-10);
    }
  });

  it('GELU activation and analytical derivative match numerical differentiation', () => {
    const x = 0.75;
    const h = 1e-6;
    const numDeriv = (gelu(x + h) - gelu(x - h)) / (2 * h);
    const anaDeriv = geluDerivative(x);
    expect(Math.abs(numDeriv - anaDeriv)).toBeLessThan(1e-5);
  });
});
