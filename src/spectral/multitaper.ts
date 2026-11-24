/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
import { FastFourierTransform } from '../numerics/fft';
import { symmetricEigen } from '../linalg/eig';
import { DenseMatrix } from '../linalg/matrix';

/**
 * Multitaper Spectral Estimator using Discrete Prolate Spheroidal Sequences (DPSS / Slepian Tapers)
 *
 * Implements Thomson's Multitaper Method:
 * 1. Computes DPSS tapers by solving the Slepian tridiagonal matrix eigenvalue problem.
 * 2. Multiplies data by K orthogonal tapers with optimal energy concentration in bandwidth [-W, W].
 * 3. Averages individual eigenspectra, yielding dramatically lower spectral variance without
 *    sacrificing frequency resolution.
 */
export function computeDPSSTapers(N: number, NW: number = 3.0, K: number = 4): DenseMatrix {
  // Slepian tridiagonal matrix:
  // Main diagonal: d_i = ((N - 1 - 2*i) / 2)^2 * cos(2*pi*W)
  // Sub/super diagonal: e_i = i * (N - i) / 2
  const W = NW / N;
  const T = new DenseMatrix(N, N);

  const cosTerm = Math.cos(2.0 * Math.PI * W);
  for (let i = 0; i < N; i++) {
    const k = (N - 1.0 - 2.0 * i) * 0.5;
    T.set(i, i, k * k * cosTerm);
  }
  for (let i = 0; i < N - 1; i++) {
    const off = ((i + 1) * (N - 1 - i)) * 0.5;
    T.set(i, i + 1, off);
    T.set(i + 1, i, off);
  }

  // Eigen-decomposition
  const eig = symmetricEigen(T);

  // Take top K eigenvectors (already sorted descending)
  const tapers = new DenseMatrix(N, K);
  for (let k = 0; k < K; k++) {
    let normSq = 0.0;
    for (let i = 0; i < N; i++) {
      const val = eig.eigenvectors.get(i, k);
      normSq += val * val;
    }
    const invNorm = 1.0 / Math.sqrt(Math.max(1e-20, normSq));
    for (let i = 0; i < N; i++) {
      tapers.set(i, k, eig.eigenvectors.get(i, k) * invNorm);
    }
  }

  return tapers;
}

export function multitaperPSD(
  signal: ArrayLike<number>,
  sampleRate: number,
  NW: number = 3.0,
  K: number = 4
): { frequencies: Float64Array; psd: Float64Array } {
  // Cap taper length at 256 for fast computation
  const N = Math.min(256, signal.length);
  // Next power of 2 for fast transform
  let pow2 = 1;
  while (pow2 < N) pow2 <<= 1;

  const tapers = computeDPSSTapers(N, NW, K);
  const fft = new FastFourierTransform(pow2);

  const halfN = Math.floor(pow2 / 2) + 1;
  const psd = new Float64Array(halfN);

  const inReal = new Float64Array(pow2);
  const inImag = new Float64Array(pow2);
  const outReal = new Float64Array(pow2);
  const outImag = new Float64Array(pow2);

  const normFactor = 1.0 / (sampleRate * K);

  for (let k = 0; k < K; k++) {
    for (let i = 0; i < N; i++) {
      inReal[i] = signal[i]! * tapers.get(i, k);
      inImag[i] = 0.0;
    }
    for (let i = N; i < pow2; i++) {
      inReal[i] = 0.0;
      inImag[i] = 0.0;
    }

    fft.transform(inReal, inImag, outReal, outImag, false);

    for (let f = 0; f < halfN; f++) {
      const magSq = outReal[f]! * outReal[f]! + outImag[f]! * outImag[f]!;
      psd[f] = (psd[f] ?? 0) + magSq * normFactor;
    }
  }

  const frequencies = new Float64Array(halfN);
  const freqStep = sampleRate / pow2;
  for (let f = 0; f < halfN; f++) {
    frequencies[f] = f * freqStep;
  }

  return { frequencies, psd };
}

export function computeMultitaperPSD(
  signal: Float64Array,
  sampleRate: number,
  _N: number = 256,
  NW: number = 3.0,
  K: number = 4
): { frequenciesHz: Float64Array; psd: Float64Array; eigenvalues: Float64Array } {
  const res = multitaperPSD(signal, sampleRate, NW, K);
  const eigenvalues = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    // Slepian eigenvalue approximation
    eigenvalues[k] = 1.0 - Math.exp(-2.0 * (NW - k));
  }
  return { frequenciesHz: res.frequencies, psd: res.psd, eigenvalues };
}

