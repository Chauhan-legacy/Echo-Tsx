/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
import { FastFourierTransform } from '../numerics/fft';
import { computeWindow, WindowType } from './windows';

export interface WelchPSDResult {
  frequencies: Float64Array; // Frequency axis in Hz (0 to fs/2)
  psd: Float64Array; // Power Spectral Density (V^2 / Hz)
  peakFrequencyHz: number;
  peakPower: number;
}

/**
 * Welch's Averaged Periodogram Method
 *
 * Divides time-series into overlapping windowed segments, computes FFTs,
 * and averages power spectral densities to minimize variance by factor 1/M.
 */
export function welchPSD(
  signal: ArrayLike<number>,
  sampleRate: number,
  options?: {
    windowSize?: number;
    overlapFraction?: number; // 0.5 default (50% overlap)
    windowType?: 'hann' | 'hamming' | 'blackman-harris' | 'kaiser' | 'flat-top';
  }
): WelchPSDResult {
  const n = signal.length;
  const winSize = options?.windowSize ?? Math.min(2048, Math.max(256, 1 << Math.floor(Math.log2(n))));
  const overlap = options?.overlapFraction ?? 0.5;
  const winType = options?.windowType ?? 'hann';

  const hop = Math.max(1, Math.floor(winSize * (1.0 - overlap)));
  const numSegments = Math.max(1, Math.floor((n - winSize) / hop) + 1);

  const { window: winData, coherentGain } = computeWindow(winType, winSize);

  // Normalization factor for power spectral density: S2 = sum(w^2)
  let s2 = 0.0;
  for (let i = 0; i < winSize; i++) {
    s2 += winData[i]! * winData[i]!;
  }
  const normFactor = 1.0 / (sampleRate * s2);

  const halfN = Math.floor(winSize / 2) + 1;
  const psdAccum = new Float64Array(halfN);

  const fft = new FastFourierTransform(winSize);
  const inReal = new Float64Array(winSize);
  const inImag = new Float64Array(winSize);
  const outReal = new Float64Array(winSize);
  const outImag = new Float64Array(winSize);

  for (let m = 0; m < numSegments; m++) {
    const offset = m * hop;
    for (let i = 0; i < winSize; i++) {
      inReal[i] = signal[offset + i]! * winData[i]!;
      inImag[i] = 0.0;
    }

    fft.transform(inReal, inImag, outReal, outImag, false);

    // One-sided power spectrum accumulation
    psdAccum[0] = (psdAccum[0] ?? 0) + (outReal[0]! * outReal[0]! + outImag[0]! * outImag[0]!) * normFactor;
    for (let k = 1; k < halfN - 1; k++) {
      const magSq = outReal[k]! * outReal[k]! + outImag[k]! * outImag[k]!;
      psdAccum[k] = (psdAccum[k] ?? 0) + 2.0 * magSq * normFactor; // Factor of 2 for one-sided conservation
    }
    const lastK = halfN - 1;
    psdAccum[lastK] = (psdAccum[lastK] ?? 0) + (outReal[lastK]! * outReal[lastK]! + outImag[lastK]! * outImag[lastK]!) * normFactor;
  }

  const invM = 1.0 / numSegments;
  const frequencies = new Float64Array(halfN);
  const psd = new Float64Array(halfN);
  const freqStep = sampleRate / winSize;

  let peakPower = 0.0;
  let peakIdx = 0;

  for (let k = 0; k < halfN; k++) {
    const p = psdAccum[k]! * invM;
    psd[k] = p;
    frequencies[k] = k * freqStep;
    if (p > peakPower) {
      peakPower = p;
      peakIdx = k;
    }
  }

  // Refine peak with 3-point parabolic peak interpolation
  let peakFrequencyHz = peakIdx * freqStep;
  if (peakIdx > 0 && peakIdx < halfN - 1) {
    const alpha = psd[peakIdx - 1]!;
    const beta = psd[peakIdx]!;
    const gamma = psd[peakIdx + 1]!;
    const delta = (0.5 * (alpha - gamma)) / (alpha - 2.0 * beta + gamma + 1e-20);
    peakFrequencyHz = (peakIdx + delta) * freqStep;
  }

  return {
    frequencies,
    psd,
    peakFrequencyHz,
    peakPower,
  };
}

export function computeWelchPSD(
  signal: Float64Array,
  options: {
    sampleRate: number;
    segmentLength?: number;
    overlapFraction?: number;
    windowType?: WindowType;
  }
): {
  frequenciesHz: Float64Array;
  psd: Float64Array;
  peakFrequencyHz: number;
  peakPower: number;
} {
  const res = welchPSD(signal, options.sampleRate, {
    windowSize: options.segmentLength,
    overlapFraction: options.overlapFraction,
    windowType: options.windowType,
  });
  return {
    frequenciesHz: res.frequencies,
    psd: res.psd,
    peakFrequencyHz: res.peakFrequencyHz,
    peakPower: res.peakPower,
  };
}

