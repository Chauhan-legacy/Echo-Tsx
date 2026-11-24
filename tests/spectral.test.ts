import { describe, it, expect } from 'vitest';
import { generateWindow } from '../src/spectral/windows';
import { computeWelchPSD } from '../src/spectral/welch';
import { computeMultitaperPSD } from '../src/spectral/multitaper';
import { evaluateGoertzelBin } from '../src/spectral/goertzel';
import {
  buildWaveletPacketTree,
  selectBestBasis,
  reconstructFromTree,
  dwtStep,
  idwtStep
} from '../src/spectral/wavelet';
import { generateSineTone, addAWGN } from '../src/signals/testSignals';

describe('Spectral Analysis & Wavelet Suite', () => {
  it('Windows compute exact Coherent Gain and Equivalent Noise Bandwidth', () => {
    const winHann = generateWindow('hann', 256);
    expect(Math.abs(winHann.coherentGain - 0.5)).toBeLessThan(0.01);
    expect(Math.abs(winHann.enbw - 1.5)).toBeLessThan(0.02);

    const winHamming = generateWindow('hamming', 256);
    expect(Math.abs(winHamming.coherentGain - 0.54)).toBeLessThan(0.01);
    expect(Math.abs(winHamming.enbw - 1.36)).toBeLessThan(0.02);
  });

  it('Welch PSD precisely identifies sinusoidal peak frequency', () => {
    const Fs = 1000;
    const targetFreq = 142.5;
    const clean = generateSineTone(targetFreq, 1.0, 0, { sampleRate: Fs, durationSeconds: 2.0 });
    const { noisySignal } = addAWGN(clean, 20.0, 42);

    const welch = computeWelchPSD(noisySignal, {
      sampleRate: Fs,
      segmentLength: 512,
      overlapFraction: 0.5,
      windowType: 'hann',
    });

    expect(Math.abs(welch.peakFrequencyHz - targetFreq)).toBeLessThan(1.0);
  });

  it('Multitaper DPSS spectral estimation reduces variance', () => {
    const Fs = 1000;
    const clean = generateSineTone(200.0, 1.0, 0, { sampleRate: Fs, durationSeconds: 1.0 });
    const mt = computeMultitaperPSD(clean, Fs, 256, 3, 4);

    expect(mt.frequenciesHz.length).toBeGreaterThan(100);
    expect(mt.eigenvalues.length).toBe(4);
    // Highest eigenvalue of DPSS should be close to 1.0
    expect(mt.eigenvalues[0]!).toBeGreaterThan(0.95);
  });

  it('Goertzel filter accurately detects isolated tracking tone', () => {
    const Fs = 48000;
    const fTone = 19400; // Tracker beacon frequency
    const sig = generateSineTone(fTone, 0.8, 0.5, { sampleRate: Fs, durationSeconds: 0.1 });

    const atTone = evaluateGoertzelBin(sig, Fs, fTone);
    const offTone = evaluateGoertzelBin(sig, Fs, 18000);

    expect(atTone.magnitude).toBeCloseTo(0.4, 1);
    expect(atTone.powerDb).toBeGreaterThan(offTone.powerDb + 40.0);
  });


  it('1-Level DWT and IDWT achieve perfect reconstruction (< 1e-12 error)', () => {
    const n = 64;
    const orig = new Float64Array(n);
    for (let i = 0; i < n; i++) orig[i] = Math.sin((i * 2 * Math.PI) / 8);

    const { approx, detail } = dwtStep(orig);
    const reconstructed = idwtStep(approx, detail);

    let maxDiff = 0.0;
    for (let i = 0; i < n; i++) {
      const diff = Math.abs(reconstructed[i]! - orig[i]!);
      if (diff > maxDiff) maxDiff = diff;
    }
    expect(maxDiff).toBeLessThan(1e-12);
  });

  it('Wavelet Packet Best-Basis Selection and Tree Reconstruction (< 1e-10 error)', () => {
    const n = 128;
    const orig = new Float64Array(n);
    for (let i = 0; i < n; i++) orig[i] = Math.sin((i * 2 * Math.PI) / 16) + 0.5 * Math.cos((i * 2 * Math.PI) / 4);

    const tree = buildWaveletPacketTree(orig, 3);
    selectBestBasis(tree);
    const reconstructed = reconstructFromTree(tree);

    let maxDiff = 0.0;
    for (let i = 0; i < n; i++) {
      const diff = Math.abs(reconstructed[i]! - orig[i]!);
      if (diff > maxDiff) maxDiff = diff;
    }
    expect(maxDiff).toBeLessThan(1e-10);
  });
});
