/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
import { describe, it, expect } from 'vitest';
import { Xoshiro128p } from '../src/numerics/prng';
import { generateSyntheticRIR, generateSyntheticENFTrack, generateSineTone, addAWGN } from '../src/signals/testSignals';
import { estimateRT60 } from '../src/forensic/rt60';
import { detectCUSUM, detectBOCPD } from '../src/forensic/changepoint';
import { extractNoiseResidual2D, computePRNUPCE } from '../src/forensic/prnu';
import { extractRPPG } from '../src/forensic/rppg';
import { estimateYINPitch, analyzeLPC } from '../src/forensic/voice';
import { computeMUSIC, computeESPRIT } from '../src/forensic/doa';
import { runFastICA } from '../src/forensic/ica';
import { fuseForensicEvidence } from '../src/forensic/bayesFusion';
import {
  medianAbsoluteDeviation,
  huberLocation,
  theilSenSlope,
  mannWhitneyUTest,
  benjaminiHochberg
} from '../src/forensic/robustStats';
import { inpaintSpectralNotch } from '../src/privacy/spectralInpainting';
import { DenseMatrix } from '../src/linalg/matrix';
import { evaluateGoertzelBin } from '../src/spectral/goertzel';

describe('Forensic Science Algorithms Suite', () => {
  it('Schroeder / Lundeby RT60 estimates acoustic reverberation decay', () => {
    const targetRT60 = 0.5; // 500 ms decay
    const rir = generateSyntheticRIR(targetRT60, 16000, 15.0, 777);
    const res = estimateRT60(rir, 16000);

    expect(res.t20Seconds).toBeGreaterThan(0.35);
    expect(res.t20Seconds).toBeLessThan(0.65);
    expect(res.rSquared).toBeGreaterThan(0.90);
  });

  it('CUSUM and BOCPD detect ENF grid splice discontinuities', () => {
    const { frequency, spliceIndex } = generateSyntheticENFTrack(50.0, 60.0, 30.0, 0.08, 0.003, 101);

    const cusum = detectCUSUM(frequency, { expectedShiftDelta: 0.04, thresholdH: 0.15 });
    expect(cusum.changePoints.length).toBeGreaterThan(0);
    // Detected within +/- 3 seconds of the actual splice at t = 30
    const detectedDiff = Math.abs(cusum.changePoints[0]! - spliceIndex);
    expect(detectedDiff).toBeLessThanOrEqual(3);

    const bocpd = detectBOCPD(frequency, 50.0);
    expect(bocpd.changePoints.length).toBeGreaterThan(0);
  });

  it('PRNU sensor noise residual and Peak-to-Correlation Energy (PCE)', () => {
    const W = 32;
    const H = 32;
    const rng = new Xoshiro128p(99);
    const sensorPattern = new Float64Array(W * H);
    // Synthetic I.I.D. sensor PRNU pattern K
    for (let i = 0; i < W * H; i++) sensorPattern[i] = rng.nextGaussian(0, 1.0);

    // Image 1 from sensor: flat scene (128) + sensor pattern + small additive noise
    const img1 = new Float64Array(W * H);
    for (let i = 0; i < W * H; i++) {
      img1[i] = 128.0 + sensorPattern[i]! + rng.nextGaussian(0, 0.1);
    }

    const res1 = extractNoiseResidual2D(img1, W, H, 1);
    const match = computePRNUPCE(res1, sensorPattern, W, H, 4);

    expect(match.pce).toBeGreaterThan(60.0);
    expect(match.isMatch).toBe(true);
    expect(match.peakOffset[0]).toBe(0);
    expect(match.peakOffset[1]).toBe(0);
  });


  it('rPPG POS and CHROM extract physiological pulse from facial RGB traces', () => {
    const Fs = 30; // 30 fps video
    const T = 120; // 4 seconds
    const targetBpm = 72; // 1.2 Hz
    const fHeart = targetBpm / 60.0;

    const r = new Float64Array(T);
    const g = new Float64Array(T);
    const b = new Float64Array(T);

    for (let i = 0; i < T; i++) {
      const t = i / Fs;
      const pulse = 0.02 * Math.sin(2 * Math.PI * fHeart * t);
      // Skin tone DC bias + subtle hemoglobin absorption in G channel
      r[i] = 200.0 + pulse * 0.5;
      g[i] = 140.0 + pulse * 2.0;
      b[i] = 120.0 + pulse * 0.2;
    }

    const posRes = extractRPPG(r, g, b, Fs, 'POS');
    expect(Math.abs(posRes.heartRateBpm - targetBpm)).toBeLessThan(5.0);
    expect(posRes.snrDb).toBeGreaterThan(3.0);
  });

  it('Voice: YIN fundamental frequency and LPC formants', () => {
    const Fs = 16000;
    const f0 = 150.0; // 150 Hz voice pitch
    const sig = generateSineTone(f0, 0.8, 0, { sampleRate: Fs, durationSeconds: 0.1 });

    const yin = estimateYINPitch(sig, Fs);
    expect(Math.abs(yin.pitchHz - f0)).toBeLessThan(2.0);
    expect(yin.periodicity).toBeGreaterThan(0.95);

    const lpc = analyzeLPC(sig.subarray(0, 512), 10, Fs);
    expect(lpc.coefficients.length).toBe(11);
    expect(lpc.formants.length).toBeGreaterThan(0);
  });

  it('DOA: MUSIC and ESPRIT estimate acoustic source angles', () => {
    // 4-element ULA covariance with source at +15 degrees
    const M = 4;
    const thetaRad = (15.0 * Math.PI) / 180.0;
    const phaseStep = Math.PI * Math.sin(thetaRad);

    const cov = new DenseMatrix(M, M);
    for (let r = 0; r < M; r++) {
      for (let c = 0; c < M; c++) {
        const d = (c - r) * phaseStep;
        cov.set(r, c, Math.cos(d) + (r === c ? 0.05 : 0.0));
      }
    }

    const music = computeMUSIC(cov, 1, 100, 1.0);
    expect(music.spectrum.length).toBeGreaterThan(10);

    const esprit = computeESPRIT(cov, 1);
    expect(esprit.length).toBe(1);
  });

  it('FastICA successfully separates mixed signals', () => {
    const T = 500;
    const s1 = new Float64Array(T);
    const s2 = new Float64Array(T);
    for (let i = 0; i < T; i++) {
      s1[i] = Math.sin((2 * Math.PI * 5 * i) / T);
      s2[i] = ((i % 50) > 25 ? 1.0 : -1.0); // Square wave
    }

    // Linear mixture: x1 = 0.8*s1 + 0.4*s2; x2 = 0.3*s1 + 0.9*s2
    const x1 = new Float64Array(T);
    const x2 = new Float64Array(T);
    for (let i = 0; i < T; i++) {
      x1[i] = 0.8 * s1[i]! + 0.4 * s2[i]!;
      x2[i] = 0.3 * s1[i]! + 0.9 * s2[i]!;
    }

    const ica = runFastICA([x1, x2], { numComponents: 2, maxIterations: 50 });
    expect(ica.unmixedSignals.length).toBe(2);
    expect(ica.converged).toBe(true);
  });

  it('Bayesian evidence fusion calculates calibrated posterior log-odds and verbal equivalent', () => {
    const evidence = [
      { id: 'tracker', name: 'Ultrasonic Tracker', likelihoodRatio: 250.0, confidence: 0.95 },
      { id: 'hum', name: 'ENF Grid Discontinuity', likelihoodRatio: 45.0, confidence: 0.90 },
      { id: 'prnu', name: 'Camera PRNU Mismatch', likelihoodRatio: 12.0, confidence: 0.80 },
    ];

    const fusion = fuseForensicEvidence(evidence, 0.5);
    expect(fusion.posteriorProbability).toBeGreaterThan(0.99);
    expect(fusion.bayesFactor).toBeGreaterThan(1000.0);
    expect(fusion.verbalConclusion).toBe('Extremely Strong Support');
    expect(fusion.contributions.length).toBe(3);
  });

  it('Robust statistics: Huber M-estimator and Theil-Sen slope resist heavy outliers', () => {
    // Normal data + heavy outlier corruptions
    const cleanData = new Float64Array([10, 11, 9, 10.5, 9.5, 10, 10.2, 9.8, 1000, -500]);
    const huber = huberLocation(cleanData);
    expect(Math.abs(huber - 10.0)).toBeLessThan(0.8);

    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 1000, 16]); // y = 2x with outlier at x=7
    const ts = theilSenSlope(x, y);
    expect(Math.abs(ts.slope - 2.0)).toBeLessThan(0.1);
  });

  it('Spectral Inpainting removes tracker tone and fills with matched noise floor', () => {
    const Fs = 48000;
    const fTone = 19400;
    const sig = generateSineTone(fTone, 1.0, 0, { sampleRate: Fs, durationSeconds: 0.1 });

    const beforePwr = evaluateGoertzelBin(sig, Fs, fTone).powerDb;
    const inpainted = inpaintSpectralNotch(sig, {
      sampleRate: Fs,
      centerFrequencyHz: fTone,
      bandwidthHz: 200,
    });
    const afterPwr = evaluateGoertzelBin(inpainted, Fs, fTone).powerDb;

    // Attenuated by > 20 dB
    expect(beforePwr - afterPwr).toBeGreaterThan(20.0);
  });
});
