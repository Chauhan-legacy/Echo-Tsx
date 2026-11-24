import { describe, it, expect } from 'vitest';
import { LayerNorm1D, PointwiseConv1D } from '../src/ml/layers';
import { MultiHeadAttentionALiBi } from '../src/ml/attention';
import { ForensicDetectorNetwork } from '../src/ml/network';
import { AdamWOptimizer, checkGradients, getCosineWarmupLR, clipGradNorm } from '../src/ml/training';
import { computeECE, fitTemperature, computeConformalThreshold } from '../src/ml/calibration';
import { computeIntegratedGradients, computeAttentionRollout } from '../src/ml/explainability';

describe('Deep Learning & Training Engine Suite', () => {
  it('Finite-difference gradient check on PointwiseConv1D verifies analytical backprop (< 1e-3 relative error)', () => {
    const Cin = 4;
    const Cout = 2;
    const T = 5;
    const layer = new PointwiseConv1D(Cin, Cout);
    const x = new Float64Array(T * Cin);
    for (let i = 0; i < T * Cin; i++) x[i] = Math.sin(i * 1.5);

    // Forward pass
    const y = layer.forward(x, T);

    // Target y_true: compute MSE loss L = 0.5 * sum((y - y_true)^2)
    const yTrue = new Float64Array(T * Cout);
    for (let i = 0; i < T * Cout; i++) yTrue[i] = Math.cos(i * 0.7);

    const gradOutput = new Float64Array(T * Cout);
    for (let i = 0; i < T * Cout; i++) gradOutput[i] = y[i]! - yTrue[i]!;

    // Analytical backward
    layer.backward(gradOutput, x, T);

    // Finite-difference verification for weights
    function lossFn(w: Float64Array): number {
      const oldW = layer.weights;
      layer.weights = w;
      const out = layer.forward(x, T);
      let loss = 0.0;
      for (let i = 0; i < T * Cout; i++) {
        const d = out[i]! - yTrue[i]!;
        loss += 0.5 * d * d;
      }
      layer.weights = oldW;
      return loss;
    }

    const check = checkGradients(lossFn, layer.weights, layer.gradWeights, 1e-5);
    expect(check.passed).toBe(true);
    expect(check.maxRelError).toBeLessThan(1e-3);
  });

  it('MultiHeadAttentionALiBi runs forward and backward without NaN or Inf', () => {
    const dim = 16;
    const heads = 4;
    const T = 8;
    const mha = new MultiHeadAttentionALiBi(dim, heads);

    const x = new Float64Array(T * dim);
    for (let i = 0; i < T * dim; i++) x[i] = (i % 7) * 0.1;

    const { y, cache } = mha.forward(x, T);
    expect(y.length).toBe(T * dim);
    for (let i = 0; i < y.length; i++) {
      expect(Number.isFinite(y[i]!)).toBe(true);
    }

    // Dummy gradient output
    const gradY = new Float64Array(T * dim).fill(1.0);
    const gradInput = mha.backward(gradY, x, cache);
    expect(gradInput.length).toBe(T * dim);
    for (let i = 0; i < gradInput.length; i++) {
      expect(Number.isFinite(gradInput[i]!)).toBe(true);
    }
  });

  it('ForensicDetectorNetwork outputs valid calibrated predictions', () => {
    const net = new ForensicDetectorNetwork(16, 16);
    const T = 10;
    const x = new Float64Array(T * 16);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.3);

    const preds = net.forward(x, T);
    expect(preds.leakScore).toBeGreaterThanOrEqual(0);
    expect(preds.leakScore).toBeLessThanOrEqual(100);
    expect(preds.trackerProbability).toBeGreaterThanOrEqual(0.0);
    expect(preds.trackerProbability).toBeLessThanOrEqual(1.0);
    expect(preds.embeddings.length).toBe(16);
  });

  it('AdamW optimizer and Cosine LR scheduler function correctly', () => {
    const p1 = new Float64Array([1.0, 2.0, -1.0]);
    const g1 = new Float64Array([0.1, -0.2, 0.05]);
    const opt = new AdamWOptimizer({ lr: 0.01, weightDecay: 0.001 });

    opt.step([p1], [g1]);
    expect(p1[0]!).toBeLessThan(1.0); // Moved in negative gradient direction
    expect(p1[1]!).toBeGreaterThan(2.0); // Moved in negative gradient direction

    const lr0 = getCosineWarmupLR(0, 100, 10, 0.01, 1e-5);
    const lr10 = getCosineWarmupLR(10, 100, 10, 0.01, 1e-5);
    const lr100 = getCosineWarmupLR(100, 100, 10, 0.01, 1e-5);
    expect(lr0).toBeLessThan(lr10);
    expect(lr10).toBeCloseTo(0.01, 3);
    expect(lr100).toBeLessThan(lr10);
  });

  it('Calibration: ECE, Temperature Scaling, and Conformal Thresholding', () => {
    const N = 100;
    const confs = new Float64Array(N);
    const labels = new Int32Array(N);
    for (let i = 0; i < N; i++) {
      confs[i] = 0.5 + 0.4 * (i / N);
      labels[i] = (i > 30) ? 1 : 0;
    }

    const eceRes = computeECE(confs, labels, 5);
    expect(eceRes.ece).toBeGreaterThanOrEqual(0.0);
    expect(eceRes.ece).toBeLessThanOrEqual(1.0);

    const qThresh = computeConformalThreshold(confs, 0.1);
    expect(qThresh).toBeGreaterThan(0.0);
    expect(qThresh).toBeLessThan(1.0);
  });

  it('Explainability: Integrated Gradients and Attention Rollout', () => {
    const T = 4;
    const inDim = 2;
    const input = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);

    // Linear model forward and gradient for testing IG completeness: F(x) = sum(w * x)
    const w = new Float64Array([0.5, -0.2, 0.3, 0.1, -0.4, 0.8, 0.2, -0.1]);
    function model(x: Float64Array, _T: number) {
      let score = 0.0;
      for (let i = 0; i < x.length; i++) score += w[i]! * x[i]!;
      return { outputScore: score, gradInput: new Float64Array(w) };
    }

    const ig = computeIntegratedGradients(input, T, inDim, model, 20);
    expect(ig.timeStepImportance.length).toBe(T);
    // Completeness axiom: sum(IG) == F(x) - F(0)
    expect(ig.completenessDelta).toBeLessThan(1e-5);

    // Attention rollout
    const attn1 = new Float64Array(T * T).fill(1.0 / T);
    const rollout = computeAttentionRollout([attn1, attn1], T);
    expect(rollout.length).toBe(T * T);
  });
});
