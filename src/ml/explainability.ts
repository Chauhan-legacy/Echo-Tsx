/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Transparent Forensic Explainability:
 * 1. Integrated Gradients (Sundararajan, Taly, Yan 2017)
 *    Computes path integral of gradients along straight-line trajectory from baseline:
 *      IG_i(x) = (x_i - x'_i) * \frac{1}{m} \sum_{k=1}^m \frac{\partial F(x' + \frac{k}{m}(x - x'))}{\partial x_i}
 * 2. Attention Rollout (Abnar & Zuidema 2020)
 *    Recursively propagates self-attention across transformer layers:
 *      \hat{A} = 0.5 * (A + I)
 *      R^{(l)} = \hat{A}^{(l)} * R^{(l-1)}
 */

export interface AttributionResult {
  attributions: Float64Array; // [T, inDim] feature attribution map
  timeStepImportance: Float64Array; // [T] 1D timeline importance
  completenessDelta: number; // Difference between sum(IG) and (F(x) - F(baseline))
}

/**
 * Compute Integrated Gradients for time series forensic inputs.
 */
export function computeIntegratedGradients(
  input: Float64Array,
  T: number,
  inDim: number,
  modelForwardGrad: (x: Float64Array, T: number) => { outputScore: number; gradInput: Float64Array },
  steps: number = 20
): AttributionResult {
  const totalFeatures = T * inDim;
  const integratedGrads = new Float64Array(totalFeatures);

  // Baseline x' is zero signal (clean silence / neutral background)
  const baseline = new Float64Array(totalFeatures);
  const baselineRes = modelForwardGrad(baseline, T);
  const targetRes = modelForwardGrad(input, T);
  const expectedDifference = targetRes.outputScore - baselineRes.outputScore;

  // Riemann summation over m steps: alpha_k = k / m
  for (let k = 1; k <= steps; k++) {
    const alpha = k / steps;
    const interpolated = new Float64Array(totalFeatures);
    for (let i = 0; i < totalFeatures; i++) {
      interpolated[i] = baseline[i]! + alpha * (input[i]! - baseline[i]!);
    }

    const { gradInput } = modelForwardGrad(interpolated, T);
    for (let i = 0; i < totalFeatures; i++) {
      integratedGrads[i] = (integratedGrads[i] ?? 0) + gradInput[i]!;
    }
  }

  // Multiply by (x - x') / steps
  const attributions = new Float64Array(totalFeatures);
  let igSum = 0.0;
  for (let i = 0; i < totalFeatures; i++) {
    const diff = input[i]! - baseline[i]!;
    const val = (diff * integratedGrads[i]!) / steps;
    attributions[i] = val;
    igSum += val;
  }

  // 1D timeline importance: sum absolute attributions across channels at each timestep t
  const timeStepImportance = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    let tSum = 0.0;
    const off = t * inDim;
    for (let d = 0; d < inDim; d++) {
      tSum += Math.abs(attributions[off + d]!);
    }
    timeStepImportance[t] = tSum;
  }

  const completenessDelta = Math.abs(igSum - expectedDifference);

  return {
    attributions,
    timeStepImportance,
    completenessDelta,
  };
}

/**
 * Attention Rollout across L transformer layers (Abnar & Zuidema 2020).
 * Input: array of attention weight matrices, each [T, T] averaged over heads.
 */
export function computeAttentionRollout(layerAttentions: Float64Array[], T: number): Float64Array {
  const L = layerAttentions.length;
  if (L === 0) return new Float64Array(T * T);

  // Initialize rollout with identity matrix
  let rollout = new Float64Array(T * T);
  for (let i = 0; i < T; i++) rollout[i * T + i] = 1.0;

  for (let l = 0; l < L; l++) {
    const attn = layerAttentions[l]!;
    // A_hat = 0.5 * (A + I) to account for residual connections
    const aHat = new Float64Array(T * T);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        const idVal = (i === j) ? 1.0 : 0.0;
        aHat[i * T + j] = 0.5 * (attn[i * T + j]! + idVal);
      }
      // Re-normalize row
      let rowSum = 0.0;
      for (let j = 0; j < T; j++) rowSum += aHat[i * T + j]!;
      for (let j = 0; j < T; j++) aHat[i * T + j] = (aHat[i * T + j] ?? 0) / Math.max(1e-12, rowSum);
    }

    // rollout = aHat * rollout
    const nextRollout = new Float64Array(T * T);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        let sum = 0.0;
        for (let k = 0; k < T; k++) {
          sum += aHat[i * T + k]! * rollout[k * T + j]!;
        }
        nextRollout[i * T + j] = sum;
      }
    }
    rollout = nextRollout;
  }

  return rollout;
}
