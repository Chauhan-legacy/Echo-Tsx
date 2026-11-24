/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Wavelet Packet Decomposition (WPD) with Coifman-Wickerhauser Best-Basis Selection.
 * Implements Shannon Entropy optimization and Daubechies D4 (db2) orthogonal filter bank.
 *
 * Mathematical Derivation:
 * Let h[n] be the low-pass scaling filter and g[n] = (-1)^n h[L-1-n] be the high-pass wavelet filter.
 * For Daubechies 4 (L=4):
 *   h[0] = (1 + sqrt(3)) / (4*sqrt(2))
 *   h[1] = (3 + sqrt(3)) / (4*sqrt(2))
 *   h[2] = (3 - sqrt(3)) / (4*sqrt(2))
 *   h[3] = (1 - sqrt(3)) / (4*sqrt(2))
 *
 * Perfect reconstruction property:
 *   sum_n h[n] h[n - 2k] = delta[k]
 *   sum_n g[n] g[n - 2k] = delta[k]
 *   sum_n h[n] g[n - 2k] = 0
 *
 * Best-basis selection minimizes additive information cost (Shannon entropy):
 *   E(x) = - sum_i p_i ln(p_i), where p_i = x_i^2 / ||x||_2^2
 * Recursive tree pruning: if E(parent) <= E(left_child) + E(right_child), keep parent node;
 * otherwise split and retain children.
 */

// Daubechies 4 filter coefficients
const SQRT3 = Math.sqrt(3.0);
const DENOM = 4.0 * Math.SQRT2;
export const DB4_H = new Float64Array([
  (1.0 + SQRT3) / DENOM,
  (3.0 + SQRT3) / DENOM,
  (3.0 - SQRT3) / DENOM,
  (1.0 - SQRT3) / DENOM,
]);

// High-pass filter g[n] = (-1)^n * h[3 - n]
export const DB4_G = new Float64Array([
  DB4_H[3]!,
  -DB4_H[2]!,
  DB4_H[1]!,
  -DB4_H[0]!,
]);

export interface WaveletNode {
  depth: number;
  index: number; // node index at this depth (0 to 2^depth - 1)
  coefficients: Float64Array;
  cost: number;
  isLeaf: boolean;
  left?: WaveletNode;
  right?: WaveletNode;
}

/**
 * Compute normalized Shannon entropy cost of a coefficient array:
 * E(x) = - sum (p_i * ln(p_i)) where p_i = x[i]^2 / normSq
 */
export function computeShannonEntropy(coeffs: Float64Array): number {
  let normSq = 0.0;
  const n = coeffs.length;
  for (let i = 0; i < n; i++) {
    normSq += coeffs[i]! * coeffs[i]!;
  }
  if (normSq <= 1e-15) return 0.0;

  let entropy = 0.0;
  for (let i = 0; i < n; i++) {
    const val = coeffs[i]!;
    const p = (val * val) / normSq;
    if (p > 1e-14) {
      entropy -= p * Math.log(p);
    }
  }
  return entropy;
}

/**
 * 1-Level Forward Discrete Wavelet Transform (Periodic boundary conditions)
 * Downsamples by 2.
 */
export function dwtStep(input: Float64Array): { approx: Float64Array; detail: Float64Array } {
  const n = input.length;
  const half = n >> 1;
  const approx = new Float64Array(half);
  const detail = new Float64Array(half);
  const L = DB4_H.length;

  for (let k = 0; k < half; k++) {
    let aSum = 0.0;
    let dSum = 0.0;
    for (let l = 0; l < L; l++) {
      const idx = (2 * k + l) % n;
      const x = input[idx]!;
      aSum += DB4_H[l]! * x;
      dSum += DB4_G[l]! * x;
    }
    approx[k] = aSum;
    detail[k] = dSum;
  }
  return { approx, detail };
}

/**
 * 1-Level Inverse Discrete Wavelet Transform (Periodic boundary conditions)
 * Upsamples by 2 and synthesizes.
 */
export function idwtStep(approx: Float64Array, detail: Float64Array): Float64Array {
  const half = approx.length;
  const n = half << 1;
  const out = new Float64Array(n);
  const L = DB4_H.length;

  for (let i = 0; i < n; i++) {
    let sum = 0.0;
    for (let l = 0; l < L; l++) {
      const p = i - l;
      if (p % 2 === 0) {
        const k = ((p >> 1) % half + half) % half;
        sum += DB4_H[l]! * approx[k]! + DB4_G[l]! * detail[k]!;
      }
    }
    out[i] = sum;
  }
  return out;
}

/**
 * Build a full Wavelet Packet Tree down to maxDepth.
 * Input length must be a multiple of 2^maxDepth.
 */
export function buildWaveletPacketTree(signal: Float64Array, maxDepth: number): WaveletNode {
  function buildNode(data: Float64Array, depth: number, index: number): WaveletNode {
    const cost = computeShannonEntropy(data);
    if (depth >= maxDepth || data.length < 4) {
      return { depth, index, coefficients: data, cost, isLeaf: true };
    }
    const { approx, detail } = dwtStep(data);
    const left = buildNode(approx, depth + 1, 2 * index);
    const right = buildNode(detail, depth + 1, 2 * index + 1);

    return {
      depth,
      index,
      coefficients: data,
      cost,
      isLeaf: false,
      left,
      right,
    };
  }

  return buildNode(signal, 0, 0);
}

/**
 * Coifman-Wickerhauser Best-Basis Selection algorithm:
 * Prunes the tree from bottom up, choosing between node and children based on entropy cost.
 */
export function selectBestBasis(root: WaveletNode): void {
  function prune(node: WaveletNode): void {
    if (node.isLeaf || !node.left || !node.right) {
      node.isLeaf = true;
      return;
    }

    prune(node.left);
    prune(node.right);

    // Sum costs of children
    const childCost = node.left.cost + node.right.cost;
    if (node.cost <= childCost) {
      // Parent has lower or equal entropy -> prune children
      node.isLeaf = true;
      node.left = undefined;
      node.right = undefined;
    } else {
      // Children have lower entropy -> update node cost to sum of children
      node.cost = childCost;
      node.isLeaf = false;
    }
  }

  prune(root);
}

/**
 * Reconstruct signal from a Wavelet Packet Tree.
 * Recursively applies IDWT on leaf nodes.
 */
export function reconstructFromTree(node: WaveletNode): Float64Array {
  if (node.isLeaf || !node.left || !node.right) {
    return node.coefficients;
  }

  const leftData = reconstructFromTree(node.left);
  const rightData = reconstructFromTree(node.right);
  return idwtStep(leftData, rightData);
}
