/**
 * Multi-Head Self-Attention with ALiBi (Attention with Linear Biases - Press et al. 2022).
 * Analytical forward and backward passes with zero external dependencies.
 */
import { PointwiseConv1D } from './layers';

export interface AttentionForwardCache {
  q: Float64Array; // [T, H, Dk]
  k: Float64Array;
  v: Float64Array;
  attnWeights: Float64Array; // [H, T, T]
  context: Float64Array;     // [T, D]
  T: number;
}

export class MultiHeadAttentionALiBi {
  readonly dim: number;
  readonly numHeads: number;
  readonly headDim: number;
  readonly scale: number;
  readonly alibiSlopes: Float64Array;

  // Projections
  qProj: PointwiseConv1D;
  kProj: PointwiseConv1D;
  vProj: PointwiseConv1D;
  outProj: PointwiseConv1D;

  constructor(dim: number, numHeads: number = 4) {
    if (dim % numHeads !== 0) {
      throw new Error(`Embedding dimension ${dim} must be divisible by numHeads ${numHeads}`);
    }
    this.dim = dim;
    this.numHeads = numHeads;
    this.headDim = dim / numHeads;
    this.scale = 1.0 / Math.sqrt(this.headDim);

    this.qProj = new PointwiseConv1D(dim, dim);
    this.kProj = new PointwiseConv1D(dim, dim);
    this.vProj = new PointwiseConv1D(dim, dim);
    this.outProj = new PointwiseConv1D(dim, dim);

    // Compute ALiBi linear penalty slopes: m_h = 2^(-8 * (h+1) / H)
    this.alibiSlopes = new Float64Array(numHeads);
    for (let h = 0; h < numHeads; h++) {
      this.alibiSlopes[h] = Math.pow(2.0, -8.0 * (h + 1) / numHeads);
    }
  }

  forward(x: Float64Array, T: number): { y: Float64Array; cache: AttentionForwardCache } {
    const D = this.dim;
    const H = this.numHeads;
    const Dk = this.headDim;

    // Linear projections: [T, D]
    const qFlat = this.qProj.forward(x, T);
    const kFlat = this.kProj.forward(x, T);
    const vFlat = this.vProj.forward(x, T);

    const q = new Float64Array(T * H * Dk);
    const k = new Float64Array(T * H * Dk);
    const v = new Float64Array(T * H * Dk);

    // Reshape to [T, H, Dk]
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        for (let d = 0; d < Dk; d++) {
          const flatIdx = t * D + h * Dk + d;
          const headIdx = (t * H + h) * Dk + d;
          q[headIdx] = qFlat[flatIdx]!;
          k[headIdx] = kFlat[flatIdx]!;
          v[headIdx] = vFlat[flatIdx]!;
        }
      }
    }

    // Attention weights: [H, T, T]
    const attnWeights = new Float64Array(H * T * T);
    const context = new Float64Array(T * D);

    for (let h = 0; h < H; h++) {
      const slope = this.alibiSlopes[h]!;
      const hAttnOff = h * T * T;

      for (let i = 0; i < T; i++) {
        const qOff = (i * H + h) * Dk;
        const rowOff = hAttnOff + i * T;

        // Compute dot products + ALiBi bias
        let maxScore = -1e9;
        for (let j = 0; j < T; j++) {
          const kOff = (j * H + h) * Dk;
          let dot = 0.0;
          for (let d = 0; d < Dk; d++) {
            dot += q[qOff + d]! * k[kOff + d]!;
          }
          const alibiBias = -slope * Math.abs(i - j);
          const score = dot * this.scale + alibiBias;
          attnWeights[rowOff + j] = score;
          if (score > maxScore) maxScore = score;
        }

        // Softmax over j
        let expSum = 0.0;
        for (let j = 0; j < T; j++) {
          const e = Math.exp(attnWeights[rowOff + j]! - maxScore);
          attnWeights[rowOff + j] = e;
          expSum += e;
        }
        for (let j = 0; j < T; j++) {
          attnWeights[rowOff + j] = (attnWeights[rowOff + j] ?? 0) / Math.max(1e-12, expSum);
        }

        // Context = sum_j attnWeights[i, j] * v[j]
        const ctxOff = i * D + h * Dk;
        for (let d = 0; d < Dk; d++) {
          let sum = 0.0;
          for (let j = 0; j < T; j++) {
            const vOff = (j * H + h) * Dk;
            sum += attnWeights[rowOff + j]! * v[vOff + d]!;
          }
          context[ctxOff + d] = sum;
        }
      }
    }

    // Output projection: [T, D] -> [T, D]
    const y = this.outProj.forward(context, T);

    return {
      y,
      cache: { q, k, v, attnWeights, context, T },
    };
  }

  backward(
    gradOutput: Float64Array,
    x: Float64Array,
    cache: AttentionForwardCache
  ): Float64Array {
    const { q, k, v, attnWeights, context, T } = cache;
    const D = this.dim;
    const H = this.numHeads;
    const Dk = this.headDim;

    // 1. Backprop through outProj
    const gradContext = this.outProj.backward(gradOutput, context, T);

    // 2. Gradients for Q, K, V
    const gradQ = new Float64Array(T * H * Dk);
    const gradK = new Float64Array(T * H * Dk);
    const gradV = new Float64Array(T * H * Dk);

    for (let h = 0; h < H; h++) {
      const hAttnOff = h * T * T;

      for (let i = 0; i < T; i++) {
        const rowOff = hAttnOff + i * T;
        const gcOff = i * D + h * Dk;

        // gradAttn[i, j] = sum_d gradContext[i, h, d] * v[j, h, d]
        const gradAttnRow = new Float64Array(T);
        let sumGradAttnTimesA = 0.0;

        for (let j = 0; j < T; j++) {
          const vOff = (j * H + h) * Dk;
          let dot = 0.0;
          for (let d = 0; d < Dk; d++) {
            dot += gradContext[gcOff + d]! * v[vOff + d]!;
            // gradV[j, h, d] += attnWeights[i, j] * gradContext[i, h, d]
            const vIdx = vOff + d;
            gradV[vIdx] = (gradV[vIdx] ?? 0) + attnWeights[rowOff + j]! * gradContext[gcOff + d]!;
          }
          gradAttnRow[j] = dot;
          sumGradAttnTimesA += dot * attnWeights[rowOff + j]!;
        }

        // Backprop through softmax: dS[i, j] = A[i, j] * (gradAttn[i, j] - sum(gradAttn * A))
        const qOff = (i * H + h) * Dk;
        for (let j = 0; j < T; j++) {
          const aij = attnWeights[rowOff + j]!;
          const dS = aij * (gradAttnRow[j]! - sumGradAttnTimesA);
          const scaledDS = dS * this.scale;
          const kOff = (j * H + h) * Dk;

          for (let d = 0; d < Dk; d++) {
            const qIdx = qOff + d;
            const kIdx = kOff + d;
            gradQ[qIdx] = (gradQ[qIdx] ?? 0) + scaledDS * k[kIdx]!;
            gradK[kIdx] = (gradK[kIdx] ?? 0) + scaledDS * q[qIdx]!;
          }
        }
      }
    }

    // Flatten gradQ, gradK, gradV to [T, D]
    const gradQFlat = new Float64Array(T * D);
    const gradKFlat = new Float64Array(T * D);
    const gradVFlat = new Float64Array(T * D);

    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        for (let d = 0; d < Dk; d++) {
          const flatIdx = t * D + h * Dk + d;
          const headIdx = (t * H + h) * Dk + d;
          gradQFlat[flatIdx] = gradQ[headIdx]!;
          gradKFlat[flatIdx] = gradK[headIdx]!;
          gradVFlat[flatIdx] = gradV[headIdx]!;
        }
      }
    }

    // Backprop through Q, K, V projections
    const gradX_q = this.qProj.backward(gradQFlat, x, T);
    const gradX_k = this.kProj.backward(gradKFlat, x, T);
    const gradX_v = this.vProj.backward(gradVFlat, x, T);

    const gradInput = new Float64Array(T * D);
    for (let i = 0; i < T * D; i++) {
      gradInput[i] = gradX_q[i]! + gradX_k[i]! + gradX_v[i]!;
    }

    return gradInput;
  }
}
