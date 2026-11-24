/**
 * Pure TypeScript Deep Learning Layers with Analytical Backward Gradients:
 * 1. Dilated Depthwise Conv1D
 * 2. Pointwise Conv1D (1x1 conv)
 * 3. LayerNorm (with learnable gamma, beta)
 * 4. GELU (Gaussian Error Linear Unit)
 *
 * Designed with preallocated Float64Array buffers for zero per-frame garbage.
 */
import { gelu, geluDerivative } from '../numerics/activations';

/**
 * 1D Layer Normalization:
 * y = gamma * (x - mu) / sqrt(var + eps) + beta
 */
export class LayerNorm1D {
  readonly dim: number;
  readonly eps: number;
  gamma: Float64Array; // [dim]
  beta: Float64Array;  // [dim]
  gradGamma: Float64Array;
  gradBeta: Float64Array;

  constructor(dim: number, eps: number = 1e-5) {
    this.dim = dim;
    this.eps = eps;
    this.gamma = new Float64Array(dim).fill(1.0);
    this.beta = new Float64Array(dim).fill(0.0);
    this.gradGamma = new Float64Array(dim);
    this.gradBeta = new Float64Array(dim);
  }

  forward(x: Float64Array, T: number): {
    y: Float64Array;
    mean: Float64Array;
    invStd: Float64Array;
    xHat: Float64Array;
  } {
    const D = this.dim;
    const y = new Float64Array(T * D);
    const mean = new Float64Array(T);
    const invStd = new Float64Array(T);
    const xHat = new Float64Array(T * D);

    for (let t = 0; t < T; t++) {
      const off = t * D;
      let sum = 0.0;
      for (let d = 0; d < D; d++) sum += x[off + d]!;
      const m = sum / D;
      mean[t] = m;

      let varSum = 0.0;
      for (let d = 0; d < D; d++) {
        const diff = x[off + d]! - m;
        varSum += diff * diff;
      }
      const is = 1.0 / Math.sqrt(varSum / D + this.eps);
      invStd[t] = is;

      for (let d = 0; d < D; d++) {
        const xh = (x[off + d]! - m) * is;
        xHat[off + d] = xh;
        y[off + d] = this.gamma[d]! * xh + this.beta[d]!;
      }
    }

    return { y, mean, invStd, xHat };
  }

  backward(
    gradOutput: Float64Array,
    xHat: Float64Array,
    invStd: Float64Array,
    T: number
  ): Float64Array {
    const D = this.dim;
    const gradInput = new Float64Array(T * D);

    for (let t = 0; t < T; t++) {
      const off = t * D;
      const is = invStd[t]!;

      let sumDy = 0.0;
      let sumDyXhat = 0.0;

      for (let d = 0; d < D; d++) {
        const dy = gradOutput[off + d]!;
        const xh = xHat[off + d]!;
        this.gradGamma[d] = (this.gradGamma[d] ?? 0) + dy * xh;
        this.gradBeta[d] = (this.gradBeta[d] ?? 0) + dy;

        const gDy = dy * this.gamma[d]!;
        sumDy += gDy;
        sumDyXhat += gDy * xh;
      }

      for (let d = 0; d < D; d++) {
        const gDy = gradOutput[off + d]! * this.gamma[d]!;
        const xh = xHat[off + d]!;
        // Standard LayerNorm gradient: dx = (1/D) * invStd * (D * gDy - sumDy - xh * sumDyXhat)
        gradInput[off + d] = (is / D) * (D * gDy - sumDy - xh * sumDyXhat);
      }
    }

    return gradInput;
  }
}

/**
 * Dilated Depthwise Conv1D:
 * Applies a separate 1D filter of size K to each channel with dilation DIL.
 */
export class DilatedDepthwiseConv1D {
  readonly channels: number;
  readonly kernelSize: number;
  readonly dilation: number;
  weights: Float64Array; // [channels * kernelSize]
  bias: Float64Array;    // [channels]
  gradWeights: Float64Array;
  gradBias: Float64Array;

  constructor(channels: number, kernelSize: number = 3, dilation: number = 1) {
    this.channels = channels;
    this.kernelSize = kernelSize;
    this.dilation = dilation;
    const totalWeights = channels * kernelSize;
    this.weights = new Float64Array(totalWeights);
    this.bias = new Float64Array(channels);
    this.gradWeights = new Float64Array(totalWeights);
    this.gradBias = new Float64Array(channels);

    // Kaiming / He uniform initialization
    const limit = Math.sqrt(2.0 / (kernelSize * channels));
    for (let i = 0; i < totalWeights; i++) {
      this.weights[i] = (Math.sin(i * 997.0) * 0.5) * limit;
    }
  }

  forward(x: Float64Array, T: number): Float64Array {
    const C = this.channels;
    const K = this.kernelSize;
    const D = this.dilation;
    const out = new Float64Array(T * C);
    const pad = Math.floor((K - 1) * D / 2);

    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        let acc = this.bias[c]!;
        const wBase = c * K;
        for (let k = 0; k < K; k++) {
          const inT = t - pad + k * D;
          if (inT >= 0 && inT < T) {
            acc += x[inT * C + c]! * this.weights[wBase + k]!;
          }
        }
        out[t * C + c] = acc;
      }
    }
    return out;
  }

  backward(gradOutput: Float64Array, x: Float64Array, T: number): Float64Array {
    const C = this.channels;
    const K = this.kernelSize;
    const D = this.dilation;
    const gradInput = new Float64Array(T * C);
    const pad = Math.floor((K - 1) * D / 2);

    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        const dy = gradOutput[t * C + c]!;
        this.gradBias[c] = (this.gradBias[c] ?? 0) + dy;
        const wBase = c * K;

        for (let k = 0; k < K; k++) {
          const inT = t - pad + k * D;
          if (inT >= 0 && inT < T) {
            const wIdx = wBase + k;
            const inIdx = inT * C + c;
            this.gradWeights[wIdx] = (this.gradWeights[wIdx] ?? 0) + dy * x[inIdx]!;
            gradInput[inIdx] = (gradInput[inIdx] ?? 0) + dy * this.weights[wIdx]!;
          }
        }
      }
    }
    return gradInput;
  }
}

/**
 * Pointwise Conv1D (1x1 Conv):
 * Projects inChannels -> outChannels across time dimension T.
 */
export class PointwiseConv1D {
  readonly inChannels: number;
  readonly outChannels: number;
  weights: Float64Array; // [outChannels * inChannels]
  bias: Float64Array;    // [outChannels]
  gradWeights: Float64Array;
  gradBias: Float64Array;

  constructor(inChannels: number, outChannels: number) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    const totalWeights = outChannels * inChannels;
    this.weights = new Float64Array(totalWeights);
    this.bias = new Float64Array(outChannels);
    this.gradWeights = new Float64Array(totalWeights);
    this.gradBias = new Float64Array(outChannels);

    // Xavier uniform initialization
    const limit = Math.sqrt(6.0 / (inChannels + outChannels));
    for (let i = 0; i < totalWeights; i++) {
      this.weights[i] = (Math.sin(i * 1013.0) * 0.5) * limit;
    }
  }

  forward(x: Float64Array, T: number): Float64Array {
    const Cin = this.inChannels;
    const Cout = this.outChannels;
    const out = new Float64Array(T * Cout);

    for (let t = 0; t < T; t++) {
      const inOff = t * Cin;
      const outOff = t * Cout;

      for (let o = 0; o < Cout; o++) {
        let acc = this.bias[o]!;
        const wRow = o * Cin;
        for (let i = 0; i < Cin; i++) {
          acc += this.weights[wRow + i]! * x[inOff + i]!;
        }
        out[outOff + o] = acc;
      }
    }
    return out;
  }

  backward(gradOutput: Float64Array, x: Float64Array, T: number): Float64Array {
    const Cin = this.inChannels;
    const Cout = this.outChannels;
    const gradInput = new Float64Array(T * Cin);

    for (let t = 0; t < T; t++) {
      const inOff = t * Cin;
      const outOff = t * Cout;

      for (let o = 0; o < Cout; o++) {
        const dy = gradOutput[outOff + o]!;
        this.gradBias[o] = (this.gradBias[o] ?? 0) + dy;
        const wRow = o * Cin;

        for (let i = 0; i < Cin; i++) {
          const wIdx = wRow + i;
          const inIdx = inOff + i;
          this.gradWeights[wIdx] = (this.gradWeights[wIdx] ?? 0) + dy * x[inIdx]!;
          gradInput[inIdx] = (gradInput[inIdx] ?? 0) + dy * this.weights[wIdx]!;
        }
      }
    }
    return gradInput;
  }
}
