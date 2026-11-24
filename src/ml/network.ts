/**
 * Pre-LayerNorm Forensic Neural Network Architecture:
 * Stacks Dilated Convolutions (receptive field expansion) + Multi-Head Self-Attention (ALiBi)
 * with multi-task forensic classification heads (tracker detection, hum detection, tampering probability).
 */
import { DilatedDepthwiseConv1D, PointwiseConv1D, LayerNorm1D } from './layers';
import { MultiHeadAttentionALiBi, AttentionForwardCache } from './attention';
import { gelu, geluDerivative } from '../numerics/activations';

export interface ForensicPredictions {
  leakScore: number;          // 0 to 100
  trackerProbability: number; // 0 to 1
  humProbability: number;     // 0 to 1
  spliceProbability: number;  // 0 to 1
  embeddings: Float64Array;   // Pooled latent representation
}

export class ForensicTransformerBlock {
  ln1: LayerNorm1D;
  attn: MultiHeadAttentionALiBi;
  ln2: LayerNorm1D;
  conv1: DilatedDepthwiseConv1D;
  pw1: PointwiseConv1D;
  pw2: PointwiseConv1D;
  readonly dim: number;

  constructor(dim: number, dilation: number = 1, numHeads: number = 4) {
    this.dim = dim;
    this.ln1 = new LayerNorm1D(dim);
    this.attn = new MultiHeadAttentionALiBi(dim, numHeads);
    this.ln2 = new LayerNorm1D(dim);
    this.conv1 = new DilatedDepthwiseConv1D(dim, 3, dilation);
    this.pw1 = new PointwiseConv1D(dim, dim * 2);
    this.pw2 = new PointwiseConv1D(dim * 2, dim);
  }

  forward(x: Float64Array, T: number): {
    y: Float64Array;
    cache: {
      ln1Cache: any;
      attnCache: AttentionForwardCache;
      ln2Cache: any;
      xMid: Float64Array;
      convOut: Float64Array;
      ffnMid: Float64Array;
      geluMid: Float64Array;
      T: number;
    };
  } {
    const D = this.dim;

    // 1. Pre-LayerNorm Self-Attention with Residual
    const ln1Res = this.ln1.forward(x, T);
    const { y: attnOut, cache: attnCache } = this.attn.forward(ln1Res.y, T);
    const xMid = new Float64Array(T * D);
    for (let i = 0; i < T * D; i++) {
      xMid[i] = x[i]! + attnOut[i]!;
    }

    // 2. Pre-LayerNorm Dilated Conv + FFN with Residual
    const ln2Res = this.ln2.forward(xMid, T);
    const convOut = this.conv1.forward(ln2Res.y, T);
    const ffnMid = this.pw1.forward(convOut, T);

    const ffnDim = D * 2;
    const geluMid = new Float64Array(T * ffnDim);
    for (let i = 0; i < T * ffnDim; i++) {
      geluMid[i] = gelu(ffnMid[i]!);
    }

    const ffnOut = this.pw2.forward(geluMid, T);
    const y = new Float64Array(T * D);
    for (let i = 0; i < T * D; i++) {
      y[i] = xMid[i]! + ffnOut[i]!;
    }

    return {
      y,
      cache: {
        ln1Cache: ln1Res,
        attnCache,
        ln2Cache: ln2Res,
        xMid,
        convOut,
        ffnMid,
        geluMid,
        T,
      },
    };
  }

  backward(gradOutput: Float64Array, x: Float64Array, cache: any): Float64Array {
    const { ln1Cache, attnCache, ln2Cache, xMid, convOut, ffnMid, geluMid, T } = cache;
    const D = this.dim;
    const ffnDim = D * 2;

    // Grad through FFN
    const gradGelu = this.pw2.backward(gradOutput, geluMid, T);
    const gradFfnMid = new Float64Array(T * ffnDim);
    for (let i = 0; i < T * ffnDim; i++) {
      gradFfnMid[i] = gradGelu[i]! * geluDerivative(ffnMid[i]!);
    }

    const gradConv = this.pw1.backward(gradFfnMid, convOut, T);
    const gradLn2Y = this.conv1.backward(gradConv, ln2Cache.y, T);
    const gradLn2In = this.ln2.backward(gradLn2Y, ln2Cache.xHat, ln2Cache.invStd, T);

    // Sum gradients at residual join: gradMid = gradOutput + gradLn2In
    const gradMid = new Float64Array(T * D);
    for (let i = 0; i < T * D; i++) {
      gradMid[i] = gradOutput[i]! + gradLn2In[i]!;
    }

    // Grad through Attention branch
    const gradAttnY = this.attn.backward(gradMid, ln1Cache.y, attnCache);
    const gradLn1In = this.ln1.backward(gradAttnY, ln1Cache.xHat, ln1Cache.invStd, T);

    // Sum gradients at input residual join: gradInput = gradMid + gradLn1In
    const gradInput = new Float64Array(T * D);
    for (let i = 0; i < T * D; i++) {
      gradInput[i] = gradMid[i]! + gradLn1In[i]!;
    }

    return gradInput;
  }
}

/**
 * Full Forensic Detection Network:
 * Combines Input Projection, Dilated Transformer Blocks, Global Pooling, and Multi-Task Heads.
 */
export class ForensicDetectorNetwork {
  inProj: PointwiseConv1D;
  block1: ForensicTransformerBlock; // dilation = 1
  block2: ForensicTransformerBlock; // dilation = 2
  headTracker: PointwiseConv1D;
  headHum: PointwiseConv1D;
  headSplice: PointwiseConv1D;
  readonly inDim: number;
  readonly hiddenDim: number;

  constructor(inDim: number = 32, hiddenDim: number = 32) {
    this.inDim = inDim;
    this.hiddenDim = hiddenDim;
    this.inProj = new PointwiseConv1D(inDim, hiddenDim);
    this.block1 = new ForensicTransformerBlock(hiddenDim, 1, 4);
    this.block2 = new ForensicTransformerBlock(hiddenDim, 2, 4);

    this.headTracker = new PointwiseConv1D(hiddenDim, 1);
    this.headHum = new PointwiseConv1D(hiddenDim, 1);
    this.headSplice = new PointwiseConv1D(hiddenDim, 1);
  }

  forward(x: Float64Array, T: number): ForensicPredictions {
    const H = this.hiddenDim;
    const xProj = this.inProj.forward(x, T);
    const b1 = this.block1.forward(xProj, T);
    const b2 = this.block2.forward(b1.y, T);

    // Global Average Pooling across time T
    const pooled = new Float64Array(H);
    for (let t = 0; t < T; t++) {
      const off = t * H;
      for (let h = 0; h < H; h++) {
        pooled[h] = (pooled[h] ?? 0) + b2.y[off + h]!;
      }
    }
    for (let h = 0; h < H; h++) pooled[h] = (pooled[h] ?? 0) / T;

    // Multi-task classification logits
    const tOut = this.headTracker.forward(pooled, 1);
    const hOut = this.headHum.forward(pooled, 1);
    const sOut = this.headSplice.forward(pooled, 1);

    const sig = (z: number) => 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, z))));
    const trackerProb = sig(tOut[0]!);
    const humProb = sig(hOut[0]!);
    const spliceProb = sig(sOut[0]!);

    // Calibrated Leak Score 0 - 100
    const rawScore = 40.0 * trackerProb + 30.0 * humProb + 30.0 * spliceProb;
    const leakScore = Math.round(Math.min(100, Math.max(0, rawScore)));

    return {
      leakScore,
      trackerProbability: trackerProb,
      humProbability: humProb,
      spliceProbability: spliceProb,
      embeddings: pooled,
    };
  }
}
