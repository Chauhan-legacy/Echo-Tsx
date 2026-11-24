/**
 * Pure TypeScript Model Training Engine:
 * 1. AdamW Optimizer with Decoupled Weight Decay (Loshchilov & Hutter 2019)
 * 2. Cosine Annealing Learning Rate Schedule with Linear Warmup
 * 3. Global-Norm Gradient Clipping
 * 4. Exponential Moving Average (EMA) of Weights
 * 5. Analytical vs Finite-Difference Gradient Checker
 */

export interface AdamWConfig {
  lr: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
}

export class AdamWOptimizer {
  config: Required<AdamWConfig>;
  m: Map<Float64Array, Float64Array>;
  v: Map<Float64Array, Float64Array>;
  stepCount: number;

  constructor(config: AdamWConfig) {
    this.config = {
      lr: config.lr,
      beta1: config.beta1 ?? 0.9,
      beta2: config.beta2 ?? 0.999,
      eps: config.eps ?? 1e-8,
      weightDecay: config.weightDecay ?? 0.01,
    };
    this.m = new Map();
    this.v = new Map();
    this.stepCount = 0;
  }

  step(params: Float64Array[], grads: Float64Array[], effectiveLr?: number): void {
    this.stepCount++;
    const lr = effectiveLr ?? this.config.lr;
    const { beta1, beta2, eps, weightDecay } = this.config;

    // Bias correction terms
    const bc1 = 1.0 - Math.pow(beta1, this.stepCount);
    const bc2 = 1.0 - Math.pow(beta2, this.stepCount);

    for (let i = 0; i < params.length; i++) {
      const p = params[i]!;
      const g = grads[i]!;
      const len = p.length;

      let mArr = this.m.get(p);
      if (!mArr) {
        mArr = new Float64Array(len);
        this.m.set(p, mArr);
      }
      let vArr = this.v.get(p);
      if (!vArr) {
        vArr = new Float64Array(len);
        this.v.set(p, vArr);
      }

      for (let j = 0; j < len; j++) {
        const grad = g[j]!;
        // 1st moment update
        mArr[j] = beta1 * mArr[j]! + (1.0 - beta1) * grad;
        // 2nd moment update
        vArr[j] = beta2 * vArr[j]! + (1.0 - beta2) * (grad * grad);

        const mHat = mArr[j]! / bc1;
        const vHat = vArr[j]! / bc2;

        // Decoupled weight decay + Adam update
        p[j] = p[j]! - lr * (mHat / (Math.sqrt(vHat) + eps) + weightDecay * p[j]!);
      }
    }
  }
}

/**
 * Cosine Annealing Learning Rate with Linear Warmup:
 *   t < T_warmup: lr_min + t/T_warmup * (lr_max - lr_min)
 *   t >= T_warmup: lr_min + 0.5*(lr_max - lr_min) * (1 + cos(pi * (t - T_warmup)/(T_total - T_warmup)))
 */
export function getCosineWarmupLR(
  currentStep: number,
  totalSteps: number,
  warmupSteps: number,
  baseLR: number,
  minLR: number = 1e-6
): number {
  if (currentStep < warmupSteps) {
    return minLR + (currentStep / Math.max(1, warmupSteps)) * (baseLR - minLR);
  }
  const progress = (currentStep - warmupSteps) / Math.max(1, totalSteps - warmupSteps);
  const clampedProgress = Math.min(1.0, Math.max(0.0, progress));
  return minLR + 0.5 * (baseLR - minLR) * (1.0 + Math.cos(Math.PI * clampedProgress));
}

/**
 * Global-Norm Gradient Clipping:
 * Clips gradients in-place so that ||g_all||_2 <= maxNorm.
 */
export function clipGradNorm(grads: Float64Array[], maxNorm: number = 1.0): number {
  let totalSqNorm = 0.0;
  for (let i = 0; i < grads.length; i++) {
    const g = grads[i]!;
    for (let j = 0; j < g.length; j++) {
      totalSqNorm += g[j]! * g[j]!;
    }
  }
  const totalNorm = Math.sqrt(totalSqNorm);

  if (totalNorm > maxNorm) {
    const scale = maxNorm / (totalNorm + 1e-12);
    for (let i = 0; i < grads.length; i++) {
      const g = grads[i]!;
      for (let j = 0; j < g.length; j++) {
        g[j] = (g[j] ?? 0) * scale;
      }
    }
  }

  return totalNorm;
}

/**
 * Exponential Moving Average (EMA) of Weights:
 * Maintains shadow weights: theta_EMA = alpha * theta_EMA + (1 - alpha) * theta.
 */
export class WeightEMA {
  readonly alpha: number;
  shadowWeights: Map<Float64Array, Float64Array>;

  constructor(alpha: number = 0.999) {
    this.alpha = alpha;
    this.shadowWeights = new Map();
  }

  update(params: Float64Array[]): void {
    for (let i = 0; i < params.length; i++) {
      const p = params[i]!;
      let shadow = this.shadowWeights.get(p);
      if (!shadow) {
        shadow = new Float64Array(p);
        this.shadowWeights.set(p, shadow);
      } else {
        const a = this.alpha;
        for (let j = 0; j < p.length; j++) {
          shadow[j] = a * shadow[j]! + (1.0 - a) * p[j]!;
        }
      }
    }
  }

  applyShadow(params: Float64Array[]): void {
    for (let i = 0; i < params.length; i++) {
      const p = params[i]!;
      const shadow = this.shadowWeights.get(p);
      if (shadow) {
        p.set(shadow);
      }
    }
  }
}

/**
 * Finite-Difference Gradient Checker:
 * Verifies analytical backward implementation against two-sided numerical difference:
 *   g_num = (L(theta + eps) - L(theta - eps)) / (2 * eps)
 * Returns max absolute and relative error.
 */
export function checkGradients(
  lossFn: (param: Float64Array) => number,
  param: Float64Array,
  analyticalGrad: Float64Array,
  eps: number = 1e-7,
  tolerance: number = 1e-6
): { maxRelError: number; maxAbsError: number; passed: boolean } {
  const n = Math.min(param.length, 25); // Check subset of weights for efficiency
  let maxRel = 0.0;
  let maxAbs = 0.0;

  for (let i = 0; i < n; i++) {
    const orig = param[i]!;

    param[i] = orig + eps;
    const lossPlus = lossFn(param);

    param[i] = orig - eps;
    const lossMinus = lossFn(param);

    param[i] = orig; // Restore

    const numGrad = (lossPlus - lossMinus) / (2.0 * eps);
    const anaGrad = analyticalGrad[i]!;

    const absErr = Math.abs(numGrad - anaGrad);
    const relErr = absErr / Math.max(1e-6, Math.abs(numGrad) + Math.abs(anaGrad));

    if (absErr > maxAbs) maxAbs = absErr;
    if (relErr > maxRel) maxRel = relErr;
  }

  return {
    maxRelError: maxRel,
    maxAbsError: maxAbs,
    passed: maxRel < tolerance || maxAbs < tolerance,
  };
}
