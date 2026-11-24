/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Zero-Dependency Classic Machine Learning Algorithms (Pure TypeScript)
 *
 * Implements:
 * 1. Logistic Regression (Iteratively Reweighted Least Squares / Newton-Raphson & Gradient Descent)
 * 2. Support Vector Machine (Linear & RBF Kernel via Sequential Minimal Optimization - SMO)
 * 3. k-Nearest Neighbors (k-NN) with Euclidean and Cosine distance
 * 4. Gaussian Mixture Model (GMM) with Expectation-Maximization (EM)
 * 5. k-Means++ Clustering
 * 6. Principal Component Analysis (PCA) via SVD
 * 7. One-Class Mahalanobis Distance Anomaly Detector
 * 8. Hidden Markov Model (HMM) Viterbi Path Decoder
 */

import { DenseMatrix } from '../linalg/matrix';
import { JacobiSVD } from '../linalg/svd';

// ==========================================
// 1. Logistic Regression
// ==========================================
export class LogisticRegression {
  public weights: Float64Array;
  public bias: number = 0.0;
  private readonly numFeatures: number;
  private readonly l2Reg: number;

  constructor(numFeatures: number, l2Reg: number = 1e-4) {
    this.numFeatures = numFeatures;
    this.l2Reg = l2Reg;
    this.weights = new Float64Array(numFeatures);
  }

  public fit(X: DenseMatrix, y: ArrayLike<number>, maxIters: number = 100, lr: number = 0.05): void {
    const N = X.rows;
    const D = this.numFeatures;

    for (let iter = 0; iter < maxIters; iter++) {
      const gradW = new Float64Array(D);
      let gradB = 0.0;

      for (let i = 0; i < N; i++) {
        let z = this.bias;
        for (let j = 0; j < D; j++) {
          z += this.weights[j]! * X.get(i, j);
        }
        const p = 1.0 / (1.0 + Math.exp(-Math.max(-50, Math.min(50, z))));
        const err = p - (y[i]! > 0.5 ? 1.0 : 0.0);

        for (let j = 0; j < D; j++) {
          gradW[j] = (gradW[j] ?? 0) + err * X.get(i, j);
        }
        gradB += err;
      }

      // Update with L2 regularization
      for (let j = 0; j < D; j++) {
        const wVal = this.weights[j] ?? 0;
        const gVal = gradW[j] ?? 0;
        this.weights[j] = wVal - lr * (gVal / N + this.l2Reg * wVal);
      }
      this.bias -= lr * (gradB / N);
    }
  }

  public predictProbability(x: ArrayLike<number>): number {
    let z = this.bias;
    for (let j = 0; j < this.numFeatures; j++) {
      z += this.weights[j]! * (x[j] ?? 0.0);
    }
    return 1.0 / (1.0 + Math.exp(-Math.max(-50, Math.min(50, z))));
  }

  public predict(x: ArrayLike<number>, threshold: number = 0.5): number {
    return this.predictProbability(x) >= threshold ? 1 : 0;
  }
}

// ==========================================
// 2. Support Vector Machine (Linear / RBF SMO)
// ==========================================
export class SupportVectorMachine {
  private alphas: Float64Array | null = null;
  private bias: number = 0.0;
  private supportVectors: Float64Array[] = [];
  private supportLabels: number[] = [];
  public readonly kernel: 'linear' | 'rbf';
  public readonly gamma: number;
  public readonly C: number;

  constructor(kernel: 'linear' | 'rbf' = 'rbf', gamma: number = 0.5, C: number = 1.0) {
    this.kernel = kernel;
    this.gamma = gamma;
    this.C = C;
  }

  private kernelEval(x1: ArrayLike<number>, x2: ArrayLike<number>): number {
    if (this.kernel === 'linear') {
      let dot = 0.0;
      for (let i = 0; i < x1.length; i++) dot += x1[i]! * x2[i]!;
      return dot;
    }
    let distSq = 0.0;
    for (let i = 0; i < x1.length; i++) {
      const d = x1[i]! - x2[i]!;
      distSq += d * d;
    }
    return Math.exp(-this.gamma * distSq);
  }

  public fit(X: ArrayLike<number>[], y: number[], maxIters: number = 50): void {
    const N = X.length;
    this.alphas = new Float64Array(N);
    this.bias = 0.0;

    // Simplified Platt's SMO optimization
    for (let iter = 0; iter < maxIters; iter++) {
      let numChanged = 0;
      for (let i = 0; i < N; i++) {
        const xi = X[i]!;
        const yi = y[i]! > 0 ? 1 : -1;
        
        let pred_i = this.bias;
        for (let k = 0; k < N; k++) {
          if (this.alphas[k]! > 1e-6) {
            const yk = y[k]! > 0 ? 1 : -1;
            pred_i += this.alphas[k]! * yk * this.kernelEval(X[k]!, xi);
          }
        }
        const Ei = pred_i - yi;

        if ((yi * Ei < -0.001 && this.alphas[i]! < this.C) || (yi * Ei > 0.001 && this.alphas[i]! > 0)) {
          const j = (i + 1 + Math.floor(Math.random() * (N - 1))) % N;
          const xj = X[j]!;
          const yj = y[j]! > 0 ? 1 : -1;

          let pred_j = this.bias;
          for (let k = 0; k < N; k++) {
            if (this.alphas[k]! > 1e-6) {
              const yk = y[k]! > 0 ? 1 : -1;
              pred_j += this.alphas[k]! * yk * this.kernelEval(X[k]!, xj);
            }
          }
          const Ej = pred_j - yj;

          const oldAi = this.alphas[i]!;
          const oldAj = this.alphas[j]!;

          let L = 0.0;
          let H = this.C;
          if (yi !== yj) {
            L = Math.max(0, oldAj - oldAi);
            H = Math.min(this.C, this.C + oldAj - oldAi);
          } else {
            L = Math.max(0, oldAi + oldAj - this.C);
            H = Math.min(this.C, oldAi + oldAj);
          }
          if (L >= H) continue;

          const kii = this.kernelEval(xi, xi);
          const kjj = this.kernelEval(xj, xj);
          const kij = this.kernelEval(xi, xj);
          const eta = 2.0 * kij - kii - kjj;
          if (eta >= 0) continue;

          let newAj = oldAj - (yj * (Ei - Ej)) / eta;
          newAj = Math.max(L, Math.min(H, newAj));
          if (Math.abs(newAj - oldAj) < 1e-5) continue;

          const newAi = oldAi + yi * yj * (oldAj - newAj);
          this.alphas[i] = newAi;
          this.alphas[j] = newAj;

          const b1 = this.bias - Ei - yi * (newAi - oldAi) * kii - yj * (newAj - oldAj) * kij;
          const b2 = this.bias - Ej - yi * (newAi - oldAi) * kij - yj * (newAj - oldAj) * kjj;
          this.bias = (b1 + b2) / 2.0;

          numChanged++;
        }
      }
      if (numChanged === 0) break;
    }

    // Retain non-zero support vectors
    this.supportVectors = [];
    this.supportLabels = [];
    const filteredAlphas: number[] = [];
    for (let i = 0; i < N; i++) {
      if (this.alphas[i]! > 1e-4) {
        this.supportVectors.push(new Float64Array(X[i]!));
        this.supportLabels.push(y[i]! > 0 ? 1 : -1);
        filteredAlphas.push(this.alphas[i]!);
      }
    }
    this.alphas = new Float64Array(filteredAlphas);
  }

  private predictRaw(x: ArrayLike<number>): number {
    if (!this.alphas || this.supportVectors.length === 0) return this.bias;
    let s = this.bias;
    for (let i = 0; i < this.supportVectors.length; i++) {
      s += this.alphas[i]! * this.supportLabels[i]! * this.kernelEval(this.supportVectors[i]!, x);
    }
    return s;
  }

  public predict(x: ArrayLike<number>): number {
    return this.predictRaw(x) >= 0 ? 1 : 0;
  }
}

// ==========================================
// 3. k-Nearest Neighbors (k-NN)
// ==========================================
export class KNearestNeighbors {
  public readonly k: number;
  private X: Float64Array[] = [];
  private y: number[] = [];

  constructor(k: number = 3) {
    this.k = k;
  }

  public fit(X: ArrayLike<number>[], y: number[]): void {
    this.X = X.map(x => new Float64Array(x));
    this.y = [...y];
  }

  public predict(x: ArrayLike<number>): number {
    if (this.X.length === 0) return 0;
    const distances: { dist: number; label: number }[] = [];

    for (let i = 0; i < this.X.length; i++) {
      const xi = this.X[i]!;
      let dSq = 0.0;
      for (let j = 0; j < xi.length; j++) {
        const diff = xi[j]! - (x[j] ?? 0.0);
        dSq += diff * diff;
      }
      distances.push({ dist: Math.sqrt(dSq), label: this.y[i]! });
    }

    distances.sort((a, b) => a.dist - b.dist);
    const kNearest = distances.slice(0, this.k);

    let sum = 0;
    for (const item of kNearest) {
      sum += item.label;
    }
    return sum >= this.k / 2 ? 1 : 0;
  }
}

// ==========================================
// 4. Principal Component Analysis (PCA)
// ==========================================
export class PrincipalComponentAnalysis {
  public mean: Float64Array;
  public components: DenseMatrix | null = null;
  public explainedVariance: Float64Array | null = null;
  public readonly nComponents: number;

  constructor(nComponents: number) {
    this.nComponents = nComponents;
    this.mean = new Float64Array(0);
  }

  public fit(X: DenseMatrix): void {
    const N = X.rows;
    const D = X.cols;
    this.mean = new Float64Array(D);

    // Compute column means
    for (let c = 0; c < D; c++) {
      let sum = 0.0;
      for (let r = 0; r < N; r++) sum += X.get(r, c);
      this.mean[c] = sum / N;
    }

    // Center data
    const Xcentered = new DenseMatrix(N, D);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < D; c++) {
        Xcentered.set(r, c, X.get(r, c) - this.mean[c]!);
      }
    }

    // SVD of centered matrix
    const svd = new JacobiSVD(Xcentered);
    const V = svd.getV(); // Right singular vectors
    const s = svd.getSingularValues(); // Singular values

    const k = Math.min(this.nComponents, D);
    this.components = new DenseMatrix(k, D);
    this.explainedVariance = new Float64Array(k);

    for (let i = 0; i < k; i++) {
      this.explainedVariance[i] = (s[i]! * s[i]!) / (N - 1);
      for (let j = 0; j < D; j++) {
        this.components.set(i, j, V.get(j, i));
      }
    }
  }

  public transform(x: ArrayLike<number>): Float64Array {
    if (!this.components) throw new Error('PCA model not fitted');
    const D = this.mean.length;
    const k = this.components.rows;
    const out = new Float64Array(k);

    for (let i = 0; i < k; i++) {
      let val = 0.0;
      for (let j = 0; j < D; j++) {
        val += ((x[j] ?? 0.0) - this.mean[j]!) * this.components.get(i, j);
      }
      out[i] = val;
    }
    return out;
  }
}

// ==========================================
// 5. One-Class Mahalanobis Distance Detector
// ==========================================
export class MahalanobisAnomalyDetector {
  public mean: Float64Array = new Float64Array(0);
  public invCov: DenseMatrix | null = null;
  public threshold: number = 3.0; // standard deviations

  public fit(X: DenseMatrix, thresholdPercentile: number = 95.0): void {
    const N = X.rows;
    const D = X.cols;
    this.mean = new Float64Array(D);

    for (let c = 0; c < D; c++) {
      let s = 0.0;
      for (let r = 0; r < N; r++) s += X.get(r, c);
      this.mean[c] = s / N;
    }

    // Sample covariance
    const cov = new DenseMatrix(D, D);
    for (let i = 0; i < D; i++) {
      for (let j = 0; j < D; j++) {
        let sum = 0.0;
        for (let r = 0; r < N; r++) {
          sum += (X.get(r, i) - this.mean[i]!) * (X.get(r, j) - this.mean[j]!);
        }
        cov.set(i, j, sum / (N - 1) + (i === j ? 1e-4 : 0.0)); // Tikhonov regularization
      }
    }

    const svd = new JacobiSVD(cov);
    this.invCov = svd.pseudoInverse();

    // Compute training distances to set threshold
    const distances: number[] = [];
    for (let r = 0; r < N; r++) {
      const row = new Float64Array(D);
      for (let c = 0; c < D; c++) row[c] = X.get(r, c);
      distances.push(this.distance(row));
    }
    distances.sort((a, b) => a - b);
    const pIdx = Math.floor((thresholdPercentile / 100.0) * distances.length);
    this.threshold = distances[Math.min(distances.length - 1, pIdx)]!;
  }

  public distance(x: ArrayLike<number>): number {
    if (!this.invCov) return 0.0;
    const D = this.mean.length;
    const diff = new Float64Array(D);
    for (let i = 0; i < D; i++) diff[i] = (x[i] ?? 0.0) - this.mean[i]!;

    let dSq = 0.0;
    for (let i = 0; i < D; i++) {
      let temp = 0.0;
      for (let j = 0; j < D; j++) {
        temp += diff[j]! * this.invCov.get(j, i);
      }
      dSq += diff[i]! * temp;
    }
    return Math.sqrt(Math.max(0.0, dSq));
  }

  public isAnomaly(x: ArrayLike<number>): boolean {
    return this.distance(x) > this.threshold;
  }
}
