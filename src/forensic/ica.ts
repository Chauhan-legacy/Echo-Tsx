/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Independent Component Analysis: FastICA (Hyvärinen 1999).
 *
 * Separates mixed acoustic sensor channels or unmixes covert tracking tones from voice/music.
 *
 * Algorithm:
 * 1. Centering: \mu = E[x], x_c = x - \mu
 * 2. Whitening: Covariance C = E[x_c x_c^T] = E D E^T.
 *    Whitening matrix V = E D^{-1/2} E^T, z = V x_c so that E[z z^T] = I.
 * 3. Fixed-Point Negentropy Iteration (Tanh contrast g(u) = \tanh(u), g'(u) = 1 - \tanh^2(u)):
 *      w^+ = E[z \tanh(w^T z)] - E[1 - \tanh^2(w^T z)] w
 *      w = w^+ / ||w^+||_2
 * 4. Symmetric Orthogonalization across all components:
 *      W = (W W^T)^{-1/2} W
 */
import { DenseMatrix } from '../linalg/matrix';
import { JacobiEigen } from '../linalg/eig';
import { Xoshiro128p } from '../numerics/prng';

export interface FastICAResult {
  unmixedSignals: Float64Array[]; // Independent components (each of length T)
  unmixingMatrix: DenseMatrix;    // W
  mixingMatrix: DenseMatrix;      // A = W^{-1}
  whiteningMatrix: DenseMatrix;   // V
  iterations: number;
  converged: boolean;
}

export function runFastICA(
  mixedSignals: Float64Array[], // M channels, each of length T
  options: {
    numComponents?: number;
    maxIterations?: number;
    tolerance?: number;
    seed?: number;
    contrast?: 'tanh' | 'kurtosis';
  } = {}
): FastICAResult {
  const M = mixedSignals.length;
  if (M === 0) throw new Error('At least one channel required for ICA');
  const T = mixedSignals[0]!.length;
  const numComponents = Math.min(M, options.numComponents ?? M);
  const maxIter = options.maxIterations ?? 100;
  const tol = options.tolerance ?? 1e-6;
  const contrast = options.contrast ?? 'tanh';
  const rng = new Xoshiro128p(options.seed ?? 1234);

  // 1. Centering
  const X = new DenseMatrix(M, T);
  for (let r = 0; r < M; r++) {
    const ch = mixedSignals[r]!;
    let mean = 0.0;
    for (let t = 0; t < T; t++) mean += ch[t]!;
    mean /= T;
    for (let t = 0; t < T; t++) {
      X.set(r, t, ch[t]! - mean);
    }
  }

  // 2. Whitening: Covariance C = (1/T) * X * X^T
  const cov = X.multiply(X.transpose());
  for (let r = 0; r < M; r++) {
    for (let c = 0; c < M; c++) {
      cov.set(r, c, cov.get(r, c) / T);
    }
  }

  const eig = new JacobiEigen(cov);
  const evals = eig.getEigenvalues();
  const evecs = eig.getEigenvectors();

  // V = E * D^{-1/2} * E^T
  const DinvSqrt = new DenseMatrix(M, M);
  for (let i = 0; i < M; i++) {
    const val = Math.max(1e-10, evals[i]!);
    DinvSqrt.set(i, i, 1.0 / Math.sqrt(val));
  }
  const whiteningMatrix = evecs.multiply(DinvSqrt).multiply(evecs.transpose());
  const Z = whiteningMatrix.multiply(X); // M x T whitened data

  // 3. Initialize unmixing matrix W (numComponents x M) with random orthonormal vectors
  const W = new DenseMatrix(numComponents, M);
  for (let r = 0; r < numComponents; r++) {
    for (let c = 0; c < M; c++) {
      W.set(r, c, rng.nextGaussian(0, 1.0));
    }
  }

  // Helper for symmetric orthogonalization: W = (W W^T)^{-1/2} * W
  function orthogonalize(wMat: DenseMatrix): DenseMatrix {
    const WWT = wMat.multiply(wMat.transpose());
    const eigW = new JacobiEigen(WWT);
    const wEvals = eigW.getEigenvalues();
    const wEvecs = eigW.getEigenvectors();

    const invSqrt = new DenseMatrix(numComponents, numComponents);
    for (let i = 0; i < numComponents; i++) {
      const v = Math.max(1e-12, wEvals[i]!);
      invSqrt.set(i, i, 1.0 / Math.sqrt(v));
    }
    const factor = wEvecs.multiply(invSqrt).multiply(wEvecs.transpose());
    return factor.multiply(wMat);
  }

  let curW = orthogonalize(W);
  let iter = 0;
  let converged = false;

  // 4. FastICA Fixed-Point Loop
  for (iter = 0; iter < maxIter; iter++) {
    const nextW = new DenseMatrix(numComponents, M);

    for (let p = 0; p < numComponents; p++) {
      // Compute w_p^T * Z for all t
      const u = new Float64Array(T);
      for (let t = 0; t < T; t++) {
        let val = 0.0;
        for (let m = 0; m < M; m++) {
          val += curW.get(p, m) * Z.get(m, t);
        }
        u[t] = val;
      }

      // E[z * g(u)] and E[g'(u)]
      const term1 = new Float64Array(M);
      let gPrimeExpectation = 0.0;

      for (let t = 0; t < T; t++) {
        const ut = u[t]!;
        let gVal = 0.0;
        let gpVal = 0.0;

        if (contrast === 'tanh') {
          const tanhU = Math.tanh(ut);
          gVal = tanhU;
          gpVal = 1.0 - tanhU * tanhU;
        } else {
          // Kurtosis: g(u) = u^3, g'(u) = 3*u^2
          gVal = ut * ut * ut;
          gpVal = 3.0 * ut * ut;
        }

        gPrimeExpectation += gpVal;
        for (let m = 0; m < M; m++) {
          term1[m] = (term1[m] ?? 0) + Z.get(m, t) * gVal;
        }
      }

      gPrimeExpectation /= T;
      for (let m = 0; m < M; m++) {
        const eZG = term1[m]! / T;
        const wVal = eZG - gPrimeExpectation * curW.get(p, m);
        nextW.set(p, m, wVal);
      }
    }

    // Symmetric orthogonalization
    const orthW = orthogonalize(nextW);

    // Check convergence: max(1 - |w_p^T * w_old_p|) < tol
    let maxDiff = 0.0;
    for (let p = 0; p < numComponents; p++) {
      let dot = 0.0;
      for (let m = 0; m < M; m++) {
        dot += orthW.get(p, m) * curW.get(p, m);
      }
      const diff = 1.0 - Math.abs(dot);
      if (diff > maxDiff) maxDiff = diff;
    }

    curW = orthW;
    if (maxDiff < tol) {
      converged = true;
      break;
    }
  }

  // 5. Compute separated independent components: S = curW * Z
  const S = curW.multiply(Z);
  const unmixed: Float64Array[] = [];
  for (let p = 0; p < numComponents; p++) {
    const ch = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      ch[t] = S.get(p, t);
    }
    unmixed.push(ch);
  }

  // Full unmixing matrix relative to raw X: W_full = curW * whiteningMatrix
  const W_full = curW.multiply(whiteningMatrix);
  // Mixing matrix A = pinv(W_full)
  const A = W_full.transpose(); // Since whitened ICA is orthonormal

  return {
    unmixedSignals: unmixed,
    unmixingMatrix: W_full,
    mixingMatrix: A,
    whiteningMatrix,
    iterations: iter,
    converged,
  };
}
