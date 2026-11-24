/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
import { DenseMatrix } from './matrix';

/**
 * Cholesky Factorization: A = L * L^T
 *
 * For symmetric positive-definite matrices.
 * Includes diagonal jitter regularization (delta * I) if matrix is near-singular.
 */
export function cholesky(A: DenseMatrix, regularizeDelta: number = 1e-10): DenseMatrix {
  const n = A.rows;
  if (A.cols !== n) throw new Error(`Matrix must be square for Cholesky: ${n}x${A.cols}`);

  const L = new DenseMatrix(n, n);
  const lData = L.data;
  const aData = A.data;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0.0;
      for (let k = 0; k < j; k++) {
        sum += lData[i * n + k]! * lData[j * n + k]!;
      }

      if (i === j) {
        let diag = aData[i * n + i]! - sum;
        if (diag <= 0.0) {
          diag = regularizeDelta; // Diagonal jitter to preserve positive definiteness
        }
        lData[i * n + j] = Math.sqrt(diag);
      } else {
        const diagVal = lData[j * n + j]!;
        lData[i * n + j] = diagVal > 1e-15 ? (aData[i * n + j]! - sum) / diagVal : 0.0;
      }
    }
  }

  return L;
}

/**
 * Solves L * L^T * x = b via forward and backward substitution.
 */
export function choleskySolve(L: DenseMatrix, b: ArrayLike<number>): Float64Array {
  const n = L.rows;
  const lData = L.data;

  // Forward solve: L * y = b
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i]!;
    for (let k = 0; k < i; k++) {
      sum -= lData[i * n + k]! * y[k]!;
    }
    const diag = lData[i * n + i]!;
    y[i] = diag > 1e-15 ? sum / diag : 0.0;
  }

  // Backward solve: L^T * x = y
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i]!;
    for (let k = i + 1; k < n; k++) {
      sum -= lData[k * n + i]! * x[k]!;
    }
    const diag = lData[i * n + i]!;
    x[i] = diag > 1e-15 ? sum / diag : 0.0;
  }

  return x;
}

export class CholeskyDecomposition {
  private L: DenseMatrix;

  constructor(A: DenseMatrix, regularizeDelta: number = 1e-10) {
    this.L = cholesky(A, regularizeDelta);
  }

  getL(): DenseMatrix {
    return this.L;
  }

  solve(b: ArrayLike<number>): Float64Array {
    return choleskySolve(this.L, b);
  }
}

