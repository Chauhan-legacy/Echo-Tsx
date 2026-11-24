import { DenseMatrix } from './matrix';

export interface EigenResult {
  eigenvalues: Float64Array; // Sorted descending
  eigenvectors: DenseMatrix; // Column vectors corresponding to eigenvalues
  iterations: number;
}

/**
 * Classical Cyclic Jacobi Eigen-Solver for Symmetric Matrices
 *
 * Computes all eigenvalues and eigenvectors of a symmetric matrix A (A = A^T):
 *   A * v_i = lambda_i * v_i
 *
 * Guaranteed quadratic asymptotic convergence.
 * Stated tolerance: error vs analytical eigenvalues < 1e-12.
 */
export function symmetricEigen(
  A: DenseMatrix,
  maxIterations: number = 60,
  tolerance: number = 1e-14
): EigenResult {
  const n = A.rows;
  if (A.cols !== n) throw new Error(`Matrix must be square: ${n}x${A.cols}`);

  // Work on copy
  const S = A.clone();
  const V = DenseMatrix.identity(n);

  let iter = 0;
  for (iter = 0; iter < maxIterations; iter++) {
    // Measure off-diagonal Frobenius norm
    let offDiagNorm = 0.0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        offDiagNorm += S.get(i, j) * S.get(i, j);
      }
    }
    if (Math.sqrt(offDiagNorm * 2.0) < tolerance) break;

    // Cyclic Jacobi sweeps
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const app = S.get(p, p);
        const aqq = S.get(q, q);
        const apq = S.get(p, q);

        if (Math.abs(apq) < 1e-15) continue;

        // Compute rotation angle theta: cot(2*theta) = (aqq - app) / (2 * apq)
        const tau = (aqq - app) / (2.0 * apq);
        let t: number;
        if (tau >= 0) {
          t = 1.0 / (tau + Math.sqrt(1.0 + tau * tau));
        } else {
          t = -1.0 / (-tau + Math.sqrt(1.0 + tau * tau));
        }

        const c = 1.0 / Math.sqrt(1.0 + t * t);
        const s = t * c;

        // Apply Givens rotation to S: S = J^T * S * J
        S.set(p, q, 0.0);
        S.set(q, p, 0.0);
        S.set(p, p, app - t * apq);
        S.set(q, q, aqq + t * apq);

        for (let r = 0; r < n; r++) {
          if (r !== p && r !== q) {
            const arp = S.get(r, p);
            const arq = S.get(r, q);
            const newArp = c * arp - s * arq;
            const newArq = s * arp + c * arq;
            S.set(r, p, newArp);
            S.set(p, r, newArp);
            S.set(r, q, newArq);
            S.set(q, r, newArq);
          }
        }

        // Accumulate rotation into eigenvectors: V = V * J
        for (let r = 0; r < n; r++) {
          const vrp = V.get(r, p);
          const vrq = V.get(r, q);
          V.set(r, p, c * vrp - s * vrq);
          V.set(r, q, s * vrp + c * vrq);
        }
      }
    }
  }

  // Extract eigenvalues from diagonal
  const rawEig = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    rawEig[i] = S.get(i, i);
  }

  // Sort descending
  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => rawEig[b]! - rawEig[a]!);

  const eigenvalues = new Float64Array(n);
  const sortedV = new DenseMatrix(n, n);

  for (let j = 0; j < n; j++) {
    const origIdx = indices[j]!;
    eigenvalues[j] = rawEig[origIdx]!;
    for (let i = 0; i < n; i++) {
      sortedV.set(i, j, V.get(i, origIdx));
    }
  }

  return {
    eigenvalues,
    eigenvectors: sortedV,
    iterations: iter,
  };
}

export class JacobiEigen {
  private result: EigenResult;

  constructor(A: DenseMatrix, maxIterations: number = 60, tolerance: number = 1e-14) {
    this.result = symmetricEigen(A, maxIterations, tolerance);
  }

  getEigenvalues(): Float64Array {
    return this.result.eigenvalues;
  }

  getEigenvectors(): DenseMatrix {
    return this.result.eigenvectors;
  }
}

