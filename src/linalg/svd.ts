import { DenseMatrix } from './matrix';

export interface SVDResult {
  U: DenseMatrix; // m x n thin or m x m
  singularValues: Float64Array; // Sorted descending
  V: DenseMatrix; // n x n
  conditionNumber: number; // s_max / s_min
  pseudoInverse: DenseMatrix; // n x m
}

/**
 * One-Sided Jacobi Singular Value Decomposition (SVD)
 *
 * Orthogonalizes columns of A via pairwise plane rotations:
 *   A * V = U * Sigma
 *
 * Robust against rank-deficient and ill-conditioned systems.
 */
export function svd(A: DenseMatrix, maxSweeps: number = 40, tol: number = 1e-13): SVDResult {
  const m = A.rows;
  const n = A.cols;
  const B = A.clone();
  const V = DenseMatrix.identity(n);

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let maxOffDiag = 0.0;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        // Dot products between column p and column q
        let alpha = 0.0; // ||col_p||^2
        let beta = 0.0;  // ||col_q||^2
        let gamma = 0.0; // col_p . col_q

        for (let i = 0; i < m; i++) {
          const bp = B.get(i, p);
          const bq = B.get(i, q);
          alpha += bp * bp;
          beta += bq * bq;
          gamma += bp * bq;
        }

        const absGamma = Math.abs(gamma);
        if (absGamma > maxOffDiag) maxOffDiag = absGamma;

        if (absGamma < tol * Math.sqrt(alpha * beta + 1e-20)) continue;

        // Compute rotation angle to orthogonalize column p and q
        const zeta = (beta - alpha) / (2.0 * gamma);
        const t = Math.sign(zeta) / (Math.abs(zeta) + Math.sqrt(1.0 + zeta * zeta));
        const c = 1.0 / Math.sqrt(1.0 + t * t);
        const s = t * c;

        // Update B columns p and q
        for (let i = 0; i < m; i++) {
          const bp = B.get(i, p);
          const bq = B.get(i, q);
          B.set(i, p, c * bp - s * bq);
          B.set(i, q, s * bp + c * bq);
        }

        // Update V columns p and q
        for (let i = 0; i < n; i++) {
          const vp = V.get(i, p);
          const vq = V.get(i, q);
          V.set(i, p, c * vp - s * vq);
          V.set(i, q, s * vp + c * vq);
        }
      }
    }

    if (maxOffDiag < tol) break;
  }

  // Singular values are the norms of columns of B
  const rawSingular = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let sumSq = 0.0;
    for (let i = 0; i < m; i++) {
      const val = B.get(i, j);
      sumSq += val * val;
    }
    rawSingular[j] = Math.sqrt(sumSq);
  }

  // Sort singular values descending
  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => rawSingular[b]! - rawSingular[a]!);

  const singularValues = new Float64Array(n);
  const sortedU = new DenseMatrix(m, n);
  const sortedV = new DenseMatrix(n, n);

  for (let j = 0; j < n; j++) {
    const origIdx = indices[j]!;
    const s = rawSingular[origIdx]!;
    singularValues[j] = s;

    const invS = s > 1e-14 ? 1.0 / s : 0.0;
    for (let i = 0; i < m; i++) {
      sortedU.set(i, j, B.get(i, origIdx) * invS);
    }
    for (let i = 0; i < n; i++) {
      sortedV.set(i, j, V.get(i, origIdx));
    }
  }

  const sMax = singularValues[0]!;
  const sMin = singularValues[n - 1]!;
  const conditionNumber = sMin > 1e-15 ? sMax / sMin : Infinity;

  // Compute Moore-Penrose Pseudo-Inverse: A^+ = V * Sigma^+ * U^T
  const pseudoInverse = new DenseMatrix(n, m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let sum = 0.0;
      for (let k = 0; k < n; k++) {
        const s = singularValues[k]!;
        if (s > 1e-12 * sMax) {
          sum += sortedV.get(i, k) * (1.0 / s) * sortedU.get(j, k);
        }
      }
      pseudoInverse.set(i, j, sum);
    }
  }

  return {
    U: sortedU,
    singularValues,
    V: sortedV,
    conditionNumber,
    pseudoInverse,
  };
}

export class JacobiSVD {
  private result: SVDResult;

  constructor(A: DenseMatrix, maxSweeps: number = 40, tol: number = 1e-13) {
    this.result = svd(A, maxSweeps, tol);
  }

  getU(): DenseMatrix {
    return this.result.U;
  }

  getV(): DenseMatrix {
    return this.result.V;
  }

  getSingularValues(): Float64Array {
    return this.result.singularValues;
  }

  pseudoInverse(): DenseMatrix {
    return this.result.pseudoInverse;
  }
}

