import { DenseMatrix } from './matrix';

/**
 * Householder QR Factorization: A = Q * R
 *
 * Q is orthogonal (m x m or m x n thin), R is upper triangular (m x n).
 * Stable orthogonal transformations prevent condition number squaring.
 */
export function qr(A: DenseMatrix): { Q: DenseMatrix; R: DenseMatrix } {
  const m = A.rows;
  const n = A.cols;
  const R = A.clone();
  const Q = DenseMatrix.identity(m);

  const minDim = Math.min(m, n);

  for (let k = 0; k < minDim; k++) {
    // Compute Householder vector for column k below diagonal
    let normSq = 0.0;
    for (let i = k; i < m; i++) {
      const val = R.get(i, k);
      normSq += val * val;
    }
    const norm = Math.sqrt(normSq);
    if (norm < 1e-15) continue;

    const alpha = R.get(k, k) >= 0 ? -norm : norm;
    const u0 = R.get(k, k) - alpha;

    const v = new Float64Array(m);
    v[k] = 1.0;
    for (let i = k + 1; i < m; i++) {
      v[i] = R.get(i, k) / u0;
    }

    const beta = -u0 / alpha; // 2 / (v^T * v)

    // Apply Householder reflector to R: R = (I - beta * v * v^T) * R
    for (let j = k; j < n; j++) {
      let dot = 0.0;
      for (let i = k; i < m; i++) {
        dot += v[i]! * R.get(i, j);
      }
      for (let i = k; i < m; i++) {
        R.set(i, j, R.get(i, j) - beta * v[i]! * dot);
      }
    }

    // Accumulate into Q: Q = Q * (I - beta * v * v^T)
    for (let i = 0; i < m; i++) {
      let dot = 0.0;
      for (let j = k; j < m; j++) {
        dot += Q.get(i, j) * v[j]!;
      }
      for (let j = k; j < m; j++) {
        Q.set(i, j, Q.get(i, j) - beta * dot * v[j]!);
      }
    }
  }

  return { Q, R };
}

export class HouseholderQR {
  private Q: DenseMatrix;
  private R: DenseMatrix;

  constructor(A: DenseMatrix) {
    const res = qr(A);
    this.Q = res.Q;
    this.R = res.R;
  }

  getQ(): DenseMatrix {
    return this.Q;
  }

  getR(): DenseMatrix {
    return this.R;
  }
}

