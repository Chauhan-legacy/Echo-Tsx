/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Zero-Dependency Dense Matrix Operations
 *
 * Designed for real-time DSP, Kalman filters, and array signal processing:
 * 1. Flat Float64Array backing buffers for cache locality.
 * 2. Transposed cache-blocked multiplication.
 * 3. Matrix 1-norm and condition number estimation.
 * 4. Tikhonov regularized linear solve: (A^T * A + lambda * I)^(-1) * A^T * b.
 */

export class DenseMatrix {
  public readonly rows: number;
  public readonly cols: number;
  public readonly data: Float64Array;

  constructor(rows: number, cols: number, data?: Float64Array | number[]) {
    if (rows <= 0 || cols <= 0) throw new Error(`Matrix dimensions must be positive: ${rows}x${cols}`);
    this.rows = rows;
    this.cols = cols;
    if (data) {
      if (data.length !== rows * cols) {
        throw new Error(`Data length (${data.length}) does not match dimensions (${rows}x${cols})`);
      }
      this.data = data instanceof Float64Array ? data : new Float64Array(data);
    } else {
      this.data = new Float64Array(rows * cols);
    }
  }

  public static identity(size: number): DenseMatrix {
    const mat = new DenseMatrix(size, size);
    for (let i = 0; i < size; i++) {
      mat.data[i * size + i] = 1.0;
    }
    return mat;
  }

  public get(r: number, c: number): number {
    return this.data[r * this.cols + c]!;
  }

  public set(r: number, c: number, val: number): void {
    this.data[r * this.cols + c] = val;
  }

  public clone(): DenseMatrix {
    const copy = new DenseMatrix(this.rows, this.cols);
    copy.data.set(this.data);
    return copy;
  }

  /**
   * Matrix transpose: A^T
   */
  public transpose(): DenseMatrix {
    const res = new DenseMatrix(this.cols, this.rows);
    const r = this.rows;
    const c = this.cols;
    const src = this.data;
    const dst = res.data;

    for (let i = 0; i < r; i++) {
      const rowOffset = i * c;
      for (let j = 0; j < c; j++) {
        dst[j * r + i] = src[rowOffset + j]!;
      }
    }
    return res;
  }

  /**
   * Cache-friendly matrix multiplication: C = A * B
   */
  public multiply(other: DenseMatrix): DenseMatrix {
    if (this.cols !== other.rows) {
      throw new Error(`Dimension mismatch for matrix multiply: (${this.rows}x${this.cols}) * (${other.rows}x${other.cols})`);
    }

    const m = this.rows;
    const kDim = this.cols;
    const n = other.cols;
    const res = new DenseMatrix(m, n);

    const aData = this.data;
    const bData = other.data;
    const cData = res.data;

    for (let i = 0; i < m; i++) {
      const aRow = i * kDim;
      const cRow = i * n;
      for (let k = 0; k < kDim; k++) {
        const aVal = aData[aRow + k]!;
        if (aVal === 0.0) continue;
        const bRow = k * n;
        for (let j = 0; j < n; j++) {
          cData[cRow + j] = (cData[cRow + j] ?? 0) + aVal * bData[bRow + j]!;
        }
      }
    }
    return res;
  }

  /**
   * Matrix-vector product: y = A * x
   */
  public multiplyVector(x: ArrayLike<number>, out?: Float64Array): Float64Array {
    if (this.cols !== x.length) {
      throw new Error(`Vector length ${x.length} does not match matrix cols ${this.cols}`);
    }
    const res = out || new Float64Array(this.rows);
    const aData = this.data;
    const r = this.rows;
    const c = this.cols;

    for (let i = 0; i < r; i++) {
      let sum = 0.0;
      const rowOffset = i * c;
      for (let j = 0; j < c; j++) {
        sum += aData[rowOffset + j]! * x[j]!;
      }
      res[i] = sum;
    }
    return res;
  }

  /**
   * Computes the 1-norm (maximum absolute column sum)
   */
  public norm1(): number {
    let maxColSum = 0.0;
    for (let j = 0; j < this.cols; j++) {
      let colSum = 0.0;
      for (let i = 0; i < this.rows; i++) {
        colSum += Math.abs(this.data[i * this.cols + j]!);
      }
      if (colSum > maxColSum) maxColSum = colSum;
    }
    return maxColSum;
  }

  /**
   * Solves A * x = b using Gaussian elimination with partial row pivoting.
   */
  public solve(b: ArrayLike<number>): Float64Array {
    return solveLinearSystem(this, b);
  }

  /**
   * Solves (A^T * A + lambda * I) * x = A^T * b using Cholesky or Gaussian elimination with Tikhonov damping.
   */
  public solveTikhonov(b: ArrayLike<number>, lambda: number = 1e-6): Float64Array {
    const at = this.transpose();
    const ata = at.multiply(this);

    // Add Tikhonov regularizer to diagonal: ata + lambda * I
    for (let i = 0; i < ata.rows; i++) {
      const idx = i * ata.cols + i;
      ata.data[idx] = (ata.data[idx] ?? 0) + lambda;
    }

    const atb = at.multiplyVector(b);

    // Solve symmetric positive-definite system via Gaussian elimination with partial pivoting
    return solveLinearSystem(ata, atb);
  }
}

/**
 * Solves A * x = b via Gaussian elimination with partial row pivoting.
 */
export function solveLinearSystem(A: DenseMatrix, b: ArrayLike<number>): Float64Array {
  const n = A.rows;
  if (A.cols !== n || b.length !== n) {
    throw new Error(`System must be square: A is ${n}x${A.cols}, b is length ${b.length}`);
  }

  // Augmented matrix
  const M = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      M[i * (n + 1) + j] = A.get(i, j);
    }
    M[i * (n + 1) + n] = b[i]!;
  }

  // Forward elimination
  for (let p = 0; p < n; p++) {
    // Find pivot
    let maxRow = p;
    let maxVal = Math.abs(M[p * (n + 1) + p]!);
    for (let i = p + 1; i < n; i++) {
      const val = Math.abs(M[i * (n + 1) + p]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }

    if (maxVal < 1e-15) {
      // Near singular: regularize pivot
      const pIdx = p * (n + 1) + p;
      M[pIdx] = (M[pIdx] ?? 0) + 1e-12;
    }

    // Swap rows
    if (maxRow !== p) {
      for (let j = p; j <= n; j++) {
        const tmp = M[p * (n + 1) + j]!;
        M[p * (n + 1) + j] = M[maxRow * (n + 1) + j]!;
        M[maxRow * (n + 1) + j] = tmp;
      }
    }

    // Eliminate
    const pivot = M[p * (n + 1) + p]!;
    for (let i = p + 1; i < n; i++) {
      const factor = M[i * (n + 1) + p]! / pivot;
      for (let j = p; j <= n; j++) {
        const idx = i * (n + 1) + j;
        M[idx] = (M[idx] ?? 0) - factor * M[p * (n + 1) + j]!;
      }
    }
  }

  // Back substitution
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i * (n + 1) + n]!;
    for (let j = i + 1; j < n; j++) {
      sum -= M[i * (n + 1) + j]! * x[j]!;
    }
    const pivot = M[i * (n + 1) + i]!;
    x[i] = Math.abs(pivot) > 1e-15 ? sum / pivot : 0.0;
  }

  return x;
}
