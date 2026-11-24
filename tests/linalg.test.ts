/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
import { describe, it, expect } from 'vitest';
import { DenseMatrix } from '../src/linalg/matrix';
import { CholeskyDecomposition } from '../src/linalg/cholesky';
import { HouseholderQR } from '../src/linalg/qr';
import { JacobiEigen } from '../src/linalg/eig';
import { JacobiSVD } from '../src/linalg/svd';

describe('Linear Algebra Suite', () => {
  it('DenseMatrix multiplication and Gaussian elimination solve', () => {
    // 3x3 system: A * x = b
    // [ 3  2 -1 ] [ x0 ]   [ 1 ]
    // [ 2 -2  4 ] [ x1 ] = [-2 ]
    // [-1 0.5 -1] [ x2 ]   [ 0 ]
    // Solution: x = [1, -2, -2]
    const A = new DenseMatrix(3, 3, new Float64Array([
      3,  2, -1,
      2, -2,  4,
     -1, 0.5, -1
    ]));
    const b = new Float64Array([1, -2, 0]);
    const x = A.solve(b);

    expect(Math.abs(x[0]! - 1.0)).toBeLessThan(1e-10);
    expect(Math.abs(x[1]! - (-2.0))).toBeLessThan(1e-10);
    expect(Math.abs(x[2]! - (-2.0))).toBeLessThan(1e-10);
  });

  it('Cholesky decomposition A = L * L^T on symmetric positive-definite matrix', () => {
    // A = [ 4  12 -16 ]
    //     [ 12 37 -43 ]
    //     [-16 -43 98 ]
    // L = [ 2  0  0 ]
    //     [ 6  1  0 ]
    //     [-8  5  3 ]
    const A = new DenseMatrix(3, 3, new Float64Array([
       4,  12, -16,
      12,  37, -43,
     -16, -43,  98
    ]));
    const chol = new CholeskyDecomposition(A);
    const L = chol.getL();

    expect(Math.abs(L.get(0, 0) - 2.0)).toBeLessThan(1e-10);
    expect(Math.abs(L.get(1, 0) - 6.0)).toBeLessThan(1e-10);
    expect(Math.abs(L.get(1, 1) - 1.0)).toBeLessThan(1e-10);
    expect(Math.abs(L.get(2, 0) - (-8.0))).toBeLessThan(1e-10);
    expect(Math.abs(L.get(2, 1) - 5.0)).toBeLessThan(1e-10);
    expect(Math.abs(L.get(2, 2) - 3.0)).toBeLessThan(1e-10);

    // Verify L * L^T == A
    const LLT = L.multiply(L.transpose());
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        expect(Math.abs(LLT.get(r, c) - A.get(r, c))).toBeLessThan(1e-10);
      }
    }
  });

  it('Householder QR decomposition A = Q * R satisfies orthogonality and reconstruction', () => {
    const A = new DenseMatrix(3, 2, new Float64Array([
      12, -51,
       6, 167,
      -4,  24
    ]));
    const qr = new HouseholderQR(A);
    const Q = qr.getQ();
    const R = qr.getR();

    // Verify Q * R == A
    const QR = Q.multiply(R);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 2; c++) {
        expect(Math.abs(QR.get(r, c) - A.get(r, c))).toBeLessThan(1e-10);
      }
    }

    // Verify Q^T * Q == I_3
    const QTQ = Q.transpose().multiply(Q);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const expected = (r === c) ? 1.0 : 0.0;
        expect(Math.abs(QTQ.get(r, c) - expected)).toBeLessThan(1e-10);
      }
    }
  });

  it('Jacobi symmetric eigenvalue decomposition recovers known eigenvalues and eigenvectors', () => {
    // Symmetric matrix with known eigenvalues [1, 2, 3]
    // A = [ 2  -1   0 ]
    //     [-1   2  -1 ]
    //     [ 0  -1   2 ]
    // Eigenvalues of tridiagonal 2, -1: 2 - 2*cos(k*pi/4) -> [2 - sqrt(2), 2, 2 + sqrt(2)]
    const A = new DenseMatrix(3, 3, new Float64Array([
       2, -1,  0,
      -1,  2, -1,
       0, -1,  2
    ]));
    const eig = new JacobiEigen(A);
    const evals = eig.getEigenvalues();

    expect(Math.abs(evals[0]! - (2 + Math.SQRT2))).toBeLessThan(1e-8);
    expect(Math.abs(evals[1]! - 2.0)).toBeLessThan(1e-8);
    expect(Math.abs(evals[2]! - (2 - Math.SQRT2))).toBeLessThan(1e-8);

    // Verify V^T * A * V == diag(evals)
    const V = eig.getEigenvectors();
    const VTAV = V.transpose().multiply(A).multiply(V);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const expected = (r === c) ? evals[r]! : 0.0;
        expect(Math.abs(VTAV.get(r, c) - expected)).toBeLessThan(1e-8);
      }
    }
  });

  it('Jacobi SVD and Moore-Penrose Pseudo-Inverse', () => {
    const A = new DenseMatrix(3, 2, new Float64Array([
      1, 2,
      3, 4,
      5, 6
    ]));
    const svd = new JacobiSVD(A);
    const pinv = svd.pseudoInverse();

    // Penrose condition 1: A * A^+ * A == A
    const cond1 = A.multiply(pinv).multiply(A);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 2; c++) {
        expect(Math.abs(cond1.get(r, c) - A.get(r, c))).toBeLessThan(1e-8);
      }
    }

    // Penrose condition 2: A^+ * A * A^+ == A^+
    const cond2 = pinv.multiply(A).multiply(pinv);
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 3; c++) {
        expect(Math.abs(cond2.get(r, c) - pinv.get(r, c))).toBeLessThan(1e-8);
      }
    }
  });
});
