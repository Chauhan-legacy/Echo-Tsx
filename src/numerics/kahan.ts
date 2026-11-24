/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * High-Precision Compensated Summation
 *
 * Implements Neumaier's improved Kahan summation algorithm:
 * Sums an array of IEEE-754 floating point numbers with O(eps) precision,
 * tracking intermediate roundoff errors and preventing catastrophic cancellation.
 */

export function kahanSum(values: ArrayLike<number>): number {
  let sum = 0.0;
  let c = 0.0; // Compensation accumulator

  const len = values.length;
  for (let i = 0; i < len; i++) {
    const val = values[i]!;
    const t = sum + val;
    if (Math.abs(sum) >= Math.abs(val)) {
      c += (sum - t) + val;
    } else {
      c += (val - t) + sum;
    }
    sum = t;
  }

  return sum + c;
}

/**
 * Compensated Dot Product (Kahan-Babuška-Neumaier)
 */
export function kahanDotProduct(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let sum = 0.0;
  let c = 0.0;

  for (let i = 0; i < n; i++) {
    const p = a[i]! * b[i]!;
    const t = sum + p;
    if (Math.abs(sum) >= Math.abs(p)) {
      c += (sum - t) + p;
    } else {
      c += (p - t) + sum;
    }
    sum = t;
  }

  return sum + c;
}
