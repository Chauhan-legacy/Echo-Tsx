/**
 * Robust Statistics and Hypothesis Testing for Audio & Video Forensics:
 * 1. Median & Median Absolute Deviation (MAD)
 * 2. Huber M-Estimator of Location
 * 3. Theil-Sen Robust Linear Regression
 * 4. Welch's Two-Sample t-Test (unequal variances)
 * 5. Mann-Whitney U Test (non-parametric rank sum)
 * 6. Benjamini-Hochberg False Discovery Rate (FDR)
 * 7. Discrete Kullback-Leibler (KL) Divergence and Mutual Information
 */

/**
 * Compute sample median of an array.
 */
export function median(values: Float64Array): number {
  const n = values.length;
  if (n === 0) return 0.0;
  const copy = new Float64Array(values).sort();
  const mid = n >> 1;
  return (n % 2 !== 0) ? copy[mid]! : (copy[mid - 1]! + copy[mid]!) / 2.0;
}

/**
 * Median Absolute Deviation (MAD) with normal consistency scale factor 1.4826:
 *   MAD = 1.482602218505602 * median(|x_i - median(x)|)
 */
export function medianAbsoluteDeviation(values: Float64Array): { med: number; mad: number } {
  const n = values.length;
  if (n === 0) return { med: 0.0, mad: 0.0 };
  const med = median(values);
  const diffs = new Float64Array(n);
  for (let i = 0; i < n; i++) diffs[i] = Math.abs(values[i]! - med);
  const mad = 1.482602218505602 * median(diffs);
  return { med, mad };
}

/**
 * Huber M-estimator of location:
 * Minimizes sum rho((x_i - mu) / c), where psi(u) = clip(u, -k, k).
 * Computed via Iteratively Reweighted Least Squares (IRLS).
 */
export function huberLocation(values: Float64Array, k: number = 1.345, maxIter: number = 30): number {
  const n = values.length;
  if (n === 0) return 0.0;
  let { med: mu, mad } = medianAbsoluteDeviation(values);
  const s = Math.max(1e-8, mad);

  for (let iter = 0; iter < maxIter; iter++) {
    let weightSum = 0.0;
    let weightedValSum = 0.0;

    for (let i = 0; i < n; i++) {
      const u = (values[i]! - mu) / s;
      const w = Math.abs(u) <= k ? 1.0 : k / Math.abs(u);
      weightSum += w;
      weightedValSum += w * values[i]!;
    }

    const nextMu = weightedValSum / Math.max(1e-12, weightSum);
    if (Math.abs(nextMu - mu) < 1e-7) {
      return nextMu;
    }
    mu = nextMu;
  }
  return mu;
}

/**
 * Theil-Sen Robust Estimator of Linear Slope:
 * Slope is the median of all pairwise slopes: (y_j - y_i) / (x_j - x_i) for i < j.
 * Breakdown point: 29.3% (immune to extreme outliers).
 */
export function theilSenSlope(x: Float64Array, y: Float64Array): { slope: number; intercept: number } {
  const n = Math.min(x.length, y.length);
  if (n < 2) return { slope: 0.0, intercept: 0.0 };

  const slopes: number[] = [];
  // Sample pairs to cap complexity at O(N^2) max 5000 pairs
  const step = n > 100 ? Math.floor(n / 50) : 1;

  for (let i = 0; i < n; i += step) {
    for (let j = i + 1; j < n; j += step) {
      const dx = x[j]! - x[i]!;
      if (Math.abs(dx) > 1e-12) {
        slopes.push((y[j]! - y[i]!) / dx);
      }
    }
  }

  const slope = median(new Float64Array(slopes));
  const intercepts = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    intercepts[i] = y[i]! - slope * x[i]!;
  }
  const intercept = median(intercepts);

  return { slope, intercept };
}

/**
 * Welch's Two-Sample t-Test (unequal sample sizes and unequal variances).
 * Returns t-statistic, Satterthwaite degrees of freedom, and p-value.
 */
export function welchTTest(x: Float64Array, y: Float64Array): { tStat: number; df: number; pValue: number } {
  const n1 = x.length;
  const n2 = y.length;
  if (n1 < 2 || n2 < 2) return { tStat: 0.0, df: 1.0, pValue: 1.0 };

  let m1 = 0, m2 = 0;
  for (let i = 0; i < n1; i++) m1 += x[i]!;
  for (let i = 0; i < n2; i++) m2 += y[i]!;
  m1 /= n1; m2 /= n2;

  let v1 = 0, v2 = 0;
  for (let i = 0; i < n1; i++) { const d = x[i]! - m1; v1 += d * d; }
  for (let i = 0; i < n2; i++) { const d = y[i]! - m2; v2 += d * d; }
  v1 /= (n1 - 1); v2 /= (n2 - 1);

  const se1 = v1 / n1;
  const se2 = v2 / n2;
  const seDiff = Math.sqrt(Math.max(1e-14, se1 + se2));
  const tStat = (m1 - m2) / seDiff;

  // Satterthwaite approximation for degrees of freedom
  const denom = (se1 * se1) / (n1 - 1) + (se2 * se2) / (n2 - 1);
  const df = denom > 1e-14 ? ((se1 + se2) * (se1 + se2)) / denom : 1.0;

  // Approximate two-tailed p-value via regularized incomplete beta / Gaussian approximation
  const absT = Math.abs(tStat);
  // Approximation using normal distribution for moderate/large df
  const z = absT * (1.0 - 1.0 / (4.0 * df));
  const pOneTail = 0.5 * Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const pValue = Math.min(1.0, Math.max(0.0, 2.0 * pOneTail));

  return { tStat, df, pValue };
}

/**
 * Mann-Whitney U Test (Wilcoxon rank-sum test for non-parametric difference).
 */
export function mannWhitneyUTest(x: Float64Array, y: Float64Array): { uStat: number; pValue: number } {
  const n1 = x.length;
  const n2 = y.length;
  const combined: { val: number; group: 0 | 1 }[] = [];

  for (let i = 0; i < n1; i++) combined.push({ val: x[i]!, group: 0 });
  for (let i = 0; i < n2; i++) combined.push({ val: y[i]!, group: 1 });
  combined.sort((a, b) => a.val - b.val);

  let rankSum1 = 0.0;
  for (let i = 0; i < combined.length; i++) {
    if (combined[i]!.group === 0) {
      rankSum1 += (i + 1);
    }
  }

  const u1 = rankSum1 - (n1 * (n1 + 1)) / 2.0;
  const u2 = n1 * n2 - u1;
  const uStat = Math.min(u1, u2);

  // Normal approximation for p-value: mu = n1*n2/2, sigma = sqrt(n1*n2*(n1+n2+1)/12)
  const muU = (n1 * n2) / 2.0;
  const sigmaU = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0);
  const z = Math.abs(uStat - muU) / Math.max(1e-12, sigmaU);
  const pValue = Math.min(1.0, Math.max(0.0, 2.0 * (1.0 - 0.5 * (1.0 + Math.sign(z) * Math.sqrt(1.0 - Math.exp(-2.0 * z * z / Math.PI))))));

  return { uStat, pValue };
}

/**
 * Benjamini-Hochberg False Discovery Rate (FDR) procedure:
 * Controls expected proportion of false discoveries among rejected hypotheses.
 */
export function benjaminiHochberg(pValues: number[], falseDiscoveryRate: number = 0.05): boolean[] {
  const m = pValues.length;
  const indexed = pValues.map((p, idx) => ({ p, idx })).sort((a, b) => a.p - b.p);

  let maxK = -1;
  for (let k = 0; k < m; k++) {
    const threshold = ((k + 1) / m) * falseDiscoveryRate;
    if (indexed[k]!.p <= threshold) {
      maxK = k;
    }
  }

  const isSignificant = new Array(m).fill(false);
  for (let k = 0; k <= maxK; k++) {
    isSignificant[indexed[k]!.idx] = true;
  }
  return isSignificant;
}

/**
 * Discrete Kullback-Leibler (KL) Divergence: D_KL(P || Q) = \sum_i P(i) \ln(P(i) / Q(i))
 */
export function klDivergence(p: Float64Array, q: Float64Array): number {
  const n = Math.min(p.length, q.length);
  let dkl = 0.0;
  for (let i = 0; i < n; i++) {
    const pi = p[i]!;
    const qi = q[i]!;
    if (pi > 1e-15) {
      const qSafe = Math.max(1e-15, qi);
      dkl += pi * Math.log(pi / qSafe);
    }
  }
  return Math.max(0.0, dkl);
}
