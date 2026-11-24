import { kahanDotProduct } from '../numerics/kahan';

export interface GridMatchResult {
  bestLagIndex: number;
  bestTimeOffsetSec: number;
  maxCorrelation: number; // -1.0 to 1.0
  meanSquareError: number;
  pValSignificance: number; // p-value of correlation vs null hypothesis
  isAuthenticMatch: boolean;
}

/**
 * Sliding Normalized Cross-Correlation (NCC) & MMSE ENF Grid Matcher
 *
 * Compares an extracted query ENF frequency sequence with a reference power grid log.
 */
export function matchENFSequence(
  queryHz: ArrayLike<number>,
  referenceHz: ArrayLike<number>,
  dtSec: number = 1.0
): GridMatchResult {
  const M = queryHz.length;
  const N = referenceHz.length;

  if (M === 0 || N < M) {
    return {
      bestLagIndex: 0,
      bestTimeOffsetSec: 0,
      maxCorrelation: 0,
      meanSquareError: Infinity,
      pValSignificance: 1.0,
      isAuthenticMatch: false,
    };
  }

  // Demean query
  let qSum = 0.0;
  for (let i = 0; i < M; i++) qSum += queryHz[i]!;
  const qMean = qSum / M;

  const qZero = new Float64Array(M);
  let qNormSq = 0.0;
  for (let i = 0; i < M; i++) {
    const d = queryHz[i]! - qMean;
    qZero[i] = d;
    qNormSq += d * d;
  }
  const qStd = Math.sqrt(qNormSq);

  let maxCorr = -1.0;
  let bestLag = 0;
  let minMmse = Infinity;

  const numLags = N - M + 1;

  for (let lag = 0; lag < numLags; lag++) {
    // Demean reference segment
    let rSum = 0.0;
    for (let i = 0; i < M; i++) rSum += referenceHz[lag + i]!;
    const rMean = rSum / M;

    const rZero = new Float64Array(M);
    let rNormSq = 0.0;
    let mseSum = 0.0;

    for (let i = 0; i < M; i++) {
      const origRef = referenceHz[lag + i]!;
      const d = origRef - rMean;
      rZero[i] = d;
      rNormSq += d * d;

      const diff = queryHz[i]! - origRef;
      mseSum += diff * diff;
    }

    const rStd = Math.sqrt(rNormSq);
    const denom = qStd * rStd;

    const dot = kahanDotProduct(qZero, rZero);
    const corr = denom > 1e-12 ? dot / denom : 0.0;
    const mse = mseSum / M;

    if (corr > maxCorr) {
      maxCorr = corr;
      bestLag = lag;
      minMmse = mse;
    }
  }

  // Statistical significance: Student's t distribution approximation for correlation coefficient
  // t = r * sqrt((M - 2) / (1 - r^2))
  const rClamped = Math.max(-0.9999, Math.min(0.9999, maxCorr));
  const tStat = Math.abs(rClamped) * Math.sqrt(Math.max(1.0, (M - 2.0) / (1.0 - rClamped * rClamped)));
  // Two-tailed p-value estimate
  const pVal = 2.0 * (1.0 - Math.min(1.0, 0.5 + 0.5 * Math.tanh(tStat / Math.sqrt(2.0))));

  const isAuthentic = maxCorr >= 0.70 && pVal < 0.001;

  return {
    bestLagIndex: bestLag,
    bestTimeOffsetSec: bestLag * dtSec,
    maxCorrelation: Number(maxCorr.toFixed(4)),
    meanSquareError: Number(minMmse.toFixed(6)),
    pValSignificance: Number(pVal.toExponential(2)),
    isAuthenticMatch: isAuthentic,
  };
}
