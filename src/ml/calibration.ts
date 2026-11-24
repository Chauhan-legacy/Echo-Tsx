/**
 * Probabilistic Calibration & Conformal Prediction:
 * 1. Temperature Scaling (Guo et al. 2017)
 * 2. Isotonic Regression (Pool Adjacent Violators Algorithm - PAVA)
 * 3. Expected Calibration Error (ECE) & Reliability Diagrams
 * 4. Conformal Prediction Sets with Finite-Sample Coverage Guarantees (Vovk et al. 2005)
 */

export interface ReliabilityBin {
  confidenceMean: number;
  accuracyMean: number;
  count: number;
}

export interface CalibrationResult {
  ece: number; // Expected Calibration Error [0, 1]
  mce: number; // Maximum Calibration Error [0, 1]
  bins: ReliabilityBin[];
}

/**
 * Expected Calibration Error (ECE) and Reliability Bins:
 * Divides [0, 1] confidence range into numBins equal intervals and computes
 * weighted average of |accuracy - confidence|.
 */
export function computeECE(
  confidences: Float64Array,
  groundTruthBinary: Int32Array,
  numBins: number = 10
): CalibrationResult {
  const n = confidences.length;
  const binEdges = new Float64Array(numBins + 1);
  for (let i = 0; i <= numBins; i++) binEdges[i] = i / numBins;

  const binCounts = new Int32Array(numBins);
  const binConfSums = new Float64Array(numBins);
  const binAccSums = new Float64Array(numBins);

  for (let i = 0; i < n; i++) {
    const c = confidences[i]!;
    const y = groundTruthBinary[i]!;
    const b = Math.min(numBins - 1, Math.max(0, Math.floor(c * numBins)));
    binCounts[b] = (binCounts[b] ?? 0) + 1;
    binConfSums[b] = (binConfSums[b] ?? 0) + c;
    binAccSums[b] = (binAccSums[b] ?? 0) + y;
  }

  let ece = 0.0;
  let mce = 0.0;
  const bins: ReliabilityBin[] = [];

  for (let b = 0; b < numBins; b++) {
    const count = binCounts[b]!;
    if (count > 0) {
      const confMean = binConfSums[b]! / count;
      const accMean = binAccSums[b]! / count;
      const diff = Math.abs(accMean - confMean);
      ece += (count / n) * diff;
      if (diff > mce) mce = diff;
      bins.push({ confidenceMean: confMean, accuracyMean: accMean, count });
    } else {
      bins.push({ confidenceMean: (b + 0.5) / numBins, accuracyMean: 0.0, count: 0 });
    }
  }

  return { ece, mce, bins };
}

/**
 * Temperature Scaling:
 * Optimizes single temperature parameter T > 0 on validation logits using Brent/secant method.
 */
export function fitTemperature(
  validationLogits: Float64Array,
  validationLabels: Int32Array
): number {
  const n = validationLogits.length;
  const sig = (z: number) => 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, z))));

  function nllLoss(T: number): number {
    let loss = 0.0;
    for (let i = 0; i < n; i++) {
      const p = Math.max(1e-12, Math.min(1.0 - 1e-12, sig(validationLogits[i]! / T)));
      const y = validationLabels[i]!;
      loss -= (y === 1) ? Math.log(p) : Math.log(1.0 - p);
    }
    return loss / n;
  }

  // Golden section search for optimal T in [0.1, 5.0]
  let a = 0.1;
  let b = 5.0;
  const phi = (Math.sqrt(5) - 1) / 2;
  let c = b - phi * (b - a);
  let d = a + phi * (b - a);

  for (let iter = 0; iter < 25; iter++) {
    if (nllLoss(c) < nllLoss(d)) {
      b = d;
      d = c;
      c = b - phi * (b - a);
    } else {
      a = c;
      c = d;
      d = a + phi * (b - a);
    }
  }

  return (a + b) / 2.0;
}

/**
 * Isotonic Regression using Pool Adjacent Violators Algorithm (PAVA).
 * Yields optimal non-decreasing calibrated probability mapping.
 */
export function fitIsotonicRegression(
  scores: Float64Array,
  labels: Int32Array
): { thresholds: Float64Array; calibratedProbs: Float64Array } {
  const n = scores.length;
  const items = [];
  for (let i = 0; i < n; i++) {
    items.push({ x: scores[i]!, y: labels[i]! });
  }
  items.sort((a, b) => a.x - b.x);

  // Blocks of constant values: { weight, sum, value }
  const blocks: { weight: number; sum: number; value: number; xMax: number }[] = [];

  for (let i = 0; i < n; i++) {
    let cur = { weight: 1.0, sum: items[i]!.y, value: items[i]!.y, xMax: items[i]!.x };

    while (blocks.length > 0 && blocks[blocks.length - 1]!.value >= cur.value) {
      const prev = blocks.pop()!;
      cur.weight += prev.weight;
      cur.sum += prev.sum;
      cur.value = cur.sum / cur.weight;
      cur.xMax = Math.max(prev.xMax, cur.xMax);
    }

    blocks.push(cur);
  }

  const thresholds = new Float64Array(blocks.length);
  const calibratedProbs = new Float64Array(blocks.length);
  for (let i = 0; i < blocks.length; i++) {
    thresholds[i] = blocks[i]!.xMax;
    calibratedProbs[i] = Math.max(0.0, Math.min(1.0, blocks[i]!.value));
  }

  return { thresholds, calibratedProbs };
}

/**
 * Split Conformal Prediction for Forensic Decision Bounds:
 * Guaranteed coverage: P(Y in C(X)) >= 1 - alpha.
 * Nonconformity score: s_i = 1 - f(x_i)_{y_i}.
 * Quantile: q_val = (1 - alpha) * (1 + 1/n) empirical quantile of calibration scores.
 */
export function computeConformalThreshold(
  calibrationScores: Float64Array, // f(x_i)_{true_class}
  significanceAlpha: number = 0.05
): number {
  const n = calibrationScores.length;
  const nonconformity = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    nonconformity[i] = 1.0 - calibrationScores[i]!;
  }
  nonconformity.sort();

  // Quantile index with conservative finite-sample correction: ceil((n + 1) * (1 - alpha)) / n
  const p = Math.min(1.0, ((n + 1) * (1.0 - significanceAlpha)) / n);
  const idx = Math.min(n - 1, Math.max(0, Math.floor(p * n)));

  return nonconformity[idx]!;
}
