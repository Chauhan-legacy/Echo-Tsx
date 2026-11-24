/**
 * Change-point detection algorithms for audio and sensor forensics:
 * 1. Two-Sided CUSUM (Page's Cumulative Sum Test)
 * 2. Bayesian Online Change-Point Detection (BOCPD - Adams & MacKay 2007)
 *
 * Applications:
 * - Detecting ENF grid splices or audio edits
 * - Detecting background noise floor transitions (recording location cut)
 * - Detecting gain/compression adjustments
 */

export interface CUSUMResult {
  changePoints: number[];
  sPositive: Float64Array;
  sNegative: Float64Array;
  threshold: number;
}

/**
 * Two-sided CUSUM detector for shift in mean:
 *   S^+_k = max(0, S^+_{k-1} + (x_k - \mu_0 - \delta / 2))
 *   S^-_k = max(0, S^-_{k-1} - (x_k - \mu_0 + \delta / 2))
 * Alarm triggered if S^+_k > h or S^-_k > h.
 */
export function detectCUSUM(
  series: Float64Array,
  options: {
    expectedShiftDelta?: number; // Minimum detectable shift delta
    thresholdH?: number;        // Decision boundary threshold
    warmupSamples?: number;     // Samples used to estimate baseline mean/std
  } = {}
): CUSUMResult {
  const n = series.length;
  const warmup = Math.min(n, Math.max(5, options.warmupSamples ?? Math.min(20, Math.floor(n * 0.2))));

  // Estimate baseline mean and std from warmup
  let sum = 0.0;
  for (let i = 0; i < warmup; i++) sum += series[i]!;
  const mu0 = sum / warmup;

  let sumSqDiff = 0.0;
  for (let i = 0; i < warmup; i++) {
    const diff = series[i]! - mu0;
    sumSqDiff += diff * diff;
  }
  const sigma = Math.sqrt(Math.max(1e-10, sumSqDiff / Math.max(1, warmup - 1)));

  // Normalized shift delta (default: 1.0 * sigma)
  const delta = (options.expectedShiftDelta ?? sigma);
  const slack = delta / 2.0;
  // Threshold default: 4 * sigma
  const threshold = options.thresholdH ?? (4.0 * sigma);

  const sPos = new Float64Array(n);
  const sNeg = new Float64Array(n);
  const changePoints: number[] = [];

  let curPos = 0.0;
  let curNeg = 0.0;
  let lastAlarmIdx = -10;

  for (let i = 0; i < n; i++) {
    const x = series[i]!;
    curPos = Math.max(0.0, curPos + (x - mu0 - slack));
    curNeg = Math.max(0.0, curNeg - (x - mu0 + slack));

    sPos[i] = curPos;
    sNeg[i] = curNeg;

    if (curPos > threshold || curNeg > threshold) {
      if (i - lastAlarmIdx > 5) {
        changePoints.push(i);
        lastAlarmIdx = i;
      }
      // Reset after detection
      curPos = 0.0;
      curNeg = 0.0;
    }
  }

  return {
    changePoints,
    sPositive: sPos,
    sNegative: sNeg,
    threshold,
  };
}

export interface BOCPDResult {
  runLengthProbabilities: Float64Array[]; // P(r_t | x_{1:t})
  mostLikelyRunLengths: Int32Array;
  changePoints: number[]; // Indices where run length resets to 0 with high confidence
}

/**
 * Bayesian Online Change-Point Detection (BOCPD - Adams & MacKay 2007)
 * Uses Normal-Inverse-Gamma conjugate prior for streaming Gaussian data with unknown mean & variance.
 * Constant hazard function: H(r) = 1 / lambda_prior.
 */
export function detectBOCPD(
  series: Float64Array,
  hazardLambda: number = 50.0,
  changePointProbThreshold: number = 0.10
): BOCPDResult {
  const T = series.length;
  if (T === 0) {
    return {
      runLengthProbabilities: [],
      mostLikelyRunLengths: new Int32Array(0),
      changePoints: [],
    };
  }

  const H = 1.0 / hazardLambda; // Constant hazard rate

  // Conjugate prior parameters for Normal-Inverse-Gamma:
  // mu ~ N(mu0, sigma^2 / kappa0)
  // sigma^2 ~ Inv-Gamma(alpha0, beta0)
  // Estimate initial prior from first few samples
  const warmup = Math.min(T, 15);
  let initMean = 0.0;
  for (let i = 0; i < warmup; i++) initMean += series[i]!;
  initMean /= warmup;

  let initVar = 0.0;
  for (let i = 0; i < warmup; i++) {
    const d = series[i]! - initMean;
    initVar += d * d;
  }
  initVar = Math.max(1e-6, initVar / Math.max(1, warmup - 1));

  const mu0 = initMean;
  const kappa0 = 1.0;
  const alpha0 = 1.5;
  const beta0 = initVar * alpha0;

  // Track hyper-parameters for each run length r = 0, 1, ..., t
  let mu = [mu0];
  let kappa = [kappa0];
  let alpha = [alpha0];
  let beta = [beta0];

  // R[r] = P(r_t = r | x_{1:t})
  let R = new Float64Array([1.0]);

  const runLengthProbs: Float64Array[] = [];
  const mostLikely = new Int32Array(T);
  const changePoints: number[] = [];

  for (let t = 0; t < T; t++) {
    const x = series[t]!;
    const curLen = R.length;

    // 1. Evaluate predictive probability for each run length: Student-t distribution
    const predProbs = new Float64Array(curLen);
    for (let r = 0; r < curLen; r++) {
      const a = alpha[r]!;
      const b = beta[r]!;
      const k = kappa[r]!;
      const m = mu[r]!;

      // Student-t with df = 2*a, loc = m, scale = sqrt(b * (k + 1) / (a * k))
      const df = 2.0 * a;
      const varScale = (b * (k + 1.0)) / (a * k);
      const diff = x - m;
      const tSq = (diff * diff) / varScale;
      // log Student-t density (unnormalized constant omitted since we normalize across r)
      const logP = -0.5 * (df + 1.0) * Math.log(1.0 + tSq / df) - 0.5 * Math.log(varScale);
      predProbs[r] = Math.exp(Math.max(-50.0, logP));
    }

    // 2. Growth probabilities: P(r_t = r-1 + 1) = P(r_{t-1}) * P(x) * (1 - H)
    const newR = new Float64Array(curLen + 1);
    let changePointMass = 0.0;

    for (let r = 0; r < curLen; r++) {
      const p = R[r]! * predProbs[r]!;
      newR[r + 1] = p * (1.0 - H);
      changePointMass += p * H;
    }
    // Change-point probability: P(r_t = 0)
    newR[0] = changePointMass;

    // 3. Normalize newR to sum to 1
    let sumR = 0.0;
    for (let r = 0; r <= curLen; r++) sumR += newR[r]!;
    if (sumR > 1e-15) {
      for (let r = 0; r <= curLen; r++) newR[r] = (newR[r] ?? 0) / sumR;
    } else {
      newR[0] = 1.0;
    }

    // 4. Update hyperparameters
    const nextMu = [mu0];
    const nextKappa = [kappa0];
    const nextAlpha = [alpha0];
    const nextBeta = [beta0];

    for (let r = 0; r < curLen; r++) {
      const k = kappa[r]!;
      const m = mu[r]!;
      const a = alpha[r]!;
      const b = beta[r]!;

      const kNew = k + 1.0;
      const mNew = (k * m + x) / kNew;
      const aNew = a + 0.5;
      const diff = x - m;
      const bNew = b + 0.5 * (k * diff * diff) / kNew;

      nextKappa.push(kNew);
      nextMu.push(mNew);
      nextAlpha.push(aNew);
      nextBeta.push(bNew);
    }

    // Prune very low probability tail to bound memory (keep top 100 hypotheses)
    const maxRetain = Math.min(newR.length, 80);
    R = newR.slice(0, maxRetain);
    mu = nextMu.slice(0, maxRetain);
    kappa = nextKappa.slice(0, maxRetain);
    alpha = nextAlpha.slice(0, maxRetain);
    beta = nextBeta.slice(0, maxRetain);

    // Record most likely run length
    let maxP = -1.0;
    let bestR = 0;
    for (let r = 0; r < R.length; r++) {
      if (R[r]! > maxP) {
        maxP = R[r]!;
        bestR = r;
      }
    }
    mostLikely[t] = bestR;
    runLengthProbs.push(R.slice());

    // Flag change point if run-length drops sharply or r=0 has elevated probability
    const runLengthDropped = t > 5 && mostLikely[t]! < (mostLikely[t - 1] ?? 0) - 3;
    if (t > 5 && (newR[0]! > changePointProbThreshold || bestR === 0 || runLengthDropped)) {
      if (changePoints.length === 0 || t - changePoints[changePoints.length - 1]! > 5) {
        changePoints.push(t);
      }
    }
  }

  return {
    runLengthProbabilities: runLengthProbs,
    mostLikelyRunLengths: mostLikely,
    changePoints,
  };
}
