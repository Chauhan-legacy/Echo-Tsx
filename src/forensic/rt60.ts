/**
 * Forensic Room Acoustic Reverberation Estimator: Schroeder Backward Integration & Lundeby Method.
 *
 * Mathematical Derivation:
 * 1. Schroeder Backward Integration (1965):
 *    The continuous energy decay curve (EDC) from an impulse response h(t) is:
 *      E(t) = \int_t^\infty h^2(\tau) d\tau
 *    In discrete time with N samples:
 *      E[k] = \sum_{j=k}^{N-1} h^2[j]
 *    Normalized energy in dB:
 *      L[k] = 10 \log_{10} \left( \frac{E[k]}{E[0]} \right)
 *
 * 2. Lundeby Noise Floor Truncation (1995):
 *    Real measurements have additive noise floor \sigma_n^2. Integrating noise past the
 *    intersection of the decay curve and noise floor introduces a curvature bias ("dog leg").
 *    The Lundeby method:
 *      - Computes average noise energy in the last 10-20% of the impulse response.
 *      - Finds the preliminary decay slope via linear regression between -5 dB and -20 dB.
 *      - Identifies cross-point t_c where the fitted decay line meets the noise floor.
 *      - Backward integration starts at t_c with noise energy subtraction:
 *        E[k] = \sum_{j=k}^{t_c} [ h^2[j] - \sigma_n^2 ]
 *
 * 3. RT60 Metrics:
 *    - T20: Regression from -5 dB to -25 dB: RT60 = 3 * (t_{-25} - t_{-5})
 *    - T30: Regression from -5 dB to -35 dB: RT60 = 2 * (t_{-35} - t_{-5})
 *    - Linear fit slope m (dB/s) -> RT60 = -60 / m
 *    - 95% Confidence Interval computed from standard error of the regression slope:
 *      SE(m) = \sqrt{ \frac{\sum (y_i - \hat{y}_i)^2 / (n - 2)}{\sum (t_i - \bar{t})^2} }
 *      CI_{95%} = \pm t_{crit, 0.05/2} \times \frac{60}{m^2} \times SE(m)
 */

export interface RT60Result {
  t20Seconds: number;
  t30Seconds: number;
  rSquared: number;
  confidenceInterval95: [number, number]; // [lower, upper]
  noiseFloorDb: number;
  crossoverTimeSeconds: number;
  decayCurveDb: Float64Array;
  timeAxisSeconds: Float64Array;
}

/**
 * Estimate RT60 from an acoustic impulse response using Schroeder integration and Lundeby method.
 */
export function estimateRT60(
  impulseResponse: Float64Array,
  sampleRate: number = 16000
): RT60Result {
  const n = impulseResponse.length;
  if (n < 128) {
    throw new Error('Impulse response too short for RT60 analysis (minimum 128 samples)');
  }

  // Find direct sound peak
  let peakIdx = 0;
  let maxAbs = 0.0;
  for (let i = 0; i < n; i++) {
    const val = Math.abs(impulseResponse[i]!);
    if (val > maxAbs) {
      maxAbs = val;
      peakIdx = i;
    }
  }

  // Work from peak onward
  const ir = impulseResponse.subarray(peakIdx);
  const m = ir.length;

  // 1. Estimate noise floor from the tail (last 15% of samples)
  const tailStart = Math.floor(m * 0.85);
  let tailEnergySum = 0.0;
  let tailCount = 0;
  for (let i = tailStart; i < m; i++) {
    const val = ir[i]!;
    tailEnergySum += val * val;
    tailCount++;
  }
  const noiseFloorSq = Math.max(1e-18, tailEnergySum / Math.max(1, tailCount));
  const peakSq = maxAbs * maxAbs;
  const noiseFloorDb = 10.0 * Math.log10(noiseFloorSq / Math.max(1e-18, peakSq));

  // 2. Lundeby cross-point search
  // Cumulative energy from back to front
  const cumulativeEnergy = new Float64Array(m);
  let acc = 0.0;
  for (let i = m - 1; i >= 0; i--) {
    const energy = ir[i]! * ir[i]!;
    // Subtract noise floor bias if above noise floor
    acc += Math.max(0.0, energy - noiseFloorSq);
    cumulativeEnergy[i] = acc;
  }

  const maxEnergy = Math.max(1e-15, cumulativeEnergy[0]!);
  const decayDb = new Float64Array(m);
  const timeSec = new Float64Array(m);

  for (let i = 0; i < m; i++) {
    timeSec[i] = i / sampleRate;
    const ratio = Math.max(1e-12, cumulativeEnergy[i]! / maxEnergy);
    decayDb[i] = 10.0 * Math.log10(ratio);
  }

  // Determine crossover time where decay curve drops below noiseFloorDb + 5 dB
  let crossoverIdx = m - 1;
  for (let i = 0; i < m; i++) {
    if (decayDb[i]! <= noiseFloorDb + 5.0) {
      crossoverIdx = i;
      break;
    }
  }
  const crossoverTime = crossoverIdx / sampleRate;

  // Helper for linear regression over a dB interval [dB_start, dB_end]
  function regressInterval(dbStart: number, dbEnd: number): { slope: number; r2: number; se: number; nPoints: number } {
    let sumT = 0.0;
    let sumY = 0.0;
    let count = 0;

    for (let i = 0; i <= crossoverIdx; i++) {
      const y = decayDb[i]!;
      if (y <= dbStart && y >= dbEnd) {
        sumT += timeSec[i]!;
        sumY += y;
        count++;
      }
    }

    if (count < 5) {
      return { slope: -60.0, r2: 0.0, se: 1.0, nPoints: count };
    }

    const meanT = sumT / count;
    const meanY = sumY / count;

    let ssTT = 0.0;
    let ssYY = 0.0;
    let ssTY = 0.0;

    for (let i = 0; i <= crossoverIdx; i++) {
      const y = decayDb[i]!;
      if (y <= dbStart && y >= dbEnd) {
        const dt = timeSec[i]! - meanT;
        const dy = y - meanY;
        ssTT += dt * dt;
        ssYY += dy * dy;
        ssTY += dt * dy;
      }
    }

    const slope = ssTT > 1e-12 ? ssTY / ssTT : -60.0;
    const r2 = (ssTT > 1e-12 && ssYY > 1e-12) ? (ssTY * ssTY) / (ssTT * ssYY) : 0.0;

    // Standard error of slope
    let residualSq = 0.0;
    for (let i = 0; i <= crossoverIdx; i++) {
      const y = decayDb[i]!;
      if (y <= dbStart && y >= dbEnd) {
        const yPred = meanY + slope * (timeSec[i]! - meanT);
        residualSq += (y - yPred) * (y - yPred);
      }
    }
    const sResidual = Math.sqrt(residualSq / Math.max(1, count - 2));
    const seSlope = ssTT > 1e-12 ? sResidual / Math.sqrt(ssTT) : 1.0;

    return { slope, r2, se: seSlope, nPoints: count };
  }

  // T20 regression (-5 dB to -25 dB)
  const reg20 = regressInterval(-5.0, -25.0);
  const t20 = reg20.slope < -1e-6 ? -60.0 / reg20.slope : 0.5;

  // T30 regression (-5 dB to -35 dB)
  const reg30 = regressInterval(-5.0, -35.0);
  const t30 = reg30.slope < -1e-6 ? -60.0 / reg30.slope : t20;

  // Best slope evaluation for CI
  const bestReg = reg30.nPoints >= 10 ? reg30 : reg20;
  const bestSlope = Math.abs(bestReg.slope) > 1e-6 ? bestReg.slope : -60.0 / Math.max(0.01, t20);
  const bestRT60 = -60.0 / bestSlope;

  // 95% Confidence Interval: delta_RT60 approx = (60 / slope^2) * 1.96 * SE(slope)
  const margin = (60.0 / (bestSlope * bestSlope)) * 1.96 * bestReg.se;
  const ciLower = Math.max(0.01, bestRT60 - margin);
  const ciUpper = bestRT60 + margin;

  return {
    t20Seconds: Math.max(0.01, t20),
    t30Seconds: Math.max(0.01, t30),
    rSquared: Math.min(1.0, Math.max(0.0, bestReg.r2)),
    confidenceInterval95: [ciLower, ciUpper],
    noiseFloorDb,
    crossoverTimeSeconds: crossoverTime,
    decayCurveDb: decayDb,
    timeAxisSeconds: timeSec,
  };
}
