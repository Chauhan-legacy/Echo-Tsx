/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Photo-Response Non-Uniformity (PRNU) Camera Sensor Forensic Fingerprinting.
 *
 * Mathematical Foundations:
 * 1. Sensor Output Model (Lukas, Goljan, Fridrich 2006):
 *    I = I^0 + I^0 * K + \Theta
 *    where I is the captured image, I^0 is the ideal noise-free scene,
 *    K is the zero-mean PRNU multiplicative sensor pattern, and \Theta is additive random noise.
 *
 * 2. Noise Residual Extraction:
 *    W = I - F(I)
 *    where F(.) is a denoising filter. In practice, a local adaptive variance Wiener filter
 *    removes scene content while preserving high-frequency PRNU sensor imperfections.
 *
 * 3. Peak-to-Correlation Energy (PCE):
 *    PCE measures the prominence of the cross-correlation peak relative to the energy of the cross-correlation plane:
 *      PCE = \frac{c(s_{peak})^2}{\frac{1}{|A| - |\mathcal{N}|} \sum_{s \notin \mathcal{N}} c(s)^2}
 *    where \mathcal{N} is an 11x11 square window centered at the peak.
 *    A PCE > 60.0 provides overwhelming statistical evidence of matching sensor origin.
 */

export interface PRNUResult {
  pce: number;
  peakCorrelation: number;
  peakOffset: [number, number]; // [dy, dx]
  isMatch: boolean;
  correlationPlane: Float64Array; // (2*maxLagY + 1) x (2*maxLagX + 1)
}

/**
 * Extract noise residual from a 2D grayscale image using local adaptive Wiener filtering.
 * Filter: \hat{I}_{x,y} = \mu + \max(0, \sigma^2 - \sigma_n^2)/\sigma^2 * (I_{x,y} - \mu)
 * Residual: W = I - \hat{I}
 */
export function extractNoiseResidual2D(
  image: Float64Array,
  width: number,
  height: number,
  windowRadius: number = 2
): Float64Array {
  const residual = new Float64Array(width * height);
  const winSize = 2 * windowRadius + 1;
  const numPixels = winSize * winSize;

  // First pass: compute global minimum variance to estimate noise floor sigma_n^2
  let minVar = 1e9;
  for (let y = windowRadius; y < height - windowRadius; y += 4) {
    for (let x = windowRadius; x < width - windowRadius; x += 4) {
      let sum = 0.0;
      let sumSq = 0.0;
      for (let dy = -windowRadius; dy <= windowRadius; dy++) {
        const rowOff = (y + dy) * width;
        for (let dx = -windowRadius; dx <= windowRadius; dx++) {
          const val = image[rowOff + x + dx]!;
          sum += val;
          sumSq += val * val;
        }
      }
      const mean = sum / numPixels;
      const v = Math.max(1e-8, sumSq / numPixels - mean * mean);
      if (v < minVar) minVar = v;
    }
  }
  const noiseVar = Math.max(1e-4, minVar);

  // Second pass: apply Wiener filter and extract residual
  for (let y = 0; y < height; y++) {
    const yStart = Math.max(0, y - windowRadius);
    const yEnd = Math.min(height - 1, y + windowRadius);
    const rowOff = y * width;

    for (let x = 0; x < width; x++) {
      const xStart = Math.max(0, x - windowRadius);
      const xEnd = Math.min(width - 1, x + windowRadius);

      let sum = 0.0;
      let sumSq = 0.0;
      let count = 0;

      for (let wy = yStart; wy <= yEnd; wy++) {
        const wRow = wy * width;
        for (let wx = xStart; wx <= xEnd; wx++) {
          const val = image[wRow + wx]!;
          sum += val;
          sumSq += val * val;
          count++;
        }
      }

      const mean = sum / count;
      const variance = Math.max(0.0, sumSq / count - mean * mean);
      const pixel = image[rowOff + x]!;

      // Wiener denoised estimate
      const ratio = Math.max(0.0, variance - noiseVar) / Math.max(1e-6, variance);
      const denoised = mean + ratio * (pixel - mean);

      // Residual is pixel minus denoised
      residual[rowOff + x] = pixel - denoised;
    }
  }

  // Zero-mean normalization
  let rSum = 0.0;
  const total = width * height;
  for (let i = 0; i < total; i++) rSum += residual[i]!;
  const rMean = rSum / total;
  for (let i = 0; i < total; i++) residual[i] = (residual[i] ?? 0) - rMean;

  return residual;
}

/**
 * Compute Peak-to-Correlation Energy (PCE) between a test residual W and reference pattern K.
 * Evaluates 2D cross-correlation over a search lag window [-maxLag, +maxLag].
 */
export function computePRNUPCE(
  testResidual: Float64Array,
  refPattern: Float64Array,
  width: number,
  height: number,
  maxLag: number = 8
): PRNUResult {
  const lagDim = 2 * maxLag + 1;
  const corrPlane = new Float64Array(lagDim * lagDim);

  let peakVal = -1.0;
  let peakDy = 0;
  let peakDx = 0;

  // Precompute refPattern norm
  let refNormSq = 0.0;
  for (let i = 0; i < refPattern.length; i++) {
    refNormSq += refPattern[i]! * refPattern[i]!;
  }
  const refNorm = Math.sqrt(Math.max(1e-12, refNormSq));

  // Compute 2D Normalized Cross-Correlation over search grid
  for (let dy = -maxLag; dy <= maxLag; dy++) {
    const planeY = dy + maxLag;
    const yStart = Math.max(0, -dy);
    const yEnd = Math.min(height, height - dy);

    for (let dx = -maxLag; dx <= maxLag; dx++) {
      const planeX = dx + maxLag;
      const xStart = Math.max(0, -dx);
      const xEnd = Math.min(width, width - dx);

      let dot = 0.0;
      let testNormSq = 0.0;

      for (let y = yStart; y < yEnd; y++) {
        const testRow = y * width;
        const refRow = (y + dy) * width;
        for (let x = xStart; x < xEnd; x++) {
          const tVal = testResidual[testRow + x]!;
          const rVal = refPattern[refRow + x + dx]!;
          dot += tVal * rVal;
          testNormSq += tVal * tVal;
        }
      }

      const denom = Math.sqrt(Math.max(1e-12, testNormSq)) * refNorm;
      const ncc = denom > 1e-12 ? dot / denom : 0.0;
      corrPlane[planeY * lagDim + planeX] = ncc;

      if (ncc > peakVal) {
        peakVal = ncc;
        peakDy = dy;
        peakDx = dx;
      }
    }
  }

  // Peak-to-Correlation Energy calculation:
  // Exclude 3x3 window around peak in the lag search grid
  const peakY = peakDy + maxLag;
  const peakX = peakDx + maxLag;
  const exclRadius = 1;

  let outEnergySum = 0.0;
  let outCount = 0;

  for (let py = 0; py < lagDim; py++) {
    for (let px = 0; px < lagDim; px++) {
      if (Math.abs(py - peakY) <= exclRadius && Math.abs(px - peakX) <= exclRadius) {
        continue; // In peak exclusion neighborhood
      }
      const c = corrPlane[py * lagDim + px]!;
      outEnergySum += c * c;
      outCount++;
    }
  }

  const meanOutEnergy = outEnergySum / Math.max(1, outCount);
  const pce = meanOutEnergy > 1e-12 ? (peakVal * peakVal) / meanOutEnergy : 0.0;

  return {
    pce,
    peakCorrelation: peakVal,
    peakOffset: [peakDy, peakDx],
    isMatch: pce >= 60.0,
    correlationPlane: corrPlane,
  };
}
