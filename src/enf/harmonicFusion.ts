import { goertzelTone } from '../spectral/goertzel';
import { welchPSD } from '../spectral/welch';

export interface HarmonicPeak {
  harmonicNumber: number; // 1, 2, 3, 4
  expectedHz: number;
  measuredHz: number;
  snrDb: number;
  weight: number;
}

export interface ENFFusionResult {
  nominalGridHz: 50 | 60;
  estimatedFundamentalHz: number;
  confidence: number;
  snrWeightedDeviationHz: number;
  harmonics: HarmonicPeak[];
}

/**
 * SNR-Weighted Multi-Harmonic ENF Estimator
 *
 * Combines fundamental and harmonics (1x, 2x, 3x, 4x) weighted by their individual SNR.
 * Sub-harmonic estimates are mapped back to fundamental: f_fundamental = f_k / k.
 */
export function fuseENFHarmonics(
  signal: ArrayLike<number>,
  sampleRate: number,
  nominalGrid: 50 | 60 = 50
): ENFFusionResult {
  const harmonicsCount = 4;
  const harmonics: HarmonicPeak[] = [];

  // Initial coarse Welch PSD to find noise floor
  const welch = welchPSD(signal, sampleRate, { windowSize: 2048, windowType: 'blackman-harris' });
  const psd = welch.psd;
  const freqs = welch.frequencies;

  let totalWeight = 0.0;
  let weightedFreqSum = 0.0;

  for (let k = 1; k <= harmonicsCount; k++) {
    const targetHz = nominalGrid * k;
    if (targetHz >= sampleRate * 0.45) break;

    // Evaluate peak near targetHz +/- 1.5 Hz
    let peakPower = 0.0;
    let peakF = targetHz;

    for (let f = 0; f < freqs.length; f++) {
      const curF = freqs[f]!;
      if (Math.abs(curF - targetHz) <= 2.0) {
        if (psd[f]! > peakPower) {
          peakPower = psd[f]!;
          peakF = curF;
        }
      }
    }

    // Estimate local noise floor +/- 5 Hz away
    let noiseSum = 0.0;
    let noiseCount = 0;
    for (let f = 0; f < freqs.length; f++) {
      const curF = freqs[f]!;
      const df = Math.abs(curF - targetHz);
      if (df >= 3.0 && df <= 8.0) {
        noiseSum += psd[f]!;
        noiseCount++;
      }
    }
    const noiseFloor = noiseCount > 0 ? noiseSum / noiseCount : 1e-12;
    const snrLinear = peakPower / Math.max(1e-15, noiseFloor);
    const snrDb = 10.0 * Math.log10(Math.max(1.0, snrLinear));

    // Refine with Goertzel 3-point search around peakF
    const gStep = 0.02;
    let bestGPower = -Infinity;
    let refinedF = peakF;
    for (let step = -4; step <= 4; step++) {
      const testF = peakF + step * gStep;
      const g = goertzelTone(signal, sampleRate, testF);
      if (g.powerDb > bestGPower) {
        bestGPower = g.powerDb;
        refinedF = testF;
      }
    }

    // Weight by SNR (harmonics with SNR < 3 dB get 0 weight)
    const weight = snrDb > 3.0 ? Math.pow(10.0, snrDb / 10.0) * (1.0 / (k * k)) : 0.0;
    const fundamentalCandidate = refinedF / k;

    harmonics.push({
      harmonicNumber: k,
      expectedHz: targetHz,
      measuredHz: Number(refinedF.toFixed(4)),
      snrDb: Number(snrDb.toFixed(1)),
      weight: Number(weight.toFixed(2)),
    });

    if (weight > 0) {
      weightedFreqSum += fundamentalCandidate * weight;
      totalWeight += weight;
    }
  }

  const fundamental = totalWeight > 0 ? weightedFreqSum / totalWeight : nominalGrid;
  const deviation = fundamental - nominalGrid;
  const confidence = totalWeight > 0 ? Math.min(0.99, Number((totalWeight / (totalWeight + 10.0)).toFixed(2))) : 0.05;

  return {
    nominalGridHz: nominalGrid,
    estimatedFundamentalHz: Number(fundamental.toFixed(4)),
    confidence,
    snrWeightedDeviationHz: Number(deviation.toFixed(4)),
    harmonics,
  };
}
