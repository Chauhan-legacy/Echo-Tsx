/**
 * Forensic Evidence Fusion via Bayesian Log-Odds and Likelihood Ratios.
 * Follows the European Network of Forensic Science Institutes (ENFSI) guideline.
 *
 * Mathematical Framework:
 * Let H_1 be the hypothesis that the media is leaking or tampered, and H_0 authentic.
 * By Bayes' Rule in log-odds formulation:
 *   logit(P(H_1 | E_1, ..., E_K)) = logit(P(H_1)) + \sum_{i=1}^K \ln(LR_i)
 * where LR_i = P(E_i | H_1) / P(E_i | H_0) is the Likelihood Ratio of forensic test i.
 *
 * Posterior probability:
 *   P(H_1 | E) = \frac{1}{1 + \exp(-\text{posterior\_log\_odds})}
 */

export interface ForensicEvidenceVector {
  id: string;
  name: string;
  likelihoodRatio: number; // LR > 1 supports H_1; LR < 1 supports H_0
  weight?: number;         // Reliability weight [0, 1]
  confidence: number;      // Confidence in detector [0, 1]
  description?: string;
}

export type VerbalSupportLevel =
  | 'Inconclusive'
  | 'Slight Support'
  | 'Moderate Support'
  | 'Strong Support'
  | 'Very Strong Support'
  | 'Extremely Strong Support';

export interface BayesianFusionResult {
  posteriorProbability: number;
  posteriorLogOdds: number;
  priorProbability: number;
  bayesFactor: number;
  verbalConclusion: VerbalSupportLevel;
  contributions: { id: string; name: string; logOddsDelta: number; percentContribution: number }[];
}

/**
 * Map Bayes factor to ENFSI verbal scale of evidential strength.
 */
export function getVerbalEquivalent(bayesFactor: number): VerbalSupportLevel {
  if (bayesFactor < 1.0) {
    if (bayesFactor > 0.1) return 'Inconclusive';
    return 'Inconclusive';
  }
  if (bayesFactor < 10.0) return 'Slight Support';
  if (bayesFactor < 100.0) return 'Moderate Support';
  if (bayesFactor < 1000.0) return 'Strong Support';
  if (bayesFactor < 10000.0) return 'Very Strong Support';
  return 'Extremely Strong Support';
}

/**
 * Fuse heterogeneous forensic evidence vectors into a unified calibrated probability.
 */
export function fuseForensicEvidence(
  evidence: ForensicEvidenceVector[],
  priorProbabilityH1: number = 0.5
): BayesianFusionResult {
  const p0 = Math.min(0.999, Math.max(0.001, priorProbabilityH1));
  const priorLogOdds = Math.log(p0 / (1.0 - p0));

  let totalLogLR = 0.0;
  const rawDeltas: { id: string; name: string; delta: number }[] = [];

  for (const ev of evidence) {
    // Regularize LR to avoid -Infinity / +Infinity
    const lr = Math.max(1e-6, Math.min(1e6, ev.likelihoodRatio));
    const w = (ev.weight ?? 1.0) * ev.confidence;
    const delta = w * Math.log(lr);
    totalLogLR += delta;
    rawDeltas.push({ id: ev.id, name: ev.name, delta });
  }

  const posteriorLogOdds = priorLogOdds + totalLogLR;
  // Stable sigmoid: 1 / (1 + exp(-z))
  const posteriorProbability = posteriorLogOdds >= 0
    ? 1.0 / (1.0 + Math.exp(-posteriorLogOdds))
    : Math.exp(posteriorLogOdds) / (1.0 + Math.exp(posteriorLogOdds));

  const bayesFactor = Math.exp(Math.max(-20.0, Math.min(20.0, totalLogLR)));
  const verbalConclusion = getVerbalEquivalent(bayesFactor);

  // Compute percentage contributions
  let absSum = 0.0;
  for (const d of rawDeltas) absSum += Math.abs(d.delta);

  const contributions = rawDeltas.map(d => ({
    id: d.id,
    name: d.name,
    logOddsDelta: Math.round(d.delta * 1000) / 1000,
    percentContribution: absSum > 1e-6 ? Math.round((Math.abs(d.delta) / absSum) * 1000) / 10 : 0,
  }));

  return {
    posteriorProbability: Math.round(posteriorProbability * 10000) / 10000,
    posteriorLogOdds: Math.round(posteriorLogOdds * 1000) / 1000,
    priorProbability: p0,
    bayesFactor: Math.round(bayesFactor * 100) / 100,
    verbalConclusion,
    contributions,
  };
}
