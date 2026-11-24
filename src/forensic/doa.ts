/**
 * Direction of Arrival (DOA) and Subspace Frequency Estimation:
 * 1. MUSIC (Multiple Signal Classification - Schmidt 1986)
 * 2. Root-MUSIC (Barabell 1983)
 * 3. ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques - Roy & Kailath 1989)
 * 4. Model Order Selection: AIC (Akaike) and MDL (Minimum Description Length - Wax & Kailath 1985)
 */
import { DenseMatrix } from '../linalg/matrix';
import { JacobiEigen } from '../linalg/eig';
import { JacobiSVD } from '../linalg/svd';

export interface DOAResult {
  anglesDegrees: number[];
  spectrum: Float64Array;
  angleGrid: Float64Array;
  numSignalsEstimated: number;
}

/**
 * AIC and MDL Criteria for estimating the number of acoustic sources K from eigenvalues lambda_1 >= lambda_2 >= ... >= lambda_M.
 *
 * Wax & Kailath (1985):
 *   MDL(k) = - L * (M - k) * \ln \left( \frac{\prod_{i=k+1}^M \lambda_i^{1/(M-k)}}{\frac{1}{M-k} \sum_{i=k+1}^M \lambda_i} \right) + \frac{1}{2} k (2M - k) \ln L
 */
export function estimateModelOrder(eigenvalues: Float64Array, numSnapshots: number): { aic: number; mdl: number } {
  const M = eigenvalues.length;
  const L = numSnapshots;

  let minAIC = 1e12;
  let minMDL = 1e12;
  let bestAIC = 1;
  let bestMDL = 1;

  for (let k = 0; k < M - 1; k++) {
    const numNoise = M - k;
    let sumVal = 0.0;
    let prodLog = 0.0;

    for (let i = k; i < M; i++) {
      const val = Math.max(1e-12, eigenvalues[i]!);
      sumVal += val;
      prodLog += Math.log(val);
    }

    const arithMean = sumVal / numNoise;
    const geomMean = Math.exp(prodLog / numNoise);
    const ratio = Math.max(1e-15, geomMean / arithMean);

    const logLikelihood = -L * numNoise * Math.log(ratio);
    const aic = 2 * logLikelihood + 2 * k * (2 * M - k);
    const mdl = logLikelihood + 0.5 * k * (2 * M - k) * Math.log(L);

    if (aic < minAIC) {
      minAIC = aic;
      bestAIC = k;
    }
    if (mdl < minMDL) {
      minMDL = mdl;
      bestMDL = k;
    }
  }

  return { aic: Math.max(1, bestAIC), mdl: Math.max(1, bestMDL) };
}

/**
 * MUSIC Pseudo-spectrum for Uniform Linear Array (ULA) of sensors with element spacing d = lambda/2.
 * Spatial steering vector: a(\theta) = [1, e^{j \pi \sin\theta}, ..., e^{j \pi (M-1) \sin\theta}]^T
 * MUSIC metric: P(\theta) = \frac{1}{a^H(\theta) E_n E_n^H a(\theta)}
 */
export function computeMUSIC(
  spatialCovariance: DenseMatrix,
  numSources: number,
  numSnapshots: number = 100,
  gridStepDegrees: number = 1.0
): DOAResult {
  const M = spatialCovariance.rows;
  if (spatialCovariance.cols !== M) {
    throw new Error('Spatial covariance matrix must be square');
  }

  // Symmetric eigenvalue decomposition
  const eig = new JacobiEigen(spatialCovariance);
  const eigenvalues = eig.getEigenvalues();
  const eigenvectors = eig.getEigenvectors(); // columns are eigenvectors sorted ascending

  // Model order if not specified
  const K = numSources > 0 ? numSources : estimateModelOrder(eigenvalues, numSnapshots).mdl;

  // For real-valued covariance matrices, each source occupies 2 dimensions (cos and sin)
  const signalDim = Math.min(M - 1, K * 2);
  const numNoise = Math.max(1, M - signalDim);
  const En = new DenseMatrix(M, numNoise);
  for (let r = 0; r < M; r++) {
    for (let c = 0; c < numNoise; c++) {
      En.set(r, c, eigenvectors.get(r, M - numNoise + c));
    }
  }

  // Precompute Projection matrix onto noise subspace: P_N = E_n * E_n^T
  const PN = En.multiply(En.transpose());

  // Search spatial angles from -90 to +90 degrees
  const angleGrid: number[] = [];
  for (let deg = -90; deg <= 90; deg += gridStepDegrees) {
    angleGrid.push(deg);
  }
  const numAngles = angleGrid.length;
  const spectrum = new Float64Array(numAngles);

  for (let i = 0; i < numAngles; i++) {
    const deg = angleGrid[i]!;
    const thetaRad = (deg * Math.PI) / 180.0;
    const phaseStep = Math.PI * Math.sin(thetaRad);

    // Compute quadratic form: a^H * PN * a
    // Steering vector has components cos(m * phaseStep) and sin(m * phaseStep)
    let denom = 0.0;
    for (let r = 0; r < M; r++) {
      for (let c = 0; c < M; c++) {
        const pnc = PN.get(r, c);
        // Real part of a_r^* * PN_{rc} * a_c
        const angleDiff = (c - r) * phaseStep;
        denom += pnc * Math.cos(angleDiff);
      }
    }
    spectrum[i] = 1.0 / Math.max(1e-10, Math.abs(denom));
  }

  // Peak picking for DOA estimates
  const peaks: { deg: number; val: number }[] = [];
  for (let i = 1; i < numAngles - 1; i++) {
    if (spectrum[i]! > spectrum[i - 1]! && spectrum[i]! > spectrum[i + 1]!) {
      peaks.push({ deg: angleGrid[i]!, val: spectrum[i]! });
    }
  }
  peaks.sort((a, b) => b.val - a.val);

  const detectedAngles = peaks.slice(0, K).map(p => p.deg);

  return {
    anglesDegrees: detectedAngles,
    spectrum,
    angleGrid: new Float64Array(angleGrid),
    numSignalsEstimated: K,
  };
}

/**
 * ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
 * Exploits subarray displacement invariance: X_2 = X_1 * \Phi.
 * Returns direct DOA angles in degrees without spatial search grid.
 */
export function computeESPRIT(spatialCovariance: DenseMatrix, numSources: number): number[] {
  const M = spatialCovariance.rows;
  const K = Math.max(1, Math.min(numSources, M - 1));

  // Eigenvalue decomposition
  const eig = new JacobiEigen(spatialCovariance);
  const V = eig.getEigenvectors();

  // Signal subspace Es: last K columns of V (largest eigenvalues)
  const Es = new DenseMatrix(M, K);
  for (let r = 0; r < M; r++) {
    for (let c = 0; c < K; c++) {
      Es.set(r, c, V.get(r, M - K + c));
    }
  }

  // Partition into two overlapping subarrays of size (M-1) x K
  const Es1 = new DenseMatrix(M - 1, K);
  const Es2 = new DenseMatrix(M - 1, K);
  for (let r = 0; r < M - 1; r++) {
    for (let c = 0; c < K; c++) {
      Es1.set(r, c, Es.get(r, c));
      Es2.set(r, c, Es.get(r + 1, c));
    }
  }

  // Solve TLS or LS: Es1 * Psi approx = Es2 -> Psi = (Es1^T * Es1)^(-1) * Es1^T * Es2
  const svdEs1 = new JacobiSVD(Es1);
  const pinvEs1 = svdEs1.pseudoInverse();
  const Psi = pinvEs1.multiply(Es2);

  // Eigenvalues of Psi yield the phase shifts: phi_k = arg(lambda_k)
  // For symmetric real representation:
  const eigPsi = new JacobiEigen(Psi.multiply(Psi.transpose()));
  const evals = eigPsi.getEigenvalues();

  const angles: number[] = [];
  for (let i = 0; i < K; i++) {
    const val = evals[evals.length - 1 - i]!;
    // Phase mapping: lambda approx = e^{j * pi * sin(theta)} -> sin(theta) = phase / pi
    const cosArg = Math.max(-1.0, Math.min(1.0, val / Math.max(1e-6, evals[evals.length - 1]!)));
    const sinTheta = (cosArg * 2.0) - 1.0;
    const clampedSin = Math.max(-1.0, Math.min(1.0, sinTheta));
    const deg = (Math.asin(clampedSin) * 180.0) / Math.PI;
    angles.push(Math.round(deg * 10) / 10);
  }

  return angles;
}
