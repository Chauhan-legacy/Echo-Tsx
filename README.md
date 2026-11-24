# echotrace-math

> **echotrace-math** is a high-assurance, zero-runtime-dependency TypeScript library for forensic signal processing, state estimation, and client-side deep learning.

Designed for Web Audio applications, Web Workers, Node.js services, and browser forensic analyst workbenches.

---

## 1. Installation

### Via npm
```bash
npm install echotrace-math
```

### Direct Browser / Web Worker Single-File Bundle
```html
<script src="dist/echotrace-math.min.js"></script>
<!-- Or in a Web Worker: importScripts('echotrace-math.min.js') -->
```
*Bundle size: 53.98 kB raw (20.10 kB gzip) with full tree-shaking support.*

---

## 2. Quick Code Examples

### Example 1: High-Precision Single-Bin Tone Detection (Goertzel)
```typescript
import { evaluateGoertzelBin } from 'echotrace-math';

const signal = new Float64Array(44100).map((_, i) => Math.sin(2 * Math.PI * 19400 * (i / 44100)));
const result = evaluateGoertzelBin(signal, 44100, 19400);

console.log(`Detected 19.4 kHz magnitude: ${result.magnitude.toFixed(4)}, power: ${result.powerDb.toFixed(2)} dBFS`);
```

### Example 2: Adaptive Kalman Filter with RTS Smoothing (ENF Tracking)
```typescript
import { ENFKalmanTracker } from 'echotrace-math';

const tracker = new ENFKalmanTracker(1.0); // 1.0s sample interval
const noisyMeasurements = [50.02, 50.05, 49.98, 50.01, 50.04];

for (let i = 0; i < noisyMeasurements.length; i++) {
  tracker.step(i, noisyMeasurements[i]!);
}

const smoothed = tracker.smoothRTS();
console.log(`Smoothed ENF at step 4: ${smoothed[4].smoothedFrequencyHz.toFixed(3)} Hz (95% CI: [${smoothed[4].smoothedConfidenceInterval95.map(x => x.toFixed(3))}])`);
```

### Example 3: Camera PRNU Sensor Pattern Peak-to-Correlation Energy (PCE)
```typescript
import { extractNoiseResidual2D, computePCE } from 'echotrace-math';

const residual = extractNoiseResidual2D(rawFrameGrayscale, 1280, 720);
const pceResult = computePCE(residual, referenceFingerprint, 1280, 720);

console.log(`PCE: ${pceResult.pce.toFixed(2)} (Threshold: > 60.0 indicates matching camera sensor)`);
```

---

## 3. Module Overview

- **`numerics/fft`:** Radix-2 decimation-in-time and Bluestein's chirp-Z algorithm for arbitrary non-power-of-two lengths.
- **`numerics/kahan`:** Compensated summation guaranteeing machine-epsilon precision on long floating-point accumulators.
- **`numerics/prng`:** Deterministic, seedable `xoshiro128**` pseudo-random number generator.
- **`linalg/matrix`:** Pure TypeScript dense matrices with Cholesky, Householder QR, Jacobi Eigen, and one-sided Jacobi SVD.
- **`spectral/windows`:** Standard windows (Hann, Hamming, Blackman-Harris, Kaiser, Flat-top) with exact coherent gain and noise bandwidth calculations.
- **`spectral/welch`:** Welch power spectral density estimation with arbitrary segment lengths and windowing.
- **`spectral/multitaper`:** Thomson discrete prolate spheroidal sequence (DPSS) multitaper spectral estimation.
- **`enf/kalman`:** 2D frequency/drift adaptive Kalman filter with Mahalanobis innovation outlier gating and RTS backward smoothing.
- **`enf/crlb`:** Cramer-Rao Lower Bound analytical calculation for multiharmonic frequency estimators.
- **`forensic/rt60`:** Schroeder energy backward integration with Lundeby automated noise-floor truncation.
- **`forensic/prnu`:** Photo-Response Non-Uniformity sensor noise residual extraction and 2D normalized cross-correlation PCE.
- **`forensic/rppg`:** Remote photoplethysmography using CHROM and POS plane-orthogonal-to-skin projection algorithms.
- **`forensic/doa`:** Multiple Signal Classification (MUSIC) and ESPRIT subspace direction-of-arrival estimators.
- **`ml/network`:** Temporal attention neural network with depthwise separable convolutions and ALiBi positional slopes.
- **`ml/classic`:** Logistic Regression (IRLS), SVM (SMO), k-NN, PCA, and Mahalanobis anomaly detection.

---

## 4. Stated Accuracy & Numerical Tolerance

| Algorithm | Stated Tolerance | Measured Deviation | Verdict |
| :--- | :--- | :--- | :---: |
| **Bluestein vs Naive DFT** | Error $< 10^{-10}$ | $3.089 \times 10^{-12}$ | **PASS** |
| **Kahan Summation** | Error $< 10^{-12}$ | $0.000 \times 10^{-16}$ | **PASS** |
| **GELU Analytical vs Numerical Gradient** | Error $< 10^{-5}$ | $1.24 \times 10^{-6}$ | **PASS** |
| **PointwiseConv1D Finite-Diff Gradient** | Error $< 10^{-6}$ | $4.18 \times 10^{-7}$ | **PASS** |
| **Orthogonal Wavelet Reconstruction** | Error $< 10^{-9}$ | $1.82 \times 10^{-10}$ | **PASS** |

---

## 5. Measured Host Execution Benchmarks

Measured on host workstation via `scripts/benchmark.ts`:

- **Radix-2 FFT ($N=1024$):** 26.77 µs / transform
- **Radix-2 FFT ($N=2048$):** 50.78 µs / transform
- **Bluestein FFT ($N=1009$ prime):** 173.58 µs / transform
- **Kalman ENF Track (600s history):** 0.84 µs / update step
- **MUSIC Pseudo-Spectrum (8 ULA sensors, 181 angles):** 0.148 ms / op
- **Temporal Attention Forward Pass (Batch 1, $T=50, D=16$):** 0.753 ms

---

## 6. Mathematical Foundations

Complete equations, paper derivations, and parameter bounds are documented in [docs/MATH.md](file:///c:/Users/Priyansh/Desktop/NEW_PROJECTS/packages/echotrace-math/docs/MATH.md).

---

## 7. How to Cite

If you use `echotrace-math` in academic research or industrial forensics, please cite using [CITATION.cff](file:///c:/Users/Priyansh/Desktop/NEW_PROJECTS/packages/echotrace-math/CITATION.cff):

```bibtex
@software{echotrace_math_2026,
  author = {ECHO//TRACE Team},
  title = {echotrace-math: High-Assurance Client-Side Forensic Signal Processing and Machine Learning Library in TypeScript},
  year = {2026},
  url = {https://github.com/echotrace/echotrace-math},
  version = {1.0.0}
}
```

---

## 8. License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](file:///c:/Users/Priyansh/Desktop/NEW_PROJECTS/packages/echotrace-math/LICENSE) and [NOTICE](file:///c:/Users/Priyansh/Desktop/NEW_PROJECTS/packages/echotrace-math/NOTICE).
