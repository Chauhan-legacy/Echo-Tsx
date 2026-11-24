# echotrace-math

Zero-dependency TypeScript library for mathematical signal analysis, acoustic forensics, and client-side neural estimation.

## Architecture

- **Numerics**: Xoshiro128 PRNG, Kahan compensated summation, Bluestein arbitrary-length FFT.
- **Linear Algebra**: Cache-blocked DenseMatrix, Cholesky, Householder QR, Jacobi Eigen-solver, Jacobi SVD with Moore-Penrose pseudo-inverse.
- **Spectral Analysis**: Hann, Hamming, Blackman-Harris, Kaiser windows, Welch PSD, Multitaper DPSS, Goertzel filter, Daubechies D4 Wavelet Packets with Shannon entropy best-basis selection.
- **Forensic Estimators**: Schroeder and Lundeby RT60 reverberation, CUSUM and BOCPD change-point detection, PRNU sensor noise residual extraction with Peak-to-Correlation Energy, rPPG facial pulse recovery (CHROM and POS), YIN pitch estimation, LPC formants, MUSIC and ESPRIT DOA, FastICA signal unmixing, Bayesian log-odds evidence fusion, Huber and Theil-Sen robust statistics.
- **Deep Learning**: Dilated depthwise separable 1D convolutions, multi-head attention with ALiBi relative position bias, AdamW optimizer, cosine annealing schedule, finite-difference gradient verification, temperature scaling, and integrated gradients.

## Usage

```typescript
import {
  FastFourierTransform,
  computeWelchPSD,
  detectCUSUM,
  estimateRT60,
  ForensicDetectorNetwork
} from 'echotrace-math';

// 1. Arbitrary-length FFT
const fft = new FastFourierTransform(100);
const { realOut, imagOut } = fft.forward(inputSignal);

// 2. Room reverberation estimation
const rt60 = estimateRT60(impulseResponse, 16000);
console.log('RT60:', rt60.t20Seconds);

// 3. Multi-task forensic model
const model = new ForensicDetectorNetwork(32, 32);
const prediction = model.forward(features, 10);
console.log('Leak score:', prediction.leakScore);
```

## Verification

The library contains a full test suite with 31 unit tests validating analytical derivations against known mathematical references.
