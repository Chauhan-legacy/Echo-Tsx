# Echo-Tsx (echotrace-math)

<p align="center">
  <img src="https://raw.githubusercontent.com/Chauhan-legacy/Echo-Tsx/main/docs/banner.png" alt="Echo-Tsx Banner" width="100%" onerror="this.style.display='none'"/>
</p>

<p align="center">
  <strong>High-assurance, zero-runtime-dependency TypeScript library for physical signal forensics, statistical state estimation, numerical linear algebra, and client-side deep learning.</strong>
</p>

<p align="center">
  <a href="https://github.com/Chauhan-legacy/Echo-Tsx/actions/workflows/ci.yml"><img src="https://github.com/Chauhan-legacy/Echo-Tsx/actions/workflows/ci.yml/badge.svg" alt="CI Build Status" /></a>
  <img src="https://img.shields.io/badge/Dependencies-0%20Runtime-22c55e?style=flat-square" alt="Zero Dependencies" />
  <img src="https://img.shields.io/badge/TypeScript-Strict%205.x%2F7.x-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript Strict" />
  <img src="https://img.shields.io/badge/Minified%20Gzip-22.2%20kB-brightgreen?style=flat-square" alt="Bundle Size" />
  <img src="https://img.shields.io/badge/Platform-Node.js%20•%20Browser%20•%20Worker-orange?style=flat-square" alt="Platform" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License" /></a>
</p>

---

## ⚡ Why Echo-Tsx?

In modern web, mobile, and serverless engineering, performing scientific-grade digital signal processing, numerical linear algebra (SVD, Eigenvalues), or state estimation (Kalman filters) traditionally required:
1. **Bulky C++ WebAssembly binaries** that add megabytes of overhead and suffer cold-start compilation lag.
2. **Heavy Python backend microservices** (NumPy / SciPy) that destroy user privacy by requiring sensitive raw media uploads.
3. **Outdated, unmaintained npm packages** with floating-point drift, memory leaks, and missing TypeScript types.

**Echo-Tsx eliminates this tradeoff.** It provides a pure, zero-dependency mathematical and signal processing foundation built strictly with standard JavaScript typed arrays (`Float32Array`, `Float64Array`).

- 🔒 **100% Client-Side Privacy:** Process microphone, camera, and sensor data directly in the user's browser or device sandbox.
- ⚡ **GC-Free Execution:** Pre-allocated buffer reuse and in-place algorithms guarantee zero garbage collection pauses during real-time 60 FPS animation or 48 kHz audio streams.
- 📐 **Rigorous Numerical Assurance:** Kahan compensated summation, Bluestein arbitrary-length chirp-Z transforms, and RTS backward smoothing verified to within $< 10^{-10}$ error against analytical ground truth.
- 📦 **Featherweight Footprint:** Entire library bundles to **22.2 kB minified gzip** (dual ESM + CommonJS + UMD).

---

## 🚀 Quick Start & Installation

### Via npm
```bash
npm install echo-tsx
# or
npm install Chauhan-legacy/Echo-Tsx
```

### Direct Browser / CDN / Web Worker
```html
<!-- Standalone minified UMD bundle (zero build step required) -->
<script src="https://cdn.jsdelivr.net/gh/Chauhan-legacy/Echo-Tsx@main/dist/echotrace-math.min.js"></script>
<script>
  // Global EchoTraceMath is immediately available
  const { evaluateGoertzelBin, fitAircraftDoppler } = window.EchoTraceMath;
</script>
```

In a dedicated Web Worker:
```javascript
importScripts('https://cdn.jsdelivr.net/gh/Chauhan-legacy/Echo-Tsx@main/dist/echotrace-math.min.js');
```

---

## 🌐 Universal Interoperability Guide

Echo-Tsx is engineered to be **framework-agnostic** and seamlessly integrate with any existing JavaScript or TypeScript codebase:

### 1. Web Audio API (Microphone & AudioBuffer Streams)
Feed standard browser `AudioBuffer` or `AudioWorklet` PCM samples directly into Echo-Tsx:

```typescript
import { evaluateGoertzelBin, computeWelchPsd } from 'echo-tsx';

// 1. Decode audio file or capture live microphone
const audioCtx = new AudioContext();
const response = await fetch('/audio/recording.wav');
const audioBuffer = await audioCtx.decodeAudioData(await response.arrayBuffer());

// 2. Extract standard Float32Array PCM channel samples
const channelData = audioBuffer.getChannelData(0);

// 3. Scan for inaudible 19.4 kHz tracking beacons with Goertzel
const bin = evaluateGoertzelBin(channelData, audioBuffer.sampleRate, 19400);
if (bin.powerDb > -45.0) {
  console.warn(`Ultrasonic beacon detected at 19.4 kHz! Power: ${bin.powerDb.toFixed(1)} dBFS`);
}

// 4. Compute Welch Power Spectral Density
const psd = computeWelchPsd(channelData, audioBuffer.sampleRate, { nfft: 2048, window: 'hann' });
```

---

### 2. Node.js & Serverless Backends (Express, Fastify, AWS Lambda)
Analyze uploaded media streams or sensor telemetry on the backend with zero native compilation dependencies:

```typescript
import { ENFKalmanTracker, fitAircraftDoppler } from 'echo-tsx';
import fs from 'fs';

// Process incoming telemetry or frequency readings from IoT sensors
const tracker = new ENFKalmanTracker(1.0); // 1.0s step interval

sensorReadings.forEach((reading, index) => {
  const state = tracker.step(index, reading.measuredHz);
  console.log(`Step ${index} | Filtered: ${state.estimatedFrequencyHz.toFixed(3)} Hz | Innovation: ${state.innovationResidual.toFixed(4)}`);
});

// Run Rauch-Tung-Striebel (RTS) backward smoother over full history
const smoothedTrajectory = tracker.smoothRTS();
```

---

### 3. Web Workers (Zero-Jank Background Processing)
Offload heavy FFTs and multi-head attention models off the browser main thread:

```typescript
// worker.ts
import { computeRealSpectrogram, TemporalAttentionModel } from 'echo-tsx';

self.onmessage = (e: MessageEvent<{ pcm: Float32Array; sampleRate: number }>) => {
  const { pcm, sampleRate } = e.data;
  
  // Fast 2D STFT Spectrogram Matrix
  const spectrogram = computeRealSpectrogram(pcm, sampleRate, 360, 160);
  
  // Post computed matrix back to main UI thread at 60 FPS
  self.postMessage({ spectrogram }, [spectrogram.matrix.buffer]);
};
```

---

### 4. Computer Vision & HTML5 Canvas / OpenCV.js
Extract camera sensor fingerprints and Photo-Response Non-Uniformity (PRNU) from video frames:

```typescript
import { extractNoiseResidual2D, computePCE } from 'echo-tsx';

// Grab ImageData from an HTML5 Canvas or video stream
const ctx = canvas.getContext('2d')!;
const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

// Extract sensor noise residual
const residual = extractNoiseResidual2D(imgData.data, canvas.width, canvas.height);

// Compare against camera reference sensor fingerprint
const match = computePCE(residual, cameraFingerprint, canvas.width, canvas.height);
console.log(`Peak-to-Correlation Energy (PCE): ${match.pce.toFixed(2)} (Match: ${match.pce > 60.0})`);
```

---

### 5. SciPy & NumPy to Echo-Tsx Migration Cheat-Sheet

| Python / SciPy / NumPy | Echo-Tsx (TypeScript) | Description |
| :--- | :--- | :--- |
| `np.fft.fft(x)` | `new FFT(N).transform(re, im)` | Radix-2 Cooley-Tukey DIT FFT |
| `scipy.signal.czt(x)` | `bluesteinFFT(real, imag)` | Arbitrary-length prime / non-$2^N$ transform |
| `scipy.signal.welch(x, fs)` | `computeWelchPsd(pcm, fs, opts)` | Welch power spectral density |
| `scipy.signal.windows.hann(N)` | `hannWindow(N)` | Deterministic window functions |
| `scipy.linalg.svd(A)` | `svd(matrix)` | One-sided Jacobi Singular Value Decomposition |
| `scipy.linalg.cholesky(A)` | `cholesky(matrix)` | Symmetric positive-definite decomposition |
| `scipy.linalg.qr(A)` | `householderQR(matrix)` | Orthogonal Householder QR factorization |
| `scipy.linalg.eigh(A)` | `jacobiEigen(symmetricMatrix)` | Real symmetric eigenvalue decomposition |

---

## 🛠 Real-World Code Recipes

### Recipe 1: Commercial Aircraft Acoustic Doppler Tracking (SkyLeak OSINT)
Fits non-linear acoustic Doppler S-curves ($f(t) = f_0 \left(1 - \frac{v}{c} \frac{v(t - t_0)}{\sqrt{d_{\min}^2 + v^2(t - t_0)^2}}\right)$) to detect high-altitude commercial jet flyovers and triangulate observer ground position to within $\approx 95\text{m}$:

```typescript
import { fitAircraftDoppler, evaluateDopplerFreq } from 'echo-tsx';

// Extracted spectral ridge timestamps and peak frequencies from audio STFT
const timePoints = new Float64Array([1.0, 3.0, 5.0, 7.5, 10.0, 12.0, 14.0]);
const frequencies = new Float64Array([182.4, 180.1, 175.2, 160.0, 144.8, 139.9, 137.6]);

// Perform bounded non-linear kinematic curve fit
const fit = fitAircraftDoppler(timePoints, frequencies);

console.log(`Aircraft Speed:       ${(fit.v * 3.6).toFixed(1)} km/h (${(fit.v * 1.94384).toFixed(1)} knots)`);
console.log(`Closest Approach:     ${fit.dMin.toFixed(0)} meters slant range`);
console.log(`Blade-Pass Fundamental: ${fit.f0.toFixed(1)} Hz`);
console.log(`Goodness-of-Fit (R²): ${fit.r2.toFixed(4)}`);
```

---

### Recipe 2: Dense Linear Algebra without WASM
Zero-dependency matrix operations running with machine precision:

```typescript
import { Matrix, svd, cholesky, householderQR } from 'echo-tsx';

// Create 3x3 matrix
const A = new Matrix(3, 3, new Float64Array([
  4, 12, -16,
  12, 37, -43,
  -16, -43, 98
]));

// Cholesky decomposition: A = L * L^T
const L = cholesky(A);

// Singular Value Decomposition: A = U * S * V^T
const { U, S, V } = svd(A);
console.log('Singular Values:', S);
```

---

### Recipe 3: Client-Side Deep Learning (Temporal Attention)
Run sequence classification and anomaly detection using depthwise separable convolutions and multi-head attention:

```typescript
import { TemporalAttentionModel } from 'echo-tsx';

// Initialize model: input_dim=8, d_model=16, num_heads=2, num_classes=3
const model = new TemporalAttentionModel(8, 16, 2, 3);

// Input batch: 1 sequence of 25 timesteps, 8 features per step
const inputBuffer = new Float32Array(1 * 25 * 8).fill(0.5);
const logits = model.forward(inputBuffer, 1, 25);

console.log('Class Logits:', logits);
```

---

## 📊 Numerical Verification & Benchmarks

Every algorithm in Echo-Tsx is benchmarked against rigorous analytical criteria:

| Mathematical Primitive | Reference Standard | Stated Tolerance | Measured Deviation | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Bluestein DFT** | Analytical Naive DFT | Error $< 10^{-10}$ | $3.089 \times 10^{-12}$ | **PASS** |
| **Kahan Compensated Sum** | Extended Precision | Error $< 10^{-12}$ | $0.000 \times 10^{-16}$ | **PASS** |
| **GELU Activation Gradient** | Numerical Finite-Diff | Error $< 10^{-5}$ | $1.24 \times 10^{-6}$ | **PASS** |
| **PointwiseConv1D Gradient** | Numerical Finite-Diff | Error $< 10^{-6}$ | $4.18 \times 10^{-7}$ | **PASS** |
| **Orthogonal Wavelet DWT** | Perfect Reconstruction | Error $< 10^{-9}$ | $1.82 \times 10^{-10}$ | **PASS** |

### Execution Performance Benchmarks
Measured on standard workstation hardware via `npx tsx scripts/benchmark.ts`:
- **Radix-2 FFT ($N=1024$):** `26.77 µs` / transform
- **Radix-2 FFT ($N=2048$):** `50.78 µs` / transform
- **Bluestein Arbitrary FFT ($N=1009$ prime):** `173.58 µs` / transform
- **Kalman 2D Filter Step:** `0.84 µs` / step
- **MUSIC Direction-of-Arrival (8 sensors, 181 angles):** `0.148 ms` / frame
- **Temporal Attention Forward Pass ($T=50, D=16$):** `0.753 ms`

---

## 📦 Module Directory

- **`numerics/`:** Radix-2 FFT, Bluestein chirp-Z, Kahan summation, `xoshiro128**` PRNG, numerical activations.
- **`spectral/`:** Welch PSD, DPSS multitaper, Goertzel single-bin detector, Wavelet packet decomposition, Window functions (Hann, Hamming, Blackman-Harris, Kaiser).
- **`enf/`:** Adaptive 2D Kalman filter, RTS backward smoother, multiharmonic CRLB bounds, grid pool matching.
- **`forensic/`:** Aircraft Doppler kinematics, Schroeder RT60 reverberation, PRNU sensor noise & PCE, rPPG heart rate extraction, MUSIC/ESPRIT direction of arrival, FastICA.
- **`linalg/`:** Dense matrix operations, SVD, Cholesky, Householder QR, Jacobi Eigenvalues.
- **`ml/`:** Depthwise separable convolutions, temporal multi-head attention, ALiBi slopes, Platt scaling calibration, classical classifiers.

---

## 📄 License & Attribution

Licensed under the **Apache License, Version 2.0**. You are free to use, embed, modify, and distribute this software for personal, academic, or commercial purposes. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2026 Priyansh Chauhan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

If you use Echo-Tsx in research or engineering publications, please cite using [CITATION.cff](CITATION.cff):

```bibtex
@software{echo_tsx_2026,
  author = {Priyansh Chauhan and ECHO//TRACE Team},
  title = {Echo-Tsx (echotrace-math): High-Assurance Zero-Dependency Forensic Signal Processing and Machine Learning in TypeScript},
  year = {2026},
  url = {https://github.com/Chauhan-legacy/Echo-Tsx},
  version = {1.0.0}
}
```
