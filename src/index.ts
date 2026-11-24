/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * ECHO//TRACE Math & ML Engine
 * High-precision, zero-dependency forensic signal processing and deep learning library.
 * Designed for browser, Node.js, and Web Worker environments.
 */

// Numerics
export * from './numerics/prng';
export * from './numerics/kahan';
export * from './numerics/activations';
export * from './numerics/fft';

// Linear Algebra
export * from './linalg/matrix';
export * from './linalg/cholesky';
export * from './linalg/qr';
export * from './linalg/eig';
export * from './linalg/svd';

// Spectral Analysis
export * from './spectral/windows';
export * from './spectral/welch';
export * from './spectral/multitaper';
export * from './spectral/goertzel';
export * from './spectral/wavelet';

// Electric Network Frequency (ENF) Forensics
export * from './enf/crlb';
export * from './enf/harmonicFusion';
export * from './enf/kalman';
export * from './enf/matching';

// Synthetic Forensic Ground-Truth Test Signals
export * from './signals/testSignals';

// Forensic Acoustics, Vision & Sensor Modules
export * from './forensic/rt60';
export * from './forensic/changepoint';
export * from './forensic/prnu';
export * from './forensic/rppg';
export * from './forensic/voice';
export * from './forensic/doa';
export * from './forensic/ica';
export * from './forensic/bayesFusion';
export * from './forensic/robustStats';
export * from './forensic/aircraftDoppler';

// Forensic Privacy & Redaction
export * from './privacy/spectralInpainting';

// Pure TypeScript Deep Learning & Explainability
export * from './ml/layers';
export * from './ml/attention';
export * from './ml/network';
export * from './ml/training';
export * from './ml/calibration';
export * from './ml/explainability';
export * from './ml/classic';
