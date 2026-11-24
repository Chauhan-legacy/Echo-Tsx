/**
 * Ground-truth synthetic test signals for validating forensic DSP and ML algorithms.
 * Zero external dependencies. Uses Xoshiro128p for deterministic PRNG.
 */
import { Xoshiro128p } from '../numerics/prng';

export interface SignalGeneratorOptions {
  sampleRate: number;
  durationSeconds: number;
  seed?: number;
}

/**
 * Generate a pure sinusoidal tone: x(t) = A * sin(2*pi*f*t + phi)
 */
export function generateSineTone(
  frequencyHz: number,
  amplitude: number = 1.0,
  phase: number = 0,
  options: SignalGeneratorOptions = { sampleRate: 1000, durationSeconds: 1.0 }
): Float64Array {
  const n = Math.floor(options.sampleRate * options.durationSeconds);
  const out = new Float64Array(n);
  const twoPiF_Fs = (2.0 * Math.PI * frequencyHz) / options.sampleRate;
  for (let i = 0; i < n; i++) {
    out[i] = amplitude * Math.sin(twoPiF_Fs * i + phase);
  }
  return out;
}

/**
 * Generate a multi-harmonic tone: fundamental + harmonic overtones
 * Useful for ENF testing (50/60 Hz and harmonics at 100/120, 150/180, 200/240 Hz).
 */
export function generateHarmonicSeries(
  fundamentalHz: number,
  harmonicAmplitudes: number[],
  options: SignalGeneratorOptions = { sampleRate: 1000, durationSeconds: 1.0 }
): Float64Array {
  const n = Math.floor(options.sampleRate * options.durationSeconds);
  const out = new Float64Array(n);
  const Fs = options.sampleRate;

  for (let h = 0; h < harmonicAmplitudes.length; h++) {
    const f = fundamentalHz * (h + 1);
    const amp = harmonicAmplitudes[h]!;
    if (amp === 0 || f >= Fs / 2) continue;
    const omega = (2.0 * Math.PI * f) / Fs;
    for (let i = 0; i < n; i++) {
      out[i] = (out[i] ?? 0) + amp * Math.sin(omega * i);
    }
  }
  return out;
}

/**
 * Add Additive White Gaussian Noise (AWGN) to a signal targeting a specific SNR in dB.
 * SNR_dB = 10 * log10(P_signal / P_noise)
 * => P_noise = P_signal / 10^(SNR_dB / 10)
 * => sigma_noise = sqrt(P_noise)
 */
export function addAWGN(
  cleanSignal: Float64Array,
  snrDb: number,
  seed: number = 42
): { noisySignal: Float64Array; noiseSigma: number; signalPower: number } {
  const n = cleanSignal.length;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const v = cleanSignal[i]!;
    sumSq += v * v;
  }
  const signalPower = sumSq / Math.max(1, n);
  const noisePower = signalPower / Math.pow(10, snrDb / 10);
  const noiseSigma = Math.sqrt(Math.max(1e-12, noisePower));

  const rng = new Xoshiro128p(seed);
  const noisy = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    noisy[i] = cleanSignal[i]! + rng.nextGaussian(0, noiseSigma);
  }

  return { noisySignal: noisy, noiseSigma, signalPower };
}

/**
 * Generate a linear or exponential chirp signal from f0 to f1 over duration T.
 */
export function generateChirp(
  f0: number,
  f1: number,
  type: 'linear' | 'exponential' = 'linear',
  amplitude: number = 1.0,
  options: SignalGeneratorOptions = { sampleRate: 1000, durationSeconds: 1.0 }
): Float64Array {
  const n = Math.floor(options.sampleRate * options.durationSeconds);
  const out = new Float64Array(n);
  const T = options.durationSeconds;
  const Fs = options.sampleRate;

  if (type === 'linear') {
    // Phase phi(t) = 2*pi * (f0*t + (f1 - f0)/(2*T) * t^2)
    const k = (f1 - f0) / T;
    for (let i = 0; i < n; i++) {
      const t = i / Fs;
      const phase = 2 * Math.PI * (f0 * t + 0.5 * k * t * t);
      out[i] = amplitude * Math.sin(phase);
    }
  } else {
    // Exponential chirp: f(t) = f0 * (f1/f0)^(t/T)
    // phi(t) = 2*pi*f0 * ( (f1/f0)^(t/T) - 1 ) / ln(f1/f0) * T
    const ratio = f1 / f0;
    const lnRatio = Math.log(ratio);
    for (let i = 0; i < n; i++) {
      const t = i / Fs;
      const phase = 2 * Math.PI * f0 * ((Math.pow(ratio, t / T) - 1) / lnRatio);
      out[i] = amplitude * Math.sin(phase);
    }
  }
  return out;
}

/**
 * Generate a Synthetic Room Impulse Response (RIR) with exact exponential energy decay.
 * Target RT60 is the time required for sound energy to decay by 60 dB (factor of 10^-6, or amplitude decay by 10^-3).
 * Energy decay: E(t) = E0 * exp(-gamma * t), where gamma = 6 * ln(10) / RT60 = 13.8155 / RT60.
 * Amplitude envelope: A(t) = exp(-gamma/2 * t) = exp(-6.907755 * t / RT60).
 */
export function generateSyntheticRIR(
  rt60Seconds: number,
  sampleRate: number = 16000,
  directToReverbRatioDb: number = 12.0,
  seed: number = 1337
): Float64Array {
  // Duration should cover at least RT60 + 0.2s margin
  const duration = Math.min(3.0, Math.max(0.5, rt60Seconds * 1.2));
  const n = Math.floor(sampleRate * duration);
  const rir = new Float64Array(n);
  const rng = new Xoshiro128p(seed);

  // Direct sound spike at t = 2ms (32 samples at 16kHz)
  const directIdx = Math.floor(sampleRate * 0.002);
  const directAmp = Math.pow(10, directToReverbRatioDb / 20);
  rir[directIdx] = directAmp;

  // Early reflections (first 50 ms): sparse Poisson specular arrivals
  const earlyLimit = Math.floor(sampleRate * 0.05);
  for (let i = directIdx + 5; i < earlyLimit; i += Math.floor(rng.nextFloat() * 15 + 5)) {
    const delaySec = i / sampleRate;
    const reflAmp = (rng.nextFloat() * 0.8 + 0.2) * Math.exp(-6.907755 * delaySec / rt60Seconds) * (rng.nextFloat() > 0.5 ? 1 : -1);
    rir[i] = reflAmp;
  }

  // Late diffuse reverberation: Gaussian noise modulated by exact exponential decay envelope
  const decayRate = 6.907755278982137 / rt60Seconds; // 3 * ln(10) / RT60
  for (let i = earlyLimit; i < n; i++) {
    const t = i / sampleRate;
    const envelope = Math.exp(-decayRate * t);
    rir[i] = (rir[i] ?? 0) + rng.nextGaussian(0, envelope * 0.4);
  }

  return rir;
}

/**
 * Generate a synthetic ENF track with known continuous drift and an optional step splice discontinuity.
 * Used for testing changepoint detectors, Kalman filters, and CUSUM.
 */
export function generateSyntheticENFTrack(
  nominalHz: number = 50.0,
  durationSeconds: number = 60.0,
  spliceTimeSeconds: number = 30.0,
  spliceStepHz: number = 0.08,
  driftStdHz: number = 0.005,
  seed: number = 2026
): { time: Float64Array; frequency: Float64Array; spliceIndex: number } {
  const Fs = 1.0; // 1 reading per second
  const n = Math.floor(durationSeconds * Fs);
  const time = new Float64Array(n);
  const freq = new Float64Array(n);
  const rng = new Xoshiro128p(seed);

  let curFreq = nominalHz;
  const spliceIndex = Math.floor(spliceTimeSeconds * Fs);

  for (let i = 0; i < n; i++) {
    time[i] = i / Fs;
    if (i === spliceIndex) {
      curFreq += spliceStepHz; // Discrete jump at splice
    }
    // Random walk + mean reversion towards nominal
    const meanReversion = -0.05 * (curFreq - nominalHz);
    curFreq += meanReversion + rng.nextGaussian(0, driftStdHz);
    freq[i] = curFreq;
  }

  return { time, frequency: freq, spliceIndex };
}
