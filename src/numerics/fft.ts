/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Zero-Dependency Arbitrary-Length Fast Fourier Transform (FFT)
 *
 * Implements:
 * 1. Radix-2 Cooley-Tukey Decimation-in-Time for powers of 2.
 * 2. Bluestein's Chirp-Z Transform for arbitrary non-power-of-2 lengths N.
 *
 * Derivation of Bluestein's Algorithm:
 *   Using the identity 2kn = k^2 + n^2 - (k - n)^2, the DFT sum becomes:
 *   X[k] = exp(-j*pi*k^2/N) * sum_{n=0}^{N-1} [ x[n] * exp(-j*pi*n^2/N) ] * exp(j*pi*(k-n)^2/N)
 *   This is a cyclic convolution of sequences a[n] and b[n], which is evaluated
 *   using a power-of-2 Radix-2 FFT of length M >= 2*N - 1 with O(N log N) complexity.
 *
 * Stated Numerical Tolerance: Error vs Naive DFT < 1e-10.
 */

export class FastFourierTransform {
  public readonly n: number;
  private readonly isPowerOfTwo: boolean;

  // Radix-2 Precomputed Tables (Float64 for high accuracy)
  private readonly bitRev: Uint32Array;
  private readonly cosTwiddle: Float64Array;
  private readonly sinTwiddle: Float64Array;

  // Bluestein workspace (if non-power-of-2)
  private readonly bluesteinM: number = 0;
  private readonly chirpCos: Float64Array | null = null;
  private readonly chirpSin: Float64Array | null = null;
  private readonly bFft: FastFourierTransform | null = null;
  private readonly bSpectrumReal: Float64Array | null = null;
  private readonly bSpectrumImag: Float64Array | null = null;

  constructor(size: number) {
    if (size <= 0) throw new Error(`FFT size must be positive, got ${size}`);
    this.n = size;
    this.isPowerOfTwo = (size & (size - 1)) === 0;

    if (this.isPowerOfTwo) {
      // Precompute bit reversal permutation
      const bitRev = new Uint32Array(size);
      let j = 0;
      for (let i = 0; i < size; i++) {
        bitRev[i] = j;
        let bit = size >> 1;
        while (bit > 0 && (j & bit) !== 0) {
          j ^= bit;
          bit >>= 1;
        }
        j ^= bit;
      }
      this.bitRev = bitRev;

      // Precompute twiddle factors: exp(-2*pi*i*k / N)
      const halfN = size >> 1;
      this.cosTwiddle = new Float64Array(halfN);
      this.sinTwiddle = new Float64Array(halfN);
      const angleStep = (-2.0 * Math.PI) / size;
      for (let k = 0; k < halfN; k++) {
        this.cosTwiddle[k] = Math.cos(angleStep * k);
        this.sinTwiddle[k] = Math.sin(angleStep * k);
      }
    } else {
      // Bluestein setup for arbitrary N
      this.bitRev = new Uint32Array(0);
      this.cosTwiddle = new Float64Array(0);
      this.sinTwiddle = new Float64Array(0);

      // Find smallest power of 2 >= 2*N - 1
      let m = 1;
      while (m < 2 * size - 1) m <<= 1;
      this.bluesteinM = m;

      this.chirpCos = new Float64Array(size);
      this.chirpSin = new Float64Array(size);
      for (let i = 0; i < size; i++) {
        const angle = (Math.PI * (i * i)) / size;
        this.chirpCos[i] = Math.cos(angle);
        this.chirpSin[i] = Math.sin(angle);
      }

      this.bFft = new FastFourierTransform(m);
      const bReal = new Float64Array(m);
      const bImag = new Float64Array(m);

      bReal[0] = this.chirpCos[0]!;
      bImag[0] = this.chirpSin[0]!;
      for (let i = 1; i < size; i++) {
        const c = this.chirpCos[i]!;
        const s = this.chirpSin[i]!;
        bReal[i] = c;
        bImag[i] = s;
        bReal[m - i] = c;
        bImag[m - i] = s;
      }

      this.bSpectrumReal = new Float64Array(m);
      this.bSpectrumImag = new Float64Array(m);
      this.bFft.transform(bReal, bImag, this.bSpectrumReal, this.bSpectrumImag, false);
    }
  }

  /**
   * Complex In-Place or Out-of-Place FFT:
   * outReal, outImag are filled with DFT(inReal, inImag).
   * If inverse is true, computes IDFT (scaled by 1/N).
   */
  public transform(
    inReal: ArrayLike<number>,
    inImag: ArrayLike<number>,
    outReal: Float64Array,
    outImag: Float64Array,
    inverse: boolean = false
  ): void {
    const n = this.n;

    if (this.isPowerOfTwo) {
      this.radix2Transform(inReal, inImag, outReal, outImag, inverse);
    } else {
      this.bluesteinTransform(inReal, inImag, outReal, outImag, inverse);
    }
  }

  private radix2Transform(
    inReal: ArrayLike<number>,
    inImag: ArrayLike<number>,
    outReal: Float64Array,
    outImag: Float64Array,
    inverse: boolean
  ): void {
    const n = this.n;
    const bitRev = this.bitRev;

    // Bit-reversal copy
    for (let i = 0; i < n; i++) {
      const rev = bitRev[i]!;
      outReal[rev] = inReal[i]!;
      outImag[rev] = inImag[i]!;
    }

    // Cooley-Tukey butterflies
    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const step = n / len;

      for (let i = 0; i < n; i += len) {
        for (let j = 0; j < halfLen; j++) {
          const twiddleIdx = j * step;
          const cos = this.cosTwiddle[twiddleIdx]!;
          const sin = inverse ? -this.sinTwiddle[twiddleIdx]! : this.sinTwiddle[twiddleIdx]!;

          const uReal = outReal[i + j]!;
          const uImag = outImag[i + j]!;

          const vReal = outReal[i + j + halfLen]!;
          const vImag = outImag[i + j + halfLen]!;

          const tReal = vReal * cos - vImag * sin;
          const tImag = vReal * sin + vImag * cos;

          outReal[i + j] = uReal + tReal;
          outImag[i + j] = uImag + tImag;

          outReal[i + j + halfLen] = uReal - tReal;
          outImag[i + j + halfLen] = uImag - tImag;
        }
      }
    }

    if (inverse) {
      const invN = 1.0 / n;
      for (let i = 0; i < n; i++) {
        outReal[i] = outReal[i]! * invN;
        outImag[i] = outImag[i]! * invN;
      }
    }
  }

  private bluesteinTransform(
    inReal: ArrayLike<number>,
    inImag: ArrayLike<number>,
    outReal: Float64Array,
    outImag: Float64Array,
    inverse: boolean
  ): void {
    const n = this.n;
    const m = this.bluesteinM;
    const bFft = this.bFft!;
    const chirpCos = this.chirpCos!;
    const chirpSin = this.chirpSin!;

    const aReal = new Float64Array(m);
    const aImag = new Float64Array(m);

    // a[n] = x[n] * chirp*(n)
    for (let i = 0; i < n; i++) {
      const xr = inReal[i]!;
      const xi = inImag[i]!;
      const c = chirpCos[i]!;
      const s = inverse ? -chirpSin[i]! : chirpSin[i]!;

      // x[i] * (c - j*s)
      aReal[i] = xr * c + xi * s;
      aImag[i] = xi * c - xr * s;
    }

    // A = FFT(a)
    const aSpecReal = new Float64Array(m);
    const aSpecImag = new Float64Array(m);
    bFft.transform(aReal, aImag, aSpecReal, aSpecImag, false);

    // Multiply A * B
    const bSpecReal = this.bSpectrumReal!;
    const bSpecImag = this.bSpectrumImag!;
    const prodReal = new Float64Array(m);
    const prodImag = new Float64Array(m);

    for (let i = 0; i < m; i++) {
      const ar = aSpecReal[i]!;
      const ai = aSpecImag[i]!;
      const br = bSpecReal[i]!;
      const bi = inverse ? -bSpecImag[i]! : bSpecImag[i]!;

      prodReal[i] = ar * br - ai * bi;
      prodImag[i] = ar * bi + ai * br;
    }

    // IFFT(prod)
    const convReal = new Float64Array(m);
    const convImag = new Float64Array(m);
    bFft.transform(prodReal, prodImag, convReal, convImag, true);

    // Final multiply by chirp*(k) and optional 1/N scaling
    const scale = inverse ? 1.0 / n : 1.0;
    for (let i = 0; i < n; i++) {
      const cr = convReal[i]!;
      const ci = convImag[i]!;
      const c = chirpCos[i]!;
      const s = inverse ? -chirpSin[i]! : chirpSin[i]!;

      outReal[i] = (cr * c + ci * s) * scale;
      outImag[i] = (ci * c - cr * s) * scale;
    }
  }

  /**
   * Convenience forward transform allocating output buffers.
   */
  public forward(realIn: ArrayLike<number>, imagIn?: ArrayLike<number>): { realOut: Float64Array; imagOut: Float64Array } {
    const realOut = new Float64Array(this.n);
    const imagOut = new Float64Array(this.n);
    const imag = imagIn ?? new Float64Array(this.n);
    this.transform(realIn, imag, realOut, imagOut, false);
    return { realOut, imagOut };
  }

  /**
   * Convenience inverse transform allocating output buffers.
   */
  public inverse(realIn: ArrayLike<number>, imagIn: ArrayLike<number>): { realOut: Float64Array; imagOut: Float64Array } {
    const realOut = new Float64Array(this.n);
    const imagOut = new Float64Array(this.n);
    this.transform(realIn, imagIn, realOut, imagOut, true);
    return { realOut, imagOut };
  }

  /**
   * Reference naive O(N^2) Discrete Fourier Transform for exact numerical unit testing.
   */
  public static naiveDFT(
    inReal: ArrayLike<number>,
    inImag: ArrayLike<number>,
    outReal: Float64Array,
    outImag: Float64Array,
    inverse: boolean = false
  ): void {
    const n = inReal.length;
    const sign = inverse ? 1.0 : -1.0;
    const invN = inverse ? 1.0 / n : 1.0;

    for (let k = 0; k < n; k++) {
      let sumRe = 0.0;
      let sumIm = 0.0;
      for (let t = 0; t < n; t++) {
        const angle = (2.0 * Math.PI * k * t * sign) / n;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const re = inReal[t]!;
        const im = inImag[t]!;
        sumRe += re * cos - im * sin;
        sumIm += re * sin + im * cos;
      }
      outReal[k] = sumRe * invN;
      outImag[k] = sumIm * invN;
    }
  }
}

export function naiveDFT(
  inReal: ArrayLike<number>,
  inImag: ArrayLike<number>,
  inverse: boolean = false
): { realOut: Float64Array; imagOut: Float64Array } {
  const n = inReal.length;
  const realOut = new Float64Array(n);
  const imagOut = new Float64Array(n);
  FastFourierTransform.naiveDFT(inReal, inImag, realOut, imagOut, inverse);
  return { realOut, imagOut };
}

export { FastFourierTransform as ArbitraryLengthFFT };

