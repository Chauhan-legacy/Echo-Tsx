/**
 * Xoshiro128** (xoshiro128plusplus 1.1) PRNG
 *
 * Designed by David Blackman and Sebastiano Vigna (2019).
 * State: 4 x 32-bit unsigned integers (128-bit period: 2^128 - 1).
 * Excellent statistical distribution, passes BigCrush tests.
 * Zero dependency, 100% deterministic with seeded initialization.
 */

export class Xoshiro128 {
  private s0: number = 0;
  private s1: number = 0;
  private s2: number = 0;
  private s3: number = 0;

  // Cached spare value for Box-Muller normal transform
  private hasSpareGaussian: boolean = false;
  private spareGaussian: number = 0;

  constructor(seed: number = 0x12345678) {
    this.seed(seed);
  }

  /**
   * SplitMix32 to seed all 4 state words from a single 32-bit seed.
   */
  public seed(seed: number): void {
    let smState = (seed >>> 0) || 1;
    const splitmix32 = (): number => {
      smState = (smState + 0x9e3779b9) >>> 0;
      let z = smState;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };

    this.s0 = splitmix32();
    this.s1 = splitmix32();
    this.s2 = splitmix32();
    this.s3 = splitmix32();
    this.hasSpareGaussian = false;
  }

  /**
   * Returns next 32-bit unsigned integer in [0, 2^32 - 1].
   */
  public nextUint32(): number {
    let s0 = this.s0;
    let s1 = this.s1;
    let s2 = this.s2;
    let s3 = this.s3;

    // rotl(s0 + s3, 7) + s0
    const sum03 = (s0 + s3) >>> 0;
    const result = (((sum03 << 7) | (sum03 >>> 25)) + s0) >>> 0;

    const t = (s1 << 9) >>> 0;

    s2 = (s2 ^ s0) >>> 0;
    s3 = (s3 ^ s1) >>> 0;
    s1 = (s1 ^ s2) >>> 0;
    s0 = (s0 ^ s3) >>> 0;

    s2 = (s2 ^ t) >>> 0;
    // rotl(s3, 11)
    s3 = ((s3 << 11) | (s3 >>> 21)) >>> 0;

    this.s0 = s0;
    this.s1 = s1;
    this.s2 = s2;
    this.s3 = s3;

    return result;
  }

  /**
   * Returns random floating point number in [0, 1).
   */
  public nextFloat(): number {
    return this.nextUint32() * 2.3283064365386963e-10; // 1 / 2^32
  }

  /**
   * Standard normal Gaussian distribution N(mean, std) via Box-Muller transform.
   */
  public nextGaussian(mean: number = 0.0, std: number = 1.0): number {
    if (this.hasSpareGaussian) {
      this.hasSpareGaussian = false;
      return mean + std * this.spareGaussian;
    }

    let u = 0, v = 0, s = 0;
    do {
      u = this.nextFloat() * 2.0 - 1.0;
      v = this.nextFloat() * 2.0 - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s === 0.0);

    const mul = Math.sqrt((-2.0 * Math.log(s)) / s);
    this.spareGaussian = v * mul;
    this.hasSpareGaussian = true;
    return mean + std * (u * mul);
  }
}

export { Xoshiro128 as Xoshiro128p };

