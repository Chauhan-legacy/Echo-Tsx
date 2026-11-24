/**
 * Adaptive Kalman Filter & Rauch-Tung-Striebel (RTS) Smoother for ENF Tracking
 *
 * State Vector:
 *   x_k = [ f_k,  f_dot_k ]^T  (Frequency in Hz, Frequency drift rate in Hz/s)
 *
 * Dynamic Model:
 *   x_k = F * x_{k-1} + w_k,  where F = [ [1, dt], [0, 1] ]
 *   z_k = H * x_k + v_k,      where H = [ 1, 0 ]
 *
 * Features:
 * 1. Adaptive measurement noise R_k estimated from innovation variance.
 * 2. Mahalanobis gating (chi-square 1-DOF at 99% confidence: d_M^2 <= 6.63) to reject transients.
 * 3. Rauch-Tung-Striebel (RTS) optimal backward smoothing.
 * 4. 95% Confidence bounds (+/- 1.96 * sqrt(P_00)) on every time step.
 */

export interface KalmanStepResult {
  timeSec: number;
  filteredFrequencyHz: number;
  frequencyDriftHzPerSec: number;
  confidenceInterval95: [number, number]; // [lower, upper]
  innovation: number;
  mahalanobisDistance: number;
  isOutlierGated: boolean;
}

export interface RTSSmoothedResult {
  timeSec: number;
  smoothedFrequencyHz: number;
  smoothedDriftHzPerSec: number;
  smoothedConfidenceInterval95: [number, number];
}

export class ENFKalmanTracker {
  private readonly dt: number;
  private readonly gateThreshold: number; // chi-square 99% = 6.635

  // State: x = [f, f_dot]^T
  private x0: number;
  private x1: number;

  // Covariance: P = [ [p00, p01], [p10, p11] ]
  private p00: number;
  private p01: number;
  private p10: number;
  private p11: number;

  // Process noise Q
  private q00: number;
  private q01: number;
  private q10: number;
  private q11: number;

  // Measurement noise R
  private r: number;

  // History for RTS smoother
  public readonly history: {
    time: number;
    xPrior: [number, number];
    xPost: [number, number];
    pPrior: [number, number, number, number];
    pPost: [number, number, number, number];
  }[] = [];

  constructor(
    initialFrequencyHz: number = 50.0,
    dt: number = 1.0,
    options?: {
      processNoiseQ?: number;
      measurementNoiseR?: number;
    }
  ) {
    this.dt = dt;
    this.gateThreshold = 6.635;

    this.x0 = initialFrequencyHz;
    this.x1 = 0.0;

    // Initial covariance
    this.p00 = 0.1;
    this.p01 = 0.0;
    this.p10 = 0.0;
    this.p11 = 0.01;

    const qBase = options?.processNoiseQ ?? 1e-4;
    // Continuous white noise acceleration model for Q
    const dt2 = dt * dt;
    const dt3 = dt2 * dt;
    const dt4 = dt3 * dt;
    this.q00 = (dt4 / 4.0) * qBase;
    this.q01 = (dt3 / 2.0) * qBase;
    this.q10 = (dt3 / 2.0) * qBase;
    this.q11 = dt2 * qBase;

    this.r = options?.measurementNoiseR ?? 0.005;
  }

  /**
   * Predict and Update step on measurement z_k.
   */
  public step(timeSec: number, measurementHz: number): KalmanStepResult {
    const dt = this.dt;

    // 1. Predict (Prior)
    // x_prior = F * x
    const x0Prior = this.x0 + dt * this.x1;
    const x1Prior = this.x1;

    // P_prior = F * P * F^T + Q
    const p00Prior = this.p00 + dt * (this.p10 + this.p01) + dt * dt * this.p11 + this.q00;
    const p01Prior = this.p01 + dt * this.p11 + this.q01;
    const p10Prior = this.p10 + dt * this.p11 + this.q10;
    const p11Prior = this.p11 + this.q11;

    // 2. Innovation: y = z - H * x_prior
    const innovation = measurementHz - x0Prior;

    // Innovation covariance: S = H * P_prior * H^T + R = p00Prior + R
    const S = p00Prior + this.r;

    // Mahalanobis distance squared: d_M^2 = y^2 / S
    const dMSq = (innovation * innovation) / Math.max(1e-15, S);
    const isOutlier = dMSq > this.gateThreshold;

    let x0Post = x0Prior;
    let x1Post = x1Prior;
    let p00Post = p00Prior;
    let p01Post = p01Prior;
    let p10Post = p10Prior;
    let p11Post = p11Prior;

    if (!isOutlier) {
      // 3. Kalman Gain: K = P_prior * H^T / S
      const k0 = p00Prior / S;
      const k1 = p10Prior / S;

      // 4. Update (Posterior)
      x0Post = x0Prior + k0 * innovation;
      x1Post = x1Prior + k1 * innovation;

      // P_post = (I - K * H) * P_prior
      p00Post = (1.0 - k0) * p00Prior;
      p01Post = (1.0 - k0) * p01Prior;
      p10Post = p10Prior - k1 * p00Prior;
      p11Post = p11Prior - k1 * p01Prior;

      // Adaptive R update (exponential moving average)
      this.r = 0.95 * this.r + 0.05 * Math.max(1e-5, innovation * innovation - p00Prior);
    }

    this.x0 = x0Post;
    this.x1 = x1Post;
    this.p00 = p00Post;
    this.p01 = p01Post;
    this.p10 = p10Post;
    this.p11 = p11Post;

    // Record for RTS smoothing
    this.history.push({
      time: timeSec,
      xPrior: [x0Prior, x1Prior],
      xPost: [x0Post, x1Post],
      pPrior: [p00Prior, p01Prior, p10Prior, p11Prior],
      pPost: [p00Post, p01Post, p10Post, p11Post],
    });

    const sigma = Math.sqrt(Math.max(1e-12, p00Post));
    const ci95: [number, number] = [x0Post - 1.96 * sigma, x0Post + 1.96 * sigma];

    return {
      timeSec,
      filteredFrequencyHz: x0Post,
      frequencyDriftHzPerSec: x1Post,
      confidenceInterval95: ci95,
      innovation,
      mahalanobisDistance: Math.sqrt(dMSq),
      isOutlierGated: isOutlier,
    };
  }

  /**
   * Rauch-Tung-Striebel (RTS) Backward Smoother
   */
  public smoothRTS(): RTSSmoothedResult[] {
    const N = this.history.length;
    if (N === 0) return [];

    const smoothed: RTSSmoothedResult[] = new Array(N);

    // Final state is already optimal
    const last = this.history[N - 1]!;
    let xSmooth0 = last.xPost[0];
    let xSmooth1 = last.xPost[1];
    let pSmooth00 = last.pPost[0];
    let pSmooth01 = last.pPost[1];
    let pSmooth10 = last.pPost[2];
    let pSmooth11 = last.pPost[3];

    let sigma = Math.sqrt(Math.max(1e-12, pSmooth00));
    smoothed[N - 1] = {
      timeSec: last.time,
      smoothedFrequencyHz: xSmooth0,
      smoothedDriftHzPerSec: xSmooth1,
      smoothedConfidenceInterval95: [xSmooth0 - 1.96 * sigma, xSmooth0 + 1.96 * sigma],
    };

    const dt = this.dt;

    for (let k = N - 2; k >= 0; k--) {
      const cur = this.history[k]!;
      const next = this.history[k + 1]!;

      // C_k = P_{k|k} * F^T * P_{k+1|k}^(-1)
      // F^T = [ [1, 0], [dt, 1] ]
      const pF00 = cur.pPost[0] + dt * cur.pPost[1];
      const pF01 = cur.pPost[1];
      const pF10 = cur.pPost[2] + dt * cur.pPost[3];
      const pF11 = cur.pPost[3];

      // Invert 2x2 next.pPrior
      const det = next.pPrior[0] * next.pPrior[3] - next.pPrior[1] * next.pPrior[2];
      const invDet = 1.0 / Math.max(1e-15, det);
      const invP00 = next.pPrior[3] * invDet;
      const invP01 = -next.pPrior[1] * invDet;
      const invP10 = -next.pPrior[2] * invDet;
      const invP11 = next.pPrior[0] * invDet;

      const c00 = pF00 * invP00 + pF01 * invP10;
      const c01 = pF00 * invP01 + pF01 * invP11;
      const c10 = pF10 * invP00 + pF11 * invP10;
      const c11 = pF10 * invP01 + pF11 * invP11;

      // x_{k|N} = x_{k|k} + C_k * (x_{k+1|N} - x_{k+1|k})
      const dx0 = xSmooth0 - next.xPrior[0];
      const dx1 = xSmooth1 - next.xPrior[1];

      xSmooth0 = cur.xPost[0] + c00 * dx0 + c01 * dx1;
      xSmooth1 = cur.xPost[1] + c10 * dx0 + c11 * dx1;

      // P_{k|N} = P_{k|k} + C_k * (P_{k+1|N} - P_{k+1|k}) * C_k^T
      const dP00 = pSmooth00 - next.pPrior[0];
      const dP01 = pSmooth01 - next.pPrior[1];
      const dP10 = pSmooth10 - next.pPrior[2];
      const dP11 = pSmooth11 - next.pPrior[3];

      const m00 = c00 * dP00 + c01 * dP10;
      const m01 = c00 * dP01 + c01 * dP11;
      const m10 = c10 * dP00 + c11 * dP10;
      const m11 = c10 * dP01 + c11 * dP11;

      pSmooth00 = cur.pPost[0] + m00 * c00 + m01 * c01;
      pSmooth01 = cur.pPost[1] + m00 * c10 + m01 * c11;
      pSmooth10 = cur.pPost[2] + m10 * c00 + m11 * c01;
      pSmooth11 = cur.pPost[3] + m10 * c10 + m11 * c11;

      sigma = Math.sqrt(Math.max(1e-12, pSmooth00));
      smoothed[k] = {
        timeSec: cur.time,
        smoothedFrequencyHz: xSmooth0,
        smoothedDriftHzPerSec: xSmooth1,
        smoothedConfidenceInterval95: [xSmooth0 - 1.96 * sigma, xSmooth0 + 1.96 * sigma],
      };
    }

    return smoothed;
  }
}
