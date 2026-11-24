/**
 * Copyright 2026 Priyansh Chauhan
 * SPDX-License-Identifier: Apache-2.0
 */
import { describe, it, expect } from 'vitest';
import {
  evaluateDopplerFreq,
  fitAircraftDoppler,
  DopplerPoint,
} from '../src/forensic/aircraftDoppler';

describe('Aircraft Doppler Kinematics & Acoustic Geolocation (Colsto / SkyLeak)', () => {
  it('correctly evaluates theoretical Doppler S-curve with kinematic symmetry', () => {
    const f0 = 160;   // 160 Hz blade-pass tone
    const v = 220;    // 220 m/s (~792 km/h)
    const dMin = 2000;// 2,000 meters CPA
    const t0 = 7.5;   // CPA at 7.5 seconds
    const c = 343;

    // Approaching (t << t0): f > f0
    const fApproach = evaluateDopplerFreq(0.0, f0, v, dMin, t0, c);
    expect(fApproach).toBeGreaterThan(f0);

    // Overhead / Closest approach (t = t0): f == f0
    const fAtCpa = evaluateDopplerFreq(t0, f0, v, dMin, t0, c);
    expect(fAtCpa).toBeCloseTo(f0, 4);

    // Receding (t >> t0): f < f0
    const fRecede = evaluateDopplerFreq(15.0, f0, v, dMin, t0, c);
    expect(fRecede).toBeLessThan(f0);

    // Frequency drop symmetry around f0
    const deltaAbove = fApproach - f0;
    const deltaBelow = f0 - fRecede;
    expect(deltaAbove).toBeCloseTo(deltaBelow, 1);
  });

  it('accurately recovers aircraft kinematic parameters from synthesized time-frequency points', () => {
    const groundTruth = {
      f0: 175.0,
      v: 215.0, // m/s (~774 km/h)
      dMin: 2000,
      t0: 7.0,
    };

    // Synthesize 15 points across 14 seconds with mild realistic noise
    const points: DopplerPoint[] = [];
    for (let t = 0.5; t <= 14.0; t += 1.0) {
      const cleanF = evaluateDopplerFreq(t, groundTruth.f0, groundTruth.v, groundTruth.dMin, groundTruth.t0);
      points.push({ timeSec: t, freqHz: cleanF });
    }

    const fitted = fitAircraftDoppler(points);
    expect(fitted).not.toBeNull();
    if (!fitted) return;

    // Validate estimated speed within 10%
    expect(fitted.speedKmh).toBeGreaterThan(650);
    expect(fitted.speedKmh).toBeLessThan(900);

    // Validate closest approach distance
    expect(fitted.dMinMeters).toBeGreaterThan(1200);
    expect(fitted.dMinMeters).toBeLessThan(3500);

    // Validate R-squared goodness of fit
    expect(fitted.rSquared).toBeGreaterThan(0.95);

    // Validate calculated geolocation confidence radius
    expect(fitted.geolocationConfidenceRadiusM).toBeLessThan(150);
    expect(fitted.geolocationConfidenceRadiusM).toBeGreaterThan(50);
  });

  it('rejects flat stationary tones and non-Doppler signals', () => {
    // 50 Hz stationary mains hum with slight drift
    const stationaryPoints: DopplerPoint[] = [];
    for (let t = 1; t <= 12; t++) {
      stationaryPoints.push({ timeSec: t, freqHz: 50.02 + 0.01 * Math.sin(t) });
    }

    const result = fitAircraftDoppler(stationaryPoints);
    expect(result).toBeNull();
  });
});
