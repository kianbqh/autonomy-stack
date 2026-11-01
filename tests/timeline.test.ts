import { describe, expect, it } from 'vitest';
import { indexAtTime, poseAt, rmseUpTo } from '../src/lib/timeline';
import type { Sample } from '../src/lib/types';

const mk = (t: number, x: number, yaw: number, ex = x): Sample => ({ t, truth: [x, 0, yaw], ekf: [ex, 0, yaw], cov: [1, 0, 1], dr: [x, 0], speed: 1 });
const S = [mk(0, 0, 0), mk(1, 1, 3.0), mk(2, 2, -3.0, 2.3)];

describe('timeline', () => {
  it('binary search clamps and finds the floor sample', () => {
    expect(indexAtTime(S, -5)).toBe(0); expect(indexAtTime(S, 0.99)).toBe(0);
    expect(indexAtTime(S, 1)).toBe(1); expect(indexAtTime(S, 1.5)).toBe(1); expect(indexAtTime(S, 99)).toBe(2);
  });
  it('interpolates position linearly and yaw across the +-pi wrap', () => {
    const p = poseAt(S, 1.5, 'truth');
    expect(p[0]).toBeCloseTo(1.5);
    // 3.0 -> -3.0 is a +0.283 rad turn through pi, not -6 rad
    expect(Math.abs(p[2])).toBeGreaterThan(3.0);
  });
  it('RMSE of EKF vs truth', () => {
    expect(rmseUpTo(S, 1)).toBe(0);
    expect(rmseUpTo(S, 2)).toBeCloseTo(Math.sqrt(0.09 / 3));
  });
});
