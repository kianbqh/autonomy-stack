import type { Sample } from './types';

/** Largest index i with samples[i].t <= t (clamped). O(log n). */
export function indexAtTime(samples: Sample[], t: number): number {
  if (t <= samples[0].t) return 0;
  if (t >= samples[samples.length - 1].t) return samples.length - 1;
  let lo = 0, hi = samples.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t <= t) lo = mid; else hi = mid;
  }
  return lo;
}

const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
function lerpAngle(a: number, b: number, u: number) {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * u;
}

/** Interpolated pose [x, y, yaw] of the given track at time t. */
export function poseAt(samples: Sample[], t: number, key: 'truth' | 'ekf'): [number, number, number] {
  const i = indexAtTime(samples, t);
  const s0 = samples[i], s1 = samples[Math.min(i + 1, samples.length - 1)];
  const u = s1.t > s0.t ? Math.min(Math.max((t - s0.t) / (s1.t - s0.t), 0), 1) : 0;
  const a = s0[key], b = s1[key];
  return [lerp(a[0], b[0], u), lerp(a[1], b[1], u), lerpAngle(a[2], b[2], u)];
}

/** Position RMSE (m) of EKF vs ground truth over samples[0..i]. */
export function rmseUpTo(samples: Sample[], i: number): number {
  let s = 0;
  for (let k = 0; k <= i; k++) s += (samples[k].ekf[0] - samples[k].truth[0]) ** 2 + (samples[k].ekf[1] - samples[k].truth[1]) ** 2;
  return Math.sqrt(s / (i + 1));
}
