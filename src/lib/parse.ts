import type { Run } from './types';

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isVec = (v: unknown, n: number) => Array.isArray(v) && v.length === n && v.every(isNum);

/** Validate an exported run; throws with a precise message so bad files fail loudly instead of rendering garbage. */
export function parseRun(raw: unknown): Run {
  if (typeof raw !== 'object' || raw === null) throw new Error('run: not an object');
  const r = raw as Record<string, unknown>;
  if (!Array.isArray(r.samples) || r.samples.length < 2) throw new Error('run: need >= 2 samples');
  if (!Array.isArray(r.gps)) throw new Error('run: gps must be an array');
  let prev = -Infinity;
  r.samples.forEach((s: any, i: number) => {
    if (!isNum(s?.t) || s.t <= prev) throw new Error(`sample ${i}: t must be increasing`);
    prev = s.t;
    if (!isVec(s.truth, 3) || !isVec(s.ekf, 3) || !isVec(s.cov, 3) || !isVec(s.dr, 2)) throw new Error(`sample ${i}: bad vector`);
    if (s.cov[0] < 0 || s.cov[2] < 0 || s.cov[0] * s.cov[2] - s.cov[1] ** 2 < -1e-9) throw new Error(`sample ${i}: covariance not PSD`);
  });
  r.gps.forEach((g: any, i: number) => {
    if (!isNum(g?.t) || !isVec(g.xy, 2) || typeof g.accepted !== 'boolean') throw new Error(`gps ${i}: bad fix`);
  });
  return { meta: (r.meta as Record<string, unknown>) ?? {}, samples: r.samples as Run['samples'], gps: r.gps as Run['gps'] };
}
