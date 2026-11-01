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
  const samples = r.samples.map((s: any, i: number) => {
    const estimate = s?.estimated_pose ?? s?.ekf;
    if (!isNum(s?.t) || s.t <= prev) throw new Error(`sample ${i}: t must be increasing`);
    prev = s.t;
    if (!isVec(s.truth, 3) || !isVec(estimate, 3) || !isVec(s.cov, 3) || !isVec(s.dr, 2)) throw new Error(`sample ${i}: bad vector`);
    if (s.cov[0] < 0 || s.cov[2] < 0 || s.cov[0] * s.cov[2] - s.cov[1] ** 2 < -1e-9) throw new Error(`sample ${i}: covariance not PSD`);
    if (s.faults !== undefined && (s.faults === null || typeof s.faults !== 'object' || Object.values(s.faults).some((v) => typeof v !== 'boolean'))) throw new Error(`sample ${i}: bad fault flags`);
    if (s.clearance !== undefined && !isNum(s.clearance)) throw new Error(`sample ${i}: bad clearance`);
    if (s.outcome !== undefined && !['running', 'success', 'collision', 'timeout'].includes(s.outcome)) throw new Error(`sample ${i}: bad outcome`);
    return { ...s, ekf: estimate };
  });
  r.gps.forEach((g: any, i: number) => {
    if (!isNum(g?.t) || !isVec(g.xy, 2) || typeof g.accepted !== 'boolean') throw new Error(`gps ${i}: bad fix`);
  });
  if (r.world !== undefined) {
    const w = r.world as any;
    if (!isVec(w?.goal, 2) || !Array.isArray(w?.obstacles) || !w.obstacles.every((o: unknown) => isVec(o, 3) && (o as number[])[2] > 0) || !isNum(w.robot_radius) || w.robot_radius <= 0) throw new Error('world: invalid geometry');
  }
  return { meta: (r.meta as Record<string, unknown>) ?? {}, samples, gps: r.gps as Run['gps'], world: r.world as Run['world'] };
}
