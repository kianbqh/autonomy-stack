import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseRun } from '../src/lib/parse';
import { boundsOf, generateCloud, mulberry32 } from '../src/lib/pointcloud';
import { FrameStats } from '../src/lib/fps';

describe('parseRun', () => {
  const real = JSON.parse(readFileSync(new URL('../public/data/run.json', import.meta.url), 'utf8'));
  it('accepts the exported ros2-sensor-fusion-sim run', () => {
    const r = parseRun(real);
    expect(r.samples.length).toBe(1200); expect(r.gps.length).toBe(120);
    expect(r.gps.filter((g) => !g.accepted).length).toBeGreaterThan(0);
  });
  it('rejects non-monotonic time and non-PSD covariance', () => {
    const bad1 = structuredClone(real); bad1.samples[5].t = bad1.samples[4].t;
    expect(() => parseRun(bad1)).toThrow(/increasing/);
    const bad2 = structuredClone(real); bad2.samples[3].cov = [1, 5, 1];
    expect(() => parseRun(bad2)).toThrow(/PSD/);
    expect(() => parseRun({ samples: [] })).toThrow();
  });
  it('loads both actual safety exports and preserves their outcome, world and covariance', () => {
    for (const [id, outcome] of [['normal', 'success'], ['fault-collision', 'collision']]) {
      const raw = JSON.parse(readFileSync(new URL(`../public/data/replays/${id}.json`, import.meta.url), 'utf8'));
      const run = parseRun(raw);
      expect(run.meta.id).toBe(id); expect(run.meta.outcome).toBe(outcome);
      expect(run.world!.obstacles.length).toBeGreaterThan(0);
      expect(run.samples[0].ekf).toEqual(raw.samples[0].estimated_pose);
      expect(run.samples.at(-1)!.outcome).toBe(outcome);
      if (outcome === 'collision') {
        expect(run.samples.at(-1)!.clearance).toBeLessThan(0);
        expect(run.samples.some((s) => s.faults?.gps_bias)).toBe(true);
      }
    }
  });
});

describe('point cloud', () => {
  it('is deterministic, sized correctly and within bounds', () => {
    const b = { minX: -5, maxX: 5, minY: 0, maxY: 20 };
    const a = generateCloud(50_000, b, 3), c = generateCloud(50_000, b, 3);
    expect(a.positions.length).toBe(150_000); expect(a.positions).toEqual(c.positions);
    for (let i = 0; i < 30_000; i++) { // first 60 % are ground points
      expect(a.positions[i * 3]).toBeGreaterThanOrEqual(-5); expect(a.positions[i * 3]).toBeLessThanOrEqual(5);
    }
    expect(a.colors.reduce((m, v) => Math.max(m, v), 0)).toBeLessThanOrEqual(1);
  });
  it('PRNG is uniform-ish', () => {
    const r = mulberry32(1); let s = 0; for (let i = 0; i < 1e5; i++) s += r();
    expect(s / 1e5).toBeCloseTo(0.5, 2);
  });
  it('bounds pad the trajectory', () => {
    expect(boundsOf([[0, 0], [10, -4]], 1)).toEqual({ minX: -1, maxX: 11, minY: -5, maxY: 1 });
  });
});

describe('FrameStats', () => {
  it('computes fps from frame intervals', () => {
    const f = new FrameStats(); for (let i = 0; i <= 60; i++) f.tick(i * 16.6667);
    const s = f.summary(); expect(s.frames).toBe(60); expect(s.fps).toBeCloseTo(60, 0); expect(s.frameMsP50).toBeCloseTo(16.67, 1);
  });
  it('retains intervals and uses linear percentile interpolation', () => {
    const f = new FrameStats();
    [0, 10, 30, 60, 100].forEach((t) => f.tick(t));
    expect(f.intervals()).toEqual([10, 20, 30, 40]);
    expect(f.summary()).toEqual({ frames: 4, fps: 40, frameMsP50: 25, frameMsP95: 38.5 });
    f.reset(); expect(f.summary().fps).toBe(0);
  });
});
