/** Deterministic PRNG (mulberry32) so benchmarks and tests are reproducible. */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Bounds { minX: number; maxX: number; minY: number; maxY: number }

/**
 * Procedural scan-like point cloud (ENU) around the trajectory: noisy ground plane (60 %) plus vertical
 * obstacle columns (40 %), coloured by height. Synthetic: the simulator has no LiDAR, this is for rendering scale tests.
 */
export function generateCloud(n: number, b: Bounds, seed = 1, obstacles = 60) {
  const rnd = mulberry32(seed);
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const obs = Array.from({ length: obstacles }, () => ({
    x: b.minX + rnd() * (b.maxX - b.minX), y: b.minY + rnd() * (b.maxY - b.minY), r: 0.5 + rnd() * 2.5, h: 1 + rnd() * 6,
  }));
  const nGround = Math.floor(n * 0.6);
  for (let i = 0; i < n; i++) {
    let x: number, y: number, z: number;
    if (i < nGround || obstacles === 0) {
      x = b.minX + rnd() * (b.maxX - b.minX); y = b.minY + rnd() * (b.maxY - b.minY); z = (rnd() - 0.5) * 0.08;
    } else {
      const o = obs[i % obstacles], th = rnd() * Math.PI * 2;
      x = o.x + o.r * Math.cos(th); y = o.y + o.r * Math.sin(th); z = rnd() * o.h;
    }
    pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
    const h = Math.min(z / 6, 1);
    col[i * 3] = 0.2 + 0.8 * h; col[i * 3 + 1] = 0.55 - 0.25 * h; col[i * 3 + 2] = 0.9 - 0.7 * h;
  }
  return { positions: pos, colors: col };
}

export function boundsOf(xy: Array<[number, number] | number[]>, pad = 10): Bounds {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of xy) { minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]); }
  return { minX: minX - pad, maxX: maxX + pad, minY: minY - pad, maxY: maxY + pad };
}
