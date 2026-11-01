/** Covariance ellipse of a 2x2 symmetric PSD matrix [[a, b], [b, c]] at nSigma: semi-axes and rotation (rad). */
export function covEllipse(a: number, b: number, c: number, nSigma = 2): { major: number; minor: number; angle: number } {
  const tr = (a + c) / 2;
  const d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const l1 = Math.max(tr + d, 0);
  const l2 = Math.max(tr - d, 0);
  const angle = Math.abs(b) < 1e-15 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  return { major: nSigma * Math.sqrt(l1), minor: nSigma * Math.sqrt(l2), angle };
}

/** Polyline points (x, y) of an ellipse, closed (first point repeated). */
export function ellipsePoints(cx: number, cy: number, e: { major: number; minor: number; angle: number }, n = 32): number[] {
  const out: number[] = [];
  const ca = Math.cos(e.angle), sa = Math.sin(e.angle);
  for (let i = 0; i <= n; i++) {
    const t = (i / n) * Math.PI * 2;
    const x = e.major * Math.cos(t), y = e.minor * Math.sin(t);
    out.push(cx + ca * x - sa * y, cy + sa * x + ca * y);
  }
  return out;
}

/** ENU (x east, y north, z up) -> three.js (x right, y up, z toward viewer). */
export const enuToThree = (x: number, y: number, z = 0): [number, number, number] => [x, z, -y];
