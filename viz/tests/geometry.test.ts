import { describe, expect, it } from 'vitest';
import { covEllipse, ellipsePoints, enuToThree } from '../src/lib/geometry';

describe('covEllipse', () => {
  it('axis-aligned covariance', () => {
    const e = covEllipse(4, 0, 1, 1);
    expect(e.major).toBeCloseTo(2); expect(e.minor).toBeCloseTo(1); expect(e.angle).toBeCloseTo(0);
  });
  it('taller than wide rotates 90 degrees', () => {
    const e = covEllipse(1, 0, 9, 2);
    expect(e.major).toBeCloseTo(6); expect(e.minor).toBeCloseTo(2); expect(e.angle).toBeCloseTo(Math.PI / 2);
  });
  it('correlated covariance: major axis along the dominant eigenvector', () => {
    // [[2,1],[1,2]] has eigenvalues 3 (along (1,1)) and 1
    const e = covEllipse(2, 1, 2, 1);
    expect(e.major).toBeCloseTo(Math.sqrt(3)); expect(e.minor).toBeCloseTo(1); expect(e.angle).toBeCloseTo(Math.PI / 4);
  });
  it('every ellipse point satisfies x^T P^-1 x = n^2', () => {
    const [a, b, c] = [0.8, -0.3, 0.4];
    const pts = ellipsePoints(0, 0, covEllipse(a, b, c, 2), 16);
    const det = a * c - b * b;
    for (let i = 0; i < pts.length; i += 2) {
      const [x, y] = [pts[i], pts[i + 1]];
      expect((c * x * x - 2 * b * x * y + a * y * y) / det).toBeCloseTo(4, 6);
    }
    expect(pts.slice(0, 2)).toEqual(pts.slice(-2).map((v) => expect.closeTo(v, 9)) as any);
  });
  it('ENU to three.js swizzle', () => { expect(enuToThree(1, 2, 3)).toEqual([1, 3, -2]); });
});
