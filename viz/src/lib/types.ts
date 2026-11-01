export interface Sample {
  t: number;
  truth: [number, number, number]; // x, y, yaw
  ekf: [number, number, number];
  cov: [number, number, number]; // Pxx, Pxy, Pyy (m^2)
  dr: [number, number];
  speed: number;
}
export interface GpsFix { t: number; xy: [number, number]; accepted: boolean }
export interface Run { meta: Record<string, unknown>; samples: Sample[]; gps: GpsFix[] }
