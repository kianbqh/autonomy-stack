export interface Sample {
  t: number;
  truth: [number, number, number]; // x, y, yaw
  ekf: [number, number, number];
  cov: [number, number, number]; // Pxx, Pxy, Pyy (m^2)
  dr: [number, number];
  speed: number;
  faults?: Record<string, boolean>;
  clearance?: number;
  min_clearance?: number;
  outcome?: 'running' | 'success' | 'collision' | 'timeout';
}
export interface GpsFix { t: number; xy: [number, number]; accepted: boolean }
export interface World { goal: [number, number]; obstacles: [number, number, number][]; robot_radius: number }
export interface Run { meta: Record<string, unknown>; samples: Sample[]; gps: GpsFix[]; world?: World }
export interface ReplayChoice { id: string; title: string; path: string; outcome?: string }
