/** Separate recorded-SLAM schema. Never coerce it into 2-D EKF/covariance/GPS. */
export type Pose12 = [number,number,number,number,number,number,number,number,number,number,number,number];
export interface SlamFrame {
  frame_id: number; timestamp_s: number; odom: Pose12;
  optimized: Pose12 | null; truth: Pose12 | null;
  registration: Record<string, number | null> | null;
}
export interface LoopDecision extends Record<string, unknown> {
  proposal_id: number; candidate_frame: number; query_frame: number;
  accepted: boolean; reason: string; endpoints_available: boolean;
}
export interface SlamReplay {
  schema: 'kitti-slam-replay/v1'; kind: string; dataset: string; sequence: string;
  status: 'complete' | 'failed'; expected_frames: number; accepted_frames: number;
  optimized_available: boolean; frames: SlamFrame[]; loops: LoopDecision[];
  events: Record<string, unknown>[]; failure: Record<string, unknown> | null;
  coordinates: Record<string, unknown>; point_cloud: { status: 'not_exported'; note: string };
  provenance: Record<string, unknown>; attribution: string;
}
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const integer = (x: unknown): x is number => finite(x) && Number.isInteger(x);
function pose(value: unknown, label: string): asserts value is Pose12 {
  if (!Array.isArray(value) || value.length !== 12 || !value.every(finite)) throw new Error(`${label}: need finite row-major 3x4`);
  for (let a=0;a<3;a++) for (let b=0;b<3;b++) {
    let dot=0; for (let k=0;k<3;k++) dot+=value[k*4+a]*value[k*4+b];
    if (Math.abs(dot-(a===b?1:0))>1e-5) throw new Error(`${label}: non-orthonormal rotation`);
  }
  const m=value;
  const det=m[0]*(m[5]*m[10]-m[6]*m[9])-m[1]*(m[4]*m[10]-m[6]*m[8])+m[2]*(m[4]*m[9]-m[5]*m[8]);
  if (Math.abs(det-1)>1e-5) throw new Error(`${label}: improper rotation`);
}
export function parseSlamReplay(raw: unknown): SlamReplay {
  if (!raw || typeof raw!=='object') throw new Error('SLAM replay must be an object');
  const r=raw as Record<string, any>;
  if (r.schema!=='kitti-slam-replay/v1') throw new Error('Unsupported SLAM replay schema');
  if (!['complete','failed'].includes(r.status) || !integer(r.expected_frames) || r.expected_frames<1 ||
      !integer(r.accepted_frames) || r.accepted_frames<0 || r.accepted_frames>r.expected_frames) throw new Error('Invalid terminal replay state/count');
  if (r.kind==='kitti_full' && ({'07':1101,'00':4541} as Record<string,number>)[r.sequence]!==r.expected_frames) throw new Error('Incomplete declared KITTI sequence');
  if (!Array.isArray(r.frames) || r.frames.length!==r.accepted_frames || !Array.isArray(r.loops) || !Array.isArray(r.events)) throw new Error('Missing replay arrays');
  if (r.optimized_available!==(r.status==='complete') || (r.status==='complete' && r.accepted_frames!==r.expected_frames)) throw new Error('Incomplete run cannot claim an optimized full trajectory');
  let previous=-Infinity;
  r.frames.forEach((f: any,i: number)=>{
    if (f.frame_id!==i || !finite(f.timestamp_s) || f.timestamp_s<0 || f.timestamp_s<=previous) throw new Error(`frame ${i}: invalid original ID/time`);
    previous=f.timestamp_s;pose(f.odom,`frame ${i} odometry`);
    if (r.status==='complete') pose(f.optimized,`frame ${i} optimized`);
    else if (f.optimized!==null) throw new Error('Failed run must not synthesize optimization');
    if (f.truth!==null) pose(f.truth,`frame ${i} truth`);
    if (f.registration!==null && (typeof f.registration!=='object' || Object.values(f.registration).some(v=>v!==null && !finite(v)))) throw new Error('Invalid per-frame diagnostics');
  });
  r.loops.forEach((l: any,i: number)=>{
    if(l.proposal_id!==i || !integer(l.candidate_frame) || !integer(l.query_frame) || l.candidate_frame<0 || l.candidate_frame>=l.query_frame || l.query_frame>=r.expected_frames || typeof l.accepted!=='boolean' || typeof l.reason!=='string') throw new Error(`loop ${i}: invalid decision`);
    if(l.endpoints_available!==(l.candidate_frame<r.accepted_frames && l.query_frame<r.accepted_frames)) throw new Error('Loop endpoint availability mismatch');
  });
  if (!r.point_cloud || r.point_cloud.status!=='not_exported' || !r.coordinates || r.coordinates.pose!=='T_camera0_initial_camera0_i row-major 3x4') throw new Error('SLAM coordinate/cloud contract mismatch');
  const display=[1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1];
  if(r.coordinates.alignment!=='initial-frame only; no ATE/scale alignment' || !Array.isArray(r.coordinates.T_three_camera0_initial) || r.coordinates.T_three_camera0_initial.length!==16 || display.some((v,i)=>r.coordinates.T_three_camera0_initial[i]!==v)) throw new Error('Unexpected replay display transform/alignment');
  if(r.status==='failed' && (!r.failure || typeof r.failure!=='object')) throw new Error('Failure evidence missing');
  return r as SlamReplay;
}
/** Proper camera0-to-Three rotation, not the legacy ENU swizzle. */
export const camera0ToThree=(p: Pose12): [number,number,number]=>[p[3],-p[7],-p[11]];
/** Nearest earlier actual sample; no fabricated timestamps or failure-frame pose. */
export function frameAtTime(frames: SlamFrame[],time: number): number {
  if (!frames.length) return -1;
  let lo=0,hi=frames.length-1;
  if(time<=frames[0].timestamp_s)return 0;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].timestamp_s<=time)lo=mid;else hi=mid-1;}
  return lo;
}
