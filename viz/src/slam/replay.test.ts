// Candidate tests: require the existing viewer's Vitest runtime; NOT run in preparation.
import { describe,expect,it } from 'vitest';
import { camera0ToThree,frameAtTime,parseSlamReplay,type SlamFrame } from './replay';
const p:[number,number,number,number,number,number,number,number,number,number,number,number]=[1,0,0,1,0,1,0,2,0,0,1,3];
describe('recorded SLAM coordinates/time',()=>{
  it('uses proper camera0 swizzle',()=>expect(camera0ToThree(p)).toEqual([1,-2,-3]));
  it('selects actual original sample and handles empty failure',()=>{
    const frames=[0,.08,.21].map((t,i)=>({frame_id:i,timestamp_s:t,odom:p,optimized:null,truth:null,registration:null})) as SlamFrame[];
    expect(frameAtTime(frames,.20)).toBe(1);expect(frameAtTime(frames,5)).toBe(2);expect(frameAtTime([],1)).toBe(-1);
  });
  it('rejects legacy EKF shape instead of inventing SLAM semantics',()=>expect(()=>parseSlamReplay({samples:[],gps:[]})).toThrow('schema'));
});
