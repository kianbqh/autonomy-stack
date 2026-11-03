"""Export actual completed/failed SLAM artifacts without manufacturing legacy EKF fields."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from common import environment,git_state,new_run,sha256,source_manifest,utc,write_json

SCHEMA="kitti-slam-replay/v1"

def read_poses(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        row=[float(x) for x in line.split()]
        if len(row)!=12 or not all(math.isfinite(x) for x in row):raise ValueError("invalid replay pose row")
        R=[row[0:3],row[4:7],row[8:11]]
        for a in range(3):
            for b in range(3):
                if abs(sum(R[k][a]*R[k][b] for k in range(3))-(1 if a==b else 0))>1e-5:raise ValueError("non-orthonormal replay rotation")
        det=R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])-R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])+R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0])
        if abs(det-1)>1e-5:raise ValueError("replay rotation not proper")
        rows.append(row)
    return rows

def multiply(a,b):
    a=[a[0:4],a[4:8],a[8:12],[0,0,0,1]];b=[b[0:4],b[4:8],b[8:12],[0,0,0,1]]
    return [sum(a[r][k]*b[k][c] for k in range(4)) for r in range(3) for c in range(4)]

def inverse(a):
    R=[[a[r*4+c] for c in range(3)] for r in range(3)];t=[a[3],a[7],a[11]]
    return [value for r in range(3) for value in [R[0][r],R[1][r],R[2][r],-sum(R[k][r]*t[k] for k in range(3))]]

def read_times(path):
    times=[float(x) for x in Path(path).read_text().splitlines()]
    if any(not math.isfinite(t) or t<0 for t in times) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError("nonincreasing/nonfinite replay timestamps")
    return times

def json_lines(path):
    if not Path(path).is_file():return []
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        rows.append(json.loads(line))
    return rows

def build_replay(run_dir,truth_path=None):
    run_dir=Path(run_dir);native=run_dir/"native"
    record=json.loads((run_dir/"run.json").read_text());config=json.loads((run_dir/"config.json").read_text())
    if record["status"] not in ("complete","failed"):raise ValueError("export a terminal attempt, never mutable live output")
    if record["config_sha256"]!=sha256(run_dir/"config.json"):raise ValueError("replay config hash mismatch")
    expected=config["expected_frames"]
    if config["kind"]=="kitti_full" and {"07":1101,"00":4541}.get(config["sequence"])!=expected:raise ValueError("formal replay count changed")
    complete=record["status"]=="complete"
    if complete:
        summary=json.loads((native/"native_summary.json").read_text())
        if summary["frames"]!=expected or summary["status"]!="complete":raise ValueError("native completion count mismatch")
        odom=read_poses(native/"poses_odom.txt");optimized=read_poses(native/"poses_slam.txt");times=read_times(native/"timestamps.txt")
        if len(odom)!=expected or len(optimized)!=expected or len(times)!=expected:raise ValueError("full replay must contain every original frame")
        failure=None
    else:
        # Old candidates had no prefix export: retain a truthful empty failure replay.
        odom=read_poses(native/"poses_odom_partial.txt") if (native/"poses_odom_partial.txt").is_file() else []
        times=read_times(native/"timestamps_partial.txt") if (native/"timestamps_partial.txt").is_file() else []
        if len(odom)!=len(times) or len(odom)>expected:raise ValueError("partial accepted-prefix files disagree")
        optimized=None
        failure=json.loads((native/"failure.json").read_text()) if (native/"failure.json").is_file() else {"message":record.get("failure",{}).get("message","native process interrupted"),"frame":len(odom),"accepted_prefix_frames":len(odom),"failed_frame_pose":None,"optimized_trajectory":None}
        if failure.get("accepted_prefix_frames",len(odom))!=len(odom):raise ValueError("failure/prefix count mismatch")
        if failure.get("failed_frame_pose") is not None or failure.get("optimized_trajectory") is not None:raise ValueError("failed run cannot supply fabricated pose/optimization")
    truth=None
    if truth_path is not None:
        truth=read_poses(truth_path)
        if len(truth)!=expected:raise ValueError("GT must match the complete declared input, even for failure-prefix replay")
        first_inverse=inverse(truth[0]);truth=[multiply(first_inverse,p) for p in truth[:len(odom)]]
    timings={}
    if (native/"frame_times.csv").is_file():
        with (native/"frame_times.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                i=int(row["frame"])
                if i in timings:raise ValueError("duplicate timing frame")
                parsed={}
                for k,v in row.items():
                    if k=="frame":continue
                    n=float(v);parsed[k]=n if math.isfinite(n) else None
                timings[i]=parsed
    frames=[{"frame_id":i,"timestamp_s":t,"odom":odom[i],"optimized":optimized[i] if optimized is not None else None,"truth":truth[i] if truth is not None else None,"registration":timings.get(i)} for i,t in enumerate(times)]
    loops=[]
    for i,row in enumerate(json_lines(native/"loops.jsonl")):
        if type(row.get("accepted")) is not bool or not isinstance(row.get("reason"),str):raise ValueError("invalid loop decision")
        a,b=row["candidate_frame"],row["query_frame"]
        if type(a)!=int or type(b)!=int or not 0<=a<b<expected:raise ValueError("invalid loop frame IDs")
        loops.append({**row,"proposal_id":i,"endpoints_available":a<len(frames) and b<len(frames),"endpoint_track":"odom","decision_scope":"front-end geometry only; no GT label"})
    source_paths=[run_dir/"run.json",run_dir/"config.json"]+[p for p in (sorted(native.iterdir()) if native.is_dir() else []) if p.is_file() and p.name in {"poses_odom.txt","poses_slam.txt","timestamps.txt","poses_odom_partial.txt","timestamps_partial.txt","loops.jsonl","events.jsonl","failure.json","native_summary.json","frame_times.csv"}]
    sources={str(p.relative_to(run_dir)).replace("\\","/"):sha256(p) for p in source_paths}
    if truth_path is not None:sources["external_ground_truth"]=sha256(truth_path)
    return {"schema":SCHEMA,"kind":config["kind"],"dataset":"KITTI odometry" if config["kind"].startswith("kitti") else "synthetic unit fixture; not KITTI metrics","sequence":config["sequence"],"status":"complete" if complete else "failed","expected_frames":expected,"accepted_frames":len(frames),"optimized_available":complete,"frames":frames,"loops":loops,"events":json_lines(native/"events.jsonl"),"failure":failure,
            "coordinates":{"pose":"T_camera0_initial_camera0_i row-major 3x4","origin":"first camera0 frame; GT normalized by inverse(GT0)*GTi","alignment":"initial-frame only; no ATE/scale alignment","units":"meters and seconds","T_three_camera0_initial":[1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1],"display":"[camera_x,-camera_y,-camera_z]; complete SE3 retained; no ENU conversion"},
            "point_cloud":{"status":"not_exported","note":"trajectory replay only; no generated/procedural cloud substitutes for KITTI"},
            "provenance":{"git_sha":record.get("git_sha"),"config_sha256":record["config_sha256"],"input_manifest_sha256":record.get("input_manifest_sha256"),"binary_sha256":record.get("binary_sha256"),"source_artifacts_sha256":sources,"native_status":record["status"],"failure_not_a_full_result":not complete},
            "attribution":"KITTI odometry by Geiger et al.; CC BY-NC-SA 3.0; derived trajectories only" if config["kind"].startswith("kitti") else "synthetic engineering fixture, not recorded driving data"}

def main():
    a=argparse.ArgumentParser();a.add_argument("--run",required=True,type=Path);a.add_argument("--truth",type=Path);a.add_argument("--output-parent",required=True,type=Path);a.add_argument("--repo-root",required=True,type=Path);a.add_argument("--git",default="git");args=a.parse_args()
    out=new_run(args.output_parent,"replay");start=time.monotonic();receipt={"status":"started","utc":utc(),"command":sys.argv,"environment":environment()}
    try:
        receipt.update(git_state(args.repo_root,args.git));write_json(out/"source-sha256.json",source_manifest(args.repo_root/"slam"))
        data=build_replay(args.run,args.truth)
        if data["provenance"]["git_sha"]!=receipt["git_sha"]:raise ValueError("exporter/native Git checkpoint mismatch")
        write_json(out/"replay.json",data);receipt.update(status="complete",exported_replay_sha256=sha256(out/"replay.json"),native_status=data["status"],accepted_frames=data["accepted_frames"],expected_frames=data["expected_frames"])
    except BaseException as e:receipt.update(status="failed",failure={"type":type(e).__name__,"message":str(e)});raise
    finally:receipt.update(finished_utc=utc(),wall_s=time.monotonic()-start);write_json(out/"export.json",receipt)
    print(out)

if __name__=="__main__":main()
