"""Frozen one-axis extrinsic sweeps; sensor-consistency diagnostic, not recovered calibration."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from common import environment,git_state,new_run,sha256,source_manifest,utc,write_json
from project_lidar import depth_edges,load_calibration,load_scan,overlay,project

def perturb(Tr,axis,offset):
    delta=np.eye(4)
    if axis.startswith("r"):
        vector=np.zeros(3);vector["xyz".index(axis[1])]=np.radians(offset)
        delta[:3,:3]=cv2.Rodrigues(vector)[0]
    else:delta["xyz".index(axis[1]),3]=offset
    return delta@Tr

def score(points,image,dt,P2,Tr,cfg):
    height,width=image.shape[:2]
    depth,support=project(points,P2,Tr,height,width,cfg["depth_min_m"],cfg["depth_max_m"])
    edges=depth_edges(depth,cfg["depth_jump_m"],cfg["edge_radius_pixels"])
    count=int(edges.sum());support["lidar_edge_pixels"]=count;support["camera_edge_pixels"]=int((dt==0).sum())
    if support["in_image_points"]<cfg["minimum_projected_points"] or support["projection_coverage"]<cfg["minimum_projection_coverage"] or count<cfg["minimum_edge_pixels"] or support["camera_edge_pixels"]<cfg["minimum_camera_edge_pixels"]:
        return {**support,"valid":False,"score":None,"reason":"insufficient_fixed_support_gate"},depth
    raw=dt[edges];norm=np.minimum(raw,cfg["distance_cap_pixels"])/cfg["distance_cap_pixels"]
    penalty=cfg["coverage_weight"]*(1-support["in_image_fraction_of_input"])
    return {**support,"valid":True,"score":float(norm.mean()+penalty),"normalized_edge_distance_mean":float(norm.mean()),
            "edge_distance_pixels_mean":float(raw.mean()),"edge_distance_pixels_p95":float(np.quantile(raw,.95)),
            "coverage_penalty":penalty,"edge_support_fraction_of_zbuffer":count/support["zbuffer_pixels"]},depth

def interval(values,cfg):
    if any(v is None for v in values):return {"mean":None,"ci95":None,"valid_frames":sum(v is not None for v in values)}
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    draws=rng.choice(np.asarray(values),size=(cfg["bootstrap_samples"],len(values)),replace=True).mean(1)
    return {"mean":float(np.mean(values)),"ci95":np.quantile(draws,[.025,.975]).tolist(),"valid_frames":len(values),"resampling_unit":"frame"}

def main():
    a=argparse.ArgumentParser();a.add_argument("--config",required=True,type=Path);a.add_argument("--input-manifest",required=True,type=Path);a.add_argument("--output-parent",required=True,type=Path);a.add_argument("--repo-root",required=True,type=Path);a.add_argument("--git",default="git");args=a.parse_args()
    out=new_run(args.output_parent,"calibration");start=time.monotonic();result={"status":"started","utc":utc(),"command":sys.argv,"environment":environment(),"config_sha256":sha256(args.config),"manifest_sha256":sha256(args.input_manifest),"claim":"offline sensor consistency only, not targetless calibration recovery"}
    try:
        result.update(git_state(args.repo_root,args.git));write_json(out/"source-sha256.json",source_manifest(args.repo_root/"slam"))
        full=json.loads(args.config.read_text());cfg=full["calibration"];manifest=json.loads(args.input_manifest.read_text())
        if manifest["sequence"]!=full["sequence"] or manifest["selected_image_frames"]!=cfg["frames"] or not manifest["verified"]:raise ValueError("input/calibration frame mismatch")
        sequence=Path(manifest["root"])/"dataset/sequences"/full["sequence"]
        P2,Tr=load_calibration(sequence/"calib.txt");inputs=[]
        for frame in cfg["frames"]:
            scan_path=sequence/"velodyne"/f"{frame:06d}.bin";image_path=sequence/"image_2"/f"{frame:06d}.png"
            for p in (scan_path,image_path,sequence/"calib.txt"):
                relative=p.relative_to(Path(manifest["root"])).as_posix()
                if sha256(p)!=manifest["files_sha256"][relative]:raise ValueError("projection input hash mismatch")
            image=cv2.imread(str(image_path));
            if image is None:raise ValueError("image decode failed")
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);edges=cv2.Canny(gray,cfg["canny_low"],cfg["canny_high"])
            dt=cv2.distanceTransform((edges==0).astype(np.uint8),cv2.DIST_L2,5)
            inputs.append((frame,load_scan(scan_path),image,dt))
        nominal=[];rows=[]
        for frame,points,image,dt in inputs:
            s,depth=score(points,image,dt,P2,Tr,cfg);nominal.append({"frame":frame,**s})
            if not cv2.imwrite(str(out/f"overlay-{frame:06d}.png"),overlay(image,depth,cfg["depth_max_m"])):raise IOError("overlay write failed")
        result["provided_calibration"]={"frames":nominal,**interval([s["score"] for s in nominal],cfg)}
        for axis in ("rx","ry","rz","tx","ty","tz"):
            offsets=cfg["rotation_offsets_deg"] if axis.startswith("r") else cfg["translation_offsets_m"]
            axis_rows=[]
            for offset in offsets:
                scores=[]
                for frame,points,image,dt in inputs:
                    s,_=score(points,image,dt,P2,perturb(Tr,axis,offset),cfg)
                    rows.append({"axis":axis,"offset":offset,"frame":frame,**s});scores.append(s["score"])
                axis_rows.append({"offset":offset,**interval(scores,cfg)})
            valid=[r for r in axis_rows if r["mean"] is not None];best=min(valid,key=lambda x:x["mean"]) if valid else None
            result[axis]={"unit":"degrees" if axis.startswith("r") else "meters","scores":axis_rows,"best":best,"boundary_optimum":best is not None and best["offset"] in (offsets[0],offsets[-1]),"provided_is_best":best is not None and best["offset"]==0}
        write_json(out/"raw-scores.json",rows)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(12,6))
        for axis,ax in zip(("rx","ry","rz","tx","ty","tz"),axes.flat):
            data=result[axis]["scores"];ax.plot([r["offset"] for r in data],[np.nan if r["mean"] is None else r["mean"] for r in data],"o-")
            ax.axvline(0,color="gray",linestyle="--");ax.set(xlabel=axis+" "+result[axis]["unit"],ylabel="fixed consistency score (lower)")
        fig.tight_layout();fig.savefig(out/"score-shape.png",dpi=160);plt.close(fig)
        result["attribution"]="KITTI odometry, A. Geiger et al.; CC BY-NC-SA 3.0; overlays derived from the named official frames"
        result["limitations"]="Only three selected recorded frames; sweeps alter one extrinsic axis at a time; frame bootstrap has low support; image/scan synchronization, sparse depth and scene edges confound scores. Metric/thresholds are frozen before scores. Supplied calibration need not win."
        result["status"]="complete"
    except BaseException as e:result.update(status="failed",failure={"type":type(e).__name__,"message":str(e)});raise
    finally:result["wall_s"]=time.monotonic()-start;result["finished_utc"]=utc();write_json(out/"calibration.json",result)
    print(out)

if __name__=="__main__":main()
