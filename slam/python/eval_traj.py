"""Own NumPy metrics; evo is called as an external independent CLI, never copied."""
from __future__ import annotations
import argparse
import bisect
import json
import io
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
from common import environment, git_state, new_run, sha256, source_manifest, utc, write_json

def load_poses(path):
    a=np.loadtxt(path,ndmin=2)
    if a.shape[1]!=12 or len(a)<2 or not np.isfinite(a).all():raise ValueError("invalid KITTI pose rows")
    poses=np.repeat(np.eye(4)[None],len(a),axis=0);poses[:,:3,:]=a.reshape(-1,3,4)
    rotations=poses[:,:3,:3]
    if not np.allclose(np.swapaxes(rotations,1,2)@rotations,np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(rotations),1,atol=1e-5):raise ValueError("non-SE3 input poses")
    return poses

def normalize(poses):return np.linalg.inv(poses[0])@poses

def rotation_angle(rotation):return float(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))

def align_no_scale(pred,truth):
    x,y=pred[:,:3,3],truth[:,:3,3];mx,my=x.mean(0),y.mean(0)
    u,_,vt=np.linalg.svd((x-mx).T@(y-my));d=np.eye(3);d[2,2]=np.linalg.det(vt.T@u.T)
    R=vt.T@d@u.T;t=my-R@mx
    A=np.eye(4);A[:3,:3]=R;A[:3,3]=t
    return A@pred,A

def distances(truth):return np.r_[0,np.cumsum(np.linalg.norm(np.diff(truth[:,:3,3],axis=0),axis=1))]

def distance_pairs(truth,length,stride=1):
    d=distances(truth);pairs=[]
    for i in range(0,len(d),stride):
        j=bisect.bisect_right(d,d[i]+length)
        if j<len(d):pairs.append((i,j))
    return pairs

def relative_error(pred,truth,i,j):
    gt=np.linalg.inv(truth[i])@truth[j];est=np.linalg.inv(pred[i])@pred[j]
    return np.linalg.inv(est)@gt

def rpe(pred,truth,pairs):
    samples=[]
    for i,j in pairs:
        e=relative_error(pred,truth,i,j)
        samples.append({"i":i,"j":j,"translation_m":float(np.linalg.norm(e[:3,3])),"rotation_rad":rotation_angle(e[:3,:3])})
    return {"count":len(samples),"translation_rmse_m":float(np.sqrt(np.mean([s["translation_m"]**2 for s in samples]))) if samples else None,
            "rotation_rmse_deg":float(np.degrees(np.sqrt(np.mean([s["rotation_rad"]**2 for s in samples])))) if samples else None,"samples":samples}

def kitti_distances(truth):
    """Official KITTI scalar precision; use input GT before origin alignment."""
    d=np.zeros(len(truth),dtype=np.float32)
    for i in range(1,len(truth)):
        delta=np.asarray(truth[i-1,:3,3]-truth[i,:3,3],dtype=np.float32)
        squared=np.float32(np.float32(delta[0]*delta[0]+delta[1]*delta[1])+delta[2]*delta[2])
        d[i]=np.float32(d[i-1]+np.sqrt(squared))
    return d

def kitti_segment_pairs(truth,lengths,stride):
    d=kitti_distances(truth);pairs=[]
    # Official ordering matters for the serial float32 overall sum.
    for i in range(0,len(d),stride):
        for length in lengths:
            nominal=np.float32(length)
            j=bisect.bisect_right(d,np.float32(d[i]+nominal))
            if j<len(d):pairs.append((i,j,float(nominal)))
    return pairs

def kitti_error_components(error):
    # The devkit matrix arithmetic is double; its scalar errors are float.
    t=np.asarray(error[:3,3],dtype=np.float32)
    squared=np.float32(np.float32(t[0]*t[0]+t[1]*t[1])+t[2]*t[2])
    translation=np.float32(np.sqrt(squared))
    diagonal=np.asarray(np.diag(error[:3,:3]),dtype=np.float32)
    trace=np.float32(np.float32(diagonal[0]+diagonal[1])+diagonal[2])
    # 0.5 and 1.0 are double literals in the official source, before float d.
    cosine=np.float32(.5*(float(trace)-1.0))
    angle=np.float32(np.arccos(np.clip(cosine,np.float32(-1),np.float32(1))))
    return translation,angle

def kitti_float32_mean(values):
    if not values:return None
    total=np.float32(0)
    for value in values:total=np.float32(total+np.float32(value))
    return float(np.float32(total/np.float32(len(values))))

def kitti_segments(pred,truth,cfg):
    samples=[]
    for i,j,length in kitti_segment_pairs(truth,cfg["segment_lengths_m"],cfg["segment_start_stride_frames"]):
        translation,angle=kitti_error_components(relative_error(pred,truth,i,j))
        ratio=float(np.float32(translation/np.float32(length)))
        rad=float(np.float32(angle/np.float32(length)))
        samples.append({"i":i,"j":j,"length_m":length,"translation_ratio":ratio,"translation_percent":ratio*100,
                        "rotation_rad_per_m":rad,"rotation_deg_per_m":float(np.degrees(rad)),"rotation_deg_per_100m":float(np.degrees(rad)*100)})
    def summary(rows):
        ratio=kitti_float32_mean([row["translation_ratio"] for row in rows])
        rad=kitti_float32_mean([row["rotation_rad_per_m"] for row in rows])
        return {"count":len(rows),"translation_ratio_mean":ratio,"rotation_rad_per_m_mean":rad,
                "translation_percent_mean":ratio*100 if ratio is not None else None,
                "rotation_deg_per_100m_mean":float(np.degrees(rad)*100) if rad is not None else None}
    result=summary(samples)
    result.update(by_length={str(length):summary([row for row in samples if row["length_m"]==length]) for length in cfg["segment_lengths_m"]},
                  samples=samples,pair_order="start_frame_then_length",scalar_precision="float32_components_distances_and_serial_sums;float64_pose_matrices",
                  official_devkit_parity="not yet run; must cross-check exact pair/aggregation before final acceptance")
    return result


def metrics(pred,truth,cfg):
    if len(pred)!=len(truth):raise ValueError("pose count mismatch")
    kitti=kitti_segments(pred,truth,cfg)
    truth=normalize(truth);pred=normalize(pred);aligned,A=align_no_scale(pred,truth)
    ate=np.linalg.norm(aligned[:,:3,3]-truth[:,:3,3],axis=1)
    return {"ate":{"alignment":"SE3_no_scale","rmse_m":float(np.sqrt(np.mean(ate**2))),"per_frame_m":ate.tolist(),"T_aligned_pred":A.tolist()},
            "rpe_1frame":rpe(pred,truth,[(i,i+1) for i in range(len(pred)-1)]),"rpe_100m":rpe(pred,truth,distance_pairs(truth,cfg["rpe_distance_m"])),
            "kitti_segments":kitti}

def eval_loops(log,truth,threshold):
    truth=normalize(truth);rows=[]
    for line in Path(log).read_text().splitlines():
        row=json.loads(line)
        if row["reason"]=="temporal_exclusion":continue
        i,j=row["candidate_frame"],row["query_frame"]
        distance=float(np.linalg.norm(truth[i,:3,3]-truth[j,:3,3]));row["posthoc_gt_distance_m"]=distance;row["posthoc_true_loop"]=distance<threshold;rows.append(row)
    return {"truth_rule":f"GT distance strictly < {threshold} m; evaluation only", "proposed":len(rows),"accepted":sum(r["accepted"] for r in rows),"true_positives":sum(r["accepted"] and r["posthoc_true_loop"] for r in rows),"false_positives":sum(r["accepted"] and not r["posthoc_true_loop"] for r in rows),"rejected_true_candidates":sum(not r["accepted"] and r["posthoc_true_loop"] for r in rows),"rows":rows,"recall":"not defined over all possible revisits; candidate diagnostics only"}

def evo_command(program,ref,est,options,result_zip,version):
    entrypoint={"evo_ape":"ape","evo_rpe":"rpe"}[program]
    # Invoke the official no-argument entry point with the checked interpreter.
    # Its parser consumes argv[1:], beginning with the KITTI subcommand below.
    code=("import importlib.metadata\n"
          f"if importlib.metadata.version('evo') != {version!r}:\n"
          "    raise RuntimeError('evo version mismatch')\n"
          f"from evo.entry_points import {entrypoint}\n{entrypoint}()\n")
    return [sys.executable,"-c",code,"kitti",str(ref),str(est),*options,"--save_results",str(result_zip)]

def evo_crosscheck(truth,pred,out,version,cfg,own):
    import importlib.metadata
    distribution=importlib.metadata.distribution("evo")
    if distribution.version!=version:raise ValueError("evo version mismatch")
    evo_identity={"python_executable":sys.executable,"distribution_version":distribution.version,"distribution_root":str(distribution.locate_file(""))}
    original_truth,original_pred=load_poses(truth),load_poses(pred)
    # Reuse the exact already-evaluated IDs; do not reselect at float boundaries.
    pairs=[(row["i"],row["j"]) for row in own["rpe_100m"]["samples"]]
    if len(original_truth)!=len(original_pred) or any(not 0<=i<j<len(original_truth) for i,j in pairs):raise ValueError("invalid explicit RPE pair IDs")
    if pairs:
        # Each selected 100m pair occupies rows 2k,2k+1. Evo evaluates every
        # adjacent pair; keep its even error indices and retain the boundary
        # samples separately. This cross-checks EXACT own pair IDs via the CLI
        # without assuming evo's meter-tolerance pairing matches KITTI's rule.
        ids=[index for pair in pairs for index in pair]
        np.savetxt(out/"100m-reference-pairs.txt",original_truth[ids,:3,:].reshape(-1,12),fmt="%.17g")
        np.savetxt(out/"100m-estimate-pairs.txt",original_pred[ids,:3,:].reshape(-1,12),fmt="%.17g")
        write_json(out/"100m-pair-selection.json",{"original_pairs":pairs,"selected_evo_error_indices":list(range(0,len(ids)-1,2)),"boundary_error_indices_ignored":list(range(1,len(ids)-1,2)),"reason":"explicit disjoint-pair expansion, exact original IDs; not a synthetic KITTI result"})
    commands=[("ape","evo_ape",truth,pred,["--align"],own["ate"]["per_frame_m"],False)]
    for relation,key in (("trans_part","translation_m"),("angle_deg","rotation_rad")):
        expected=[row[key] if key=="translation_m" else float(np.degrees(row[key])) for row in own["rpe_1frame"]["samples"]]
        commands.append(("rpe_1frame_"+relation,"evo_rpe",truth,pred,["--delta","1","--delta_unit","f","--all_pairs","--pose_relation",relation],expected,False))
        if pairs:
            expected=[row[key] if key=="translation_m" else float(np.degrees(row[key])) for row in own["rpe_100m"]["samples"]]
            commands.append(("rpe_100m_exact_"+relation,"evo_rpe",out/"100m-reference-pairs.txt",out/"100m-estimate-pairs.txt",["--delta","1","--delta_unit","f","--all_pairs","--pose_relation",relation],expected,True))
    records=[]
    for label,program,ref,est,options,expected,select_even in commands:
        result_zip=out/(label+".zip")
        command=evo_command(program,ref,est,options,result_zip,version)
        t=time.monotonic();p=subprocess.run(command,text=True,capture_output=True)
        (out/(label+".stdout.txt")).write_text(p.stdout);(out/(label+".stderr.txt")).write_text(p.stderr)
        record={"command":command,"exit_code":p.returncode,"wall_s":time.monotonic()-t,"label":label,"passed":False}
        if p.returncode==0:
            with zipfile.ZipFile(result_zip) as z:
                errors=np.load(io.BytesIO(z.read("error_array.npy")),allow_pickle=False)
                record["raw_evo_stats"]=json.loads(z.read("stats.json"))
            record["raw_evo_stats_include_pair_boundaries"]=select_even
            chosen=errors[::2] if select_even else errors
            expected_raw_count=2*len(expected)-1 if select_even else len(expected)
            record["raw_pair_count"]=len(errors);record["expected_raw_pair_count"]=expected_raw_count;record["selected_pair_count"]=len(chosen);record["expected_pair_count"]=len(expected)
            if len(errors)==expected_raw_count and len(chosen)==len(expected):
                record["max_abs_difference"]=float(np.max(np.abs(chosen-np.asarray(expected)))) if expected else 0.0
                record["passed"]=bool(np.allclose(chosen,expected,atol=1e-6,rtol=1e-6))
            record["saved_zip_sha256"]=sha256(result_zip)
        records.append(record)
    return {"version":version,"identity":evo_identity,"commands":records,"passed":all(r["passed"] for r in records),"distance_pair_count":len(pairs),"pair_parity":"frame1 all pairs; 100m explicit pair expansion with saved IDs and selected indices"}

def plot_results(predictions,truth,results,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    truth=normalize(truth)
    fig,ax=plt.subplots(figsize=(8,6));ax.plot(truth[:,0,3],truth[:,2,3],label="GT")
    for name,poses in predictions.items():
        aligned,_=align_no_scale(normalize(poses),truth)
        ax.plot(aligned[:,0,3],aligned[:,2,3],label=name)
    for row in results["loops"]["rows"]:
        if row["accepted"]:
            i,j=row["candidate_frame"],row["query_frame"]
            poses,_=align_no_scale(normalize(predictions["slam"]),truth)
            ax.plot(poses[[i,j],0,3],poses[[i,j],2,3],color="green" if row["posthoc_true_loop"] else "red",alpha=.6,linewidth=.8)
    ax.axis("equal");ax.set(xlabel="camera0 x (m)",ylabel="camera0 z (m)",title="SE3-aligned paths; loops green=true/red=false, GT labels post hoc");ax.legend();fig.tight_layout();fig.savefig(out/"trajectory-loops.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3))
    for name in predictions:ax.plot(results[name]["ate"]["per_frame_m"],label=name)
    ax.set(xlabel="original frame ID",ylabel="aligned position error (m)");ax.legend();fig.tight_layout();fig.savefig(out/"position-error.png",dpi=160);plt.close(fig)

def main():
    a=argparse.ArgumentParser();a.add_argument("--config",required=True,type=Path);a.add_argument("--native-run",required=True,type=Path);a.add_argument("--truth",required=True,type=Path);a.add_argument("--output-parent",required=True,type=Path);a.add_argument("--repo-root",required=True,type=Path);a.add_argument("--git",default="git");a.add_argument("--evo",action="store_true");args=a.parse_args()
    out=new_run(args.output_parent,"evaluation");started=time.monotonic();result={"status":"started","utc":utc(),"command":sys.argv,"environment":environment(),"config_sha256":sha256(args.config),"truth_sha256":sha256(args.truth)}
    try:
        result.update(git_state(args.repo_root,args.git));write_json(out/"source-sha256.json",source_manifest(args.repo_root/"slam"))
        native_record=json.loads((args.native_run.parent/"run.json").read_text())
        if native_record["status"]!="complete" or native_record["config_sha256"]!=result["config_sha256"] or native_record["git_sha"]!=result["git_sha"]:raise ValueError("evaluation must match completed native source/config checkpoint")
        result["native_run_sha256"]=sha256(args.native_run.parent/"run.json")
        cfg=json.loads(args.config.read_text());truth=load_poses(args.truth)
        if len(truth)!=cfg["expected_frames"]:raise ValueError("GT frame count differs from frozen config")
        predictions={}
        for name in ("odom","slam"):
            p=args.native_run/f"poses_{name}.txt";predictions[name]=load_poses(p);result[name]=metrics(predictions[name],truth,cfg["evaluation"]);result[name]["input_sha256"]=sha256(p)
            if args.evo:
                e=out/name;e.mkdir();result[name]["evo"]=evo_crosscheck(args.truth,p,e,cfg["evaluation"]["evo_version"],cfg["evaluation"],result[name])
            else:result[name]["evo"]={"status":"not run"}
        result["loops"]=eval_loops(args.native_run/"loops.jsonl",truth,cfg["evaluation"]["loop_truth_threshold_m"])
        plot_results(predictions,truth,result,out)
        evo_passed=bool(args.evo and all(result[n]["evo"]["passed"] for n in ("odom","slam")))
        result["own_metrics_status"]="complete"
        result["evo_status"]="passed" if evo_passed else ("failed" if args.evo else "not_run")
        # Official devkit verification is not implemented here; do not imply it ran.
        result["official_devkit_status"]="not_run"
        result["status"]="partial_official_devkit_parity_not_run" if evo_passed else "partial_independent_crosscheck_not_passed"
    except BaseException as e:result.update(status="failed",failure={"type":type(e).__name__,"message":str(e)});raise
    finally:result["wall_s"]=time.monotonic()-started;result["finished_utc"]=utc();write_json(out/"metrics.json",result)
    print(out)

if __name__=="__main__":main()
