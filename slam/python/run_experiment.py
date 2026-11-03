"""Config-driven immutable CPU runs. No downloads, GT-driven loops, or auto-resume."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import environment, git_state, new_run, sha256, source_manifest, summary, utc, write_json

def validate(config, manifest):
    if config["schema_version"] != 1 or manifest["schema_version"] != 1:
        raise ValueError("unsupported schema")
    expected = config["expected_frames"]
    if config["kind"] == "kitti_full":
        required = {"07": 1101, "00": 4541}
        if config["sequence"] not in required or expected != required[config["sequence"]]:
            raise ValueError("formal frame count is frozen")
        if manifest["dataset"] != "KITTI odometry" or manifest["ground_truth_rows"] != expected:
            raise ValueError("formal run requires full official input manifest")
    elif config["kind"] not in ("synthetic_unit_fixture", "kitti_partial_smoke"):
        raise ValueError("unknown experiment kind")
    if manifest["sequence"] != config["sequence"] or manifest["frames"] != expected:
        raise ValueError("input manifest mismatch")
    if manifest["frame_ids"] != list(range(expected)):
        raise ValueError("missing or reordered scan IDs")
    if config["kind"] == "kitti_partial_smoke" and expected >= {"07":1101,"00":4541}[config["sequence"]]:
        raise ValueError("smoke cannot stand in for full sequence")
    if manifest.get("verified") is not True:
        raise ValueError("unverified input manifest")

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True, type=Path)
    a.add_argument("--input-manifest", required=True, type=Path)
    a.add_argument("--binary", required=True, type=Path)
    a.add_argument("--build-receipt", required=True, type=Path)
    a.add_argument("--repo-root", required=True, type=Path)
    a.add_argument("--output-parent", required=True, type=Path)
    a.add_argument("--git", default="git")
    args = a.parse_args()
    out = new_run(args.output_parent, "slam")
    start = time.monotonic()
    record = {"status":"started", "started_utc":utc(), "start_monotonic":start,
              "launcher_command":sys.argv, "environment":environment(), "output":str(out.resolve())}
    write_json(out/"run.json", record)
    native_process = None
    try:
        config = json.loads(args.config.read_text())
        manifest = json.loads(args.input_manifest.read_text())
        validate(config, manifest)
        record.update(git_state(args.repo_root, args.git))
        build = json.loads(args.build_receipt.read_text())
        if build.get("verified") is not True or build["binary_sha256"] != sha256(args.binary):
            raise ValueError("unverified or mismatched native build receipt")
        source = source_manifest(args.repo_root/"slam")
        if build["git_sha"] != record["git_sha"] or build["source_files"] != source:
            raise ValueError("native binary source/config checkpoint mismatch")
        root = Path(manifest["root"]).resolve()
        # Verify extracted bytes before the timed native process. Never claim hashing as registration time.
        verify_start = time.monotonic()
        for relative, expected_hash in manifest["files_sha256"].items():
            p = (root/relative).resolve()
            if not p.is_relative_to(root) or sha256(p) != expected_hash:
                raise ValueError("input file missing/changed: " + relative)
        record["input_verify_wall_s"] = time.monotonic()-verify_start
        for name, path in (("config.json",args.config),("input-manifest.json",args.input_manifest),("build-receipt.json",args.build_receipt)):
            shutil.copyfile(path,out/name)
        write_json(out/"source-sha256.json",source)
        record.update({"config_sha256":sha256(args.config), "input_manifest_sha256":sha256(args.input_manifest),
                       "binary_sha256":sha256(args.binary), "build_receipt_sha256":sha256(args.build_receipt),
                       "kind":config["kind"], "sequence":config["sequence"]})
        flat = dict(config["algorithm"], expected_frames=config["expected_frames"])
        (out/"native.cfg").write_text("".join(f"{k}={v}\n" for k,v in sorted(flat.items())))
        seq = root/"dataset"/"sequences"/config["sequence"]
        command = [str(args.binary.resolve()),str(seq),str((out/"native.cfg").resolve()),str((out/"native").resolve())]
        native_env=os.environ.copy()
        loader=build.get("loader_environment")
        if isinstance(loader,dict) and isinstance(loader.get("LD_LIBRARY_PATH"),str):
            native_env["LD_LIBRARY_PATH"]=loader["LD_LIBRARY_PATH"]
        record["native_loader_environment"]={"LD_LIBRARY_PATH":native_env.get("LD_LIBRARY_PATH")}
        record["native_command"] = command
        write_json(out/"run.json",record)
        native_start=time.monotonic()
        with (out/"stdout.txt").open("w") as stdout,(out/"stderr.txt").open("w") as stderr:
            native_process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=native_env)
            record["native_pid"]=native_process.pid
            write_json(out/"run.json",record)
            code=native_process.wait()
        record["native_process_wall_s"]=time.monotonic()-native_start
        record["native_exit_code"]=code
        if code: raise RuntimeError(f"native exit {code}; retain events/stderr")
        native=json.loads((out/"native/native_summary.json").read_text())
        if native["frames"]!=config["expected_frames"]: raise ValueError("native output frame count mismatch")
        for name in ("poses_odom.txt","poses_slam.txt","timestamps.txt"):
            if len((out/"native"/name).read_text().splitlines())!=config["expected_frames"]: raise ValueError("output row count mismatch")
        rows=list(csv.DictReader((out/"native/frame_times.csv").open()))
        record["timing"]={key:summary([float(r[key]) for r in rows if int(r["frame"])>0]) for key in ("io_ms","preprocess_ms","map_normals_ms","registration_ms","loop_ms","frame_inclusive_ms","frame_cpu_ms")}
        record["timing_scope"]="all scans except origin; no warmup/repeats; raw CSV includes origin; wall frame includes scan I/O, preprocessing, registration/map/loops and accepted-prefix pose/timestamp checkpoint flush; excludes final timing/candidate/journal flush; frame_cpu_ms uses process std::clock; end graph timing separate; native process includes I/O and output; preflight hashes excluded"
        record["status"]="complete";record["native_summary"]=native
    except BaseException as e:
        if native_process is not None and native_process.poll() is None:
            # Only this launcher's own native child; no process-name matching.
            native_process.terminate()
            try:
                native_process.wait(timeout=5)
                action="terminated"
            except subprocess.TimeoutExpired:
                native_process.kill();native_process.wait();action="killed_after_timeout"
            record["owned_child_cleanup"]={"pid":native_process.pid,"action":action,"exit_code":native_process.returncode}
        record["status"]="failed";record["failure"]={"type":type(e).__name__,"message":str(e),"utc":utc(),"monotonic":time.monotonic()}
        raise
    finally:
        record["finished_utc"]=utc();record["wall_s"]=time.monotonic()-start
        write_json(out/"run.json",record)
    print(out)

if __name__=="__main__": main()
