"""Explicit later-stage local-source build. No download, package install, or source fallback."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
from common import environment,git_state,new_run,sha256,source_manifest,utc,write_json

def unpack(archive,dest):
    dest.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        for info in tar.getmembers():
            p=PurePosixPath(info.name)
            if p.is_absolute() or ".." in p.parts or info.issym() or info.islnk() or info.isdev():raise ValueError("unsafe dependency tar member")
        tar.extractall(dest,filter="data")
    roots=list(dest.iterdir())
    if len(roots)!=1 or not roots[0].is_dir():raise ValueError("source archive must have one root")
    return roots[0]

def main():
    a=argparse.ArgumentParser();a.add_argument("--lock",required=True,type=Path);a.add_argument("--archives-dir",required=True,type=Path);a.add_argument("--scratch-parent",required=True,type=Path);a.add_argument("--repo-root",required=True,type=Path);a.add_argument("--git",default="git");a.add_argument("--jobs",type=int,default=2)
    a.add_argument("--prerequisite-prefix",action="append",type=Path,default=[],help="Explicit existing prerequisite prefix; repeat for multiple isolated prefixes")
    a.add_argument("--cmake-policy-version-minimum",choices=("3.5",),help="Explicit compatibility policy for the pinned legacy GTSAM source on CMake 4")
    args=a.parse_args()
    if args.jobs not in (1,2):raise ValueError("start with at most two compiler processes")
    prerequisite_prefixes=[p.resolve(strict=True) for p in args.prerequisite_prefix]
    if any(not p.is_dir() or ";" in str(p) for p in prerequisite_prefixes):raise ValueError("prerequisite prefix must be an existing directory without a CMake list separator")
    lock=json.loads(args.lock.read_text())
    if lock.get("status")!="verified source lock":raise ValueError("candidate dependency hashes unresolved; root must freeze actual bytes first")
    expected={"eigen":"3.4.0","nanoflann":"1.6.3","gtsam":"4.2","googletest":"1.15.2"}
    if {s["name"]:s["version"] for s in lock["sources"]}!=expected:raise ValueError("dependency pin changed")
    for source in lock["sources"]:
        p=args.archives_dir/source["filename"]
        if not source["sha256"] or sha256(p)!=source["sha256"]:raise ValueError("source archive checksum mismatch")
    tools={}
    for name in ("cmake","ctest","ninja","c++"):
        tool=shutil.which(name)
        if tool is None:raise FileNotFoundError("required build tool unavailable in child PATH: "+name)
        path=Path(tool).resolve();tools[name]={"path":str(path),"sha256":sha256(path)}
    run=new_run(args.scratch_parent,"native-build");prefix=run/"prefix";prefix.mkdir()
    prefix_path=";".join(str(p) for p in [prefix,*prerequisite_prefixes])
    record={"status":"started","started_utc":utc(),"command":sys.argv,"environment":environment(),"lock_sha256":sha256(args.lock),"source_files":source_manifest(args.repo_root/"slam"),"prerequisite_prefixes":[str(p) for p in prerequisite_prefixes],"cmake_prefix_path":prefix_path,"resolved_tools":tools,**git_state(args.repo_root,args.git)}
    commands=[];start=time.monotonic()
    def command(parts):
        commands.append(parts);write_json(run/"build.json",dict(record,commands=commands))
        with (run/"commands.log").open("a") as log:
            log.write(json.dumps(parts)+"\n");log.flush();subprocess.run(parts,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        for name in ("eigen","nanoflann","googletest","gtsam"):
            source=next(s for s in lock["sources"] if s["name"]==name)
            root=unpack(args.archives_dir/source["filename"],run/(name+"-source"))
            if name=="nanoflann":
                (prefix/"include").mkdir(exist_ok=True);shutil.copyfile(root/"include/nanoflann.hpp",prefix/"include/nanoflann.hpp");continue
            build=run/(name+"-build")
            parts=["cmake","-S",str(root),"-B",str(build),"-G","Ninja","-DCMAKE_BUILD_TYPE=Release","-DCMAKE_INSTALL_PREFIX="+str(prefix),"-DCMAKE_PREFIX_PATH="+prefix_path]
            if name=="eigen":parts += ["-DBUILD_TESTING=OFF"]
            if name=="googletest":parts += ["-DBUILD_GMOCK=OFF","-DINSTALL_GTEST=ON"]
            if name=="gtsam":parts += ["-DGTSAM_BUILD_TESTS=OFF","-DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF","-DGTSAM_BUILD_UNSTABLE=OFF","-DGTSAM_BUILD_PYTHON=OFF","-DGTSAM_USE_SYSTEM_EIGEN=ON","-DGTSAM_WITH_TBB=OFF","-DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF"]
            if name=="gtsam" and args.cmake_policy_version_minimum:parts += ["-DCMAKE_POLICY_VERSION_MINIMUM="+args.cmake_policy_version_minimum]
            command(parts);command(["cmake","--build",str(build),"--parallel",str(args.jobs)]);command(["cmake","--install",str(build)])
        build=run/"slam-build";command(["cmake","-S",str(args.repo_root/"slam"),"-B",str(build),"-G","Ninja","-DCMAKE_BUILD_TYPE=Release","-DCMAKE_PREFIX_PATH="+prefix_path,"-DNANOFLANN_INCLUDE_DIR="+str(prefix/"include")])
        command(["cmake","--build",str(build),"--parallel",str(args.jobs)])
        command(["ctest","--test-dir",str(build),"--output-on-failure"])
        binary=build/"run_kitti_slam";record.update(status="complete",verified=True,binary=str(binary),binary_sha256=sha256(binary),dependencies=lock["sources"],compiler_version=subprocess.check_output(["c++","--version"],text=True),cmake_version=subprocess.check_output(["cmake","--version"],text=True),dynamic_libraries=subprocess.check_output(["ldd",str(binary)],text=True))
    except BaseException as e:record.update(status="failed",verified=False,failure={"type":type(e).__name__,"message":str(e)});raise
    finally:record.update(finished_utc=utc(),wall_s=time.monotonic()-start,commands=commands);write_json(run/"build.json",record)
    print(run/"build.json")

if __name__=="__main__":main()
