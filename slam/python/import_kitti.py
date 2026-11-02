"""Selective local ZIP import; intentionally has no downloader or network dependency."""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile
from common import new_run, sha256, utc, write_json

def safe_name(info):
    name=info.orig_filename
    path=PurePosixPath(name)
    if not name or name!=info.filename or any(ord(c)<32 for c in name) or "\\" in name or path.is_absolute() or any(p in ("..", "") for p in path.parts) or any(":" in p for p in path.parts):
        raise ValueError("unsafe ZIP member: "+name)
    mode=(info.external_attr>>16)&0xffff
    if stat.S_ISLNK(mode): raise ValueError("ZIP symlink rejected")
    if info.flag_bits&1: raise ValueError("encrypted ZIP rejected")
    return str(path)

def select_member(name, sequence, image_frames):
    base=f"dataset/sequences/{sequence}/"
    if name.startswith(base+"velodyne/"):
        leaf=name.removeprefix(base+"velodyne/")
        return len(leaf)==10 and leaf[:6].isdigit() and leaf.endswith(".bin")
    return name in {base+"calib.txt",base+"times.txt",f"dataset/poses/{sequence}.txt"} or name in {base+f"image_2/{n:06d}.png" for n in image_frames}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--archives",required=True,type=Path,help="local receipt JSON: archives [{path,bytes,sha256}]")
    a.add_argument("--config",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    a.add_argument("--allow-sequence-00",action="store_true")
    args=a.parse_args()
    cfg=json.loads(args.config.read_text());seq=cfg["sequence"]
    if cfg["kind"]!="kitti_full" or seq not in ("07","00") or (seq=="00" and not args.allow_sequence_00): raise ValueError("only authorized seq07 or explicit stretch00")
    expected={"07":1101,"00":4541}[seq]
    if cfg["expected_frames"]!=expected: raise ValueError("full input count mismatch")
    images=cfg["calibration"]["frames"]
    if len(images) not in (2,3) or len(set(images))!=len(images) or any(type(i)!=int or i<0 or i>=expected for i in images): raise ValueError("only 2-3 predeclared image_2 frames")
    receipts=json.loads(args.archives.read_text())["archives"]
    allowed={"data_odometry_calib.zip","data_odometry_poses.zip","data_odometry_velodyne.zip","data_odometry_color.zip"}
    if {Path(r["path"]).name for r in receipts}!=allowed or len(receipts)!=4: raise ValueError("require four complete official archives")
    identities={}
    for r in receipts:
        p=Path(r["path"])
        if not p.is_file() or p.suffix!=".zip" or p.stat().st_size!=r["bytes"] or not isinstance(r["sha256"],str) or len(r["sha256"])!=64:
            raise ValueError("incomplete archive/receipt")
        if sha256(p)!=r["sha256"]: raise ValueError("archive checksum mismatch")
        identities[p.name]=r["sha256"]
    out=args.output.resolve()
    if out.exists():
        old=json.loads((out/"input-manifest.json").read_text())
        if old["archives_sha256"]!=identities or old["sequence"]!=seq or old["selected_image_frames"]!=images: raise ValueError("existing extraction differs; choose fresh output")
        for relative,h in old["files_sha256"].items():
            target=(out/relative).resolve()
            if not target.is_relative_to(out) or sha256(target)!=h: raise ValueError("existing extracted bytes changed")
        print(out/"input-manifest.json");return
    out.mkdir(parents=True,exist_ok=False)
    journal={"status":"started","utc":utc(),"archives_sha256":identities,"sequence":seq}
    write_json(out/"import-journal.json",journal)
    try:
        selected=set();hashes={};crc={}
        for receipt in receipts:
            with zipfile.ZipFile(receipt["path"]) as z:
                names=set()
                for info in z.infolist():
                    name=safe_name(info)
                    if name in names: raise ValueError("duplicate ZIP member")
                    names.add(name)
                    if info.is_dir() or not select_member(name,seq,images):continue
                    if name in selected: raise ValueError("duplicate selected path across archives")
                    selected.add(name);target=(out/name).resolve()
                    if not target.is_relative_to(out):raise ValueError("extraction escapes output")
                    target.parent.mkdir(parents=True,exist_ok=True)
                    # Exclusive creation and ZipExtFile EOF CRC verification, no extractall.
                    with z.open(info) as src,target.open("xb") as dst:shutil.copyfileobj(src,dst,length=1024*1024)
                    if target.stat().st_size!=info.file_size:raise ValueError("member length mismatch")
                    hashes[name]=sha256(target);crc[name]=f"{info.CRC:08x}"
        base=f"dataset/sequences/{seq}/"
        frames=sorted(int(PurePosixPath(n).stem) for n in selected if n.startswith(base+"velodyne/"))
        if frames!=list(range(expected)):raise ValueError("scan IDs missing or extra")
        times=(out/(base+"times.txt")).read_text().splitlines()
        truth=(out/f"dataset/poses/{seq}.txt").read_text().splitlines()
        if len(times)!=expected or len(truth)!=expected:raise ValueError("times/GT count mismatch")
        for i in images:
            if base+f"image_2/{i:06d}.png" not in selected:raise ValueError("selected image missing")
        manifest={"schema_version":1,"dataset":"KITTI odometry","license":"CC BY-NC-SA 3.0", "source":"https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
                  "root":str(out),"sequence":seq,"frames":expected,"frame_ids":frames,"ground_truth_rows":len(truth),"selected_image_frames":images,
                  "archives_sha256":identities,"files_sha256":hashes,"members_crc32":crc,"verified":True,"verified_utc":utc(),"raw_data_in_git":False}
        write_json(out/"input-manifest.json",manifest);journal["status"]="complete"
    except BaseException as e:
        journal.update(status="failed",error={"type":type(e).__name__,"message":str(e)})
        raise
    finally:journal["finished_utc"]=utc();write_json(out/"import-journal.json",journal)
    print(out/"input-manifest.json")

if __name__=="__main__":main()
