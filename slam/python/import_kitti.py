"""Selective local ZIP import; intentionally has no downloader or network dependency."""
from __future__ import annotations
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile
import zlib
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

FULL_CALIBRATION="data_odometry_calib.zip"
COLOR_CALIBRATION="data_odometry_color.zip"
CAMERA_FIELDS={"P0","P1","P2","P3"}
DECIMAL_TOKEN=re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")

def calibration_fields(payload, archive):
    """Strict schema for the official full/camera-only archive pair."""
    expected=CAMERA_FIELDS|({"Tr"} if archive==FULL_CALIBRATION else set())
    fields={}
    for line in payload.decode("ascii").splitlines():
        if not line.strip():continue
        key,separator,text=line.partition(":")
        if not separator or key not in expected or key in fields:
            raise ValueError("invalid calibration field/schema: "+archive)
        tokens=text.split()
        if len(tokens)!=12 or any(not DECIMAL_TOKEN.fullmatch(token) for token in tokens):
            raise ValueError("invalid calibration numeric field: "+archive+":"+key)
        try: values=tuple(Decimal(token) for token in tokens)
        except InvalidOperation as error: raise ValueError("invalid calibration numeric field: "+archive+":"+key) from error
        if any(not value.is_finite() or not math.isfinite(float(value)) for value in values):
            raise ValueError("nonfinite calibration numeric field: "+archive+":"+key)
        fields[key]=values
    if set(fields)!=expected:
        raise ValueError("missing calibration fields: "+archive)
    return fields

def duplicate_bytes(z,info,target):
    """Read through ZIP EOF/CRC and compare actual bytes, not just two hashes."""
    digest=hashlib.sha256();checksum=0;size=0;same=True
    with z.open(info) as src,target.open("rb") as existing:
        while True:
            block=src.read(1024*1024)
            if not block:break
            size+=len(block);digest.update(block);checksum=zlib.crc32(block,checksum)
            if existing.read(len(block))!=block:same=False
        if existing.read(1):same=False
    if size!=info.file_size:raise ValueError("member length mismatch")
    if checksum!=info.CRC:raise ValueError("member CRC mismatch")
    return same,digest.hexdigest(),f"{checksum:08x}"

def import_selected(z,info,archive,out,name,sequence,hashes,crc,sources,canonical):
    """Import one selected member; differing bytes have one narrow source rule."""
    target=(out/name).resolve()
    if not target.is_relative_to(out):raise ValueError("extraction escapes output")
    source={"archive":Path(archive["path"]).name,"archive_path":str(Path(archive["path"]).resolve()),
            "archive_sha256":archive["sha256"],"member":name,"bytes":info.file_size,
            "crc32":f"{info.CRC:08x}","sha256":None,"crc_verified":False,"accepted":False,"canonical":False}
    sources.setdefault(name,[]).append(source)
    official_calib=name==f"dataset/sequences/{sequence}/calib.txt" and source["archive"] in {FULL_CALIBRATION,COLOR_CALIBRATION}
    if name not in hashes:
        target.parent.mkdir(parents=True,exist_ok=True)
        # Initial creation remains exclusive; ZipExtFile reads through EOF CRC.
        with z.open(info) as src,target.open("xb") as dst:shutil.copyfileobj(src,dst,length=1024*1024)
        if target.stat().st_size!=info.file_size:raise ValueError("member length mismatch")
        source.update(sha256=sha256(target),crc_verified=True)
        if official_calib:calibration_fields(target.read_bytes(),source["archive"])
        hashes[name]=source["sha256"];crc[name]=source["crc32"]
        source.update(accepted=True,canonical=True,resolution="first selected source")
        canonical[name]={"archive":source["archive"],"member":name,"sha256":source["sha256"],"reason":"first selected source"}
        return
    same,actual_hash,actual_crc=duplicate_bytes(z,info,target)
    source.update(sha256=actual_hash,crc32=actual_crc,crc_verified=True)
    incoming_fields=None
    if official_calib:
        # Calibration members are small; use the actual verified ZIP payload.
        with z.open(info) as stream:incoming_payload=stream.read()
        if hashlib.sha256(incoming_payload).hexdigest()!=actual_hash:
            raise ValueError("ZIP member bytes changed during import")
        incoming_fields=calibration_fields(incoming_payload,source["archive"])
    choose_incoming=False;reason="byte-identical duplicate"
    if not same:
        previous=canonical[name]["archive"]
        if not (official_calib and {previous,source["archive"]}=={FULL_CALIBRATION,COLOR_CALIBRATION}):
            raise ValueError("conflicting selected bytes across archives: "+name)
        existing_payload=target.read_bytes()
        if hashlib.sha256(existing_payload).hexdigest()!=hashes[name]:
            raise ValueError("canonical extracted bytes changed during import")
        existing_fields=calibration_fields(existing_payload,previous)
        if any(existing_fields[key]!=incoming_fields[key] for key in CAMERA_FIELDS):
            raise ValueError("conflicting calibration camera values: "+name)
        choose_incoming=source["archive"]==FULL_CALIBRATION
        reason="official full calibration canonical; exact P0-P3 camera subset"
        if choose_incoming:
            replacement=target.with_name(target.name+".canonical-part")
            with replacement.open("xb") as stream:stream.write(incoming_payload)
            replacement.replace(target)
    elif actual_hash!=hashes[name]:
        raise ValueError("canonical extracted bytes changed during import")
    elif official_calib and source["archive"]==FULL_CALIBRATION:
        choose_incoming=True;reason="official full calibration source preferred"
    source.update(accepted=True,resolution=reason)
    if choose_incoming:
        for record in sources[name]:record["canonical"]=False
        source["canonical"]=True
        hashes[name]=source["sha256"];crc[name]=source["crc32"]
        canonical[name]={"archive":source["archive"],"member":name,"sha256":source["sha256"],"reason":reason}

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
    sources={};canonical={}
    journal={"status":"started","utc":utc(),"archives_sha256":identities,"sequence":seq,
             "member_sources":sources,"canonical_selection":canonical}
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
                    import_selected(z,info,receipt,out,name,seq,hashes,crc,sources,canonical)
                    selected.add(name)
        base=f"dataset/sequences/{seq}/"
        if canonical.get(base+"calib.txt",{}).get("archive")!=FULL_CALIBRATION:
            raise ValueError("complete official calibration member required")
        frames=sorted(int(PurePosixPath(n).stem) for n in selected if n.startswith(base+"velodyne/"))
        if frames!=list(range(expected)):raise ValueError("scan IDs missing or extra")
        times=(out/(base+"times.txt")).read_text().splitlines()
        truth=(out/f"dataset/poses/{seq}.txt").read_text().splitlines()
        if len(times)!=expected or len(truth)!=expected:raise ValueError("times/GT count mismatch")
        for i in images:
            if base+f"image_2/{i:06d}.png" not in selected:raise ValueError("selected image missing")
        manifest={"schema_version":1,"dataset":"KITTI odometry","license":"CC BY-NC-SA 3.0", "source":"https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
                  "root":str(out),"sequence":seq,"frames":expected,"frame_ids":frames,"ground_truth_rows":len(truth),"selected_image_frames":images,
                  "archives_sha256":identities,"files_sha256":hashes,"members_crc32":crc,
                  "member_sources":sources,"canonical_selection":canonical,
                  "verified":True,"verified_utc":utc(),"raw_data_in_git":False}
        write_json(out/"input-manifest.json",manifest);journal["status"]="complete"
    except BaseException as e:
        journal.update(status="failed",error={"type":type(e).__name__,"message":str(e)})
        raise
    finally:journal["finished_utc"]=utc();write_json(out/"import-journal.json",journal)
    print(out/"input-manifest.json")

if __name__=="__main__":main()
