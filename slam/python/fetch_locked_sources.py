"""Explicit source retrieval for a later authorized setup/CI step; never invoked at configure time."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import urllib.request
from common import sha256

def main():
    a=argparse.ArgumentParser();a.add_argument("--lock",required=True,type=Path);a.add_argument("--output",required=True,type=Path);args=a.parse_args()
    lock=json.loads(args.lock.read_text())
    if lock.get("status")!="verified source lock":raise ValueError("unresolved lock; no download attempted")
    args.output.mkdir(parents=True,exist_ok=True)
    for s in lock["sources"]:
        if not isinstance(s.get("sha256"),str) or len(s["sha256"])!=64:raise ValueError("missing verified hash")
        if Path(s["filename"]).name!=s["filename"]:raise ValueError("unsafe source filename")
        target=args.output/s["filename"]
        if target.exists():
            if sha256(target)!=s["sha256"]:raise ValueError("cached source hash mismatch")
            continue
        temp=target.with_suffix(target.suffix+".part")
        if temp.exists():raise ValueError("partial source present; preserve and inspect it")
        with urllib.request.urlopen(s["source"],timeout=60) as response,temp.open("xb") as dst:
            while block:=response.read(1024*1024):dst.write(block)
        if sha256(temp)!=s["sha256"]:raise ValueError("download hash mismatch; partial retained")
        temp.rename(target)

if __name__=="__main__":main()
