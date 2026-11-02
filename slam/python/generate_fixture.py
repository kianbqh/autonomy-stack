"""Three-frame, pure-stdlib CPU unit fixture. It is not KITTI data or a benchmark."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
from common import sha256,utc,write_json

def main():
    a=argparse.ArgumentParser();a.add_argument("--output",required=True,type=Path);a.add_argument("--base-config",required=True,type=Path);a.add_argument("--degenerate-frame",type=int,choices=(1,2),help="explicit negative unit fixture, never a KITTI result");args=a.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);seq=out/"dataset/sequences/toy";scans=seq/"velodyne";scans.mkdir(parents=True)
    world=[]
    for i in range(-20,21):
        for j in range(-20,21):
            x,y=i*.12,j*.12
            world.extend(((5,x,y),(x,-4,y),(x,y,-2)))
    for frame in range(3):
        with (scans/f"{frame:06d}.bin").open("wb") as f:
            selected=world[2::3] if frame==args.degenerate_frame else world
            for x,y,z in selected:f.write(struct.pack("<ffff",x-frame*.04,y,z,1.0))
    (seq/"times.txt").write_text("0\n0.08\n0.21\n")
    (seq/"calib.txt").write_text("P2: 500 0 300 -50 0 500 200 0 0 0 1 0\nTr: 0 -1 0 0.2 0 0 -1 -0.1 1 0 0 0.3\n")
    truth=out/"dataset/poses";truth.mkdir();(truth/"toy.txt").write_text("".join(f"1 0 0 0 0 1 0 0 0 0 1 {i*.04}\n" for i in range(3)))
    # Tr rotates LiDAR +x into camera0 +z; this is a coordinate fixture, not measured KITTI GT.
    config=json.loads(args.base_config.read_text());config.update(kind="synthetic_unit_fixture",sequence="toy",expected_frames=3)
    config["algorithm"].update(voxel=.12,map_voxel=.12,min_range=0,normal_k=12,normal_radius=.5,normal_planarity=.05,keyframe_distance=.03)
    write_json(out/"config.json",config)
    files={p.relative_to(out).as_posix():sha256(p) for p in sorted((out/"dataset").rglob("*")) if p.is_file()}
    write_json(out/"input-manifest.json",{"schema_version":1,"dataset":"synthetic unit fixture","sequence":"toy","root":str(out),"frames":3,"frame_ids":[0,1,2],"ground_truth_rows":3,"ground_truth_frame":"camera0_initial_relative_fixture_only","files_sha256":files,"verified":True,"verified_utc":utc(),"negative_fixture":None if args.degenerate_frame is None else {"frame":args.degenerate_frame,"fault":"single_plane_scan","expected":"native degeneracy rejection; must be verified by actual execution"}})
    print(out)

if __name__=="__main__":main()
