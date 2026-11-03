"""Small stdlib artifact fixtures only; no KITTI/native/browser run occurs."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from common import sha256,write_json
from export_replay import build_replay,read_poses
import run_experiment

def p(x):return [1,0,0,x,0,1,0,0,0,0,1,0]
def rows(path,poses):path.write_text("".join(" ".join(str(x) for x in pose)+"\n" for pose in poses))

class ReplayFixtures(unittest.TestCase):
    def setUp(self):
        scratch=ROOT.parents[1]/"slam/checks/tmp";scratch.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=scratch)
        self.run=Path(self.tmp.name);self.native=self.run/"native";self.native.mkdir()
        cfg={"schema_version":1,"kind":"synthetic_unit_fixture","sequence":"toy","expected_frames":3}
        write_json(self.run/"config.json",cfg)
        write_json(self.run/"run.json",{"status":"complete","config_sha256":sha256(self.run/"config.json"),"git_sha":"a"*40})
        write_json(self.native/"native_summary.json",{"status":"complete","frames":3})
        rows(self.native/"poses_odom.txt",[p(0),p(1),p(2)])
        rows(self.native/"poses_slam.txt",[p(0),p(.9),p(1.8)])
        (self.native/"timestamps.txt").write_text("0\n0.08\n0.21\n")
        (self.native/"loops.jsonl").write_text(json.dumps({"candidate_frame":0,"query_frame":1,"accepted":False,"reason":"temporal_exclusion"})+"\n"+json.dumps({"candidate_frame":0,"query_frame":2,"accepted":True,"reason":"accepted","forward":{"rmse_m":.01}})+"\n")
        (self.native/"events.jsonl").write_text('{"event":"completed","frames":3}\n')
        self.truth=self.run/"truth.txt"
        rows(self.truth,[[0,-1,0,10,1,0,0,20+i,0,0,1,0] for i in range(3)])
    def tearDown(self):self.tmp.cleanup()
    def test_complete_keeps_exact_times_full_se3_and_decisions(self):
        r=build_replay(self.run,self.truth)
        self.assertEqual([f["timestamp_s"] for f in r["frames"]],[0,.08,.21])
        self.assertEqual(r["frames"][1]["truth"],p(1))
        self.assertEqual(r["frames"][2]["optimized"],p(1.8))
        self.assertEqual([l["accepted"] for l in r["loops"]],[False,True])
        self.assertEqual(r["point_cloud"]["status"],"not_exported")
        for key in ("cov","gps","ekf","dr"):self.assertNotIn(key,r["frames"][0])
    def test_failed_prefix_never_fills_missing_pose_or_optimization(self):
        record=json.loads((self.run/"run.json").read_text());record["status"]="failed";write_json(self.run/"run.json",record)
        rows(self.native/"poses_odom_partial.txt",[p(0),p(1)]);(self.native/"timestamps_partial.txt").write_text("0\n0.08\n")
        write_json(self.native/"failure.json",{"frame":2,"accepted_prefix_frames":2,"message":"registration_failed:degenerate_normal_equations","failed_frame_pose":None,"optimized_trajectory":None})
        r=build_replay(self.run,self.truth)
        self.assertEqual(r["status"],"failed");self.assertEqual(r["accepted_frames"],2)
        self.assertTrue(all(f["optimized"] is None for f in r["frames"]))
        self.assertFalse(r["loops"][1]["endpoints_available"])
        self.assertEqual(r["failure"]["failed_frame_pose"],None)
    def test_complete_shortened_output_is_rejected(self):
        rows(self.native/"poses_odom.txt",[p(0),p(1)])
        with self.assertRaisesRegex(ValueError,"every original frame"):build_replay(self.run,self.truth)
    def test_non_se3_pose_and_gt_count_mismatch_are_rejected(self):
        bad=p(0);bad[0]=2;rows(self.native/"poses_odom.txt",[bad,p(1),p(2)])
        with self.assertRaisesRegex(ValueError,"orthonormal"):build_replay(self.run,self.truth)
        rows(self.native/"poses_odom.txt",[p(0),p(1),p(2)]);rows(self.truth,[p(0),p(1)])
        with self.assertRaisesRegex(ValueError,"GT must match"):build_replay(self.run,self.truth)
    def test_live_attempt_is_rejected(self):
        record=json.loads((self.run/"run.json").read_text());record["status"]="started";write_json(self.run/"run.json",record)
        with self.assertRaisesRegex(ValueError,"terminal attempt"):build_replay(self.run,self.truth)
    def test_launcher_preflight_failure_preserves_journal_without_child(self):
        manifest=self.run/"input-manifest.json"
        write_json(manifest,{"schema_version":1,"dataset":"synthetic unit fixture","sequence":"toy","frames":3,"frame_ids":[0,1,2],"verified":True,"root":str(self.run),"files_sha256":{"missing.dat":"b"*64}})
        binary=self.run/"unexecuted-binary-fixture.txt";binary.write_text("never executed")
        build=self.run/"build.json";write_json(build,{"verified":True,"binary_sha256":sha256(binary),"git_sha":"a"*40,"source_files":{}})
        argv=["run_experiment.py","--config",str(self.run/"config.json"),"--input-manifest",str(manifest),"--binary",str(binary),"--build-receipt",str(build),"--repo-root",str(self.run),"--output-parent",str(self.run/"attempts")]
        with patch.object(sys,"argv",argv),patch.object(run_experiment,"environment",return_value={"fixture":"stdlib only"}),patch.object(run_experiment,"git_state",return_value={"git_sha":"a"*40}),patch.object(run_experiment,"source_manifest",return_value={}),patch.object(run_experiment.subprocess,"Popen") as launch:
            with self.assertRaises(FileNotFoundError):run_experiment.main()
            launch.assert_not_called()
        record=json.loads(next((self.run/"attempts").glob("*/run.json")).read_text())
        self.assertEqual(record["status"],"failed");self.assertEqual(record["failure"]["type"],"FileNotFoundError")
        self.assertNotIn("owned_child_cleanup",record)

if __name__=="__main__":unittest.main()
