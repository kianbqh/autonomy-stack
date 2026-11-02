"""Light stdlib checks permitted before any native/runtime resource release."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from common import percentile,summary
from import_kitti import safe_name,select_member
from run_experiment import validate

class Contracts(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/"configs/seq07.json").read_text())
        self.manifest={"schema_version":1,"dataset":"KITTI odometry","sequence":"07","frames":1101,"frame_ids":list(range(1101)),"ground_truth_rows":1101,"verified":True}
    def test_full_count_and_order_are_frozen(self):
        validate(self.cfg,self.manifest)
        for key,value in (("frames",1100),("ground_truth_rows",1100),("frame_ids",[0,2]),("verified",False)):
            m=copy.deepcopy(self.manifest);m[key]=value
            with self.assertRaises(ValueError):validate(self.cfg,m)
    def test_partial_cannot_be_formal(self):
        cfg=copy.deepcopy(self.cfg);cfg["expected_frames"]=3
        with self.assertRaises(ValueError):validate(cfg,self.manifest)
        cfg=copy.deepcopy(self.cfg);cfg["kind"]="kitti_partial_smoke"
        with self.assertRaises(ValueError):validate(cfg,self.manifest)
    def test_selective_import(self):
        self.assertTrue(select_member("dataset/sequences/07/velodyne/000000.bin","07",[0,550,1100]))
        self.assertTrue(select_member("dataset/sequences/07/image_2/000550.png","07",[0,550,1100]))
        for path in ("dataset/sequences/00/velodyne/000000.bin","dataset/sequences/07/image_2/000001.png","dataset/sequences/07/velodyne/sub/000000.bin","dataset/sequences/07/image_3/000000.png"):
            self.assertFalse(select_member(path,"07",[0,550,1100]))
    def test_unsafe_zip_members_fail(self):
        for name in ("../escape","/absolute","C:/escape","dataset/../escape","a\\b"):
            with self.assertRaises(ValueError):safe_name(zipfile.ZipInfo(name))
        symlink=zipfile.ZipInfo("dataset/link");symlink.external_attr=(0o120777<<16)
        with self.assertRaises(ValueError):safe_name(symlink)
    def test_python_files_parse_without_importing_ml(self):
        for path in ROOT.rglob("*.py"):ast.parse(path.read_text(),filename=str(path))
    def test_percentile_convention(self):
        self.assertAlmostEqual(percentile([1,2,3,4],.95),3.85)
        self.assertEqual(summary([])["mean_ms"],None)
    def test_algorithm_keys_match_native_contract(self):
        cpp=(ROOT/"src/io.cpp").read_text()
        for key in self.cfg["algorithm"]:
            self.assertTrue(f"NUM({key})" in cpp or f"COUNT({key})" in cpp,key)
        self.assertEqual(self.cfg["calibration"]["frames"],[0,550,1100])
    def test_no_ground_truth_in_frontend(self):
        # This architectural gate checks the actual CLI/pipeline interface, not numerical accuracy.
        cpp=(ROOT/"src/pipeline.cpp").read_text()
        self.assertNotIn("poses/",cpp);self.assertNotIn("ground_truth",cpp)
        self.assertIn("verifyLoop(scan,candidate",cpp)
    def test_dependency_downloads_are_explicitly_disabled(self):
        cmake=(ROOT/"CMakeLists.txt").read_text()
        self.assertNotIn("FetchContent",cmake);self.assertIn("find_package(GTSAM 4.2 EXACT REQUIRED)",cmake)

if __name__=="__main__":unittest.main()
