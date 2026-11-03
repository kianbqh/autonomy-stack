"""Preparation gate: stdlib only, with a machine-readable receipt; never imports ML runtime."""
import argparse
import ast
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    start=time.monotonic()
    python_files=sorted(root.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    configs=sorted((root/"configs").glob("*.yaml"))
    for path in configs:
        json.loads(path.read_text(encoding="utf-8"))
    suite=unittest.defaultTestLoader.discover(str(root/"tests"),pattern="test_*.py")
    log=io.StringIO()
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    forbidden=[module for module in ("numpy","torch","gymnasium","gymnasium_robotics","mujoco") if module in sys.modules]
    if forbidden:
        raise AssertionError(f"preparation imported heavy modules: {forbidden}")
    receipt={"status":"passed" if result.wasSuccessful() else "failed","layer":"native Windows stdlib/source checks only",
             "timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"command":sys.argv,
             "python":sys.version,"executable":sys.executable,"elapsed_seconds":time.monotonic()-start,
             "python_files_parsed":len(python_files),"configs_parsed":len(configs),"tests_run":result.testsRun,
             "failures":len(result.failures),"errors":len(result.errors),"skips":[reason for _,reason in result.skipped],
             "heavy_modules_imported":forbidden,"test_output":log.getvalue(),
             "source_hashes":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in python_files+configs},
             "unexecuted":["dependency installation/resolution","Fetch/MuJoCo reset/step","actual BC updates",
                            "native expert success","snapshot exactness","physical DR/gain checks",
                            "physical safety/fault matrix","all formal training/evaluation","CI"]}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps({key:receipt[key] for key in ("status","python_files_parsed","configs_parsed","tests_run","failures","errors","heavy_modules_imported")}))
    if not result.wasSuccessful():
        print(log.getvalue())
        raise SystemExit(1)

if __name__=="__main__":
    main()
