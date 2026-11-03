"""Opt-in actual simulator/torch engineering checks, separate from pure tests."""
import json
import os
from pathlib import Path
import unittest

@unittest.skipUnless(os.environ.get("M3_RUN_RUNTIME_TESTS")=="1","set M3_RUN_RUNTIME_TESTS=1 for actual simulator engineering checks")
class ActualRuntimeTests(unittest.TestCase):
    def test_actual_fetch_engineering_gates(self):
        from manip.provenance import Attempt
        from manip.runtime_checks import run_checks
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/"configs"/"push_smoke.yaml").read_text())
        with Attempt(os.environ["M3_RUN_ROOT"],"ci-smoke",config,os.environ["M3_REPO"]) as attempt:
            run_checks(config,attempt)

if __name__=="__main__":
    unittest.main()
