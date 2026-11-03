"""Finite experiment commands with explicit configuration and immutable outputs."""
import argparse
import json
from pathlib import Path
from .provenance import Attempt,sha256
def main(command):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--repo',required=True)
    p.add_argument('--run-root',required=True); p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--seed',type=int,choices=(0,1,2),default=0); p.add_argument('--budget',type=int,choices=(100,),default=100)
    for name in ('manifest','policy-manifest','grid-config','plan'): p.add_argument('--'+name)
    a=p.parse_args(); config=json.loads(Path(a.config).read_text(encoding='utf-8'))
    config={**config,'execution_device':a.device,'input_config_path':str(Path(a.config).resolve()),'input_config_file_sha256':sha256(a.config)}
    with Attempt(a.run_root,f'{command}-s{a.seed}',config,a.repo) as run:
        if command in ('collect','bc'):
            from .training import collect,train_bc
            if command=='collect': collect(config,run)
            elif not a.manifest: raise ValueError('actual demo manifest required')
            else: train_bc(config,run,a.manifest,a.seed,a.budget,a.device)
        elif command=='matrix':
            from .matrix import evaluate_matrix
            evaluate_matrix(config,run,a.policy_manifest,json.loads(Path(a.grid_config).read_text()),a.device)
        elif command=='smoke':
            from .runtime_checks import run_checks
            run_checks(config,run)
        elif command=='report':
            from .report import build
            build(json.loads(Path(a.plan).read_text()),run)
        elif command=='traces':
            from .traces import export
            export(config,run,json.loads(Path(a.plan).read_text()),a.device)
        else: raise ValueError(command)
        print(run.path)
