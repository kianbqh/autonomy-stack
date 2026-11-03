"""Finite reduced-stage runner. No RL, parameter sweep, oracle or automation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',required=True,choices=('collect','bc','panel','report'))
    p.add_argument('--repo',required=True); p.add_argument('--run-root',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu'); p.add_argument('--inputs',required=True)
    a=p.parse_args(); inputs=json.loads(Path(a.inputs).read_text()); here=Path(__file__).resolve().parents[1]
    shared=['--repo',a.repo,'--run-root',a.run_root,'--device',a.device]
    def run(script,config,extra=()):
        cmd=[sys.executable,str(here/'scripts'/script),'--config',str(here/'configs'/config),*shared,*map(str,extra)]
        print(json.dumps(cmd),flush=True); subprocess.run(cmd,check=True)
    if a.stage=='collect':
        for filename in ('push_expert.yaml','push_dr_expert.yaml'): run('collect_demos.py',filename)
    elif a.stage=='bc':
        for condition,filename in (('nominal','push_bc_mlp.yaml'),('dr','push_dr_train.yaml')):
            for seed in (0,1,2): run('train_bc.py',filename,['--seed',seed,'--manifest',inputs['demo_manifests'][condition]])
    elif a.stage=='panel': run('eval_matrix.py','push_bc_mlp.yaml',['--policy-manifest',inputs['policy_manifest'],'--grid-config',here/'configs'/'dr_eval.yaml'])
    elif a.stage=='report': run('build_report.py','push_bc_mlp.yaml',['--plan',inputs['report_inputs']])
if __name__=='__main__': main()
