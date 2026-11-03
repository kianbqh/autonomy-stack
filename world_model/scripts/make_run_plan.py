"""Freeze one finite plan from completed real BC artifacts. Does not launch runtime."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wm.provenance import artifact_root,dump_new,read_json,sha256,verify_file

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bc-results',nargs=6,required=True,help='actual completed nominal/DR BC results, seeds0/1/2 each')
    p.add_argument('--output',required=True,help='new G-backed plan directory')
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]; groups={}; receipts=[]
    for path in a.bc_results:
        result=read_json(path)
        if result['status']!='complete' or result['method'] not in ('bc','bc_dr') or result['demo_budget_pool']!=100:
            raise ValueError('only real complete main-protocol BC results are accepted')
        verify_file(result['checkpoint'],result['checkpoint_sha256'])
        groups.setdefault(result['method'],[]).append(result)
        receipts.append({'method':result['method'],'path':str(Path(path).resolve()),'sha256':sha256(path),'pointer':'evaluation'})
    if set(groups)!={'bc','bc_dr'}: raise ValueError('both BC conditions required')
    for rows in groups.values():
        if sorted(r['model_seed'] for r in rows)!=[0,1,2] or len({r['checkpoint_sha256'] for r in rows})!=3:
            raise ValueError('three distinct actual trained checkpoints per BC condition required')
    seed0=next(r for r in groups['bc'] if r['model_seed']==0)
    collection=read_json(root/'configs/collection.template.json'); collection.pop('input_status',None)
    for spec in collection['source_specs']:
        if spec['source_policy']!='expert': spec['checkpoint']={'path':seed0['checkpoint'],'sha256':seed0['checkpoint_sha256']}
    configs={'collect':collection,'train':read_json(root/'configs/small_model.json'),
             'prediction':read_json(root/'configs/prediction.json'),'mpc':read_json(root/'configs/mpc.json'),
             'report':{'mode':'formal','policy_results':receipts},'plot':{'mode':'formal','prediction_result':None}}
    output=artifact_root(a.output); output.mkdir(parents=True,exist_ok=False)
    stages=[]
    for kind,config in configs.items():
        path=output/(kind+'.json'); dump_new(path,config)
        stages.append({'name':kind,'stage':kind,'config':{'path':str(path),'sha256':sha256(path)}})
    stages[1]['input_bindings']={'intake':'collect.intake'}
    stages[2]['input_bindings']={'intake':'collect.intake','checkpoints':'train.checkpoints'}
    stages[3]['input_bindings']={'checkpoints':'train.checkpoints'}
    stages[4]['append_bindings']={'policy_results':['mpc.policy_results']}
    stages[5]['config_bindings']={'prediction_result':'prediction.result_receipt'}
    plan=output/'run-plan.json'
    dump_new(plan,{'mode':'formal','protocol':'m3-reduced-v2','planner':configs['mpc']['planner'],'stages':stages,
                   'bc_inputs':receipts,'scope':'one main configuration per three-seed learning comparison; no grid/oracle/RL'})
    print(plan)
if __name__=='__main__': main()
