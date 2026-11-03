"""Frozen11-cell paired robustness panel; no family selection or Cartesian sweep."""
import csv
import importlib
import json
from pathlib import Path
from .envs import make_env
from .evaluate import evaluate_policy
from .policies import load_policy
from .protocol import EVAL_SEEDS,aggregate_policies,digest,assert_panel
from .provenance import sha256
def validate_policy_manifest(manifest):
    groups={}
    for p in manifest['policies']:
        if p['method'] not in ('bc','bc_dr','expert'): raise ValueError('only BC comparison is required in main panel')
        if p['kind']!=p['method']: raise ValueError('BC nominal/DR condition must match its checkpoint kind')
        groups.setdefault(p['method'],[]).append(p)
        if p['kind']!='expert' and sha256(p['checkpoint'])!=p['checkpoint_sha256']: raise ValueError('checkpoint bytes changed')
    if not {'bc','bc_dr'}<=set(groups): raise ValueError('both predeclared BC conditions required')
    for method,policies in groups.items():
        if method=='expert':
            if len(policies)!=1 or policies[0]['model_seed'] is not None: raise ValueError('expert is one fixed controller')
        elif sorted(p['model_seed'] for p in policies)!=[0,1,2] or len({p['checkpoint_sha256'] for p in policies})!=3:
            raise ValueError('each BC condition requires3 independent actual checkpoints')
    return groups
def load_adapter(spec,device):
    if spec['kind']=='expert':
        from .expert import PushExpert
        return PushExpert(spec['controller'])
    if sha256(spec['checkpoint'])!=spec['checkpoint_sha256']: raise ValueError('checkpoint hash changed')
    if spec['kind']=='external':
        module=importlib.import_module(spec['adapter_module'])
        if sha256(module.__file__)!=spec['adapter_sha256']: raise ValueError('external adapter source changed')
        return module.load_policy(spec,device=device)
    policy=load_policy(spec['kind'],spec['checkpoint'],device)
    if policy.model_seed!=spec['model_seed']: raise ValueError('actual checkpoint seed mismatch')
    return policy
def evaluate_matrix(config,attempt,policy_manifest_path,grid_config,device):
    assert_panel(grid_config); groups=validate_policy_manifest(json.loads(Path(policy_manifest_path).read_text()))
    rows=[]
    for method,policies in groups.items():
        for spec in policies:
            policy=load_adapter(spec,device)
            for index,cell in enumerate(grid_config['cells']):
                perturbation={k:v for k,v in cell.items() if k!='name'}
                env=make_env({**config,'training_randomization':None},perturbation)
                try: result=evaluate_policy(env,policy,EVAL_SEEDS,spec['model_seed'],method)
                finally: env.close()
                row={'method':method,'model_seed':spec['model_seed'],'cell':index,'cell_name':cell['name'],
                     'perturbation':perturbation,'policy_checkpoint_sha256':spec.get('checkpoint_sha256'),'result':result}
                attempt.write(f"panel-{method}-s{spec['model_seed']}-{index:02d}.json",row); rows.append(row)
    path=attempt.path/'robustness_panel.csv'
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method','model_seed','cell','cell_name','successes','episodes','success_rate','perturbation'])
        writer.writeheader()
        for row in rows: writer.writerow({**{k:row[k] for k in ('method','model_seed','cell','cell_name')},
            **{k:row['result'][k] for k in ('successes','episodes','success_rate')},'perturbation':json.dumps(row['perturbation'],sort_keys=True)})
    means={}
    for method,policies in groups.items():
        summary=[{'model_seed':p['model_seed'],'success_rate':sum(r['result']['success_rate'] for r in rows if r['method']==method and r['model_seed']==p['model_seed'])/11} for p in policies]
        means[method]=({'fixed_controller_panel_mean':summary[0]['success_rate'],'policies':summary} if method=='expert' else aggregate_policies(summary))
    attempt.write('panel-summary.json',{'status':'complete','policy_manifest_sha256':sha256(policy_manifest_path),
        'panel_config_sha256':digest(grid_config),'panel_means':means,'rows':rows,'csv':str(path),'csv_sha256':sha256(path),
        'weighting':'uniform11-cell predeclared one-axis panel,100paired resets/cell; not Cartesian or full-distribution guarantee'})
