"""Finite config-driven entry points with immutable source and input receipts."""
import argparse
from pathlib import Path
from .protocol import MODEL_SEEDS,population_summary
from .provenance import attempted,checked_config,read_json,sha256
def main(command):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output-root',required=True)
    for name in ('intake','checkpoints','resume'): p.add_argument('--'+name)
    a=p.parse_args(); config=checked_config(a.config); inputs={}
    for key in ('intake','checkpoints','resume'):
        path=getattr(a,key)
        if path: inputs[key]={'path':str(Path(path).resolve()),'sha256':sha256(path)}
    with attempted(a.output_root,command,a.config,inputs) as run:
        run.write('effective-config.json',{'effective_config':config,'inputs':inputs})
        if command=='collect':
            from .collection import collect
            result=collect(config,run)
        elif command in ('train','prediction'):
            if not a.intake: raise ValueError('immutable actual episode intake required')
            from .dataset import load_dataset
            intake=read_json(a.intake); dataset=load_dataset(intake['manifests'],intake); run.write('dataset-audit.json',dataset.audit)
            if command=='train':
                from .train import train_seed
                if config['model_seeds']!=list(MODEL_SEEDS): raise ValueError('three independent small-model seeds required')
                resumes=read_json(a.resume) if a.resume else {}
                rows=[train_seed(dataset,config,s,run,config['checkpoint_root'],resumes.get(str(s))) for s in config['model_seeds']]
                checkpoints=[{'path':r['checkpoint'],'sha256':r['checkpoint_sha256'],'model_seed':r['model_seed']} for r in rows]
                run.write('checkpoints.json',{'checkpoints':checkpoints,'dataset_identity':dataset.identity})
                result={'models':rows,'checkpoints':checkpoints,'dataset_identity':dataset.identity}
            else:
                if not a.checkpoints: raise ValueError('actual trained checkpoints required')
                from .prediction import evaluate_prediction
                from .train import load_model
                checkpoints=read_json(a.checkpoints)['checkpoints']
                if sorted(c['model_seed'] for c in checkpoints)!=list(MODEL_SEEDS) or len({c['sha256'] for c in checkpoints})!=3: raise ValueError('three independent checkpoints required')
                rows=[]
                for spec in checkpoints:
                    model,checkpoint=load_model(spec,config['device']); rows.append(evaluate_prediction(model,checkpoint,dataset,config))
                across=[]
                for h in config['horizons']:
                    summaries=[next(x for x in r['summary'] if x['horizon']==h) for r in rows]
                    across.append({'horizon':h,**{k:population_summary([r[k]['mean'] for r in summaries]) for k in ('object_mm','gripper_mm','cv_object_mm','cv_gripper_mm')}})
                result={'models':rows,'across_model_seeds':across,'dataset_identity':dataset.identity,'checkpoints':checkpoints,
                    'seed_aggregation':'population summaries of the3trained model episode means; the deterministic CV baseline is shared across those same test episodes, not3trained CV models'}
        elif command=='mpc':
            from .evaluation import evaluate_nominal
            if a.checkpoints: config={**config,'checkpoints':read_json(a.checkpoints)['checkpoints']}
            result=evaluate_nominal(config,run)
        elif command=='report':
            from .reporting import compare
            result=compare(config,run)
        elif command=='plot':
            from .reporting import plot_prediction
            result=plot_prediction(config['prediction_result'],run)
        elif command=='run_all':
            from .pipeline import run_plan
            result=run_plan(config,run,a.output_root)
        else: raise ValueError('only reduced finite main stages are supported')
        run.finish(result); print(run.path)
