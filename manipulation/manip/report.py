"""Reduced evidence table. No cut algorithm is presented as a missing experiment."""
import json
from pathlib import Path
from .protocol import aggregate_policies
from .provenance import sha256
def build(inputs,attempt):
    groups={}; receipts=[]; rows=[]
    for path in inputs.get('bc_results',[]):
        item=json.loads(Path(path).read_text())
        if item['status']!='complete' or item['method'] not in ('bc','bc_dr') or item['demo_budget_pool']!=100:
            raise ValueError('only complete main-protocol BC inputs are accepted')
        groups.setdefault(item['method'],[]).append((path,item)); receipts.append({'path':path,'sha256':sha256(path)})
    panel=None
    if inputs.get('panel_summary'):
        path=inputs['panel_summary']; panel=json.loads(Path(path).read_text()); receipts.append({'path':path,'sha256':sha256(path)})
        if panel['status']!='complete': raise ValueError('panel incomplete')
    bc_specs=[]
    for method,items in groups.items():
        if sorted(i['model_seed'] for _,i in items)!=[0,1,2] or len({i['checkpoint_sha256'] for _,i in items})!=3:
            raise ValueError('three independent trained seeds required')
        if len({i['input_manifest_sha256'] for _,i in items})!=1: raise ValueError('each condition must use one frozen demo pool across model seeds')
        specs=[i['provenance']['config']['bc'] for _,i in items]
        if any(s!=specs[0] for s in specs): raise ValueError('BC training settings differ across seeds')
        bc_specs.append(specs[0])
        rows.append({'method':method,'demo_pool':100,'fit_val_unused_counts':[{k:i[k] for k in ('model_seed','actual_fit_episodes','validation_episodes','unused_test_episodes')} for _,i in items],
                     'collection_summary':items[0][1]['collection_summary'],
                     'nominal':aggregate_policies([i['evaluation'] for _,i in items]),'robustness_panel':panel['panel_means'].get(method) if panel else None,
                     'evidence':[p for p,_ in items]})
    missing=[m for m in ('bc','bc_dr') if m not in groups]
    if bc_specs and any(s!=bc_specs[0] for s in bc_specs): raise ValueError('nominal and DR BC must share architecture/optimization settings')
    if panel is None: missing.append('fixed11-cell robustness panel')
    attempt.write('table-data.json',{'status':'partial' if missing else 'reduced TrackA inputs complete; root acceptance pending',
        'rows':rows,'missing_required':missing,'inputs':receipts,'std_definition':'population ddof0 over3trained policies',
        'scope_cuts':'user-directed: diffusion/RL/scaling/full243grid/fullfaultmatrix/oracle; not failures'})
    lines=['# FetchPush BC comparison','','| Method | Successful demo pool | Nominal success mean ± population std | Fixed11-cell mean ± population std |','|---|---:|---|---|']
    for row in rows:
        n=row['nominal']; p=row['robustness_panel']; panel_text=f"{p['mean']} ± {p['population_std']}" if p else 'not run'
        lines.append(f"| {row['method']} |100|{n['mean']} ± {n['population_std']}|{panel_text}|")
    lines+=['','Actual fit/validation/unused-test counts, input hashes, per-policy100episode outcomes and links are in table-data.json.',
            'The11-cell panel is predeclared; success pools can differ because collection conditions differ. No real robot or sim-to-real result.']
    with (attempt.path/'table.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
