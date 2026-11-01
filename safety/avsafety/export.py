"""Re-run two recorded matrix cases and export their actual state/fault traces."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .runner import run_episode
from .scenario import Scenario


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=root / 'viz/public/data/replays')
    parser.add_argument('--report', type=Path, default=root / 'results/replay_integration.json')
    args = parser.parse_args()
    matrix = root / 'safety/results/episodes.csv'
    rows = list(csv.DictReader(matrix.open()))
    common = lambda row: row['estimator'] == 'ekf' and float(row['gps_sigma']) == 1.5 and row['odom_fault'] == 'ok'
    selections = [
        ('normal', 'Normal GPS · successful arrival', [r for r in rows if common(r) and r['gps_fault'] == 'none' and r['outcome'] == 'success']),
        ('fault-collision', 'GPS bias · collision', [r for r in rows if common(r) and r['gps_fault'] == 'bias_step' and r['outcome'] == 'collision' and r['fault_at_closest'].lower() == 'true']),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, reports = [], []
    for identifier, title, candidates in selections:
        assert candidates, identifier
        recorded = min(candidates, key=lambda row: int(row['seed']))
        scenario = Scenario(float(recorded['gps_sigma']), recorded['gps_fault'], recorded['odom_fault'])
        replay = {}
        summary = run_episode(scenario, recorded['estimator'], int(recorded['seed']), replay=replay)
        assert summary['outcome'] == recorded['outcome']
        for key in ('time', 'min_clearance', 'loc_rmse'):
            assert math.isclose(summary[key], float(recorded[key]), rel_tol=1e-9, abs_tol=1e-9), (identifier, key)
        replay['meta'].update({'id': identifier, 'title': title})
        path = args.output_dir / f'{identifier}.json'
        path.write_text(json.dumps(replay, separators=(',', ':'), allow_nan=False) + '\n')
        manifest.append({'id': identifier, 'title': title, 'path': f'./data/replays/{path.name}',
                         'outcome': summary['outcome'], 'scenario': scenario.name, 'seed': int(recorded['seed'])})
        reports.append({'id': identifier, 'samples': len(replay['samples']), 'gps_fixes': len(replay['gps']),
                        'summary': summary, 'reference_row': recorded,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output_dir / 'index.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'exported_utc': datetime.now(timezone.utc).isoformat(),
              'matrix_sha256': hashlib.sha256(matrix.read_bytes()).hexdigest(),
              'fusion_implementation': 'fusion_core.ekf / fusion_core.sim at repository root',
              'episodes': reports}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({entry['id']: entry['summary']['outcome'] for entry in reports}))


if __name__ == '__main__':
    main()
