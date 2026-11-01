"""Finalize the identified 8.4.196 epoch-101 bookkeeping case without training.

The original failed progress, CSV, receipt, log and compiled source stay intact.
Only a proven 100-epoch run from the recorded source is eligible for recovery.
"""
import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

TRAINING_COMMIT = '52381819736eec30d1f76cd30f4bf99bbc1dfa4d'
TRAINING_SHA256 = '24ac8f2b5a847f1d3d8cfc0991da020eefddd00e381804d7fa56e60eba333044'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def archive_file(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        assert sha(source) == sha(destination), destination
    else:
        shutil.copy2(source, destination)
    return {'file': destination.name, 'bytes': source.stat().st_size, 'sha256': sha(source)}


def verify_checkpoint_table(checkpoint, rows):
    table = checkpoint['train_results']
    assert isinstance(table, dict)
    assert set(table) == set(rows[0]), (set(table), set(rows[0]))
    assert all(len(values) == 100 for values in table.values())
    max_delta = 0.0
    for key, values in table.items():
        for index, value in enumerate(values):
            actual, expected = float(value), float(rows[index][key])
            assert math.isfinite(actual) and math.isfinite(expected)
            assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (key, index, actual, expected)
            max_delta = max(max_delta, abs(actual - expected))
    assert int(checkpoint['epoch']) == -1  # legitimate strip_optimizer deployment state
    assert checkpoint.get('optimizer') is None
    assert checkpoint.get('model') is not None
    expected = {'epochs': 100, 'batch': 8, 'imgsz': 640, 'workers': 8, 'seed': 0, 'amp': True}
    assert all(checkpoint['train_args'][key] == value for key, value in expected.items())
    return {'train_results_rows': 100, 'columns': sorted(table), 'max_abs_csv_delta': max_delta,
            'stripped_epoch': checkpoint['epoch'], 'optimizer_is_none': True,
            'metadata_verbatim': {key: checkpoint.get(key) for key in ['version', 'date', 'license', 'docs', 'author']},
            'train_args': checkpoint['train_args']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--training-source', type=Path, required=True)
    parser.add_argument('--launch-record', type=Path, required=True)
    parser.add_argument('--active-record', type=Path, required=True)
    parser.add_argument('--exit-record', type=Path, required=True)
    parser.add_argument('--training-log', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--finalizer-commit')
    args = parser.parse_args()
    assert not args.out.exists(), 'Do not repeat a completed finalization/validation'
    raw = json.loads(args.progress.read_text())
    launch = json.loads(args.launch_record.read_text())
    active = json.loads(args.active_record.read_text())
    receipt = json.loads(args.exit_record.read_text())
    assert launch['source_commit'] == TRAINING_COMMIT
    assert sha(args.training_source) == raw['current_script_sha256'] == launch['source_sha256'] == TRAINING_SHA256
    assert active['source_sha256'] == TRAINING_SHA256
    assert receipt['trainer_exit_code'] == receipt['overall_exit_code'] == 1 and receipt['tee_exit_code'] == 0
    assert raw['status'] == 'failed' and raw['error'] == 'AssertionError()'
    assert [item['epoch'] for item in raw['epochs']] == list(range(1, 102))
    for path in Path('/proc').glob('[0-9]*'):
        try:
            assert os.getpgid(int(path.name)) != active['pgid'], 'Wait for all original training-group members to exit'
        except (FileNotFoundError, ProcessLookupError):
            pass

    source_text = args.training_source.read_text()
    guard_lines = [node.lineno for node in ast.walk(ast.parse(source_text)) if isinstance(node, ast.Assert)
                   and "metadata['epochs']" in ast.get_source_segment(source_text, node)
                   and 'args.epochs + 1' in ast.get_source_segment(source_text, node)]
    assert len(guard_lines) == 1, guard_lines
    log = args.training_log.read_text(errors='replace')
    frames = re.findall(r'File "([^"\n]*/train\.py)", line (\d+), in <module>', log)
    assert frames and int(frames[-1][1]) == guard_lines[0], frames[-3:]
    assert 'AssertionError' in log[log.rfind('Traceback'):]
    csv_path = args.run / 'results.csv'
    rows = list(csv.DictReader(csv_path.open()))
    assert [int(row['epoch']) for row in rows] == list(range(1, 101))
    training_records = raw['epochs'][:100]
    for index, record in enumerate(training_records):
        assert record['batch'] == 8 and math.isfinite(record['elapsed_s']) and record['elapsed_s'] > 0
        for key, value in record['metrics'].items():
            if key in rows[index]:
                assert math.isclose(float(value), float(rows[index][key]), rel_tol=5e-6, abs_tol=5e-6)
    extra = raw['epochs'][100]
    assert extra['epoch'] == 101 and extra['elapsed_s'] >= training_records[-1]['elapsed_s']
    data = Path(raw['requested']['data'])
    assert sha(data) == raw['data_yaml_sha256']

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    preserved = []
    for source, name in [(args.progress, 'raw-progress-with-callback101.json'), (csv_path, 'native-results.csv'),
                         (args.run / 'args.yaml', 'native-args.yaml'), (args.training_source, 'compiled-training-source.py'),
                         (args.exit_record, 'original-training-exit.json'), (args.training_log, 'original-training.log'),
                         (args.launch_record, 'training-launch.json'), (args.active_record, 'training-active.json')]:
        preserved.append(archive_file(source, args.archive_dir / name))

    # Heavy imports are after exact-case, source, receipt and 100-row guards.
    import torch
    import ultralytics
    from ultralytics import YOLO
    assert ultralytics.__version__ == '8.4.196'
    library_source = Path(ultralytics.__file__).parent / 'engine/trainer.py'
    library_text = library_source.read_text()
    final_eval = next(node for node in ast.walk(ast.parse(library_text))
                      if isinstance(node, ast.FunctionDef) and node.name == 'final_eval')
    final_eval_text = ast.get_source_segment(library_text, final_eval)
    assert final_eval_text.index('self.epoch += 1') < final_eval_text.index('self.run_callbacks("on_fit_epoch_end")')
    assert final_eval_text.index('self.run_callbacks("on_fit_epoch_end")') < final_eval_text.index('self.epoch -= 1')
    torch.set_num_threads(4)
    weight_checks = {}
    for name in ['last.pt', 'best.pt']:
        weight = args.run / 'weights' / name
        preserved.append(archive_file(weight, args.archive_dir / name))
        checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
        weight_checks[name] = {'sha256': sha(weight), 'bytes': weight.stat().st_size,
                              **verify_checkpoint_table(checkpoint, rows)}
        del checkpoint

    report = copy.deepcopy(raw)
    report['epochs'] = copy.deepcopy(training_records)
    report['post_training_evaluations'] = [{'event': 'library_final_best_validation', 'callback_epoch_label': 101,
        'last_training_epoch': 100, 'raw_callback_record_verbatim': copy.deepcopy(extra),
        'timer_scope': 'raw elapsed_s starts at epoch100 and overlaps it; not pure final-validation time or another training epoch.'}]
    original_failure = {key: report.pop(key) for key in ['failed_utc', 'error', 'elapsed_before_failure_s'] if key in report}
    report['original_training_failure_verbatim'] = original_failure
    report['last_invocation_seconds'] = original_failure['elapsed_before_failure_s']
    report['last_invocation_timer_scope'] = 'Original invocation up to its recorded post-training guard failure; not full multi-resume training time.'
    report['train_seconds'] = sum(item['elapsed_s'] for item in training_records)
    report['training_timer_scope'] = 'Sum of100 completed training epoch callback windows, including logged pauses; excludes overlapping final-best-eval callback101, setup and discarded partial work.'
    validation_loader = {}

    def val_ready(validator):
        loader = validator.dataloader
        dataset = loader.dataset
        validation_loader.update({'images': len(dataset), 'boxes': sum(len(item['cls']) for item in dataset.labels),
            'batch_size': loader.batch_size, 'workers': loader.num_workers, 'cache_mode': dataset.cache,
            'ram_cache_packed': dataset.cache == 'ram' and hasattr(dataset.ims, 'buffer'),
            'ap_iou_thresholds': validator.iouv.detach().cpu().tolist()})
        assert validation_loader['images'] == 588 and validation_loader['boxes'] == 90968

    best = args.run / 'weights/best.pt'
    best_model = YOLO(str(best))
    best_model.add_callback('on_val_start', val_ready)
    validation_start = time.perf_counter()
    validation = best_model.val(data=str(data), imgsz=640, batch=8, workers=8, device='0', max_det=1000,
        plots=False, verbose=False, project=str(args.run.parent), name=args.run.name + '_val', exist_ok=True)
    validation_seconds = time.perf_counter() - validation_start
    assert validation_loader.get('images') == 588 and validation_loader.get('boxes') == 90968
    assert sha(best) == weight_checks['best.pt']['sha256']
    assert [int(row['epoch']) for row in csv.DictReader(csv_path.open())] == list(range(1, 101))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    copied = args.out.with_name('full_training_epochs.csv')
    shutil.copy2(csv_path, copied)
    shutil.copy2(args.run / 'args.yaml', args.out.with_name('full_training_args.yaml'))
    recovery = {'status': 'complete', 'kind': 'identified_final_eval_callback_bookkeeping', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'original_training_commit': TRAINING_COMMIT, 'original_training_sha256': TRAINING_SHA256,
        'finalizer_file': Path(__file__).name, 'finalizer_sha256': sha(Path(__file__)), 'finalizer_commit': args.finalizer_commit,
        'original_exit_receipt_verbatim': receipt, 'failed_metadata_guard_line': guard_lines[0],
        'raw_inputs_archived': preserved, 'native_csv_training_rows': 100, 'raw_progress_callback_records': 101,
        'library_source_sha256': sha(library_source), 'library_final_eval_source': final_eval_text,
        'checkpoint_checks': weight_checks, 'planned_standalone_best_validation_loader': validation_loader,
        'training_reexecuted': False, 'raw_progress_overwritten': False}
    report.update({'status': 'complete', 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'epochs_completed': 100, 'validation_seconds': validation_seconds, 'validation_split': 'official val',
        'max_det': 1000, 'actual_final_batch': 8, 'weights': {'file': best.name, 'sha256': sha(best)},
        'raw_epoch_csv_sha256': sha(copied), 'artifact_finalization': recovery,
        'mAP50': float(validation.box.map50), 'mAP50_95': float(validation.box.map),
        'precision': float(validation.box.mp), 'recall': float(validation.box.mr)})
    report['validation_results_dict'] = {key: float(value) for key, value in validation.results_dict.items()}
    report['validation_metric_arrays'] = {
        'class_names': validation.names, 'ap_class_index': validation.box.ap_class_index.tolist(),
        'precision': validation.box.p.tolist(), 'recall': validation.box.r.tolist(),
        'ap_by_class_and_iou': validation.box.all_ap.tolist(),
        'ap_iou_thresholds': validation_loader['ap_iou_thresholds']}
    assert all(math.isfinite(report[key]) for key in ['mAP50', 'mAP50_95', 'precision', 'recall'])
    write_json(args.archive_dir / 'FINALIZATION_RECEIPT.json', recovery)
    write_json(args.out, report)
    print('POSTTRAIN_FINALIZED ' + json.dumps({'epochs': 100, 'source_sha256': report['current_script_sha256'],
        'finalizer_sha256': recovery['finalizer_sha256'], 'mAP50': report['mAP50'], 'mAP50_95': report['mAP50_95']}), flush=True)


if __name__ == '__main__':
    main()
