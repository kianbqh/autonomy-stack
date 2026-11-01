"""Train one detector and preserve actual loader counts, epoch times and validation metrics."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import torch
import ultralytics
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolo11s.pt')
parser.add_argument('--data', default='SKU-110K.yaml')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--device', default='0')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--name', default='sku110k_full_yolo11s')
parser.add_argument('--project', default='runs')
parser.add_argument('--warmup-epochs', type=float, default=3.0)
parser.add_argument('--out', default='results/train_full_3080.json')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cache', choices=('ram', 'disk'), default=False,
                    help='Official Ultralytics image cache mode; RAM allocation remains subject to its memory check')
parser.add_argument('--resume', type=Path, help='Resume this run from its last.pt, retaining optimizer and completed epochs')
args = parser.parse_args()
out = Path(args.out).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
project = Path(args.project).resolve()
run = project / args.name
progress = out.with_name(out.stem + '_progress.json')
assert args.resume or not (run / 'results.csv').exists(), 'Existing run must not be overwritten; choose a new name or explicitly resume it'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)

model = YOLO(str(args.resume) if args.resume else args.model)
pretrained = Path(model.ckpt_path)
metadata = json.loads(progress.read_text()) if args.resume else {
    'started_utc': datetime.now(timezone.utc).isoformat(), 'requested': vars(args),
    'pretrained': {'file': pretrained.name, 'sha256': sha(pretrained)},
    'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                    'ultralytics': ultralytics.__version__, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                    'torch_threads': torch.get_num_threads(),
                    'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip() if args.device != 'cpu' else None},
    'deterministic': True, 'data_yaml_sha256': sha(args.data) if Path(args.data).is_file() else None,
    'pretrained_source': 'Official Ultralytics assets release, resolved by the installed YOLO loader',
    'status': 'preparing', 'epochs': [],
}
resume_epochs = 0
invocation_started = datetime.now(timezone.utc).isoformat()
if args.resume:
    assert args.resume.resolve() == (run / 'weights/last.pt').resolve(), 'Resume only this run\'s last checkpoint'
    resume_epochs = int(model.ckpt['epoch']) + 1
    rows = list(csv.DictReader((run / 'results.csv').open()))
    assert len(rows) == resume_epochs, (len(rows), resume_epochs)
    assert [record['epoch'] for record in metadata['epochs']] == list(range(1, resume_epochs + 1))
    assert int(model.ckpt['train_args']['epochs']) == args.epochs
    assert model.ckpt.get('optimizer') is not None, 'A resumable optimizer state is required'
    metadata.setdefault('resume_history', []).append({
        'started_utc': invocation_started, 'checkpoint_epoch': resume_epochs,
        'checkpoint_sha256': sha(args.resume), 'script_sha256': sha(__file__),
        'requested': vars(args), 'previous_status': metadata['status'],
        'reason': 'Explicit checkpoint recovery, retaining completed epochs and optimizer/EMA state.',
    })
metadata['status'] = 'preparing'
metadata['current_invocation_started_utc'] = invocation_started
metadata['current_script_sha256'] = sha(__file__)
save(progress, metadata)
epoch_start = 0

def loader_report(loader, list_path):
    dataset = loader.dataset
    source = Path(list_path)
    expected = {(source.parent / name.strip()).resolve() for name in source.read_text().splitlines() if name.strip()} if source.is_file() and source.suffix == '.txt' else set()
    usable = {Path(name).resolve() for name in dataset.im_files}
    return {'official_list_count': len(expected) or None, 'usable_images': len(dataset),
            'skipped_images': sorted(p.name for p in expected - usable),
            'usable_boxes': sum(len(label['cls']) for label in dataset.labels),
            'batch_size': loader.batch_size, 'workers': loader.num_workers,
            'cache_mode': dataset.cache, 'ram_cache_packed': dataset.cache == 'ram' and hasattr(dataset.ims, 'buffer')}

def prepared(trainer):
    assert trainer.start_epoch == resume_epochs
    assert trainer.epochs == args.epochs and trainer.batch_size == args.batch
    metadata['loaders'] = {'train': loader_report(trainer.train_loader, trainer.data['train']),
                           'val': loader_report(trainer.test_loader, trainer.data[trainer.args.split])}
    metadata['actual_device'] = str(trainer.device)
    metadata['actual_amp'] = bool(trainer.amp)
    metadata['status'] = 'training'
    metadata['actual_train_args'] = vars(trainer.args)
    save(progress, metadata)
    print('VERIFIED_LOADERS ' + json.dumps(metadata['loaders']), flush=True)

def start_epoch(trainer):
    global epoch_start
    if trainer.device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_start = time.perf_counter()

def finish_epoch(trainer):
    if metadata['epochs'] and metadata['epochs'][-1]['epoch'] == trainer.epoch + 1:
        return  # final_eval can emit this hook again for the final epoch
    if trainer.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - epoch_start
    record = {'epoch': trainer.epoch + 1, 'elapsed_s': elapsed, 'batch': trainer.batch_size,
              'metrics': {key: float(value) for key, value in trainer.metrics.items()},
              'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if trainer.device.type == 'cuda' else None}
    metadata['epochs'].append(record)
    save(progress, metadata)
    print('EPOCH_COMPLETE ' + json.dumps(record), flush=True)

model.add_callback('on_pretrain_routine_end', prepared)
model.add_callback('on_train_epoch_start', start_epoch)
model.add_callback('on_fit_epoch_end', finish_epoch)
t0 = time.perf_counter()
try:
    model.train(data=str(Path(args.data).resolve()) if Path(args.data).is_file() else args.data,
                epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
                project=str(project), name=args.name, exist_ok=True, max_det=1000, plots=False, verbose=False,
                amp=args.device != 'cpu', warmup_epochs=args.warmup_epochs, seed=args.seed, deterministic=True,
                resume=str(args.resume) if args.resume else False, cache=args.cache)
    train_elapsed = time.perf_counter() - t0
    best = run / 'weights/best.pt'
    raw_epochs = list(csv.DictReader((run / 'results.csv').open()))
    assert len(raw_epochs) == args.epochs, (len(raw_epochs), args.epochs)
    assert [record['epoch'] for record in metadata['epochs']] == list(range(1, args.epochs + 1))
    copied = out.with_name('full_training_epochs.csv')
    shutil.copyfile(run / 'results.csv', copied)
    shutil.copyfile(run / 'args.yaml', out.with_name('full_training_args.yaml'))
    t_val = time.perf_counter()
    validation = YOLO(str(best)).val(data=str(Path(args.data).resolve()) if Path(args.data).is_file() else args.data,
                                    imgsz=args.imgsz, batch=model.trainer.batch_size, device=args.device, max_det=1000,
                                    plots=False, verbose=False, project=str(project), name=args.name + '_val', exist_ok=True)
    metadata.update({'status': 'complete', 'finished_utc': datetime.now(timezone.utc).isoformat(),
                     'epochs_completed': len(raw_epochs),
                     'train_seconds': sum(record['elapsed_s'] for record in metadata['epochs']),
                     'training_timer_scope': 'Sum of completed epoch wall-time windows, including in-epoch validation; excludes setup and discarded incomplete epoch work.',
                     'last_invocation_seconds': train_elapsed,
                     'validation_seconds': time.perf_counter() - t_val,
                     'validation_split': 'official val', 'max_det': 1000,
                     'actual_final_batch': model.trainer.batch_size,
                     'weights': {'file': best.name, 'sha256': sha(best)},
                     'raw_epoch_csv_sha256': sha(copied),
                     'mAP50': float(validation.box.map50), 'mAP50_95': float(validation.box.map),
                     'precision': float(validation.box.mp), 'recall': float(validation.box.mr)})
    save(out, metadata)
    save(progress, metadata)
    print('TRAINING_COMPLETE ' + json.dumps({key: metadata[key] for key in ['epochs_completed', 'train_seconds', 'mAP50', 'mAP50_95', 'precision', 'recall']}), flush=True)
except BaseException as error:
    metadata.update({'status': 'failed', 'failed_utc': datetime.now(timezone.utc).isoformat(),
                     'error': repr(error), 'elapsed_before_failure_s': time.perf_counter() - t0})
    save(progress, metadata)
    raise
