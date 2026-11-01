"""Safely extract the official archive and convert every official annotation split."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile
import time
import fcntl

parser = argparse.ArgumentParser()
parser.add_argument('--archive', required=True)
parser.add_argument('--root', required=True)
parser.add_argument('--download-record', help='Optional downloader metadata; otherwise hash the supplied local archive')
parser.add_argument('--out', default='results/sku110k_full_provenance.json')
args = parser.parse_args()
archive, root = Path(args.archive), Path(args.root)
root.mkdir(parents=True, exist_ok=True)
lock = (root / '.prepare.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
marker = root / '.sku110k-extracted.json'
if not marker.exists():
    count, skipped, written, last = 0, 0, 0, time.monotonic()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            # Validate every member, including already complete files. Do not follow
            # archive symlinks; a resumed run preserves complete files by exact size.
            checked = tarfile.data_filter(member, str(root))
            if checked is None:
                continue
            destination = root / checked.name
            if not destination.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'Archive member escapes dataset root: {member.name}')
            if checked.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif checked.isfile():
                if destination.is_symlink():
                    raise ValueError(f'Unexpected existing symlink: {destination}')
                if destination.is_file() and destination.stat().st_size == checked.size:
                    skipped += 1
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + '.extracting')
                    with stream.extractfile(member) as source, temporary.open('wb') as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                    assert temporary.stat().st_size == checked.size
                    temporary.replace(destination)
                    written += 1
            else:
                raise ValueError(f'Unsupported archive member: {member.name}')
            count += 1
            if time.monotonic() - last >= 15:
                print(f'Archive entries {count}, complete files reused {skipped}, files written {written}', flush=True)
                last = time.monotonic()
    marker.write_text(json.dumps({'entries': count, 'complete_files_reused': skipped, 'files_written': written,
                                 'finished_utc': datetime.now(timezone.utc).isoformat()}) + '\n')
dataset = root / 'SKU110K_fixed'
assert (dataset / 'images').is_dir() and (dataset / 'annotations').is_dir()
labels = dataset / 'labels'
labels.mkdir(exist_ok=True)
report = {
    'source': 'https://github.com/eg4000/SKU110K_CVPR19',
    'source_terms': 'Official source states academic and non-commercial use',
    'download': json.loads(Path(args.download_record).read_text()) if args.download_record else {
        'url': 'https://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz',
        'bytes': archive.stat().st_size,
        'sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
        'source': 'User-supplied local official archive',
    },
    'conversion': 'All official CSV rows; single object class; normalized xywh using CSV dimensions, five decimals; no image resizing',
    'splits': {},
}
for split, expected in [('train', 8219), ('val', 588), ('test', 2936)]:
    source = dataset / 'annotations' / f'annotations_{split}.csv'
    grouped = defaultdict(list)
    invalid = []
    rows = 0
    with source.open(newline='') as handle:
        for row_number, row in enumerate(csv.reader(handle), 1):
            if len(row) != 8:
                raise ValueError(f'{source.name}:{row_number}: expected eight columns')
            name = row[0]
            if Path(name).name != name:
                raise ValueError(f'Unexpected image name: {name}')
            x1, y1, x2, y2 = map(float, row[1:5])
            width, height = map(float, row[6:8])
            if not all(math.isfinite(x) for x in (x1, y1, x2, y2, width, height)) or min(width, height) <= 0:
                raise ValueError(f'{source.name}:{row_number}: invalid numeric annotation')
            # Match Ultralytics official conversion, retaining questionable annotations for its verifier.
            normalized = ((x1 / width + x2 / width) / 2, (y1 / height + y2 / height) / 2,
                          x2 / width - x1 / width, y2 / height - y1 / height)
            if normalized[2] <= 0 or normalized[3] <= 0 or min(normalized) < 0 or max(normalized) > 1:
                invalid.append({'row': row_number, 'image': name, 'xywh': normalized})
            grouped[name].append('0 ' + ' '.join(f'{value:.5f}' for value in normalized) + '\n')
            rows += 1
    names = sorted(grouped)
    assert len(names) == expected, (split, len(names), expected)
    missing = [name for name in names if not (dataset / 'images' / name).is_file()]
    for name, lines in grouped.items():
        path = labels / Path(name).with_suffix('.txt')
        temp = path.with_suffix('.txt.tmp')
        temp.write_text(''.join(lines))
        temp.replace(path)
    (dataset / f'{split}.txt').write_text(''.join(f'./images/{name}\n' for name in names))
    report['splits'][split] = {'official_images': len(names), 'annotation_rows': rows,
                               'missing_images': missing, 'out_of_range_or_nonpositive_boxes': invalid,
                               'annotation_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                               'image_verification': 'Training loader performs image/label verification; usable and skipped counts are recorded by the trainer'}
    print(f'{split}: {len(names)} official images, {rows} boxes, missing={len(missing)}, questionable_boxes={len(invalid)}', flush=True)
(dataset / 'data.yaml').write_text(f'path: {dataset.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: object\n')
report['prepared_utc'] = datetime.now(timezone.utc).isoformat()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + '\n')
print(f'FULL_DATA_READY {dataset / "data.yaml"}', flush=True)
