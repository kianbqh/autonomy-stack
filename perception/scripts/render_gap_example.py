"""Draw an example from the saved gap candidates without running inference again."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--report', default='results/onnx_bench.json')
parser.add_argument('--images', default='data/sku110k_yolo/images/val')
parser.add_argument('--out', default='results/example_gaps.jpg')
args = parser.parse_args()
report = json.loads(Path(args.report).read_text())
row = max(report['per_image'], key=lambda item: len(item['gaps']))
image = Image.open(Path(args.images) / row['image']).convert('RGB')
draw = ImageDraw.Draw(image)
for gap in row['gaps']:
    draw.rectangle([gap['x0'], gap['y0'], gap['x1'], gap['y1']], outline=(255, 0, 0), width=4)
draw.rectangle([0, 0, min(image.width, 460), 23], fill=(0, 0, 0))
draw.text((5, 5), f"Red: candidate gaps | {row['image']} | {len(row['gaps'])} candidates", fill=(255, 255, 255))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
image.save(args.out, quality=85)
print(json.dumps({'output': args.out, 'source_image': row['image'], 'gap_candidates': len(row['gaps']),
                  'selection': f"most candidates among the {len(report['per_image'])} measured images", 'inference_rerun': False}))
