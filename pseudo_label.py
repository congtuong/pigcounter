"""Generate a separate pseudo-labeled pig-sei9k dataset with a trained teacher."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def class_mapping(names: dict) -> dict[int, int]:
    expected = {'pig': 0, 'person': 1}
    mapping = {int(i): expected[str(name).lower().strip()] for i, name in names.items()
               if str(name).lower().strip() in expected}
    if set(mapping.values()) != {0, 1}:
        raise ValueError(f'Teacher must declare pig and person classes, got {names}')
    return mapping


def label_lines(boxes, mapping: dict[int, int], threshold: float) -> tuple[list[str], list[float]]:
    lines, scores = [], []
    for cls, score, coordinates in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xywhn.tolist()):
        if int(cls) not in mapping or score < threshold:
            continue
        coords = [max(0.0, min(1.0, float(v))) for v in coordinates]
        if coords[2] <= 0 or coords[3] <= 0:
            continue
        lines.append(f'{mapping[int(cls)]} ' + ' '.join(f'{v:.8f}' for v in coords))
        scores.append(float(score))
    return lines, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Teacher .pt or TensorRT .engine')
    parser.add_argument('--source', type=Path, default=ROOT / 'datasets' / 'pig-sei9k')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets' / 'pig-sei9k-pseudo')
    parser.add_argument('--conf', type=float, default=0.7)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-det', type=int, default=1000, help='Detection cap for crowded scenes')
    parser.add_argument('--keep-empty', action='store_true', help='Include no-detection images as negatives after review')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not args.model.is_file() or args.model.suffix not in {'.pt', '.engine'}:
        parser.error('--model must be an existing .pt or .engine file')
    if not 0 < args.conf < 1 or not 0 < args.iou <= 1 or args.batch < 1 or args.max_det < 1:
        parser.error('Invalid confidence, IoU, batch size, or detection cap')
    if output == source or output in source.parents or source in output.parents:
        parser.error('--output must be separate from the source dataset')
    if output.exists():
        parser.error('Output already exists; choose a new --output to preserve prior results')
    images_by_split = {}
    for split, candidates in [('train', ('train',)), ('val', ('valid', 'val')), ('test', ('test',))]:
        folder = next((source / name / 'images' for name in candidates if (source / name / 'images').is_dir()), None)
        if folder:
            images = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if images:
                images_by_split[split] = (folder, images)
    if not images_by_split:
        parser.error(f'No images found in {source}')
    from ultralytics import YOLO
    teacher = YOLO(str(args.model.resolve()), task='detect')
    output.mkdir(parents=True)
    counts = {}
    mapping = None
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as report:
        writer = csv.writer(report)
        writer.writerow(['split', 'image', 'detections', 'min_confidence', 'included', 'at_detection_cap'])
        for split, (folder, images) in images_by_split.items():
            counts[split] = dict(scanned=0, included=0, boxes=0, empty=0)
            for start in range(0, len(images), args.batch):
                batch = images[start:start + args.batch]
                results = teacher.predict(source=[str(p) for p in batch], stream=True,
                                          imgsz=args.imgsz, batch=args.batch, device=args.device,
                                          conf=args.conf, iou=args.iou, max_det=args.max_det,
                                          rect=False, verbose=False, save=False)
                for result in results:
                    if mapping is None:
                        mapping = class_mapping(result.names)
                    image = Path(result.path).resolve()
                    relative = image.relative_to(folder)
                    lines, scores = label_lines(result.boxes, mapping, args.conf)
                    included = bool(lines) or args.keep_empty
                    counts[split]['scanned'] += 1
                    counts[split]['empty'] += not bool(lines)
                    writer.writerow([split, relative.as_posix(), len(lines), min(scores) if scores else '', included,
                                     len(result.boxes) >= args.max_det])
                    if not included:
                        continue
                    destination = output / split / 'images' / relative
                    label = output / split / 'labels' / relative.with_suffix('.txt')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    label.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(image, destination)
                    except OSError:
                        shutil.copy2(image, destination)
                    label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
                    counts[split]['included'] += 1
                    counts[split]['boxes'] += len(lines)
                print(f'{split}: {min(start + args.batch, len(images))}/{len(images)} images', flush=True)
    config = {'path': output.as_posix(), 'names': {0: 'pig', 1: 'person'}}
    for split, stats in counts.items():
        if stats['included']:
            config[split] = f'{split}/images'
    (output / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    summary = dict(model=str(args.model.resolve()), conf=args.conf, iou=args.iou,
                   imgsz=args.imgsz, batch=args.batch, keep_empty=args.keep_empty,
                   max_det=args.max_det, counts=counts)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Pseudo dataset: {output}. Review predictions.csv before using it for training.')


if __name__ == '__main__':
    main()
