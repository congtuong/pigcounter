"""Prepare the downloaded Roboflow datasets and train a single-class pig detector."""
from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
ALIASES = ('pig-sei9k', 'pig-hjhho', 'pig-detection', 'pig-pig-pig',
           'pig-tilpu', 'pig-whv6e', 'pig-vvvue', 'pig-mkd4a',
           'pig-kebke', 'pig-a2jtl', 'pig-detectv2')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def prepare(source: Path, output: Path, allow_partial: bool) -> Path:
    available = [alias for alias in ALIASES if (source / alias / 'data.yaml').is_file()]
    missing = sorted(set(ALIASES) - set(available))
    if missing and not allow_partial:
        raise ValueError('Wait for all downloads to finish. Missing: ' + ', '.join(missing))
    if not available:
        raise ValueError(f'No downloaded datasets found in {source}')
    if output.exists():
        raise ValueError(f'{output} already exists. Use --data to reuse its data.yaml, or choose a new --prepared directory.')
    counts = dict(train=0, val=0, test=0)
    for alias in available:
        folder = source / alias
        metadata = yaml.safe_load((folder / 'data.yaml').read_text(encoding='utf-8'))
        names = metadata['names']
        names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
        pig_ids = {i for i, name in names.items() if str(name).strip().lower() in {'pig', 'pigs'}}
        if alias in {'pig-hjhho', 'pig-pig-pig', 'pig-a2jtl'} and names == {0: '0'}:
            pig_ids = {0}
        if not pig_ids:
            raise ValueError(f'{alias}: unrecognized pig class names {names}; inspect before training.')
        for split, candidates in [('train', ('train',)), ('val', ('valid', 'val')), ('test', ('test',))]:
            image_dir = next((folder / s / 'images' for s in candidates if (folder / s / 'images').is_dir()), None)
            if image_dir is None:
                if split != 'test':
                    raise ValueError(f'{alias}: missing {split} images')
                continue
            images = sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in EXTENSIONS)
            if not images and split != 'test':
                raise ValueError(f'{alias}: empty {split} split; download may still be running')
            for image in images:
                relative = image.relative_to(image_dir)
                label = image_dir.parent / 'labels' / relative.with_suffix('.txt')
                if not label.is_file():
                    raise ValueError(f'Missing annotation: {label}; download may still be running')
                lines = []
                for line in label.read_text(encoding='utf-8').splitlines():
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(f'{label}: expected detection boxes with 5 fields')
                    cls = int(fields[0])
                    coords = [float(v) for v in fields[1:]]
                    if cls not in names or any(not math.isfinite(v) or not 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                        raise ValueError(f'{label}: invalid box {line}')
                    if cls in pig_ids:
                        lines.append('0 ' + ' '.join(fields[1:]))
                destination = output / split / 'images' / alias / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(image, destination)
                except OSError:
                    shutil.copy2(image, destination)
                target_label = output / split / 'labels' / alias / relative.with_suffix('.txt')
                target_label.parent.mkdir(parents=True, exist_ok=True)
                target_label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
                counts[split] += 1
    if not counts['train'] or not counts['val']:
        raise ValueError('Training and validation images are required')
    config = {'path': output.as_posix(), 'train': 'train/images', 'val': 'val/images', 'names': {0: 'pig'}}
    if counts['test']:
        config['test'] = 'test/images'
    data = output / 'data.yaml'
    data.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(f'Prepared {len(available)} datasets: {counts}. Configuration: {data}')
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'datasets' / 'combined-pigs')
    parser.add_argument('--data', type=Path, help='Reuse an already prepared data.yaml')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--model', default='yolo26l.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=float, default=-1, help='-1: automatic GPU batch size; positive integer: fixed batch')
    parser.add_argument('--device', default=None, help='e.g. 0, 0,1, or cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--name', default='yolo26l-pigs')
    parser.add_argument('--resume', type=Path, help='Resume from a last.pt checkpoint')
    args = parser.parse_args()
    if args.resume:
        from ultralytics import YOLO
        YOLO(str(args.resume.resolve())).train(resume=True)
        return
    data = args.data.resolve() if args.data else prepare(args.datasets.resolve(), args.prepared.resolve(), args.allow_partial)
    if args.prepare_only:
        return
    from ultralytics import YOLO
    batch = int(args.batch) if args.batch.is_integer() else args.batch
    YOLO(args.model).train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                           batch=batch, device=args.device, workers=args.workers,
                           patience=args.patience, seed=42, project=str(ROOT / 'runs' / 'detect'),
                           name=args.name, cache=False)


if __name__ == '__main__':
    main()
