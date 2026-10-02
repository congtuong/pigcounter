"""Train a YOLO26l pig detector using a prepared dataset YAML."""
from __future__ import annotations

import argparse
from pathlib import Path
from training_augmentation import augmentation_settings

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data.yaml', help='Prepared dataset YAML')
    parser.add_argument('--model', default='yolo26l.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=float, default=16, help='-1: automatic GPU batch size; positive integer: fixed batch')
    parser.add_argument('--device', default=None, help='e.g. 0, 0,1, or cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--name', default='yolo26l-pigs')
    parser.add_argument('--augmentation', choices=('outdoor', 'standard'), default='outdoor',
                        help='outdoor: camera shake, blur, weather and exposure; standard: YOLO defaults')
    parser.add_argument('--resume', type=Path, help='Resume from a last.pt checkpoint')
    args = parser.parse_args()
    if args.resume:
        from ultralytics import YOLO
        YOLO(str(args.resume.resolve())).train(resume=True)
        return
    data = args.data.resolve()
    if not data.is_file():
        parser.error(f'Dataset YAML does not exist: {data}. Run uv run prepare_data.py first.')
    from ultralytics import YOLO
    batch = int(args.batch) if args.batch.is_integer() else args.batch
    settings = augmentation_settings(args.augmentation)
    print(f'Training augmentation profile: {args.augmentation}')
    YOLO(args.model).train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                           batch=batch, device=args.device, workers=args.workers,
                           patience=args.patience, seed=42, project=str(ROOT / 'runs' / 'detect'),
                           name=args.name, cache=False, **settings)


if __name__ == '__main__':
    main()
