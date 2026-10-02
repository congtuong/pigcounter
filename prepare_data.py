"""Pool all source images and write only a reproducible 90:10 train/val split."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
# Exclude pig-sei9k until its noisy annotations are reviewed/pseudo-labeled.
ALIASES = ('pig-hjhho', 'pig-detection', 'pig-pig-pig',
           'pig-tilpu', 'pig-whv6e', 'pig-vvvue', 'pig-mkd4a',
           'pig-a2jtl', 'pig-detectv2', 'pig-4ri1b')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def split_images(images: list[Path], seed: int) -> tuple[list[Path], list[Path]]:
    images = sorted(set(images))
    if len(images) < 2:
        raise ValueError('At least two training/validation images are required')
    random.Random(seed).shuffle(images)
    val_count = max(1, min(len(images) - 1, round(len(images) * 0.1)))
    return images[val_count:], images[:val_count]


def prepare(source: Path, allow_partial: bool = False, seed: int = 42) -> Path:
    source = source.resolve()
    available = [alias for alias in ALIASES if (source / alias / 'data.yaml').is_file()]
    missing = sorted(set(ALIASES) - set(available))
    if missing and not allow_partial:
        raise ValueError('Missing datasets: ' + ', '.join(missing))
    pool = []
    for alias in available:
        folder = source / alias
        metadata = yaml.safe_load((folder / 'data.yaml').read_text(encoding='utf-8'))
        names = metadata.get('names', {})
        names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
        if str(names.get(0, '')).lower() in {'person', 'people', 'human'}:
            print(f'WARNING: {alias}/data.yaml declares class 0 as person. Output uses your requested 0=pig, 1=person; source labels are untouched.')
        for split, candidates in [('train', ('train',)), ('val', ('valid', 'val')), ('test', ('test',))]:
            image_dir = next((folder / s / 'images' for s in candidates if (folder / s / 'images').is_dir()), None)
            if image_dir is None:
                continue
            images = sorted(p for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            pool.extend(images)
    train, val = split_images(pool, seed)
    manifests = source / 'splits'
    manifests.mkdir(parents=True, exist_ok=True)
    splits = {'train': train, 'val': val}
    config = {'path': ROOT.as_posix(), 'names': {0: 'pig', 1: 'person'}}
    for split, images in splits.items():
        if images:
            manifest = manifests / f'{split}.txt'
            manifest.write_text(''.join(p.as_posix() + '\n' for p in images), encoding='utf-8')
            config[split] = manifest.as_posix()
    data = ROOT / 'data.yaml'
    data.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    counts = {split: len(images) for split, images in splits.items()}
    print(f'Wrote {data}: {len(available)} datasets, {counts}, seed={seed}. Source images and labels unchanged.')
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='Reproducible random split seed')
    args = parser.parse_args()
    try:
        prepare(args.datasets, args.allow_partial, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
