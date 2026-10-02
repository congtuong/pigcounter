"""Write the root multi-dataset YAML without changing source images or labels."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
# Exclude pig-sei9k until its noisy annotations are reviewed/pseudo-labeled.
ALIASES = ('pig-hjhho', 'pig-detection', 'pig-pig-pig',
           'pig-tilpu', 'pig-whv6e', 'pig-vvvue', 'pig-mkd4a',
           'pig-a2jtl', 'pig-detectv2', 'pig-4ri1b')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def prepare(source: Path, allow_partial: bool = False) -> Path:
    source = source.resolve()
    available = [alias for alias in ALIASES if (source / alias / 'data.yaml').is_file()]
    missing = sorted(set(ALIASES) - set(available))
    if missing and not allow_partial:
        raise ValueError('Missing datasets: ' + ', '.join(missing))
    splits = {'train': [], 'val': [], 'test': []}
    counts = dict(train=0, val=0, test=0)
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
            count = sum(1 for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if count:
                splits[split].append(image_dir.as_posix())
                counts[split] += count
    if not counts['train'] or not counts['val']:
        raise ValueError('Training and validation images are required overall')
    config = {'path': ROOT.as_posix(), 'train': splits['train'], 'val': splits['val'], 'names': {0: 'pig', 1: 'person'}}
    if counts['test']:
        config['test'] = splits['test']
    data = ROOT / 'data.yaml'
    data.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(f'Wrote {data}: {len(available)} datasets, {counts}. Source images and labels unchanged.')
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    try:
        prepare(args.datasets, args.allow_partial)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
