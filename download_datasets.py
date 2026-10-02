"""
Download Roboflow pig datasets for YOLO training.

Usage:
    export ROBOFLOW_API_KEY="<your_key>"
    uv run download_datasets.py            # download all
    uv run download_datasets.py --dry-run  # print plan only
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from roboflow import Roboflow


# (workspace, project_slug, version, project_alias)
# version=None lets Roboflow pick the latest.
# Datasets that appear twice in the source list are deduplicated by (workspace, slug).
DATASETS: list[tuple[str, str, int | None, str]] = [
    ("yunzhao-jiang-o9lvh", "pig-sei9k",  None, "pig-sei9k"),
    ("pigdataset-1eimk",     "pig-hjhho",  None, "pig-hjhho"),
    ("opsis",               "pig-detection-0xmro", None, "pig-detection"),
    ("loihp",               "pig-pig-pig", None, "pig-pig-pig"),
    ("li-yao-tseng",        "pig-tilpu",   None, "pig-tilpu"),
    ("track-pig",           "pig-whv6e",   None, "pig-whv6e"),
    ("pig-xcoka",           "pig-vvvue",   None, "pig-vvvue"),
    ("cong-tuong",   "pig-mkd4a-wyabd",   None, "pig-mkd4a"),
    ("experimentdata",      "pig-a2jtl",   None, "pig-a2jtl"),
    ("zhuyishuailivecom",   "pig-detectv2-lvr9i", None, "pig-detectv2"),
    ("data-qtyxx",           "pig-4ri1b", None, "pig-4ri1b"),
]

OUTPUT_ROOT = Path("datasets")
DEFAULT_FORMAT = "yolov8"  # YOLO txt labels — compatible with YOLOv5/v8/v11


@dataclass
class DownloadResult:
    alias: str
    workspace: str
    project: str
    version: int | None
    location: Path
    image_count: int | None = None


def _unique_datasets(
    datasets: list[tuple[str, str, int | None, str]],
) -> list[tuple[str, str, int | None, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, int | None, str]] = []
    for ws, slug, ver, alias in datasets:
        key = (ws, slug)
        if key in seen:
            print(f"[skip-duplicate] {ws}/{slug} already queued")
            continue
        seen.add(key)
        out.append((ws, slug, ver, alias))
    return out


def download_all(
    api_key: str,
    datasets: list[tuple[str, str, int | None, str]],
    fmt: str,
    output_root: Path,
    dry_run: bool = False,
) -> list[DownloadResult]:
    rf = Roboflow(api_key=api_key)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []

    for workspace, project_slug, version, alias in datasets:
        target = output_root / alias
        print(f"\n=== {workspace}/{project_slug} -> {target} (format={fmt}) ===")
        if dry_run:
            results.append(DownloadResult(alias, workspace, project_slug, version, target))
            continue

        try:
            proj = rf.workspace(workspace).project(project_slug)
            if version is None:
                versions = proj.versions()  # list[Version]
                if not versions:
                    raise RuntimeError("no versions available")
                version = versions[-1].version  # latest
            ds = proj.version(version).download(fmt, location=str(target))
        except Exception as exc:  # noqa: BLE001 — surface API errors verbatim
            print(f"[error] {workspace}/{project_slug}: {exc}", file=sys.stderr)
            continue

        results.append(
            DownloadResult(
                alias=alias,
                workspace=workspace,
                project=project_slug,
                version=version,
                location=Path(ds.location),
            )
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default=DEFAULT_FORMAT,
                        help=f"Roboflow dataset format (default: {DEFAULT_FORMAT})")
    parser.add_argument("--output", default=str(OUTPUT_ROOT),
                        help=f"Output root directory (default: {OUTPUT_ROOT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without downloading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key and not args.dry_run:
        print("ROBOFLOW_API_KEY env var not set. Export it then retry.", file=sys.stderr)
        return 2

    datasets = _unique_datasets(DATASETS)
    print(f"Queued {len(datasets)} unique dataset(s).")
    results = download_all(
        api_key=api_key or "",
        datasets=datasets,
        fmt=args.format,
        output_root=Path(args.output),
        dry_run=args.dry_run,
    )

    print("\n--- summary ---")
    for r in results:
        print(f"  {r.alias:18s} {r.workspace}/{r.project} v{r.version} -> {r.location}")
    print(f"\nDone. {len(results)} dataset(s) processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
