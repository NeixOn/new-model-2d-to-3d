"""
Download an Objaverse chair subset for Stage 1/2 training.

Run in a Kaggle notebook with Internet=On:
  %cd /kaggle/working/new-model-2d-to-3d/hunyuan3d
  !pip install -q objaverse
  %run prepare_objaverse_chairs.py --limit 500 --out /kaggle/working/objaverse_chairs

Outputs:
  /kaggle/working/objaverse_chairs/objects/**/*.glb
  /kaggle/working/objaverse_chairs/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/objaverse_chairs"))
    p.add_argument("--category", type=str, default="chair")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--download-processes", type=int, default=8)
    p.add_argument("--min-mb", type=float, default=0.02)
    p.add_argument("--max-mb", type=float, default=80.0)
    p.add_argument("--copy-files", action="store_true", help="Copy GLBs into out/objects instead of referencing cache paths.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    objects_dir = args.out / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    try:
        import objaverse
    except ImportError as exc:
        raise SystemExit("Install first: pip install objaverse") from exc

    print("Loading LVIS annotations...")
    annotations = objaverse.load_lvis_annotations()
    if args.category not in annotations:
        close = [k for k in annotations if args.category.lower() in k.lower()]
        raise KeyError(f"Category {args.category!r} not found. Similar keys: {close[:20]}")

    all_uids = sorted(annotations[args.category])
    selected = all_uids[args.start : args.start + args.limit]
    print(f"Category: {args.category}")
    print(f"Total category UIDs: {len(all_uids)}")
    print(f"Selected UIDs: {len(selected)} from offset {args.start}")

    print("Downloading objects...")
    uid_to_path = objaverse.load_objects(
        uids=selected,
        download_processes=args.download_processes,
    )

    manifest_path = args.out / "manifest.csv"
    kept = 0
    skipped = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uid", "category", "source_path", "local_path", "size_mb", "status"],
        )
        writer.writeheader()

        for uid in selected:
            src = Path(uid_to_path.get(uid, ""))
            status = "ok"
            local_path = src
            size_mb = 0.0

            if not src.exists():
                status = "missing"
                skipped += 1
            else:
                size_mb = src.stat().st_size / (1024 * 1024)
                if size_mb < args.min_mb:
                    status = "too_small"
                    skipped += 1
                elif size_mb > args.max_mb:
                    status = "too_large"
                    skipped += 1
                else:
                    kept += 1
                    if args.copy_files:
                        local_path = objects_dir / f"{uid}.glb"
                        if not local_path.exists():
                            shutil.copy2(src, local_path)

            writer.writerow(
                {
                    "uid": uid,
                    "category": args.category,
                    "source_path": str(src),
                    "local_path": str(local_path),
                    "size_mb": f"{size_mb:.4f}",
                    "status": status,
                }
            )

    print(f"Manifest: {manifest_path}")
    print(f"Kept: {kept}")
    print(f"Skipped: {skipped}")
    print("Next: run mesh_to_points_npz.py using this manifest.")


if __name__ == "__main__":
    main()
