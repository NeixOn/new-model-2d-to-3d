"""
Kaggle helper for running OpenLRM inference.

Run this in a Kaggle GPU notebook, not TPU:
  %run kaggle_openlrm_quickstart.py --image /kaggle/working/new-model-2d-to-3d/image/stul.jpg --model small

It clones OpenLRM if needed, installs requirements, runs inference, and prints
where generated meshes/videos were saved.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--workdir", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--model", choices=["small", "base", "large"], default="small")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-mesh", action="store_true")
    return p.parse_args()


def model_settings(name: str) -> tuple[str, str]:
    if name == "small":
        return "./configs/infer-s.yaml", "zxhezexin/openlrm-mix-small-1.1"
    if name == "base":
        return "./configs/infer-b.yaml", "zxhezexin/openlrm-mix-base-1.1"
    return "./configs/infer-l.yaml", "zxhezexin/openlrm-mix-large-1.1"


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    repo_dir = args.workdir / "OpenLRM"
    if not repo_dir.exists():
        run(["git", "clone", "https://github.com/3DTopia/OpenLRM.git", str(repo_dir)])

    if not args.skip_install:
        run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=repo_dir)

    infer_config, model_name = model_settings(args.model)
    env = os.environ.copy()
    env.update(
        {
            "EXPORT_VIDEO": "false" if args.no_video else "true",
            "EXPORT_MESH": "false" if args.no_mesh else "true",
            "INFER_CONFIG": infer_config,
            "MODEL_NAME": model_name,
            "IMAGE_INPUT": str(args.image),
        }
    )

    run(
        [
            "python",
            "-m",
            "openlrm.launch",
            "infer.lrm",
            "--infer",
            infer_config,
            f"model_name={model_name}",
            f"image_input={args.image}",
            f"export_video={env['EXPORT_VIDEO']}",
            f"export_mesh={env['EXPORT_MESH']}",
        ],
        cwd=repo_dir,
        env=env,
    )

    print("\nGenerated files:", flush=True)
    for suffix in ("*.obj", "*.ply", "*.glb", "*.mp4"):
        for path in sorted(repo_dir.rglob(suffix))[:50]:
            print(path)


if __name__ == "__main__":
    main()
