"""
Convert downloaded Objaverse meshes into points.npz files for Stage 1.

Input:
  manifest.csv from prepare_objaverse_chairs.py

Output:
  out/<uid>/points.npz

Each NPZ contains:
  points: [N, 3] float32 in [-1, 1]
  occupancy: [N, 1] float32
  shape_points: [M, 4] float32 = xyz + occupancy

Kaggle:
  %cd /kaggle/working/new-model-2d-to-3d/hunyuan3d
  !pip install -q trimesh rtree
  %run mesh_to_points_npz.py \
    --manifest /kaggle/working/objaverse_chairs/manifest.csv \
    --out /kaggle/working/objaverse_chair_points \
    --limit 500
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/objaverse_chair_points"))
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--points", type=int, default=65536)
    p.add_argument("--shape-points", type=int, default=8192)
    p.add_argument("--surface-ratio", type=float, default=0.55)
    p.add_argument("--surface-noise", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("status") == "ok" and Path(r.get("local_path", "")).exists()]
    return rows


def as_scene_mesh(obj) -> "trimesh.Trimesh":
    import trimesh

    if isinstance(obj, trimesh.Scene):
        meshes = []
        for geom in obj.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
                meshes.append(geom)
        if not meshes:
            raise ValueError("Scene contains no valid meshes")
        return trimesh.util.concatenate(meshes)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    raise TypeError(f"Unsupported trimesh load result: {type(obj)}")


def normalize_mesh(mesh):
    mesh = mesh.copy()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Empty mesh")
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) * 0.5
    scale = float((bounds[1] - bounds[0]).max())
    if scale <= 1e-8:
        raise ValueError("Degenerate mesh scale")
    mesh.vertices = (mesh.vertices - center) / scale * 1.8
    return mesh


def robust_contains(mesh, points: np.ndarray) -> np.ndarray:
    try:
        return mesh.contains(points)
    except Exception:
        # Fallback for non-watertight meshes: use signed distance when available.
        try:
            import trimesh

            signed = trimesh.proximity.signed_distance(mesh, points)
            return signed > 0
        except Exception:
            # Last-resort weak fallback: label points close to surface as occupied.
            _closest, dist, _tri = mesh.nearest.on_surface(points)
            return dist < 0.04


def sample_points(mesh, n_points: int, n_shape: int, surface_ratio: float, surface_noise: float):
    n_surface = int(n_points * surface_ratio)
    n_uniform = n_points - n_surface

    surface, _face_idx = mesh.sample(n_surface, return_index=True)
    surface = surface + np.random.normal(scale=surface_noise, size=surface.shape)
    uniform = np.random.uniform(-1.0, 1.0, size=(n_uniform, 3))
    points = np.concatenate([surface, uniform], axis=0).astype(np.float32)
    points = np.clip(points, -1.0, 1.0)

    occ = robust_contains(mesh, points).astype(np.float32)[:, None]

    n_shape_surface = min(n_shape // 2, n_surface)
    n_shape_uniform = n_shape - n_shape_surface
    shape_surface = surface[:n_shape_surface]
    shape_uniform = np.random.uniform(-1.0, 1.0, size=(n_shape_uniform, 3))
    shape_xyz = np.concatenate([shape_surface, shape_uniform], axis=0).astype(np.float32)
    shape_xyz = np.clip(shape_xyz, -1.0, 1.0)
    shape_occ = robust_contains(mesh, shape_xyz).astype(np.float32)[:, None]
    shape_points = np.concatenate([shape_xyz, shape_occ], axis=-1).astype(np.float32)

    return points, occ, shape_points


def convert_one(row: dict[str, str], out_root: Path, args: argparse.Namespace) -> tuple[bool, str]:
    import trimesh

    uid = row["uid"]
    mesh_path = Path(row["local_path"])
    out_dir = out_root / uid
    out_path = out_dir / "points.npz"
    if out_path.exists() and not args.overwrite:
        return True, "exists"

    try:
        loaded = trimesh.load(mesh_path, force="scene", process=False)
        mesh = normalize_mesh(as_scene_mesh(loaded))
        if len(mesh.faces) > 300_000:
            mesh = mesh.simplify_quadric_decimation(300_000)

        points, occupancy, shape_points = sample_points(
            mesh,
            n_points=args.points,
            n_shape=args.shape_points,
            surface_ratio=args.surface_ratio,
            surface_noise=args.surface_noise,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            points=points,
            occupancy=occupancy,
            shape_points=shape_points,
            uid=np.array(uid),
            mesh_path=np.array(str(mesh_path)),
        )
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest)
    rows = rows[args.start : args.start + args.limit]
    print(f"Converting {len(rows)} meshes from manifest: {args.manifest}")
    print(f"Output: {args.out}")
    print(f"points={args.points}, shape_points={args.shape_points}")

    log_path = args.out / "conversion_log.csv"
    ok_count = 0
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "ok", "message"])
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            ok, msg = convert_one(row, args.out, args)
            ok_count += int(ok)
            writer.writerow({"uid": row["uid"], "ok": ok, "message": msg})
            if idx % 25 == 0 or not ok:
                print(f"[{idx}/{len(rows)}] ok={ok_count} latest={row['uid']} status={msg}", flush=True)

    print(f"Done. Converted: {ok_count}/{len(rows)}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
