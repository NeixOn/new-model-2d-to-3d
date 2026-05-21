"""
Stage 3 inference for the Hunyuan-like image -> latent shape -> mesh pipeline.

Loads:
  Stage 1 ShapeVAE checkpoint
  Stage 2 ImageConditionedShapeFlowDiT checkpoint

Input:
  One rendered/RGB image.

Output:
  prediction.obj
  prediction_occupancy.npy
  prediction_preview.png

Kaggle example:
  %cd /kaggle/working/new-model-2d-to-3d/hunyuan3d
  %run infer_stage3_image_to_3d.py \
    --image /kaggle/working/shapenet_render_stage2/.../rendering/00.png \
    --stage1 /kaggle/working/shape_vae_stage1_results/shape_vae_stage1_best.pt \
    --stage2 /kaggle/working/shape_flow_stage2_results/shape_flow_stage2_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hunyuan_like_shape_flow_architecture import (
    ImageConditionedShapeFlowDiT,
    ShapeFlowConfig,
    ShapeVAE,
    euler_sample_flow,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--stage1", type=Path, default=Path("/kaggle/working/shape_vae_stage1_results/shape_vae_stage1_best.pt"))
    p.add_argument("--stage2", type=Path, default=Path("/kaggle/working/shape_flow_stage2_results/shape_flow_stage2_best.pt"))
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/hunyuan_like_stage3_prediction"))

    # Must match Stage 1/2 training config.
    p.add_argument("--latent-tokens", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--flow-heads", type=int, default=8)
    p.add_argument("--decoder-hidden", type=int, default=384)
    p.add_argument("--decoder-layers", type=int, default=5)
    p.add_argument("--fourier-bands", type=int, default=8)

    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--image-dim", type=int, default=192)
    p.add_argument("--image-heads", type=int, default=4)
    p.add_argument("--image-depth", type=int, default=2)
    p.add_argument("--flow-layers", type=int, default=4)
    p.add_argument("--flow-mlp-ratio", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)

    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


def choose_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def make_config(args: argparse.Namespace) -> ShapeFlowConfig:
    return ShapeFlowConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        image_dim=args.image_dim,
        image_depth=args.image_depth,
        image_heads=args.image_heads,
        latent_tokens=args.latent_tokens,
        latent_dim=args.latent_dim,
        flow_heads=args.flow_heads,
        vae_decoder_hidden=args.decoder_hidden,
        vae_decoder_layers=args.decoder_layers,
        fourier_bands=args.fourier_bands,
        flow_layers=args.flow_layers,
        flow_mlp_ratio=args.flow_mlp_ratio,
        dropout=args.dropout,
    )


def load_image(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (2, 0, 1))[None]
    return torch.from_numpy(arr).to(device)


def make_grid(size: int, device: torch.device) -> torch.Tensor:
    lin = torch.linspace(-1.0, 1.0, size, device=device)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing="ij")
    return torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)


@torch.no_grad()
def decode_grid(
    vae: ShapeVAE,
    latent: torch.Tensor,
    grid_size: int,
    chunk: int = 8192,
) -> np.ndarray:
    points = make_grid(grid_size, latent.device)
    preds = []
    for start in range(0, points.shape[1], chunk):
        sdf = vae.decode(latent, points[:, start : start + chunk])
        logits = -sdf
        probs = torch.sigmoid(logits)
        preds.append(probs.detach().cpu())
    occ = torch.cat(preds, dim=1)[0, :, 0].numpy()
    return occ.reshape(grid_size, grid_size, grid_size)


def save_obj_from_occupancy(occupancy: np.ndarray, path: Path, threshold: float) -> bool:
    try:
        from skimage import measure
    except Exception as exc:
        print(f"Cannot import skimage.measure for marching cubes: {exc}")
        print("Falling back to coarse voxel OBJ export.")
        return save_voxel_obj_from_occupancy(occupancy, path, threshold)

    if occupancy.max() < threshold or occupancy.min() > threshold:
        print(
            f"Cannot extract mesh: occupancy range [{occupancy.min():.4f}, {occupancy.max():.4f}] "
            f"does not cross threshold={threshold}"
        )
        return False

    verts, faces, _normals, _values = measure.marching_cubes(occupancy, level=threshold)
    scale = 2.0 / max(occupancy.shape[0] - 1, 1)
    verts = verts * scale - 1.0

    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            a, b, c = face + 1
            f.write(f"f {a} {b} {c}\n")
    print(f"Saved mesh: {path} ({len(verts)} verts, {len(faces)} faces)")
    return True


def save_voxel_obj_from_occupancy(
    occupancy: np.ndarray,
    path: Path,
    threshold: float,
    max_voxels: int = 4000,
) -> bool:
    filled = np.argwhere(occupancy >= threshold)
    if filled.size == 0:
        print(f"Cannot extract voxel OBJ: no cells above threshold={threshold}")
        return False

    if len(filled) > max_voxels:
        values = occupancy[filled[:, 0], filled[:, 1], filled[:, 2]]
        keep = np.argsort(values)[-max_voxels:]
        filled = filled[keep]

    n = occupancy.shape[0]
    step = 2.0 / max(n - 1, 1)
    half = step * 0.48

    cube_offsets = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float32,
    )
    cube_faces = [
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (4, 8, 5, 1),
    ]

    with open(path, "w", encoding="utf-8") as f:
        vertex_base = 0
        for cell in filled:
            center = cell.astype(np.float32) * step - 1.0
            verts = center[None, :] + cube_offsets
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in cube_faces:
                a, b, c, d = [vertex_base + idx for idx in face]
                f.write(f"f {a} {b} {c} {d}\n")
            vertex_base += 8

    print(f"Saved coarse voxel mesh: {path} ({len(filled)} cubes)")
    return True


def save_preview(occupancy: np.ndarray, path: Path, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vox = (occupancy >= threshold).astype(np.float32)
    proj_xy = vox.max(axis=0)
    proj_xz = vox.max(axis=1)
    proj_yz = vox.max(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, arr, title in zip(axes, [proj_xy, proj_xz, proj_yz], ["xy", "xz", "yz"]):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved preview: {path}")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = choose_device(args.force_cpu)
    cfg = make_config(args)

    print(f"Device: {device}")
    print(f"Image: {args.image}")
    print(f"Stage 1: {args.stage1}")
    print(f"Stage 2: {args.stage2}")
    print(f"Output: {args.out}")
    print(f"Sampling steps: {args.steps}, grid: {args.grid}, threshold: {args.threshold}")

    vae = ShapeVAE(cfg)
    vae.load_state_dict(torch.load(args.stage1, map_location="cpu"), strict=True)
    vae.to(device).eval()

    flow = ImageConditionedShapeFlowDiT(cfg)
    flow.load_state_dict(torch.load(args.stage2, map_location="cpu"), strict=True)
    flow.to(device).eval()

    image = load_image(args.image, args.image_size, device)

    print("Sampling latent with rectified flow...")
    latent = euler_sample_flow(flow, image, steps=args.steps)

    print("Decoding occupancy grid...")
    occupancy = decode_grid(vae, latent, args.grid)
    np.save(args.out / "prediction_occupancy.npy", occupancy)
    print(
        f"Occupancy stats: min={occupancy.min():.4f}, max={occupancy.max():.4f}, "
        f"mean={occupancy.mean():.4f}"
    )

    save_preview(occupancy, args.out / "prediction_preview.png", args.threshold)
    ok = save_obj_from_occupancy(occupancy, args.out / "prediction.obj", args.threshold)
    if not ok:
        print("Try a lower threshold, for example --threshold 0.35 or --threshold 0.25")

    print("Done.")


if __name__ == "__main__":
    main()
