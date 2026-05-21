"""
Stage 2 training for the Hunyuan-like image-conditioned Shape Flow DiT.

Goal:
  Freeze a trained Stage 1 ShapeVAE encoder, encode each ground-truth 3D shape
  into latent tokens, then train a rectified-flow transformer to generate those
  latent tokens from one rendered image.

Expected Kaggle inputs:
  /kaggle/input/.../ShapeNetRendering.tgz
  /kaggle/working/shapenet_r2n2_quick/**/*.binvox
  /kaggle/working/shape_vae_stage1_results/shape_vae_stage1_best.pt

Kaggle smoke run:
  %cd /kaggle/working/new-model-2d-to-3d/hunyuan3d
  %env STAGE2_SHAPE_ROOT=/kaggle/working/shapenet_r2n2_quick
  %env STAGE1_CKPT=/kaggle/working/shape_vae_stage1_results/shape_vae_stage1_best.pt
  %env EPOCHS=2
  %env BATCH_SIZE=2
  %env SHAPE_POINTS=512
  %env LATENT_TOKENS=256
  %env LATENT_DIM=384
  %env FLOW_HEADS=8
  %env IMAGE_SIZE=112
  %env IMAGE_DIM=192
  %env IMAGE_HEADS=4
  %env IMAGE_DEPTH=2
  %env FLOW_LAYERS=4
  %run train_stage2_shape_flow.py
"""

from __future__ import annotations

import math
import os
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from hunyuan_like_shape_flow_architecture import (
    ImageConditionedShapeFlowDiT,
    ShapeFlowConfig,
    ShapeVAE,
    rectified_flow_loss,
)


SEED = int(os.environ.get("SEED", "42"))
SHAPE_ROOT = Path(os.environ.get("STAGE2_SHAPE_ROOT", "/kaggle/working/shapenet_r2n2_quick"))
RENDER_ROOT = Path(os.environ.get("STAGE2_RENDER_ROOT", "/kaggle/working/shapenet_render_stage2"))
RESULTS_DIR = Path(os.environ.get("STAGE2_RESULTS_DIR", "/kaggle/working/shape_flow_stage2_results"))
STAGE1_CKPT = Path(os.environ.get("STAGE1_CKPT", "/kaggle/working/shape_vae_stage1_results/shape_vae_stage1_best.pt"))

EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
LR = float(os.environ.get("LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.10"))
SANITY_ONLY = os.environ.get("SANITY_ONLY", "0") == "1"
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
VERBOSE_BATCH = os.environ.get("VERBOSE_BATCH", "1") == "1"

MAX_MODELS_PER_CLASS = int(os.environ.get("MAX_MODELS_PER_CLASS", "1200"))
VIEWS_PER_MODEL = int(os.environ.get("VIEWS_PER_MODEL", "4"))
SHAPENET_CLASSES = [
    item.strip()
    for item in os.environ.get("SHAPENET_CLASSES", "03001627").split(",")
    if item.strip()
]

SHAPE_POINTS = int(os.environ.get("SHAPE_POINTS", "1024"))

LATENT_TOKENS = int(os.environ.get("LATENT_TOKENS", "256"))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "384"))
FLOW_HEADS = int(os.environ.get("FLOW_HEADS", "8"))
DECODER_HIDDEN = int(os.environ.get("DECODER_HIDDEN", "384"))
DECODER_LAYERS = int(os.environ.get("DECODER_LAYERS", "5"))
FOURIER_BANDS = int(os.environ.get("FOURIER_BANDS", "8"))

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "112"))
IMAGE_DIM = int(os.environ.get("IMAGE_DIM", "192"))
IMAGE_HEADS = int(os.environ.get("IMAGE_HEADS", "4"))
IMAGE_DEPTH = int(os.environ.get("IMAGE_DEPTH", "2"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "14"))
FLOW_LAYERS = int(os.environ.get("FLOW_LAYERS", "4"))
FLOW_MLP_RATIO = int(os.environ.get("FLOW_MLP_RATIO", "4"))
DROPOUT = float(os.environ.get("DROPOUT", "0.05"))

POSSIBLE_SHAPENET_DIRS = [
    Path("/kaggle/input/shapenet-3dr2n2"),
    Path("/kaggle/input/datasets/sirish001/shapenet-3dr2n2"),
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if FORCE_CPU:
        return torch.device("cpu")
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def optimizer_step(optimizer: torch.optim.Optimizer) -> None:
    try:
        import torch_xla.core.xla_model as xm

        xm.optimizer_step(optimizer, barrier=True)
    except Exception:
        optimizer.step()


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch_xla.core.xla_model as xm

        xm.save(model.state_dict(), path)
    except Exception:
        torch.save(model.state_dict(), path)


def find_shapenet_render_archive() -> Path:
    for dataset_dir in POSSIBLE_SHAPENET_DIRS:
        archive = dataset_dir / "ShapeNetRendering.tgz"
        if archive.exists():
            return archive

    input_root = Path("/kaggle/input")
    if input_root.exists():
        matches = list(input_root.rglob("ShapeNetRendering.tgz"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Cannot find ShapeNetRendering.tgz under /kaggle/input. "
        "Add https://www.kaggle.com/datasets/sirish001/shapenet-3dr2n2 to the notebook."
    )


def find_category_and_model(path: str, category_ids: set[str]) -> tuple[str | None, str | None]:
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        if part in category_ids and idx + 1 < len(parts):
            return part, parts[idx + 1]
    return None, None


def safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, dst: Path) -> None:
    target = (dst / member.name).resolve()
    dst_resolved = dst.resolve()
    if not str(target).startswith(str(dst_resolved)):
        raise RuntimeError(f"Unsafe archive member path: {member.name}")
    tar.extract(member, dst)


def read_binvox(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        line = f.readline().strip()
        if not line.startswith(b"#binvox"):
            raise ValueError(f"Not a binvox file: {path}")

        dims = None
        while True:
            line = f.readline().strip()
            if line.startswith(b"dim"):
                dims = tuple(map(int, line.split()[1:4]))
            elif line.startswith(b"data"):
                break

        if dims is None:
            raise ValueError(f"Missing dimensions in binvox file: {path}")

        raw = np.frombuffer(f.read(), dtype=np.uint8)
        values = raw[0::2]
        counts = raw[1::2]
        voxels = np.repeat(values, counts).astype(np.float32)
        expected = dims[0] * dims[1] * dims[2]
        if voxels.size != expected:
            raise ValueError(f"Bad binvox RLE size in {path}: got {voxels.size}, expected {expected}")
        return voxels.reshape(dims)


def voxel_grid_coords(size: int) -> np.ndarray:
    coords = np.stack(
        np.meshgrid(
            np.linspace(-1.0, 1.0, size, dtype=np.float32),
            np.linspace(-1.0, 1.0, size, dtype=np.float32),
            np.linspace(-1.0, 1.0, size, dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    )
    return coords.reshape(-1, 3)


def binvox_to_shape_points(path: Path) -> np.ndarray:
    vox = read_binvox(path).astype(np.float32)
    coords = voxel_grid_coords(vox.shape[0])
    occ = vox.reshape(-1, 1)
    pos = np.where(occ[:, 0] > 0.5)[0]
    neg = np.where(occ[:, 0] <= 0.5)[0]
    n_pos = min(len(pos), SHAPE_POINTS // 2)
    n_neg = SHAPE_POINTS - n_pos
    if n_pos > 0:
        idx = np.concatenate(
            [
                np.random.choice(pos, n_pos, replace=len(pos) < n_pos),
                np.random.choice(neg, n_neg, replace=len(neg) < n_neg),
            ]
        )
    else:
        idx = np.random.choice(len(coords), SHAPE_POINTS, replace=len(coords) < SHAPE_POINTS)
    np.random.shuffle(idx)
    return np.concatenate([coords[idx], occ[idx]], axis=-1).astype(np.float32)


def discover_binvox_by_key(root: Path) -> dict[tuple[str, str], Path]:
    category_ids = set(SHAPENET_CLASSES)
    out = {}
    for path in sorted(root.rglob("*.binvox")):
        cat_id, model_id = find_category_and_model(str(path), category_ids)
        if cat_id is not None and model_id is not None:
            out[(cat_id, model_id)] = path
    if not out:
        raise FileNotFoundError(f"No .binvox files found under {root}. Run Stage 1 extraction first.")
    return out


def prepare_render_subset(binvox_by_key: dict[tuple[str, str], Path]) -> None:
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    existing = list(RENDER_ROOT.rglob("*.png"))
    if existing:
        return

    archive = find_shapenet_render_archive()
    wanted = set(binvox_by_key.keys())
    extracted_views: dict[tuple[str, str], int] = {}

    print(f"No rendered images found in {RENDER_ROOT}", flush=True)
    print(f"Extracting render subset from: {archive}", flush=True)
    print(f"Views per model: {VIEWS_PER_MODEL}", flush=True)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".png"):
                continue
            cat_id, model_id = find_category_and_model(member.name, set(SHAPENET_CLASSES))
            key = (cat_id, model_id)
            if key not in wanted:
                continue
            if extracted_views.get(key, 0) >= VIEWS_PER_MODEL:
                continue
            safe_extract_member(tar, member, RENDER_ROOT)
            extracted_views[key] = extracted_views.get(key, 0) + 1

            total = sum(extracted_views.values())
            if total % 500 == 0:
                print(f"Extracted {total} render images...", flush=True)

            if len(extracted_views) == len(wanted) and all(v >= VIEWS_PER_MODEL for v in extracted_views.values()):
                break

    total = sum(extracted_views.values())
    if total == 0:
        raise RuntimeError("ShapeNetRendering.tgz was found, but no matching PNG renders were extracted.")
    print(f"Prepared render subset: {total} images for {len(extracted_views)} models", flush=True)


def discover_image_pairs(binvox_by_key: dict[tuple[str, str], Path]) -> list[tuple[Path, Path, str]]:
    category_ids = set(SHAPENET_CLASSES)
    images_by_key: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(RENDER_ROOT.rglob("*.png")):
        cat_id, model_id = find_category_and_model(str(path), category_ids)
        if cat_id is not None and model_id is not None:
            images_by_key.setdefault((cat_id, model_id), []).append(path)

    samples = []
    for key, vox_path in binvox_by_key.items():
        for image_path in images_by_key.get(key, [])[:VIEWS_PER_MODEL]:
            samples.append((image_path, vox_path, f"{key[0]}/{key[1]}"))

    if not samples:
        raise RuntimeError(f"No image/binvox pairs found under {RENDER_ROOT}")
    random.shuffle(samples)
    return samples


def split_by_model(samples: list[tuple[Path, Path, str]]) -> tuple[list[tuple[Path, Path, str]], list[tuple[Path, Path, str]]]:
    by_model: dict[str, list[tuple[Path, Path, str]]] = {}
    for item in samples:
        by_model.setdefault(item[2], []).append(item)
    keys = list(by_model)
    random.shuffle(keys)
    val_count = max(1, int(len(keys) * VAL_FRACTION))
    val_keys = set(keys[:val_count])
    train, val = [], []
    for key in keys:
        (val if key in val_keys else train).extend(by_model[key])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


class Stage2ShapeImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path, vox_path, _ = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        image_np = np.transpose(image_np, (2, 0, 1))
        shape_points = binvox_to_shape_points(vox_path)
        return {
            "image": torch.from_numpy(image_np.astype(np.float32)),
            "shape_points": torch.from_numpy(shape_points),
        }


def make_config() -> ShapeFlowConfig:
    if LATENT_DIM % FLOW_HEADS != 0:
        raise ValueError("LATENT_DIM must be divisible by FLOW_HEADS")
    if IMAGE_DIM % IMAGE_HEADS != 0:
        raise ValueError("IMAGE_DIM must be divisible by IMAGE_HEADS")
    return ShapeFlowConfig(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        image_dim=IMAGE_DIM,
        image_depth=IMAGE_DEPTH,
        image_heads=IMAGE_HEADS,
        latent_tokens=LATENT_TOKENS,
        latent_dim=LATENT_DIM,
        flow_heads=FLOW_HEADS,
        vae_decoder_hidden=DECODER_HIDDEN,
        vae_decoder_layers=DECODER_LAYERS,
        fourier_bands=FOURIER_BANDS,
        flow_layers=FLOW_LAYERS,
        flow_mlp_ratio=FLOW_MLP_RATIO,
        dropout=DROPOUT,
    )


def load_stage1_vae(cfg: ShapeFlowConfig, device: torch.device) -> ShapeVAE:
    if not STAGE1_CKPT.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {STAGE1_CKPT}")
    vae = ShapeVAE(cfg)
    state = torch.load(STAGE1_CKPT, map_location="cpu")
    vae.load_state_dict(state, strict=True)
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def run_epoch(
    flow: ImageConditionedShapeFlowDiT,
    vae: ShapeVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    flow.train(train)
    total_loss = 0.0
    steps = 0

    for step, batch in enumerate(loader, start=1):
        if VERBOSE_BATCH and step == 1:
            print(f"{'Train' if train else 'Val'}: first batch loaded, moving to {device}...", flush=True)
        image = batch["image"].to(device)
        shape_points = batch["shape_points"].to(device)

        if VERBOSE_BATCH and step == 1:
            print(f"{'Train' if train else 'Val'}: encoding target shape latent...", flush=True)
        with torch.no_grad():
            z_target, _ = vae.encode(shape_points)
            z_target = z_target.detach()

        with torch.set_grad_enabled(train):
            if VERBOSE_BATCH and step == 1:
                print(f"{'Train' if train else 'Val'}: running flow forward/loss...", flush=True)
            loss = rectified_flow_loss(flow, image, z_target)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                if VERBOSE_BATCH and step == 1:
                    print(f"{'Train' if train else 'Val'}: backward done, optimizer step...", flush=True)
                optimizer_step(optimizer)
                if VERBOSE_BATCH and step == 1:
                    print(f"{'Train' if train else 'Val'}: optimizer step done.", flush=True)

        total_loss += float(loss.detach().cpu())
        steps += 1

    return {"loss": total_loss / max(steps, 1)}


def main() -> None:
    seed_everything(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    binvox_by_key = discover_binvox_by_key(SHAPE_ROOT)
    # Keep Stage 2 paired with the same max model count used in Stage 1.
    binvox_by_key = dict(list(sorted(binvox_by_key.items()))[:MAX_MODELS_PER_CLASS])
    prepare_render_subset(binvox_by_key)
    samples = discover_image_pairs(binvox_by_key)
    train_samples, val_samples = split_by_model(samples)

    train_loader = DataLoader(
        Stage2ShapeImageDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        Stage2ShapeImageDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    device = choose_device()
    cfg = make_config()
    vae = load_stage1_vae(cfg, device)
    flow = ImageConditionedShapeFlowDiT(cfg).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"Shape root: {SHAPE_ROOT}", flush=True)
    print(f"Render root: {RENDER_ROOT}", flush=True)
    print(f"Stage 1 checkpoint: {STAGE1_CKPT}", flush=True)
    print(f"Results: {RESULTS_DIR}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Samples: total={len(samples)}, train={len(train_samples)}, val={len(val_samples)}", flush=True)
    print(
        "Config: "
        f"image={IMAGE_SIZE}, image_dim={IMAGE_DIM}, image_depth={IMAGE_DEPTH}, "
        f"latent_tokens={LATENT_TOKENS}, latent_dim={LATENT_DIM}, "
        f"flow_layers={FLOW_LAYERS}, batch={BATCH_SIZE}, shape_points={SHAPE_POINTS}",
        flush=True,
    )

    print("Loading one sanity batch before training...", flush=True)
    sanity_batch = next(iter(train_loader))
    print(
        "Sanity batch shapes: "
        f"image={tuple(sanity_batch['image'].shape)}, "
        f"shape_points={tuple(sanity_batch['shape_points'].shape)}",
        flush=True,
    )
    if SANITY_ONLY:
        print("SANITY_ONLY=1, stopping before training loop.", flush=True)
        return

    history_path = RESULTS_DIR / "stage2_history.csv"
    history_path.write_text("epoch,train_loss,val_loss\n", encoding="utf-8")
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(flow, vae, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(flow, vae, val_loader, None, device, train=False)

        with open(history_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_metrics['loss']},{val_metrics['loss']}\n")

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_flow_loss={train_metrics['loss']:.4f} val_flow_loss={val_metrics['loss']:.4f}",
            flush=True,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(flow, RESULTS_DIR / "shape_flow_stage2_best.pt")
        save_checkpoint(flow, RESULTS_DIR / "shape_flow_stage2_last.pt")

    print(f"Done. Best val flow loss: {best_val:.4f}", flush=True)
    print(f"Saved: {RESULTS_DIR / 'shape_flow_stage2_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
