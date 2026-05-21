"""
Stage 1 training for the Hunyuan-like ShapeVAE.

Goal:
  Train a 3D shape autoencoder:
    shape samples -> latent shape tokens -> SDF/occupancy field

Preferred dataset format:
  dataset/
    train/object_id/points.npz
    val/object_id/points.npz

Accepted NPZ keys:
  points or query_points: float32 [N, 3], coordinates in [-1, 1]
  sdf, target_sdf, or query_sdf: float32 [N] or [N, 1]
  occupancy or occ: float32 [N] or [N, 1], optional fallback
  shape_points: float32 [M, 4+], optional encoder input

Fallback:
  If no points.npz files are found, the script can read .binvox files from a
  prepared ShapeNet/3D-R2N2 extraction and train an occupancy autoencoder.

Kaggle example:
  %env STAGE1_DATA_ROOT=/kaggle/working/shapenet_r2n2_quick
  %env FIELD_TYPE=occupancy
  %env EPOCHS=30
  %env BATCH_SIZE=8
  %run train_stage1_shape_vae.py
"""

from __future__ import annotations

import math
import os
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from hunyuan_like_shape_flow_architecture import ShapeFlowConfig, ShapeVAE, shape_vae_loss


SEED = int(os.environ.get("SEED", "42"))
DATA_ROOT = Path(os.environ.get("STAGE1_DATA_ROOT", "/kaggle/working/shape_stage1_dataset"))
RESULTS_DIR = Path(os.environ.get("STAGE1_RESULTS_DIR", "/kaggle/working/shape_vae_stage1_results"))

FIELD_TYPE = os.environ.get("FIELD_TYPE", "auto").lower()  # auto, sdf, occupancy
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
LR = float(os.environ.get("LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
KL_WEIGHT = float(os.environ.get("KL_WEIGHT", "1e-4"))
POS_WEIGHT = float(os.environ.get("POS_WEIGHT", "4.0"))
DICE_WEIGHT = float(os.environ.get("DICE_WEIGHT", "0.5"))
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.10"))
RESUME_CKPT = os.environ.get("RESUME_CKPT", "")
BEST_METRIC = os.environ.get("BEST_METRIC", "iou").lower()  # iou or loss

SHAPE_POINTS = int(os.environ.get("SHAPE_POINTS", "4096"))
QUERY_POINTS = int(os.environ.get("QUERY_POINTS", "4096"))

LATENT_TOKENS = int(os.environ.get("LATENT_TOKENS", "256"))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "384"))
FLOW_HEADS = int(os.environ.get("FLOW_HEADS", "8"))
DECODER_HIDDEN = int(os.environ.get("DECODER_HIDDEN", "384"))
DECODER_LAYERS = int(os.environ.get("DECODER_LAYERS", "5"))
FOURIER_BANDS = int(os.environ.get("FOURIER_BANDS", "8"))
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
SANITY_ONLY = os.environ.get("SANITY_ONLY", "0") == "1"
VERBOSE_BATCH = os.environ.get("VERBOSE_BATCH", "1") == "1"

MAX_MODELS_PER_CLASS = int(os.environ.get("MAX_MODELS_PER_CLASS", "1200"))
SHAPENET_CLASSES = [
    item.strip()
    for item in os.environ.get("SHAPENET_CLASSES", "03001627").split(",")
    if item.strip()
]
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


def find_shapenet_vox_archive() -> Path:
    for dataset_dir in POSSIBLE_SHAPENET_DIRS:
        archive = dataset_dir / "ShapeNetVox32.tgz"
        if archive.exists():
            return archive

    input_root = Path("/kaggle/input")
    if input_root.exists():
        matches = list(input_root.rglob("ShapeNetVox32.tgz"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Cannot find ShapeNetVox32.tgz under /kaggle/input. "
        "In Kaggle, add dataset https://www.kaggle.com/datasets/sirish001/shapenet-3dr2n2"
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


def prepare_shapenet_binvox_subset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    done_file = root / f".prepared_binvox_m{MAX_MODELS_PER_CLASS}_{'_'.join(SHAPENET_CLASSES)}"
    existing = list(root.rglob("*.binvox"))
    if existing:
        return
    if done_file.exists():
        return

    archive = find_shapenet_vox_archive()
    category_ids = set(SHAPENET_CLASSES)
    chosen: dict[str, set[str]] = {cat_id: set() for cat_id in category_ids}

    print(f"No .binvox files found in {root}")
    print(f"Extracting ShapeNet voxel subset from: {archive}")
    print(f"Classes: {', '.join(SHAPENET_CLASSES)}")
    print(f"Max models per class: {MAX_MODELS_PER_CLASS}")

    extracted = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".binvox"):
                continue

            cat_id, model_id = find_category_and_model(member.name, category_ids)
            if cat_id is None or model_id is None:
                continue
            if len(chosen[cat_id]) >= MAX_MODELS_PER_CLASS:
                continue

            chosen[cat_id].add(model_id)
            safe_extract_member(tar, member, root)
            extracted += 1

            if extracted % 250 == 0:
                print(f"Extracted {extracted} binvox files...")

            if all(len(models) >= MAX_MODELS_PER_CLASS for models in chosen.values()):
                break

    if extracted == 0:
        raise RuntimeError(
            "ShapeNetVox32.tgz was found, but no matching .binvox files were extracted. "
            f"Check SHAPENET_CLASSES={SHAPENET_CLASSES}."
        )

    done_file.write_text("ok", encoding="utf-8")
    print(
        "Prepared ShapeNet binvox subset: "
        + ", ".join(f"{cat_id}={len(models)}" for cat_id, models in chosen.items())
    )


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


def sample_rows(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] >= n:
        idx = np.random.choice(arr.shape[0], size=n, replace=False)
    else:
        idx = np.random.choice(arr.shape[0], size=n, replace=True)
    return arr[idx]


def as_column(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    return x


class PointsNPZShapeDataset(Dataset):
    def __init__(self, paths: list[Path], field_type: str):
        self.paths = paths
        self.field_type = field_type

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.paths[idx]
        data = np.load(path)

        points = data.get("query_points")
        if points is None:
            points = data.get("points")
        if points is None:
            raise KeyError(f"{path} must contain points or query_points")
        points = np.asarray(points, dtype=np.float32)

        sdf = data.get("target_sdf")
        if sdf is None:
            sdf = data.get("query_sdf")
        if sdf is None:
            sdf = data.get("sdf")

        occ = data.get("occupancy")
        if occ is None:
            occ = data.get("occ")

        local_field = self.field_type
        if local_field == "auto":
            local_field = "sdf" if sdf is not None else "occupancy"

        if local_field == "sdf":
            if sdf is None:
                raise KeyError(f"{path} has no sdf/target_sdf/query_sdf")
            target = as_column(sdf)
            field_is_sdf = np.array([1.0], dtype=np.float32)
        else:
            if occ is None:
                if sdf is None:
                    raise KeyError(f"{path} has neither occupancy nor sdf")
                occ = (as_column(sdf) <= 0).astype(np.float32)
            target = as_column(occ)
            field_is_sdf = np.array([0.0], dtype=np.float32)

        shape_points = data.get("shape_points")
        if shape_points is None:
            encoder_value = target
            shape_points = np.concatenate([points, encoder_value], axis=-1)
        else:
            shape_points = np.asarray(shape_points, dtype=np.float32)

        q_idx = np.random.choice(points.shape[0], size=QUERY_POINTS, replace=points.shape[0] < QUERY_POINTS)
        query_points = points[q_idx]
        target = target[q_idx]

        shape_points = sample_rows(shape_points, SHAPE_POINTS)
        if shape_points.shape[-1] < 4:
            raise ValueError(f"shape_points in {path} must have at least 4 channels")

        return {
            "shape_points": torch.from_numpy(shape_points[:, :4].astype(np.float32)),
            "query_points": torch.from_numpy(query_points.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "field_is_sdf": torch.from_numpy(field_is_sdf),
        }


class BinvoxShapeDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vox = read_binvox(self.paths[idx]).astype(np.float32)
        coords = voxel_grid_coords(vox.shape[0])
        occ = vox.reshape(-1, 1)

        # Oversample occupied cells so the encoder sees the actual shape.
        pos = np.where(occ[:, 0] > 0.5)[0]
        neg = np.where(occ[:, 0] <= 0.5)[0]
        n_pos = min(len(pos), SHAPE_POINTS // 2)
        n_neg = SHAPE_POINTS - n_pos
        if n_pos > 0:
            shape_idx = np.concatenate(
                [
                    np.random.choice(pos, n_pos, replace=len(pos) < n_pos),
                    np.random.choice(neg, n_neg, replace=len(neg) < n_neg),
                ]
            )
        else:
            shape_idx = np.random.choice(len(coords), SHAPE_POINTS, replace=len(coords) < SHAPE_POINTS)

        q_idx = np.random.choice(len(coords), QUERY_POINTS, replace=len(coords) < QUERY_POINTS)
        shape_points = np.concatenate([coords[shape_idx], occ[shape_idx]], axis=-1)

        return {
            "shape_points": torch.from_numpy(shape_points.astype(np.float32)),
            "query_points": torch.from_numpy(coords[q_idx].astype(np.float32)),
            "target": torch.from_numpy(occ[q_idx].astype(np.float32)),
            "field_is_sdf": torch.from_numpy(np.array([0.0], dtype=np.float32)),
        }


def discover_dataset(root: Path, field_type: str) -> Dataset:
    npz_paths = sorted(root.rglob("points.npz"))
    if not npz_paths:
        npz_paths = sorted(root.rglob("*.npz"))

    if npz_paths:
        print(f"Using NPZ point dataset: {len(npz_paths)} files")
        return PointsNPZShapeDataset(npz_paths, field_type)

    binvox_paths = sorted(root.rglob("*.binvox"))
    if binvox_paths:
        print(f"Using BINVOX occupancy dataset: {len(binvox_paths)} files")
        return BinvoxShapeDataset(binvox_paths)

    if root.as_posix().startswith("/kaggle/working") or root.name.startswith("shapenet"):
        prepare_shapenet_binvox_subset(root)
        binvox_paths = sorted(root.rglob("*.binvox"))
        if binvox_paths:
            print(f"Using BINVOX occupancy dataset: {len(binvox_paths)} files")
            return BinvoxShapeDataset(binvox_paths)

    raise FileNotFoundError(
        f"No points.npz/*.npz or *.binvox files found under {root}. "
        "Set STAGE1_DATA_ROOT to your prepared shape dataset."
    )


def make_model() -> ShapeVAE:
    if LATENT_DIM % FLOW_HEADS != 0:
        raise ValueError("LATENT_DIM must be divisible by FLOW_HEADS")
    cfg = ShapeFlowConfig(
        latent_tokens=LATENT_TOKENS,
        latent_dim=LATENT_DIM,
        flow_heads=FLOW_HEADS,
        vae_decoder_hidden=DECODER_HIDDEN,
        vae_decoder_layers=DECODER_LAYERS,
        fourier_bands=FOURIER_BANDS,
    )
    return ShapeVAE(cfg)


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target"]
    is_sdf = bool(batch["field_is_sdf"][0, 0].item() > 0.5)

    if is_sdf:
        loss = shape_vae_loss(outputs["sdf"], target, outputs["mu"], outputs["logvar"], kl_weight=KL_WEIGHT)
        recon = F.smooth_l1_loss(outputs["sdf"], target)
        metrics = {"recon": float(recon.detach().cpu()), "bce": math.nan}
        return loss, metrics

    logits = -outputs["sdf"]
    weights = 1.0 + (POS_WEIGHT - 1.0) * target
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights)
    probs = torch.sigmoid(logits)
    inter_soft = (probs * target).sum(dim=(1, 2))
    union_soft = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = 1.0 - ((2.0 * inter_soft + 1e-6) / (union_soft + 1e-6)).mean()
    kl = -0.5 * torch.mean(1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())
    loss = bce + DICE_WEIGHT * dice + KL_WEIGHT * kl

    with torch.no_grad():
        pred = (probs > 0.5).float()
        inter = (pred * target).sum()
        union = ((pred + target) > 0).float().sum()
        iou = ((inter + 1e-6) / (union + 1e-6)).item()
    metrics = {
        "recon": math.nan,
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "iou": iou,
    }
    return loss, metrics


def run_epoch(model: ShapeVAE, loader: DataLoader, optimizer, device: torch.device, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "bce": 0.0, "iou": 0.0}
    totals["dice"] = 0.0
    counts = {"recon": 0, "bce": 0, "dice": 0, "iou": 0}
    steps = 0

    for step, batch in enumerate(loader, start=1):
        if VERBOSE_BATCH and step == 1:
            print(f"{'Train' if train else 'Val'}: first batch loaded, moving to {device}...", flush=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        if VERBOSE_BATCH and step == 1:
            print(f"{'Train' if train else 'Val'}: first batch on device, running forward...", flush=True)

        with torch.set_grad_enabled(train):
            outputs = model(batch["shape_points"], batch["query_points"])
            if VERBOSE_BATCH and step == 1:
                print(f"{'Train' if train else 'Val'}: forward done, computing loss...", flush=True)
            loss, metrics = compute_loss(outputs, batch)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if VERBOSE_BATCH and step == 1:
                    print(f"{'Train' if train else 'Val'}: backward done, optimizer step...", flush=True)
                optimizer_step(optimizer)
                if VERBOSE_BATCH and step == 1:
                    print(f"{'Train' if train else 'Val'}: optimizer step done.", flush=True)

        totals["loss"] += float(loss.detach().cpu())
        for key in ("recon", "bce", "dice", "iou"):
            value = metrics.get(key, math.nan)
            if not math.isnan(value):
                totals[key] += value
                counts[key] += 1
        steps += 1

    result = {"loss": totals["loss"] / max(steps, 1)}
    for key in ("recon", "bce", "dice", "iou"):
        if counts[key] > 0:
            result[key] = totals[key] / counts[key]
    return result


def main() -> None:
    seed_everything(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = discover_dataset(DATA_ROOT, FIELD_TYPE)
    val_size = max(1, int(len(dataset) * VAL_FRACTION))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    device = choose_device()
    model = make_model().to(device)
    if RESUME_CKPT:
        resume_path = Path(RESUME_CKPT)
        if not resume_path.exists():
            raise FileNotFoundError(f"RESUME_CKPT not found: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location="cpu"), strict=True)
        print(f"Resumed model weights from: {resume_path}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"Data root: {DATA_ROOT}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Device: {device}")
    print(f"Objects: total={len(dataset)}, train={len(train_ds)}, val={len(val_ds)}")
    print(
        "Config: "
        f"latent_tokens={LATENT_TOKENS}, latent_dim={LATENT_DIM}, heads={FLOW_HEADS}, "
        f"shape_points={SHAPE_POINTS}, query_points={QUERY_POINTS}, field_type={FIELD_TYPE}, "
        f"num_workers={NUM_WORKERS}, force_cpu={FORCE_CPU}, dice_weight={DICE_WEIGHT}, "
        f"best_metric={BEST_METRIC}"
    )

    print("Loading one sanity batch before training...", flush=True)
    sanity_batch = next(iter(train_loader))
    print(
        "Sanity batch shapes: "
        f"shape_points={tuple(sanity_batch['shape_points'].shape)}, "
        f"query_points={tuple(sanity_batch['query_points'].shape)}, "
        f"target={tuple(sanity_batch['target'].shape)}",
        flush=True,
    )
    if SANITY_ONLY:
        print("SANITY_ONLY=1, stopping before training loop.", flush=True)
        return

    best_score = -float("inf") if BEST_METRIC == "iou" else float("inf")
    history_path = RESULTS_DIR / "stage1_history.csv"
    history_path.write_text(
        "epoch,train_loss,val_loss,train_bce,val_bce,train_dice,val_dice,train_iou,val_iou\n",
        encoding="utf-8",
    )

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)

        row = [
            epoch,
            train_metrics.get("loss", math.nan),
            val_metrics.get("loss", math.nan),
            train_metrics.get("bce", math.nan),
            val_metrics.get("bce", math.nan),
            train_metrics.get("dice", math.nan),
            val_metrics.get("dice", math.nan),
            train_metrics.get("iou", math.nan),
            val_metrics.get("iou", math.nan),
        ]
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(",".join(str(x) for x in row) + "\n")

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"train_bce={train_metrics.get('bce', math.nan):.4f} "
            f"val_bce={val_metrics.get('bce', math.nan):.4f} | "
            f"train_dice={train_metrics.get('dice', math.nan):.4f} "
            f"val_dice={val_metrics.get('dice', math.nan):.4f} | "
            f"train_iou={train_metrics.get('iou', math.nan):.4f} "
            f"val_iou={val_metrics.get('iou', math.nan):.4f}"
        )

        score = val_metrics.get("iou", -float("inf")) if BEST_METRIC == "iou" else val_metrics["loss"]
        improved = score > best_score if BEST_METRIC == "iou" else score < best_score
        if improved:
            best_score = score
            save_checkpoint(model, RESULTS_DIR / "shape_vae_stage1_best.pt")

        save_checkpoint(model, RESULTS_DIR / "shape_vae_stage1_last.pt")

    print(f"Done. Best val {BEST_METRIC}: {best_score:.4f}")
    print(f"Saved: {RESULTS_DIR / 'shape_vae_stage1_best.pt'}")


if __name__ == "__main__":
    main()
