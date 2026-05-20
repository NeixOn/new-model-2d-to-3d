"""
Quick single-image 3D reconstruction training for Kaggle TPU v5e-8.

Dataset expected in Kaggle:
  Add dataset: https://www.kaggle.com/datasets/sirish001/shapenet-3dr2n2

It contains:
  /kaggle/input/shapenet-3dr2n2/ShapeNetRendering.tgz
  /kaggle/input/shapenet-3dr2n2/ShapeNetVox32.tgz

Run in a Kaggle Notebook with Accelerator = TPU v5e-8:
  %run /kaggle/working/kaggle_tpu_v5e8_quick_3d_recon.py

For the first smoke test, keep MAX_MODELS_PER_CLASS small.
"""

import os
import math
import random
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from torchvision import transforms
from torchvision.models import resnet18

# Kaggle TPU VM usually sets this already, but keeping it explicit makes the
# notebook more robust across runtime images.
os.environ.setdefault("PJRT_DEVICE", "TPU")


# ---------------------------
# Config
# ---------------------------

SEED = 42

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "shapenet_r2n2_quick"
RESULTS_DIR = WORK_DIR / "quick_3d_recon_results"

POSSIBLE_DATASET_DIRS = [
    Path("/kaggle/input/shapenet-3dr2n2"),
    Path("/kaggle/input/datasets/sirish001/shapenet-3dr2n2"),
]


def find_dataset_archives() -> tuple[Path, Path]:
    """Find 3D-R2N2 archives in common Kaggle input locations."""
    for dataset_dir in POSSIBLE_DATASET_DIRS:
        render_tgz = dataset_dir / "ShapeNetRendering.tgz"
        vox_tgz = dataset_dir / "ShapeNetVox32.tgz"
        if render_tgz.exists() and vox_tgz.exists():
            return render_tgz, vox_tgz

    input_root = Path("/kaggle/input")
    if input_root.exists():
        render_matches = list(input_root.rglob("ShapeNetRendering.tgz"))
        vox_matches = list(input_root.rglob("ShapeNetVox32.tgz"))
        if render_matches and vox_matches:
            return render_matches[0], vox_matches[0]

    raise FileNotFoundError(
        "Cannot find ShapeNetRendering.tgz and ShapeNetVox32.tgz under /kaggle/input. "
        "In Kaggle, add dataset 'sirish001/shapenet-3dr2n2' to the notebook."
    )

# ShapeNet synset IDs.
# Start with one class for a fast proof-of-life run. Add more when everything works.
CLASSES = {
    "chair": "03001627",
    # "airplane": "02691156",
    # "car": "02958343",
}

MAX_MODELS_PER_CLASS = 160       # 80-300 is good for a quick TPU smoke test.
VIEWS_PER_MODEL = 1              # Use one rendered image per object for single-image reconstruction.
IMAGE_SIZE = 128
VOXEL_SIZE = 32

EPOCHS = 4
BATCH_SIZE_PER_CORE = 8          # Global batch = BATCH_SIZE_PER_CORE * 8 on v5e-8.
NUM_WORKERS = 2
LR = 2e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.4


# ---------------------------
# Small utilities
# ---------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_category_and_model(path: str, category_ids: set[str]):
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


def prepare_quick_subset() -> None:
    """Extract only a tiny subset from the 3D-R2N2 tarballs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    done_file = DATA_DIR / f".prepared_{MAX_MODELS_PER_CLASS}_{'_'.join(CLASSES.values())}"
    if done_file.exists():
        print(f"Dataset subset already prepared: {DATA_DIR}")
        return

    render_tgz, vox_tgz = find_dataset_archives()
    print(f"Using render archive: {render_tgz}")
    print(f"Using voxel archive:  {vox_tgz}")

    category_ids = set(CLASSES.values())
    chosen: dict[str, set[str]] = {cat_id: set() for cat_id in category_ids}

    print("Selecting and extracting voxel targets...")
    with tarfile.open(vox_tgz, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".binvox"):
                continue
            cat_id, model_id = find_category_and_model(member.name, category_ids)
            if cat_id is None:
                continue
            if len(chosen[cat_id]) >= MAX_MODELS_PER_CLASS:
                continue
            chosen[cat_id].add(model_id)
            safe_extract_member(tar, member, DATA_DIR)
            if all(len(v) >= MAX_MODELS_PER_CLASS for v in chosen.values()):
                break

    total_models = sum(len(v) for v in chosen.values())
    if total_models == 0:
        raise RuntimeError("No voxel files were extracted. Archive layout may be different.")

    print(f"Selected {total_models} models: " + ", ".join(f"{k}={len(v)}" for k, v in chosen.items()))
    print("Extracting matching rendered images...")

    extracted_views: dict[tuple[str, str], int] = {}
    with tarfile.open(render_tgz, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".png"):
                continue
            cat_id, model_id = find_category_and_model(member.name, category_ids)
            if cat_id is None or model_id not in chosen[cat_id]:
                continue
            key = (cat_id, model_id)
            if extracted_views.get(key, 0) >= VIEWS_PER_MODEL:
                continue
            safe_extract_member(tar, member, DATA_DIR)
            extracted_views[key] = extracted_views.get(key, 0) + 1

    if len(extracted_views) == 0:
        raise RuntimeError("No rendered images were extracted. Archive layout may be different.")

    done_file.write_text("ok", encoding="utf-8")
    print(f"Prepared subset at: {DATA_DIR}")


def read_binvox(path: Path) -> np.ndarray:
    """Read a 32^3 binvox file into a float32 occupancy grid."""
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

        voxels = voxels.reshape(dims)
        return voxels


def discover_samples(root: Path):
    category_ids = set(CLASSES.values())
    binvox_files = list(root.rglob("*.binvox"))
    image_files = list(root.rglob("*.png"))

    vox_by_key = {}
    for path in binvox_files:
        cat_id, model_id = find_category_and_model(str(path), category_ids)
        if cat_id is not None:
            vox_by_key[(cat_id, model_id)] = path

    images_by_key = {}
    for path in image_files:
        cat_id, model_id = find_category_and_model(str(path), category_ids)
        if cat_id is not None:
            images_by_key.setdefault((cat_id, model_id), []).append(path)

    samples = []
    for key, vox_path in vox_by_key.items():
        for image_path in sorted(images_by_key.get(key, []))[:VIEWS_PER_MODEL]:
            samples.append((image_path, vox_path))

    if not samples:
        raise RuntimeError(f"No paired image/binvox samples found under {root}")

    random.shuffle(samples)
    return samples


class R2N2VoxelDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.image_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, voxel_path = self.samples[idx]
        img = Image.open(image_path).convert("RGB")
        img = self.image_tf(img)

        vox = read_binvox(voxel_path)
        vox = torch.from_numpy(vox).unsqueeze(0)
        return img, vox


# ---------------------------
# Model
# ---------------------------

class VoxelDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1), # 8
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # 16
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),   # 32
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z).view(z.shape[0], 256, 4, 4, 4)
        return self.net(x)


class SingleImageVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.decoder = VoxelDecoder(latent_dim=512)

    def forward(self, image):
        z = self.encoder(image)
        return self.decoder(z)


def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def reconstruction_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)
    return bce + dice


@torch.no_grad()
def batch_iou(logits, target, threshold=THRESHOLD):
    pred = (torch.sigmoid(logits) > threshold).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dims)
    union = ((pred + target) > 0).float().sum(dims)
    return ((inter + 1e-6) / (union + 1e-6)).mean()


# ---------------------------
# Visualization
# ---------------------------

def save_voxel_projection(vox: np.ndarray, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Cheap readable preview: max projections, not a full 3D renderer.
    proj_xy = vox.max(axis=0)
    proj_xz = vox.max(axis=1)
    proj_yz = vox.max(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, arr, name in zip(axes, [proj_xy, proj_xz, proj_yz], ["xy", "xz", "yz"]):
        ax.imshow(arr, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_predictions(model, dataset, device, rank: int) -> None:
    if rank != 0:
        return
    model.eval()
    preview_dir = RESULTS_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(6, len(dataset))):
        image, target = dataset[idx]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > THRESHOLD).astype(np.float32)
        gt = target[0].numpy()
        save_voxel_projection(pred, preview_dir / f"sample_{idx}_pred.png", "prediction")
        save_voxel_projection(gt, preview_dir / f"sample_{idx}_gt.png", "ground truth")


# ---------------------------
# TPU training
# ---------------------------

def train_one_epoch(model, loader, optimizer, device, xm, pl):
    model.train()
    para_loader = pl.MpDeviceLoader(loader, device)

    total_loss = 0.0
    total_iou = 0.0
    steps = 0

    for images, voxels in para_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = reconstruction_loss(logits, voxels)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)

        total_loss += float(loss.detach().cpu())
        total_iou += float(batch_iou(logits.detach(), voxels).cpu())
        steps += 1

    return total_loss / max(steps, 1), total_iou / max(steps, 1)


@torch.no_grad()
def validate(model, loader, device, pl):
    model.eval()
    para_loader = pl.MpDeviceLoader(loader, device)

    total_loss = 0.0
    total_iou = 0.0
    steps = 0

    for images, voxels in para_loader:
        logits = model(images)
        loss = reconstruction_loss(logits, voxels)
        total_loss += float(loss.detach().cpu())
        total_iou += float(batch_iou(logits, voxels).cpu())
        steps += 1

    return total_loss / max(steps, 1), total_iou / max(steps, 1)


def _mp_fn(rank, flags):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    seed_everything(SEED + rank)
    device = xm.xla_device()

    samples = discover_samples(DATA_DIR)
    dataset = R2N2VoxelDataset(samples)

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        seed=SEED,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_PER_CORE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_PER_CORE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    model = SingleImageVoxelNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if rank == 0:
        print(f"Samples: total={len(dataset)}, train={len(train_ds)}, val={len(val_ds)}")
        print(f"World size: {xm.xrt_world_size()}")
        print(f"Global batch size: {BATCH_SIZE_PER_CORE * xm.xrt_world_size()}")

    best_iou = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)

        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device, xm, pl)
        val_loss, val_iou = validate(model, val_loader, device, pl)

        # Average metrics across TPU workers.
        train_loss = xm.mesh_reduce("train_loss", train_loss, np.mean)
        train_iou = xm.mesh_reduce("train_iou", train_iou, np.mean)
        val_loss = xm.mesh_reduce("val_loss", val_loss, np.mean)
        val_iou = xm.mesh_reduce("val_iou", val_iou, np.mean)

        if rank == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
                f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
            )

        if val_iou > best_iou:
            best_iou = val_iou
            xm.save(model.state_dict(), RESULTS_DIR / "best_model.pt")

    save_predictions(model, val_ds, device, rank)

    if rank == 0:
        print(f"Done. Results saved to: {RESULTS_DIR}")
        print(f"Best validation IoU: {best_iou:.4f}")


def main():
    seed_everything(SEED)
    prepare_quick_subset()

    import torch_xla.distributed.xla_multiprocessing as xmp
    # With PJRT-based torch-xla runtimes, passing nprocs=8 is rejected.
    # nprocs=None lets XLA use all available TPU devices on v5e-8.
    xmp.spawn(_mp_fn, args=({},), nprocs=None, start_method="fork")


if __name__ == "__main__":
    main()
