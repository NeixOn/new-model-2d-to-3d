"""
JAX TPU training for a stronger single-image 3D reconstruction architecture:
Triplane Occupancy / LRM-lite.

This is the second architecture for comparison with the dense voxel baseline.

Run on Kaggle TPU v5e-8:
  %cd /kaggle/working/new-model-2d-to-3d/new_architecture_triplane_lrm_lite

  %env EPOCHS=120
  %env MAX_MODELS_PER_CLASS=6778
  %env VIEWS_PER_MODEL=8
  %env PER_DEVICE_BATCH=4
  %env QUERY_POINTS=4096
  %env TRIPLANE_RES=32
  %env TRIPLANE_CHANNELS=16
  %env LR=2e-4
  %env POS_WEIGHT=4.0

  %run kaggle_jax_triplane_occupancy_train.py

Outputs:
  /kaggle/working/triplane_lrm_lite_results/best_model_params.npz
  /kaggle/working/triplane_lrm_lite_results/training_history.csv
  /kaggle/working/triplane_lrm_lite_results/previews/
"""

import math
import os
import random
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jrandom
from jax.tree_util import tree_map


SEED = 42

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "shapenet_r2n2_quick"
RESULTS_DIR = WORK_DIR / "triplane_lrm_lite_results"

POSSIBLE_DATASET_DIRS = [
    Path("/kaggle/input/shapenet-3dr2n2"),
    Path("/kaggle/input/datasets/sirish001/shapenet-3dr2n2"),
]

CLASSES = {
    "chair": "03001627",
    # "airplane": "02691156",
    # "car": "02958343",
}

MAX_MODELS_PER_CLASS = int(os.environ.get("MAX_MODELS_PER_CLASS", "1200"))
VIEWS_PER_MODEL = int(os.environ.get("VIEWS_PER_MODEL", "4"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "128"))
VOXEL_SIZE = int(os.environ.get("VOXEL_SIZE", "32"))

TRIPLANE_RES = int(os.environ.get("TRIPLANE_RES", "32"))
TRIPLANE_CHANNELS = int(os.environ.get("TRIPLANE_CHANNELS", "16"))
QUERY_POINTS = int(os.environ.get("QUERY_POINTS", "4096"))
MLP_HIDDEN = int(os.environ.get("MLP_HIDDEN", "128"))
DROPOUT = float(os.environ.get("DROPOUT", "0.10"))
QUERY_POS_BOOST = float(os.environ.get("QUERY_POS_BOOST", "8.0"))

EPOCHS = int(os.environ.get("EPOCHS", "80"))
PER_DEVICE_BATCH = int(os.environ.get("PER_DEVICE_BATCH", "4"))
LR = float(os.environ.get("LR", "2e-4"))
MIN_LR = float(os.environ.get("MIN_LR", "2e-5"))
LR_WARMUP_EPOCHS = int(os.environ.get("LR_WARMUP_EPOCHS", "5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
POS_WEIGHT = float(os.environ.get("POS_WEIGHT", "4.0"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.4"))
PATIENCE = int(os.environ.get("PATIENCE", "30"))
MIN_DELTA = float(os.environ.get("MIN_DELTA", "1e-4"))
MAX_VAL_BATCHES = int(os.environ.get("MAX_VAL_BATCHES", "50"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_dataset_archives() -> tuple[Path, Path]:
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

    raise FileNotFoundError("Cannot find ShapeNetRendering.tgz and ShapeNetVox32.tgz under /kaggle/input.")


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


def prepare_subset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    done_file = DATA_DIR / f".prepared_m{MAX_MODELS_PER_CLASS}_v{VIEWS_PER_MODEL}_{'_'.join(CLASSES.values())}"
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

    done_file.write_text("ok", encoding="utf-8")
    print(f"Prepared subset at: {DATA_DIR}")


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


def discover_samples(root: Path):
    category_ids = set(CLASSES.values())
    vox_by_key = {}
    for path in root.rglob("*.binvox"):
        cat_id, model_id = find_category_and_model(str(path), category_ids)
        if cat_id is not None:
            vox_by_key[(cat_id, model_id)] = path

    images_by_key = {}
    for path in root.rglob("*.png"):
        cat_id, model_id = find_category_and_model(str(path), category_ids)
        if cat_id is not None:
            images_by_key.setdefault((cat_id, model_id), []).append(path)

    samples = []
    selected_items = sorted(vox_by_key.items())[:MAX_MODELS_PER_CLASS]
    for key, vox_path in selected_items:
        for image_path in sorted(images_by_key.get(key, []))[:VIEWS_PER_MODEL]:
            samples.append((image_path, vox_path))

    if not samples:
        raise RuntimeError(f"No paired image/binvox samples found under {root}")

    random.shuffle(samples)
    return samples


def split_samples_by_model(samples, val_fraction: float, global_batch: int):
    by_model = {}
    for image_path, voxel_path in samples:
        by_model.setdefault(str(voxel_path), []).append((image_path, voxel_path))

    keys = list(by_model.keys())
    random.shuffle(keys)
    val_count = max(1, int(len(keys) * val_fraction))
    val_keys = set(keys[:val_count])

    train_samples, val_samples = [], []
    for key in keys:
        if key in val_keys:
            val_samples.extend(by_model[key])
        else:
            train_samples.extend(by_model[key])

    while len(val_samples) < global_batch and train_samples:
        val_samples.append(train_samples.pop())

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    return train_samples, val_samples


def load_sample(image_path: Path, voxel_path: Path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    vox = read_binvox(voxel_path).astype(np.float32)
    return image, vox


def batch_loader(samples, batch_size: int, shuffle: bool):
    idxs = np.arange(len(samples))
    if shuffle:
        np.random.shuffle(idxs)

    usable = (len(idxs) // batch_size) * batch_size
    idxs = idxs[:usable]

    for start in range(0, usable, batch_size):
        batch_idxs = idxs[start:start + batch_size]
        images, voxels = [], []
        for idx in batch_idxs:
            image, voxel = load_sample(*samples[int(idx)])
            images.append(image)
            voxels.append(voxel)
        yield np.stack(images), np.stack(voxels)


def shard_batch(batch, n_devices: int):
    images, voxels = batch
    per_device = images.shape[0] // n_devices
    return (
        images.reshape(n_devices, per_device, IMAGE_SIZE, IMAGE_SIZE, 3),
        voxels.reshape(n_devices, per_device, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
    )


def make_query_grid():
    coords = np.stack(np.meshgrid(
        np.linspace(-1.0, 1.0, VOXEL_SIZE, dtype=np.float32),
        np.linspace(-1.0, 1.0, VOXEL_SIZE, dtype=np.float32),
        np.linspace(-1.0, 1.0, VOXEL_SIZE, dtype=np.float32),
        indexing="ij",
    ), axis=-1)
    return coords.reshape(-1, 3)


FULL_QUERY_GRID = make_query_grid()


def sample_query_points(key, voxels, num_points: int):
    batch = voxels.shape[0]
    total = VOXEL_SIZE ** 3
    flat = voxels.reshape(batch, total)
    keys = jrandom.split(key, batch)
    probs = 1.0 + (QUERY_POS_BOOST - 1.0) * flat
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)

    def sample_one(k, p):
        return jrandom.choice(k, total, shape=(num_points,), replace=True, p=p)

    idx = jax.vmap(sample_one)(keys, probs)
    labels = jnp.take_along_axis(flat, idx, axis=1)
    coords = jnp.asarray(FULL_QUERY_GRID)[idx]
    return coords, labels[..., None]


def init_conv(key, in_ch, out_ch, kernel):
    scale = math.sqrt(2.0 / (kernel * kernel * in_ch))
    return {
        "w": scale * jrandom.normal(key, (kernel, kernel, in_ch, out_ch), dtype=jnp.float32),
        "b": jnp.zeros((out_ch,), dtype=jnp.float32),
    }


def init_dense(key, in_dim, out_dim):
    scale = math.sqrt(2.0 / in_dim)
    return {
        "w": scale * jrandom.normal(key, (in_dim, out_dim), dtype=jnp.float32),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def init_params(seed=SEED):
    keys = jrandom.split(jrandom.PRNGKey(seed), 12)
    triplane_dim = 3 * TRIPLANE_RES * TRIPLANE_RES * TRIPLANE_CHANNELS
    mlp_in = 3 * TRIPLANE_CHANNELS + 3 + 6
    return {
        "conv1": init_conv(keys[0], 3, 32, 5),
        "conv2": init_conv(keys[1], 32, 64, 3),
        "conv3": init_conv(keys[2], 64, 128, 3),
        "conv4": init_conv(keys[3], 128, 256, 3),
        "conv5": init_conv(keys[4], 256, 256, 3),
        "tp_fc1": init_dense(keys[5], 256, 1024),
        "tp_fc2": init_dense(keys[6], 1024, triplane_dim),
        "mlp1": init_dense(keys[7], mlp_in, MLP_HIDDEN),
        "mlp2": init_dense(keys[8], MLP_HIDDEN, MLP_HIDDEN),
        "mlp3": init_dense(keys[9], MLP_HIDDEN, 1),
    }


def conv2d(x, p, stride):
    y = lax.conv_general_dilated(
        x,
        p["w"],
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + p["b"]


def dense(x, p):
    return x @ p["w"] + p["b"]


def encode_to_triplanes(params, images):
    x = jax.nn.relu(conv2d(images, params["conv1"], 2))
    x = jax.nn.relu(conv2d(x, params["conv2"], 2))
    x = jax.nn.relu(conv2d(x, params["conv3"], 2))
    x = jax.nn.relu(conv2d(x, params["conv4"], 2))
    x = jax.nn.relu(conv2d(x, params["conv5"], 1))
    x = jnp.mean(x, axis=(1, 2))
    x = jax.nn.gelu(dense(x, params["tp_fc1"]))
    x = dense(x, params["tp_fc2"])
    return x.reshape((-1, 3, TRIPLANE_RES, TRIPLANE_RES, TRIPLANE_CHANNELS))


def bilinear_sample(plane, uv):
    # plane: B,H,W,C; uv: B,N,2 in [-1,1]
    b, h, w, c = plane.shape
    u = (uv[..., 0] + 1.0) * 0.5 * (w - 1)
    v = (uv[..., 1] + 1.0) * 0.5 * (h - 1)
    u0 = jnp.floor(u).astype(jnp.int32)
    v0 = jnp.floor(v).astype(jnp.int32)
    u1 = jnp.clip(u0 + 1, 0, w - 1)
    v1 = jnp.clip(v0 + 1, 0, h - 1)
    u0 = jnp.clip(u0, 0, w - 1)
    v0 = jnp.clip(v0, 0, h - 1)

    batch_idx = jnp.arange(b)[:, None]
    f00 = plane[batch_idx, v0, u0]
    f01 = plane[batch_idx, v1, u0]
    f10 = plane[batch_idx, v0, u1]
    f11 = plane[batch_idx, v1, u1]

    wu = (u - u0.astype(jnp.float32))[..., None]
    wv = (v - v0.astype(jnp.float32))[..., None]
    return (
        f00 * (1 - wu) * (1 - wv)
        + f10 * wu * (1 - wv)
        + f01 * (1 - wu) * wv
        + f11 * wu * wv
    )


def positional_encoding(points):
    return jnp.concatenate([
        points,
        jnp.sin(math.pi * points),
        jnp.cos(math.pi * points),
    ], axis=-1)


def apply_dropout(x, rng, rate):
    if rate <= 0.0:
        return x
    keep = 1.0 - rate
    mask = jrandom.bernoulli(rng, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)


def query_occupancy(params, triplanes, points, rng=None, training=False):
    xy = bilinear_sample(triplanes[:, 0], points[..., [0, 1]])
    xz = bilinear_sample(triplanes[:, 1], points[..., [0, 2]])
    yz = bilinear_sample(triplanes[:, 2], points[..., [1, 2]])
    feats = jnp.concatenate([xy, xz, yz, positional_encoding(points)], axis=-1)
    x = jax.nn.gelu(dense(feats, params["mlp1"]))
    if training and DROPOUT > 0.0 and rng is not None:
        rng, drop_rng = jrandom.split(rng)
        x = apply_dropout(x, drop_rng, DROPOUT)
    x = jax.nn.gelu(dense(x, params["mlp2"]))
    if training and DROPOUT > 0.0 and rng is not None:
        rng, drop_rng = jrandom.split(rng)
        x = apply_dropout(x, drop_rng, DROPOUT)
    return dense(x, params["mlp3"])


def forward(params, images, points, rng=None, training=False):
    triplanes = encode_to_triplanes(params, images)
    return query_occupancy(params, triplanes, points, rng=rng, training=training)


def loss_and_iou(params, images, voxels, rng):
    sample_rng, dropout_rng = jrandom.split(rng)
    points, labels = sample_query_points(sample_rng, voxels, QUERY_POINTS)
    logits = forward(params, images, points, rng=dropout_rng, training=True)

    bce_per_point = jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    weights = 1.0 + (POS_WEIGHT - 1.0) * labels
    bce = jnp.sum(bce_per_point * weights) / jnp.sum(weights)

    probs = jax.nn.sigmoid(logits)
    inter_soft = jnp.sum(probs * labels, axis=(1, 2))
    union_soft = jnp.sum(probs, axis=(1, 2)) + jnp.sum(labels, axis=(1, 2))
    dice = 1.0 - jnp.mean((2.0 * inter_soft + 1e-6) / (union_soft + 1e-6))

    pred = (probs > THRESHOLD).astype(jnp.float32)
    inter = jnp.sum(pred * labels, axis=(1, 2))
    union = jnp.sum(((pred + labels) > 0).astype(jnp.float32), axis=(1, 2))
    iou = jnp.mean((inter + 1e-6) / (union + 1e-6))
    return bce + dice, iou


def full_grid_iou(params, images, voxels):
    batch = images.shape[0]
    points = jnp.asarray(FULL_QUERY_GRID, dtype=jnp.float32)
    points = jnp.broadcast_to(points[None, ...], (batch, points.shape[0], 3))
    logits = forward(params, images, points)
    probs = jax.nn.sigmoid(logits[..., 0])
    pred = (probs > THRESHOLD).astype(jnp.float32)
    labels = voxels.reshape(batch, -1)
    inter = jnp.sum(pred * labels, axis=1)
    union = jnp.sum(((pred + labels) > 0).astype(jnp.float32), axis=1)
    return jnp.mean((inter + 1e-6) / (union + 1e-6))


def init_adam_state(params):
    zeros = tree_map(jnp.zeros_like, params)
    return {"step": jnp.array(0, dtype=jnp.int32), "m": zeros, "v": zeros}


def adamw_update(params, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    step = state["step"] + 1
    m = tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, state["m"], grads)
    v = tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), state["v"], grads)
    lr_t = lr * jnp.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)

    def upd(p, m_, v_):
        return p - lr_t * m_ / (jnp.sqrt(v_) + eps) - lr * WEIGHT_DECAY * p

    return tree_map(upd, params, m, v), {"step": step, "m": m, "v": v}


def train_step(params, opt_state, images, voxels, rng, lr):
    (loss, iou), grads = jax.value_and_grad(loss_and_iou, has_aux=True)(params, images, voxels, rng)
    grads = lax.pmean(grads, axis_name="devices")
    loss = lax.pmean(loss, axis_name="devices")
    iou = lax.pmean(iou, axis_name="devices")
    params, opt_state = adamw_update(params, grads, opt_state, lr)
    return params, opt_state, loss, iou


def eval_step(params, images, voxels):
    iou = full_grid_iou(params, images, voxels)
    iou = lax.pmean(iou, axis_name="devices")
    return iou


p_train_step = jax.pmap(train_step, axis_name="devices")
p_eval_step = jax.pmap(eval_step, axis_name="devices")


def split_rng_step(rng):
    new_rng, step_rng = jrandom.split(rng)
    return new_rng, step_rng


p_split_rng = jax.pmap(split_rng_step)


def epoch_lr(epoch: int) -> float:
    if LR_WARMUP_EPOCHS > 0 and epoch <= LR_WARMUP_EPOCHS:
        return LR * epoch / LR_WARMUP_EPOCHS
    progress = (epoch - LR_WARMUP_EPOCHS) / max(EPOCHS - LR_WARMUP_EPOCHS, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return MIN_LR + (LR - MIN_LR) * cosine


def unreplicate(x):
    return tree_map(lambda y: np.asarray(y[0]), x)


def save_params(params, path: Path):
    leaves = jax.tree_util.tree_leaves(params)
    np.savez(path, **{f"arr_{i}": np.asarray(x) for i, x in enumerate(leaves)})


def save_voxel_projection(vox: np.ndarray, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def save_predictions(params, samples):
    preview_dir = RESULTS_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for idx, (image_path, voxel_path) in enumerate(samples[:6]):
        image, target = load_sample(image_path, voxel_path)
        points = jnp.asarray(FULL_QUERY_GRID[None, ...], dtype=jnp.float32)
        logits = forward(params, jnp.asarray(image[None, ...]), points)
        pred = (np.asarray(jax.nn.sigmoid(logits[0, :, 0])) > THRESHOLD).astype(np.float32)
        pred = pred.reshape(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
        save_voxel_projection(pred, preview_dir / f"sample_{idx}_pred.png", "prediction")
        save_voxel_projection(target, preview_dir / f"sample_{idx}_gt.png", "ground truth")


def main():
    seed_everything(SEED)
    prepare_subset()

    devices = jax.devices()
    n_devices = len(devices)
    if n_devices < 2:
        raise RuntimeError(f"JAX sees only {n_devices} device(s): {devices}. Check Kaggle accelerator = TPU v5e-8.")

    print("JAX devices:")
    for d in devices:
        print(f"  {d}")
    print(f"Parallel replicas: {n_devices}")

    global_batch = PER_DEVICE_BATCH * n_devices
    samples = discover_samples(DATA_DIR)
    train_samples, val_samples = split_samples_by_model(samples, val_fraction=0.15, global_batch=global_batch)
    train_steps = len(train_samples) // global_batch
    val_steps = len(val_samples) // global_batch

    print(f"Samples: total={len(samples)}, train={len(train_samples)}, val={len(val_samples)}")
    print(f"Global batch size: {global_batch} ({PER_DEVICE_BATCH} x {n_devices})")
    print(f"Steps: train={train_steps}, val={val_steps}")
    print(
        "Config: "
        f"epochs={EPOCHS}, views={VIEWS_PER_MODEL}, query_points={QUERY_POINTS}, "
        f"triplane={TRIPLANE_RES}x{TRIPLANE_RES}x{TRIPLANE_CHANNELS}, lr={LR}, "
        f"max_val_batches={MAX_VAL_BATCHES}, dropout={DROPOUT}, query_pos_boost={QUERY_POS_BOOST}"
    )

    params = init_params(SEED)
    opt_state = init_adam_state(params)
    params_repl = jax.device_put_replicated(params, devices)
    opt_state_repl = jax.device_put_replicated(opt_state, devices)
    rng_repl = jax.device_put_sharded([jrandom.PRNGKey(SEED + i) for i in range(n_devices)], devices)

    best_iou = -1.0
    best_epoch = 0
    best_params = params
    bad_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        lr_value = epoch_lr(epoch)
        train_losses, train_ious = [], []

        for batch in batch_loader(train_samples, global_batch, shuffle=True):
            images, voxels = shard_batch(batch, n_devices)
            rng_repl, step_rng = p_split_rng(rng_repl)
            lr_sharded = np.full((n_devices,), lr_value, dtype=np.float32)
            params_repl, opt_state_repl, loss, iou = p_train_step(
                params_repl, opt_state_repl, images, voxels, step_rng, lr_sharded
            )
            train_losses.append(float(np.asarray(loss[0])))
            train_ious.append(float(np.asarray(iou[0])))

        val_ious = []
        for val_step, batch in enumerate(batch_loader(val_samples, global_batch, shuffle=False), start=1):
            if MAX_VAL_BATCHES > 0 and val_step > MAX_VAL_BATCHES:
                break
            images, voxels = shard_batch(batch, n_devices)
            iou = p_eval_step(params_repl, images, voxels)
            val_ious.append(float(np.asarray(iou[0])))

        train_loss = float(np.mean(train_losses))
        train_iou = float(np.mean(train_ious))
        val_iou = float(np.mean(val_ious))
        history.append((epoch, lr_value, train_loss, train_iou, val_iou))

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | lr={lr_value:.2e} | "
            f"train_loss={train_loss:.4f} train_query_iou={train_iou:.4f} | "
            f"val_full_iou={val_iou:.4f}"
        )

        if val_iou > best_iou + MIN_DELTA:
            best_iou = val_iou
            best_epoch = epoch
            bad_epochs = 0
            best_params = unreplicate(params_repl)
            save_params(best_params, RESULTS_DIR / "best_model_params.npz")
        else:
            bad_epochs += 1

        if PATIENCE > 0 and bad_epochs >= PATIENCE:
            print(f"Early stopping: best epoch={best_epoch}, best_val_full_iou={best_iou:.4f}")
            break

    with open(RESULTS_DIR / "training_history.csv", "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,train_query_iou,val_full_iou\n")
        for row in history:
            f.write(",".join(str(x) for x in row) + "\n")

    save_predictions(best_params, val_samples)
    print(f"Done. Results saved to: {RESULTS_DIR}")
    print(f"Best validation full-grid IoU: {best_iou:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
