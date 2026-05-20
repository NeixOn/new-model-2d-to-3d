"""
Parallel Kaggle TPU v5e-8 smoke training with JAX pmap.

Use this on Kaggle when TensorFlow sees only CPU and PyTorch/XLA fails with
SliceBuilder worker address errors.

Run:
  %cd /kaggle/working/new-model-2d-to-3d
  %run kaggle_tpu_v5e8_jax_quick_3d_recon.py
"""

import math
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
RESULTS_DIR = WORK_DIR / "quick_3d_recon_results_jax"

POSSIBLE_DATASET_DIRS = [
    Path("/kaggle/input/shapenet-3dr2n2"),
    Path("/kaggle/input/datasets/sirish001/shapenet-3dr2n2"),
]

CLASSES = {
    "chair": "03001627",
    # "airplane": "02691156",
    # "car": "02958343",
}

MAX_MODELS_PER_CLASS = 160
VIEWS_PER_MODEL = 1
IMAGE_SIZE = 128
VOXEL_SIZE = 32

EPOCHS = 4
PER_DEVICE_BATCH = 4
LR = 2e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.4


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

    raise FileNotFoundError(
        "Cannot find ShapeNetRendering.tgz and ShapeNetVox32.tgz under /kaggle/input."
    )


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


def load_sample(image_path: Path, voxel_path: Path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    vox = read_binvox(voxel_path).astype(np.float32)
    vox = vox.reshape(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1)
    return image, vox


def batch_loader(samples, batch_size: int, shuffle: bool):
    idxs = np.arange(len(samples))
    if shuffle:
        np.random.shuffle(idxs)

    usable = (len(idxs) // batch_size) * batch_size
    idxs = idxs[:usable]

    for start in range(0, usable, batch_size):
        batch_idxs = idxs[start : start + batch_size]
        images, voxels = [], []
        for idx in batch_idxs:
            img, vox = load_sample(*samples[int(idx)])
            images.append(img)
            voxels.append(vox)
        yield np.stack(images), np.stack(voxels)


def shard_batch(batch, n_devices: int):
    images, voxels = batch
    per_device = images.shape[0] // n_devices
    images = images.reshape(n_devices, per_device, IMAGE_SIZE, IMAGE_SIZE, 3)
    voxels = voxels.reshape(n_devices, per_device, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1)
    return images, voxels


def init_conv(key, in_ch, out_ch, kernel):
    k1, _ = jrandom.split(key)
    scale = math.sqrt(2.0 / (kernel * kernel * in_ch))
    w = scale * jrandom.normal(k1, (kernel, kernel, in_ch, out_ch), dtype=jnp.float32)
    b = jnp.zeros((out_ch,), dtype=jnp.float32)
    return {"w": w, "b": b}


def init_dense(key, in_dim, out_dim):
    k1, _ = jrandom.split(key)
    scale = math.sqrt(2.0 / in_dim)
    w = scale * jrandom.normal(k1, (in_dim, out_dim), dtype=jnp.float32)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": w, "b": b}


def init_params(seed=SEED):
    keys = jrandom.split(jrandom.PRNGKey(seed), 7)
    return {
        "conv1": init_conv(keys[0], 3, 32, 5),
        "conv2": init_conv(keys[1], 32, 64, 3),
        "conv3": init_conv(keys[2], 64, 128, 3),
        "conv4": init_conv(keys[3], 128, 256, 3),
        "fc1": init_dense(keys[4], 256, 512),
        "fc2": init_dense(keys[5], 512, VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE),
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


def forward(params, x):
    x = jax.nn.relu(conv2d(x, params["conv1"], 2))
    x = jax.nn.relu(conv2d(x, params["conv2"], 2))
    x = jax.nn.relu(conv2d(x, params["conv3"], 2))
    x = jax.nn.relu(conv2d(x, params["conv4"], 2))
    x = jnp.mean(x, axis=(1, 2))
    x = jax.nn.relu(dense(x, params["fc1"]))
    logits = dense(x, params["fc2"])
    return logits.reshape((-1, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1))


def loss_and_iou(params, images, voxels):
    logits = forward(params, images)
    bce = jnp.mean(jnp.maximum(logits, 0) - logits * voxels + jnp.log1p(jnp.exp(-jnp.abs(logits))))

    probs = jax.nn.sigmoid(logits)
    axes = tuple(range(1, probs.ndim))
    inter_soft = jnp.sum(probs * voxels, axis=axes)
    union_soft = jnp.sum(probs, axis=axes) + jnp.sum(voxels, axis=axes)
    dice = 1.0 - jnp.mean((2.0 * inter_soft + 1e-6) / (union_soft + 1e-6))

    pred = (probs > THRESHOLD).astype(jnp.float32)
    target = (voxels > 0.5).astype(jnp.float32)
    inter = jnp.sum(pred * target, axis=axes)
    union = jnp.sum(((pred + target) > 0).astype(jnp.float32), axis=axes)
    iou = jnp.mean((inter + 1e-6) / (union + 1e-6))
    return bce + dice, iou


def init_adam_state(params):
    zeros = tree_map(jnp.zeros_like, params)
    return {"step": jnp.array(0, dtype=jnp.int32), "m": zeros, "v": zeros}


def adamw_update(params, grads, state, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
    step = state["step"] + 1
    m = tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, state["m"], grads)
    v = tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), state["v"], grads)
    lr_t = lr * jnp.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)

    def upd(p, m_, v_):
        return p - lr_t * m_ / (jnp.sqrt(v_) + eps) - lr * WEIGHT_DECAY * p

    params = tree_map(upd, params, m, v)
    return params, {"step": step, "m": m, "v": v}


def train_step(params, opt_state, images, voxels):
    (loss, iou), grads = jax.value_and_grad(loss_and_iou, has_aux=True)(params, images, voxels)
    grads = lax.pmean(grads, axis_name="devices")
    loss = lax.pmean(loss, axis_name="devices")
    iou = lax.pmean(iou, axis_name="devices")
    params, opt_state = adamw_update(params, grads, opt_state)
    return params, opt_state, loss, iou


def eval_step(params, images, voxels):
    loss, iou = loss_and_iou(params, images, voxels)
    loss = lax.pmean(loss, axis_name="devices")
    iou = lax.pmean(iou, axis_name="devices")
    return loss, iou


p_train_step = jax.pmap(train_step, axis_name="devices")
p_eval_step = jax.pmap(eval_step, axis_name="devices")


def unreplicate(x):
    return tree_map(lambda y: np.asarray(y[0]), x)


def save_voxel_projection(vox: np.ndarray, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vox = np.squeeze(vox)
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
        logits = forward(params, jnp.asarray(image[None, ...]))
        pred = (np.asarray(jax.nn.sigmoid(logits[0])) > THRESHOLD).astype(np.float32)
        save_voxel_projection(pred, preview_dir / f"sample_{idx}_pred.png", "prediction")
        save_voxel_projection(target, preview_dir / f"sample_{idx}_gt.png", "ground truth")


def main():
    seed_everything(SEED)
    prepare_quick_subset()

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
    val_size = max(global_batch, int(0.15 * len(samples)))
    val_size = min((val_size // global_batch) * global_batch, len(samples) - global_batch)
    train_samples = samples[:-val_size]
    val_samples = samples[-val_size:]

    train_steps = len(train_samples) // global_batch
    val_steps = len(val_samples) // global_batch
    print(f"Samples: total={len(samples)}, train={len(train_samples)}, val={len(val_samples)}")
    print(f"Global batch size: {global_batch} ({PER_DEVICE_BATCH} x {n_devices})")
    print(f"Steps: train={train_steps}, val={val_steps}")

    params = init_params(SEED)
    opt_state = init_adam_state(params)
    params_repl = jax.device_put_replicated(params, devices)
    opt_state_repl = jax.device_put_replicated(opt_state, devices)

    best_iou = -1.0
    best_params = params

    for epoch in range(1, EPOCHS + 1):
        train_losses, train_ious = [], []
        for batch in batch_loader(train_samples, global_batch, shuffle=True):
            images, voxels = shard_batch(batch, n_devices)
            params_repl, opt_state_repl, loss, iou = p_train_step(params_repl, opt_state_repl, images, voxels)
            train_losses.append(float(np.asarray(loss[0])))
            train_ious.append(float(np.asarray(iou[0])))

        val_losses, val_ious = [], []
        for batch in batch_loader(val_samples, global_batch, shuffle=False):
            images, voxels = shard_batch(batch, n_devices)
            loss, iou = p_eval_step(params_repl, images, voxels)
            val_losses.append(float(np.asarray(loss[0])))
            val_ious.append(float(np.asarray(iou[0])))

        train_loss = float(np.mean(train_losses))
        train_iou = float(np.mean(train_ious))
        val_loss = float(np.mean(val_losses))
        val_iou = float(np.mean(val_ious))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            best_params = unreplicate(params_repl)
            leaves = jax.tree_util.tree_leaves(best_params)
            np.savez(RESULTS_DIR / "best_model_params.npz", **{f"arr_{i}": np.asarray(x) for i, x in enumerate(leaves)})

    save_predictions(best_params, val_samples)
    print(f"Done. Results saved to: {RESULTS_DIR}")
    print(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
