"""
Parallel Kaggle TPU v5e-8 smoke training for single-image 3D reconstruction.

This version uses TensorFlow TPUStrategy because Kaggle's current PyTorch/XLA
runtime can fail multi-process initialization on v5e with:
  Expected 8 worker addresses, got 1

Dataset expected in Kaggle:
  /kaggle/input/datasets/sirish001/shapenet-3dr2n2/ShapeNetRendering.tgz
  /kaggle/input/datasets/sirish001/shapenet-3dr2n2/ShapeNetVox32.tgz

Run:
  %cd /kaggle/working/new-model-2d-to-3d
  %run kaggle_tpu_v5e8_tf_quick_3d_recon.py
"""

import os
import random
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

import tensorflow as tf


SEED = 42

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "shapenet_r2n2_quick"
RESULTS_DIR = WORK_DIR / "quick_3d_recon_results_tf"

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
GLOBAL_BATCH_SIZE = 64
LR = 2e-4
THRESHOLD = 0.4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


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
            samples.append((str(image_path), str(vox_path)))

    if not samples:
        raise RuntimeError(f"No paired image/binvox samples found under {root}")

    random.shuffle(samples)
    return samples


def load_sample(image_path: bytes, voxel_path: bytes):
    image = Image.open(image_path.decode("utf-8")).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 127.5 - 1.0

    vox = read_binvox(Path(voxel_path.decode("utf-8")))
    vox = np.expand_dims(vox, axis=-1).astype(np.float32)
    return image, vox


def make_dataset(samples, training: bool):
    image_paths = [s[0] for s in samples]
    voxel_paths = [s[1] for s in samples]
    ds = tf.data.Dataset.from_tensor_slices((image_paths, voxel_paths))
    if training:
        ds = ds.shuffle(len(samples), seed=SEED, reshuffle_each_iteration=True)

    def mapper(image_path, voxel_path):
        image, vox = tf.numpy_function(
            load_sample,
            [image_path, voxel_path],
            [tf.float32, tf.float32],
        )
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        vox.set_shape((VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1))
        return image, vox

    ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(GLOBAL_BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    image_in = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")

    x = tf.keras.layers.Conv2D(32, 5, strides=2, padding="same", activation="relu")(image_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(256 * 4 * 4 * 4, activation="relu")(x)
    x = tf.keras.layers.Reshape((4, 4, 4, 256))(x)
    x = tf.keras.layers.Conv3DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv3DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv3DTranspose(32, 4, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    logits = tf.keras.layers.Conv3D(1, 3, padding="same", name="voxel_logits")(x)

    return tf.keras.Model(image_in, logits)


def voxel_loss(y_true, y_pred):
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    bce = tf.reduce_mean(bce)

    probs = tf.sigmoid(y_pred)
    axes = tuple(range(1, len(probs.shape)))
    inter = tf.reduce_sum(probs * y_true, axis=axes)
    union = tf.reduce_sum(probs, axis=axes) + tf.reduce_sum(y_true, axis=axes)
    dice = 1.0 - tf.reduce_mean((2.0 * inter + 1e-6) / (union + 1e-6))
    return bce + dice


def voxel_iou(y_true, y_pred):
    pred = tf.cast(tf.sigmoid(y_pred) > THRESHOLD, tf.float32)
    target = tf.cast(y_true > 0.5, tf.float32)
    axes = tuple(range(1, len(pred.shape)))
    inter = tf.reduce_sum(pred * target, axis=axes)
    union = tf.reduce_sum(tf.cast((pred + target) > 0, tf.float32), axis=axes)
    return tf.reduce_mean((inter + 1e-6) / (union + 1e-6))


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


def save_predictions(model, samples):
    preview_dir = RESULTS_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for idx, (image_path, voxel_path) in enumerate(samples[:6]):
        image, target = load_sample(image_path.encode("utf-8"), voxel_path.encode("utf-8"))
        logits = model.predict(image[None, ...], verbose=0)[0]
        pred = (1.0 / (1.0 + np.exp(-logits)) > THRESHOLD).astype(np.float32)
        save_voxel_projection(pred, preview_dir / f"sample_{idx}_pred.png", "prediction")
        save_voxel_projection(target, preview_dir / f"sample_{idx}_gt.png", "ground truth")


def get_tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print(f"TPU replicas in sync: {strategy.num_replicas_in_sync}")
    return strategy


def main():
    seed_everything(SEED)
    prepare_quick_subset()

    samples = discover_samples(DATA_DIR)
    val_size = max(1, int(0.15 * len(samples)))
    train_samples = samples[:-val_size]
    val_samples = samples[-val_size:]

    train_ds = make_dataset(train_samples, training=True)
    val_ds = make_dataset(val_samples, training=False)

    strategy = get_tpu_strategy()
    with strategy.scope():
        model = build_model()
        optimizer = tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss=voxel_loss, metrics=[voxel_iou])

    print(f"Samples: total={len(samples)}, train={len(train_samples)}, val={len(val_samples)}")
    print(f"Global batch size: {GLOBAL_BATCH_SIZE}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(RESULTS_DIR / "best_model.weights.h5"),
            monitor="val_voxel_iou",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(str(RESULTS_DIR / "saved_model.keras"))
    save_predictions(model, val_samples)

    best_iou = max(history.history.get("val_voxel_iou", [0.0]))
    print(f"Done. Results saved to: {RESULTS_DIR}")
    print(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
