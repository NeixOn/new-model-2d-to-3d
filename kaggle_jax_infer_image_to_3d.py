"""
Inference for the JAX single-image 3D reconstruction baseline.

Run in Kaggle after training:
  %cd /kaggle/working/new-model-2d-to-3d
  %run kaggle_jax_infer_image_to_3d.py --image /path/to/image.png

Optional:
  %run kaggle_jax_infer_image_to_3d.py --image /path/to/image.png --threshold 0.4 --out-dir /kaggle/working/my_recon

Outputs:
  prediction.obj
  prediction_voxels.npy
  prediction_preview.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jrandom


IMAGE_SIZE = 128
VOXEL_SIZE = 32
SEED = 42
DEFAULT_WEIGHTS = Path("/kaggle/working/quick_3d_recon_results_jax/best_model_params.npz")
DEFAULT_OUT_DIR = Path("/kaggle/working/single_image_reconstruction")


def init_conv(key, in_ch, out_ch, kernel):
    scale = (2.0 / (kernel * kernel * in_ch)) ** 0.5
    w = scale * jrandom.normal(key, (kernel, kernel, in_ch, out_ch), dtype=jnp.float32)
    b = jnp.zeros((out_ch,), dtype=jnp.float32)
    return {"w": w, "b": b}


def init_dense(key, in_dim, out_dim):
    scale = (2.0 / in_dim) ** 0.5
    w = scale * jrandom.normal(key, (in_dim, out_dim), dtype=jnp.float32)
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


def load_params(weights_path: Path):
    template = init_params()
    treedef = jax.tree_util.tree_structure(template)
    n_leaves = len(jax.tree_util.tree_leaves(template))

    data = np.load(weights_path)
    leaves = [jnp.asarray(data[f"arr_{i}"]) for i in range(n_leaves)]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def load_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return image[None, ...]


def save_preview(voxels: np.ndarray, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj_xy = voxels.max(axis=0)
    proj_xz = voxels.max(axis=1)
    proj_yz = voxels.max(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, arr, title in zip(axes, [proj_xy, proj_xz, proj_yz], ["xy", "xz", "yz"]):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_voxel_obj(voxels: np.ndarray, path: Path):
    """Write occupied voxels as a simple cube mesh OBJ."""
    occupied = np.argwhere(voxels > 0)
    if occupied.size == 0:
        raise RuntimeError("No occupied voxels at this threshold. Try --threshold 0.25")

    vertices = []
    faces = []
    vertex_offset = 1
    scale = 1.0 / VOXEL_SIZE

    cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    cube_faces = [
        [1, 2, 3, 4],
        [5, 8, 7, 6],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 8, 4],
        [4, 8, 5, 1],
    ]

    for x, y, z in occupied:
        base = (np.array([x, y, z], dtype=np.float32) - VOXEL_SIZE / 2) * scale
        vertices.extend((base + cube_vertices * scale).tolist())
        faces.extend([[idx + vertex_offset - 1 for idx in face] for face in cube_faces])
        vertex_offset += 8

    with open(path, "w", encoding="utf-8") as f:
        f.write("# voxel reconstruction\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to best_model_params.npz")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for output files")
    parser.add_argument("--threshold", type=float, default=0.4, help="Occupancy threshold")
    args = parser.parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    params = load_params(weights_path)
    image = load_image(image_path)

    logits = forward(params, jnp.asarray(image))
    probs = np.asarray(jax.nn.sigmoid(logits[0, ..., 0]))
    voxels = (probs > args.threshold).astype(np.float32)

    np.save(out_dir / "prediction_voxels.npy", probs)
    save_preview(voxels, out_dir / "prediction_preview.png")
    write_voxel_obj(voxels, out_dir / "prediction.obj")

    print(f"Input image: {image_path}")
    print(f"Weights: {weights_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Occupied voxels: {int(voxels.sum())} / {VOXEL_SIZE ** 3}")
    print(f"Saved OBJ: {out_dir / 'prediction.obj'}")
    print(f"Saved preview: {out_dir / 'prediction_preview.png'}")
    print(f"Saved probabilities: {out_dir / 'prediction_voxels.npy'}")


if __name__ == "__main__":
    main()
