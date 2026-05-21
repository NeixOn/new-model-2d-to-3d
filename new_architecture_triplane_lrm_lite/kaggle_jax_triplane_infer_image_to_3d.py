"""
Inference for Triplane Occupancy / LRM-lite.

Run after training:
  %cd /kaggle/working/new-model-2d-to-3d/new_architecture_triplane_lrm_lite
  %run kaggle_jax_triplane_infer_image_to_3d.py --image /path/to/image.png

Outputs:
  /kaggle/working/triplane_lrm_lite_single_image/prediction.obj
  /kaggle/working/triplane_lrm_lite_single_image/prediction_preview.png
  /kaggle/working/triplane_lrm_lite_single_image/prediction_voxels.npy
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jrandom


SEED = 42
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "128"))
VOXEL_SIZE = int(os.environ.get("VOXEL_SIZE", "32"))
TRIPLANE_RES = int(os.environ.get("TRIPLANE_RES", "32"))
TRIPLANE_CHANNELS = int(os.environ.get("TRIPLANE_CHANNELS", "16"))
MLP_HIDDEN = int(os.environ.get("MLP_HIDDEN", "128"))

DEFAULT_WEIGHTS = Path("/kaggle/working/triplane_lrm_lite_results/best_model_params.npz")
DEFAULT_OUT_DIR = Path("/kaggle/working/triplane_lrm_lite_single_image")


def make_query_grid():
    coords = np.stack(np.meshgrid(
        np.linspace(-1.0, 1.0, VOXEL_SIZE, dtype=np.float32),
        np.linspace(-1.0, 1.0, VOXEL_SIZE, dtype=np.float32),
        np.linspace(-1.0, 1.0, VOXEL_SIZE, dtype=np.float32),
        indexing="ij",
    ), axis=-1)
    return coords.reshape(-1, 3)


FULL_QUERY_GRID = make_query_grid()


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


def load_params(weights_path: Path):
    template = init_params()
    treedef = jax.tree_util.tree_structure(template)
    n_leaves = len(jax.tree_util.tree_leaves(template))
    data = np.load(weights_path)
    leaves = [jnp.asarray(data[f"arr_{i}"]) for i in range(n_leaves)]
    return jax.tree_util.tree_unflatten(treedef, leaves)


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
    return f00 * (1 - wu) * (1 - wv) + f10 * wu * (1 - wv) + f01 * (1 - wu) * wv + f11 * wu * wv


def positional_encoding(points):
    return jnp.concatenate([points, jnp.sin(math.pi * points), jnp.cos(math.pi * points)], axis=-1)


def query_occupancy(params, triplanes, points):
    xy = bilinear_sample(triplanes[:, 0], points[..., [0, 1]])
    xz = bilinear_sample(triplanes[:, 1], points[..., [0, 2]])
    yz = bilinear_sample(triplanes[:, 2], points[..., [1, 2]])
    feats = jnp.concatenate([xy, xz, yz, positional_encoding(points)], axis=-1)
    x = jax.nn.gelu(dense(feats, params["mlp1"]))
    x = jax.nn.gelu(dense(x, params["mlp2"]))
    return dense(x, params["mlp3"])


def forward(params, images, points):
    triplanes = encode_to_triplanes(params, images)
    return query_occupancy(params, triplanes, points)


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
    occupied = np.argwhere(voxels > 0)
    if occupied.size == 0:
        raise RuntimeError("No occupied voxels at this threshold. Try --threshold 0.25")

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

    scale = 1.0 / VOXEL_SIZE
    vertex_offset = 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("# triplane occupancy voxel mesh\n")
        for x, y, z in occupied:
            base = (np.array([x, y, z], dtype=np.float32) - VOXEL_SIZE / 2) * scale
            for v in base + cube_vertices * scale:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in cube_faces:
                f.write("f " + " ".join(str(i + vertex_offset - 1) for i in face) + "\n")
            vertex_offset += 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--threshold", type=float, default=0.4)
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
    points = jnp.asarray(FULL_QUERY_GRID[None, ...], dtype=jnp.float32)
    logits = forward(params, jnp.asarray(image), points)
    probs = np.asarray(jax.nn.sigmoid(logits[0, :, 0])).reshape(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
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


if __name__ == "__main__":
    main()
