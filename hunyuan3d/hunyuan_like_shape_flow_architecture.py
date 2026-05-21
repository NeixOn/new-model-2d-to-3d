"""
Hunyuan3D-like architecture for single-image 3D shape generation.

This is a second architecture candidate, intentionally different from an
LRM/TripoSR triplane reconstructor.

Core idea:
  Stage 1: train a ShapeVAE that compresses each 3D object into latent tokens.
  Stage 2: train an image-conditioned rectified-flow DiT to generate those
           latent shape tokens from a single image.
  Stage 3: decode generated shape tokens to an SDF/occupancy field and extract
           a mesh with marching cubes.

This mirrors the public high-level Hunyuan3D family idea:
  image condition -> 3D latent shape generation with a diffusion/flow transformer
  -> shape decoder -> mesh.

It is not Tencent's code or weights. It is a research-friendly architecture
you can train in your project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ShapeFlowConfig:
    image_size: int = 224
    patch_size: int = 14
    image_dim: int = 768
    image_depth: int = 12
    image_heads: int = 12

    latent_tokens: int = 1024
    latent_dim: int = 768

    shape_encoder_points: int = 8192
    shape_encoder_width: int = 512

    vae_decoder_hidden: int = 512
    vae_decoder_layers: int = 6
    fourier_bands: int = 10

    flow_layers: int = 18
    flow_heads: int = 12
    flow_mlp_ratio: int = 4
    dropout: float = 0.05


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def fourier_encode(points: torch.Tensor, bands: int) -> torch.Tensor:
    freqs = (2.0 ** torch.arange(bands, device=points.device, dtype=points.dtype)) * math.pi
    xb = points[..., None, :] * freqs[:, None]
    enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
    return torch.cat([points, enc.flatten(-2)], dim=-1)


class ViTImageConditioner(nn.Module):
    def __init__(self, cfg: ShapeFlowConfig):
        super().__init__()
        if cfg.image_dim % cfg.image_heads != 0:
            raise ValueError("image_dim must be divisible by image_heads")
        self.grid = cfg.image_size // cfg.patch_size
        n = self.grid * self.grid
        self.patch = nn.Conv2d(3, cfg.image_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.image_dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, cfg.image_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.image_dim,
            nhead=cfg.image_heads,
            dim_feedforward=cfg.image_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.image_depth)
        self.norm = nn.LayerNorm(cfg.image_dim)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch(image).flatten(2).transpose(1, 2)
        cls = self.cls.expand(image.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        return self.norm(self.blocks(x))


class PointShapeEncoder(nn.Module):
    """Encodes a sampled shape into latent tokens.

    Input points should be sampled from/near the ground-truth mesh. Features can
    include xyz, normal, occupancy, sdf, or rgb. Minimal input is xyz + sdf.
    """

    def __init__(self, cfg: ShapeFlowConfig, point_feature_dim: int = 4):
        super().__init__()
        if cfg.latent_dim % cfg.flow_heads != 0:
            raise ValueError("latent_dim must be divisible by flow_heads")
        self.cfg = cfg
        self.point_mlp = nn.Sequential(
            nn.Linear(point_feature_dim, cfg.shape_encoder_width),
            nn.SiLU(),
            nn.Linear(cfg.shape_encoder_width, cfg.shape_encoder_width),
            nn.SiLU(),
            nn.Linear(cfg.shape_encoder_width, cfg.latent_dim),
        )
        self.latent_queries = nn.Parameter(torch.zeros(1, cfg.latent_tokens, cfg.latent_dim))
        self.cross = nn.MultiheadAttention(cfg.latent_dim, cfg.flow_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(cfg.latent_dim)
        self.norm_kv = nn.LayerNorm(cfg.latent_dim)
        self.to_mu = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

    def forward(self, shape_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = shape_points.shape[0]
        point_tokens = self.point_mlp(shape_points)
        q = self.latent_queries.expand(b, -1, -1)
        z = q + self.cross(self.norm_q(q), self.norm_kv(point_tokens), self.norm_kv(point_tokens), need_weights=False)[0]
        return self.to_mu(z), self.to_logvar(z)


class ShapeFieldDecoder(nn.Module):
    """Decodes latent shape tokens into SDF/occupancy at arbitrary 3D points."""

    def __init__(self, cfg: ShapeFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.point_proj = nn.Linear(3 + 2 * cfg.fourier_bands * 3, cfg.latent_dim)
        self.cross = nn.MultiheadAttention(cfg.latent_dim, cfg.flow_heads, batch_first=True)

        layers: list[nn.Module] = []
        in_dim = cfg.latent_dim * 2
        for i in range(cfg.vae_decoder_layers):
            layers.append(nn.Linear(in_dim if i == 0 else cfg.vae_decoder_hidden, cfg.vae_decoder_hidden))
            layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        self.sdf = nn.Linear(cfg.vae_decoder_hidden, 1)

    def forward(self, latent_tokens: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        p = self.point_proj(fourier_encode(query_points, self.cfg.fourier_bands))
        attended = self.cross(p, latent_tokens, latent_tokens, need_weights=False)[0]
        h = self.mlp(torch.cat([p, attended], dim=-1))
        return self.sdf(h)


class ShapeVAE(nn.Module):
    def __init__(self, cfg: ShapeFlowConfig):
        super().__init__()
        self.encoder = PointShapeEncoder(cfg)
        self.decoder = ShapeFieldDecoder(cfg)

    def encode(self, shape_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(shape_points)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, query_points)

    def forward(self, shape_points: torch.Tensor, query_points: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(shape_points)
        z = self.reparameterize(mu, logvar)
        sdf = self.decode(z, query_points)
        return {"sdf": sdf, "z": z, "mu": mu, "logvar": logvar}


class AdaLayerNormDiTBlock(nn.Module):
    def __init__(self, cfg: ShapeFlowConfig):
        super().__init__()
        if cfg.latent_dim % cfg.flow_heads != 0:
            raise ValueError("latent_dim must be divisible by flow_heads")
        d = cfg.latent_dim
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, cfg.flow_heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d, cfg.flow_heads, dropout=cfg.dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(d, d * cfg.flow_mlp_ratio),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * cfg.flow_mlp_ratio, d),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

    def forward(self, x: torch.Tensor, image_tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, g1, gate1, s2, g2, gate2 = self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s1, g1)
        x = x + gate1[:, None] * self.self_attn(h, h, h, need_weights=False)[0]

        h = self.norm2(x)
        x = x + self.cross_attn(h, image_tokens, image_tokens, need_weights=False)[0]

        h = modulate(self.norm3(x), s2, g2)
        x = x + gate2[:, None] * self.ff(h)
        return x


class ImageConditionedShapeFlowDiT(nn.Module):
    """Rectified-flow transformer over ShapeVAE latent tokens.

    Training target:
      z_t = (1 - t) * noise + t * z_target
      velocity_target = z_target - noise
      model predicts velocity(z_t, t, image)
    """

    def __init__(self, cfg: ShapeFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = ViTImageConditioner(cfg)
        self.image_to_latent = nn.Linear(cfg.image_dim, cfg.latent_dim)
        self.x_pos = nn.Parameter(torch.zeros(1, cfg.latent_tokens, cfg.latent_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.latent_dim * 4, cfg.latent_dim),
        )
        self.in_proj = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.blocks = nn.ModuleList([AdaLayerNormDiTBlock(cfg) for _ in range(cfg.flow_layers)])
        self.out_norm = nn.LayerNorm(cfg.latent_dim)
        self.out = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        nn.init.trunc_normal_(self.x_pos, std=0.02)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_to_latent(self.image_encoder(image))
        time = self.time_mlp(sinusoidal_timestep_embedding(t, self.cfg.latent_dim))
        x = self.in_proj(z_t) + self.x_pos
        for block in self.blocks:
            x = block(x, image_tokens, time)
        return self.out(self.out_norm(x))


def shape_vae_loss(
    pred_sdf: torch.Tensor,
    target_sdf: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-4,
) -> torch.Tensor:
    recon = F.smooth_l1_loss(pred_sdf, target_sdf)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl


def rectified_flow_loss(
    flow_model: ImageConditionedShapeFlowDiT,
    image: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    b = z_target.shape[0]
    t = torch.rand(b, device=z_target.device, dtype=z_target.dtype)
    noise = torch.randn_like(z_target)
    z_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * z_target
    velocity_target = z_target - noise
    velocity_pred = flow_model(z_t, t, image)
    return F.mse_loss(velocity_pred, velocity_target)


@torch.no_grad()
def euler_sample_flow(
    flow_model: ImageConditionedShapeFlowDiT,
    image: torch.Tensor,
    steps: int = 50,
) -> torch.Tensor:
    cfg = flow_model.cfg
    z = torch.randn(image.shape[0], cfg.latent_tokens, cfg.latent_dim, device=image.device, dtype=image.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((image.shape[0],), i / steps, device=image.device, dtype=image.dtype)
        v = flow_model(z, t, image)
        z = z + dt * v
    return z


def build_hunyuan_like_models() -> tuple[ShapeVAE, ImageConditionedShapeFlowDiT]:
    cfg = ShapeFlowConfig()
    return ShapeVAE(cfg), ImageConditionedShapeFlowDiT(cfg)


if __name__ == "__main__":
    cfg = ShapeFlowConfig(
        latent_tokens=128,
        latent_dim=256,
        image_dim=256,
        image_depth=2,
        image_heads=4,
        flow_layers=2,
        flow_heads=4,
    )
    vae = ShapeVAE(cfg)
    flow = ImageConditionedShapeFlowDiT(cfg)
    shape_points = torch.randn(2, 2048, 4)
    query_points = torch.randn(2, 4096, 3).clamp(-1, 1)
    image = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    out = vae(shape_points, query_points)
    loss = rectified_flow_loss(flow, image, out["mu"].detach())
    print(out["sdf"].shape, out["z"].shape, loss.item())
