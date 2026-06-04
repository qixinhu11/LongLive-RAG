"""Convolutional AutoEncoder for compressing [C, H, W] latent frames to a 1-D vector.

Designed for retrieval over WAN latent features with:
- GroupNorm (instance-independent, no train/eval distribution mismatch)
- Residual blocks for gradient flow
- Global Average Pooling instead of flatten (massive param reduction)
- Upsample+Conv decoder (no checkerboard artifacts from ConvTranspose2d)
- Optional L2 normalization for cosine-similarity retrieval

Note: No spatial attention — this AE operates on individual frames independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .config import AEConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _num_groups(channels: int, preferred: int = 32) -> int:
    """Pick a valid group count for GroupNorm."""
    for g in (preferred, 16, 8, 4, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Pre-norm residual block: GN → SiLU → Conv → GN → SiLU → Conv + skip."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_num_groups(dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(_num_groups(dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        # Zero-init last conv so the block starts as identity
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Progressively downsamples [C, H, W] → GAP → embedding vector."""

    def __init__(self, in_channels: int, hidden_dims: List[int],
                 latent_dim: int, spatial_h: int, spatial_w: int):
        super().__init__()
        # Build downsample + residual stack
        layers: list[nn.Module] = []
        ch = in_channels
        for hd in hidden_dims:
            layers.append(nn.Conv2d(ch, hd, 3, stride=2, padding=1))
            layers.append(nn.GroupNorm(_num_groups(hd), hd))
            layers.append(nn.SiLU(inplace=True))
            layers.append(ResBlock(hd))
            ch = hd
        self.conv_stack = nn.Sequential(*layers)

        # Final norm before pooling — stabilises feature scales across channels
        self.final_norm = nn.Sequential(
            nn.GroupNorm(_num_groups(ch), ch),
            nn.SiLU(inplace=True),
        )

        # Global Average Pooling → feature vector
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Projection to embedding space
        self.fc = nn.Linear(ch, latent_dim)

        self._bottleneck_channels = ch

        # Compute bottleneck spatial size (needed by decoder to reshape)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, spatial_h, spatial_w)
            out = self.conv_stack(dummy)
            self._feat_shape = out.shape[1:]   # (C', H', W')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C, H, W] where N = B*S (flattened batch × sequence)
        Returns:
            embed: [N, latent_dim]
        """
        h = self.conv_stack(x)
        h = self.final_norm(h)
        h = self.gap(h).flatten(1)             # [N, C']
        return self.fc(h)


class Decoder(nn.Module):
    """Maps embedding (1-D) back to [C, H, W] via upsample + conv (no checkerboard)."""

    def __init__(self, latent_dim: int, hidden_dims: List[int],
                 out_channels: int, feat_shape: Tuple[int, int, int],
                 target_h: int, target_w: int):
        super().__init__()
        self._feat_shape = feat_shape            # (C', H', W')
        flat_size = 1
        for s in feat_shape:
            flat_size *= s

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, flat_size),
            nn.SiLU(inplace=True),
        )

        # Build upsample stack (reverse of encoder dims)
        rev = list(reversed(hidden_dims))
        layers: list[nn.Module] = []
        for i in range(len(rev) - 1):
            layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(rev[i], rev[i + 1], 3, padding=1),
                nn.GroupNorm(_num_groups(rev[i + 1]), rev[i + 1]),
                nn.SiLU(inplace=True),
                ResBlock(rev[i + 1]),
            ])
        self.upsample_stack = nn.Sequential(*layers)

        # Final upsample: back to original channels, no activation
        # (latent values are unbounded, so no sigmoid/tanh)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(rev[-1], out_channels, 3, padding=1),
        )

        self._target_h = target_h
        self._target_w = target_w

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, *self._feat_shape)
        h = self.upsample_stack(h)
        h = self.final(h)
        # Crop to exact target spatial size
        h = h[:, :, :self._target_h, :self._target_w]
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Full AE
# ─────────────────────────────────────────────────────────────────────────────

class LatentAE(nn.Module):
    """
    AutoEncoder for video latent frames (retrieval-optimised).

    Input:  (B*S, C, H, W)  e.g. (B*S, 16, 60, 104)
    Latent: (B*S, latent_dim)  e.g. (B*S, 1024)
    Output: (B*S, C, H, W)  — reconstruction
    """

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(
            in_channels=cfg.in_channels,
            hidden_dims=cfg.hidden_dims,
            latent_dim=cfg.latent_dim,
            spatial_h=cfg.spatial_h,
            spatial_w=cfg.spatial_w,
        )

        self.decoder = Decoder(
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            out_channels=cfg.in_channels,
            feat_shape=self.encoder._feat_shape,
            target_h=cfg.spatial_h,
            target_w=cfg.spatial_w,
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Re-apply zero-init for residual output projections
        # (the global kaiming/xavier init above overwrites their __init__ zero-init)
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.block[-1].weight)
                nn.init.zeros_(m.block[-1].bias)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, C, H, W] where N = B*S
        Returns:
            recon: [N, C, H, W] — reconstruction
            embed: [N, latent_dim] — embedding vector
        """
        embed = self.encoder(x)
        recon = self.decoder(embed)
        return recon, embed

    # ── Encode only (for downstream retrieval) ────────────────────────────────
    @torch.no_grad()
    def encode(self, x: torch.Tensor,
               normalize: bool = True) -> torch.Tensor:
        """Return the embedding, optionally L2-normalised for retrieval.

        Args:
            x: [N, C, H, W] where N = B*S
            normalize: L2-normalize for cosine similarity retrieval
        """
        embed = self.encoder(x)
        if normalize:
            embed = F.normalize(embed, dim=-1)
        return embed

    # ── Loss ──────────────────────────────────────────────────────────────────
    @staticmethod
    def loss_function(recon: torch.Tensor, target: torch.Tensor):
        """
        Returns:
            recon_loss (scalar tensor)
        """
        return F.mse_loss(recon.float(), target.float(), reduction='mean')


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Loss (AE Regularizer)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalDeltaLoss(nn.Module):
    """
    Penalizes redundancy across time by pushing temporally adjacent frames
    (within a window) to differ in embedding space, forcing the AE to encode
    delta / motion semantics rather than redundant static similarity.
    """
    def __init__(self, margin: float = 0.85, weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(self, v_t: torch.Tensor, v_ref: torch.Tensor) -> torch.Tensor:
        """
        v_t, v_ref: [batch_size, embedding_dim]
        Hinge loss: max(0, sim - margin)
        """
        sim = F.cosine_similarity(v_t, v_ref, dim=-1)
        penalty = F.relu(sim - self.margin)
        return self.weight * penalty.mean()
