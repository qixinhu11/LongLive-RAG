"""
Single-GPU training for the LongLive-RAG retrieval autoencoder.

Usage:
    bash train_ae_delta.sh
or directly:
    python -m ae.train --config ae/configs/ae_delta.yaml

Config loading order:  dataclass defaults → YAML file → CLI flags

Structure:
    load_config / build_argparser   – config resolution
    build_dataloaders               – dataset + train/val split
    compute_losses                  – forward pass + composite loss (shared by train & val)
    train_one_epoch / validate      – one pass over a loader
    save_checkpoint                 – uniform checkpoint writer
    main                            – orchestration
"""

import argparse
import os
import time
import random

import wandb

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import AEConfig
from .model import LatentAE, TemporalDeltaLoss
from .dataset import LatentFrameDataset


# ─────────────────────────────────────────────────────────────────────────────
# Config / CLI
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    """Create an argparser whose defaults come from the AEConfig dataclass."""
    defaults = AEConfig()
    parser = argparse.ArgumentParser(description="Train Latent AE")
    for field_name, field_val in vars(defaults).items():
        ftype = type(field_val)
        if ftype is bool:
            parser.add_argument(f"--{field_name}", type=lambda v: v.lower() in ("true", "1", "yes"),
                                default=None)
        elif ftype is list:
            parser.add_argument(f"--{field_name}", type=int, nargs="+", default=None)
        else:
            parser.add_argument(f"--{field_name}", type=ftype, default=None)
    return parser


def load_config() -> AEConfig:
    """Load config with 3-tier override: defaults → YAML → CLI."""
    args = build_argparser().parse_args()

    if args.config:
        cfg = AEConfig.from_yaml(args.config)
        cfg.config = args.config
    else:
        cfg = AEConfig()

    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    return cfg


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: AEConfig):
    """Build the train/val DataLoaders from a single latent-frame dataset."""
    print(f"\nLoading dataset from {cfg.data_dir} ...")
    full_dataset = LatentFrameDataset(cfg.data_dir, data_percentage=cfg.data_percentage, seq_len=cfg.seq_len)

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * cfg.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    print(f"  Total files : {n_total}")
    print(f"  Train / Val : {n_train} / {n_val}")
    print(f"  Batch size  : {cfg.batch_size}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Loss (shared by train and validation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_losses(model, chunk, cfg: AEConfig, temporal_loss_fn, device):
    """Forward pass + composite loss for one batch of frame chunks.

    Args:
        chunk: [B, S, C, H, W] continuous frame chunks.
    Returns:
        (total, recon, delta, smooth) loss tensors.
    """
    B, S, C, H, W = chunk.shape
    x_flat = chunk.view(B * S, C, H, W)

    recon, embed = model(x_flat)
    recon_loss = LatentAE.loss_function(recon, x_flat)

    embed_seq = embed.view(B, S, -1)                       # [B, S, D]
    embed_dim = embed_seq.shape[-1]

    # ── Window Temporal Delta loss: average hinge over temporal offsets 1..w ──
    delta_loss = torch.zeros((), device=device)
    max_window = min(S - 1, cfg.delta_window)
    for k in range(1, max_window + 1):
        cur = embed_seq[:, k:, :].reshape(-1, embed_dim)
        prev = embed_seq[:, :-k, :].reshape(-1, embed_dim)
        delta_loss = delta_loss + temporal_loss_fn(cur, prev)
    if max_window > 0:
        delta_loss = delta_loss / max_window

    # ── Trajectory-smoothing: penalise embedding acceleration ──
    smooth_loss = torch.zeros((), device=device)
    if S >= 3 and cfg.smooth_weight > 0:
        accel = embed_seq[:, 2:, :] - 2 * embed_seq[:, 1:-1, :] + embed_seq[:, :-2, :]
        smooth_loss = torch.sqrt(accel.float().pow(2).sum(dim=-1) + 1e-8).mean() * cfg.smooth_weight

    total = recon_loss + delta_loss + smooth_loss
    return total, recon_loss, delta_loss, smooth_loss


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, temporal_loss_fn, cfg, device, epoch, use_wandb):
    """Run one training epoch; returns the epoch-averaged loss components."""
    model.train()
    amp_enabled = cfg.use_amp and device.type == "cuda"
    sums = {"recon": 0.0, "delta": 0.0, "smooth": 0.0}
    n_batches = 0

    for step, chunk in enumerate(loader):
        chunk = chunk.to(device, non_blocking=True)        # [B, S, C, H, W]

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss, recon, delta, smooth = compute_losses(model, chunk, cfg, temporal_loss_fn, device)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        sums["recon"] += recon.item()
        sums["delta"] += delta.item()
        sums["smooth"] += smooth.item()
        n_batches += 1

        if (step + 1) % cfg.log_every == 0:
            print(f"  [E{epoch:03d} S{step+1:04d}]  "
                  f"loss={loss.item():.6f}  recon={recon.item():.6f}  "
                  f"dt={delta.item():.4f}  sm={smooth.item():.4f}")
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/recon_loss": recon.item(),
                    "train/delta_seq": delta.item(),
                    "train/smooth": smooth.item(),
                }, step=epoch * len(loader) + step)

    return {k: v / max(n_batches, 1) for k, v in sums.items()}


@torch.no_grad()
def validate(model, loader, temporal_loss_fn, cfg, device):
    """Run one validation pass; returns the averaged loss components."""
    model.eval()
    amp_enabled = cfg.use_amp and device.type == "cuda"
    sums = {"recon": 0.0, "delta": 0.0, "smooth": 0.0}
    n_batches = 0

    for chunk in loader:
        chunk = chunk.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            _, recon, delta, smooth = compute_losses(model, chunk, cfg, temporal_loss_fn, device)
        sums["recon"] += recon.item()
        sums["delta"] += delta.item()
        sums["smooth"] += smooth.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in sums.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scaler, cfg, epoch, **extra):
    """Write a checkpoint with model/optimizer/scaler state plus optional extras."""
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": vars(cfg),
        **extra,
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    set_seed(cfg.seed)

    if cfg.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu)
        device = torch.device(f"cuda:{cfg.gpu}")
    else:
        device = torch.device("cpu")

    # ── Experiment directory ──────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = cfg.experiment_name()
    experiment_dir = os.path.join(cfg.log_dir, experiment_name, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    cfg.save(experiment_dir)

    print(cfg.summary())
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Device: {device}")

    # ── Wandb ─────────────────────────────────────────────────────────────────
    use_wandb = not cfg.disable_wandb
    if use_wandb:
        run_name = cfg.wandb_run_name or f"{experiment_name}_{cfg.key_params_str()}"
        wandb.init(
            project=cfg.wandb_project,
            name=f"{run_name}_{timestamp}",
            entity=cfg.wandb_entity or None,
            config=vars(cfg),
            dir=experiment_dir,
            tags=[cfg.method_name(),
                  os.path.basename(cfg.data_dir.strip('/')),
                  *([] if not cfg.tag else [cfg.tag])],
        )
        print(f"Wandb run: {wandb.run.name}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)
    steps_per_epoch = len(train_loader)

    # ── Model / optimizer ─────────────────────────────────────────────────────
    model = LatentAE(cfg).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Encoder feat shape: {model.encoder._feat_shape}")
    print(f"Latent dim: {cfg.latent_dim}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")
    temporal_loss_fn = TemporalDeltaLoss(margin=cfg.delta_margin, weight=cfg.delta_weight)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {cfg.resume}  (epoch {start_epoch})")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scaler,
                             temporal_loss_fn, cfg, device, epoch, use_wandb)
        val = validate(model, val_loader, temporal_loss_fn, cfg, device)
        elapsed = time.time() - t0

        tr_total = tr["recon"] + tr["delta"] + tr["smooth"]
        val_total = val["recon"] + val["delta"] + val["smooth"]

        print(f"Epoch {epoch:03d}  "
              f"tr_rec={tr['recon']:.6f} tr_dt={tr['delta']:.4f} tr_sm={tr['smooth']:.4f} | "
              f"val_rec={val['recon']:.6f} val_dt={val['delta']:.4f} val_sm={val['smooth']:.4f}  "
              f"time={elapsed:.1f}s")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch/train_recon": tr["recon"],
                "epoch/train_delta_seq": tr["delta"],
                "epoch/train_smooth": tr["smooth"],
                "epoch/train_total": tr_total,
                "epoch/val_recon": val["recon"],
                "epoch/val_delta_seq": val["delta"],
                "epoch/val_smooth": val["smooth"],
                "epoch/val_total": val_total,
                "epoch/time_s": elapsed,
            }, step=(epoch + 1) * steps_per_epoch)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_total < best_val_loss:
            best_val_loss = val_total
            ckpt_path = os.path.join(experiment_dir, "ae_epoch_best.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, cfg, epoch, val_loss=best_val_loss)
            print(f"  → Saved NEW BEST checkpoint: {ckpt_path} (val_loss: {best_val_loss:.6f})")

        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            ckpt_path = os.path.join(experiment_dir, f"ae_epoch{epoch:03d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, cfg, epoch)
            print(f"  → Saved periodic checkpoint: {ckpt_path}")

    if use_wandb:
        wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
