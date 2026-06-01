"""
Single-GPU training for the LongLive-RAG retrieval autoencoder.

Usage:
    bash train_ae_delta.sh
or directly:
    python -m ae.train --config ae/configs/ae_delta.yaml

Config loading order:  dataclass defaults → YAML file → CLI flags
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
    parser = build_argparser()
    args = parser.parse_args()

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

    # ── Dataset & DataLoaders ─────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LatentAE(cfg).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Encoder feat shape: {model.encoder._feat_shape}")
    print(f"Latent dim: {cfg.latent_dim}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {cfg.resume}  (epoch {start_epoch})")

    # ── Loss setup ────────────────────────────────────────────────────────────
    temporal_loss_fn = TemporalDeltaLoss(margin=cfg.delta_margin, weight=cfg.delta_weight)
    best_val_loss = float('inf')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        epoch_recon = 0.0
        epoch_delta_seq = 0.0
        epoch_smooth = 0.0
        n_batches = 0

        for step, chunk in enumerate(train_loader):
            chunk = chunk.to(device, non_blocking=True)  # [B, S, C, H, W]
            B, S, C, H_, W_ = chunk.shape
            x_flat = chunk.view(B * S, C, H_, W_)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp and device.type == "cuda"):
                recon, embed = model(x_flat)
                recon_loss = LatentAE.loss_function(recon, x_flat)

                # ── Window Temporal Delta Loss ──
                embed_seq = embed.view(B, S, -1)     # [B, S, D]

                loss_delta_seq = torch.tensor(0.0, device=device)
                max_window = min(S - 1, cfg.delta_window)
                for k in range(1, max_window + 1):
                    embed_t_k = embed_seq[:, k:, :]
                    embed_prev_k = embed_seq[:, :-k, :]
                    lt_k = temporal_loss_fn(
                        embed_t_k.reshape(-1, embed_t_k.shape[-1]),
                        embed_prev_k.reshape(-1, embed_prev_k.shape[-1])
                    )
                    loss_delta_seq += lt_k
                if max_window > 0:
                    loss_delta_seq /= max_window

                # ── Trajectory-smoothing (acceleration) loss ──
                loss_smooth = torch.tensor(0.0, device=device)
                if S >= 3 and cfg.smooth_weight > 0:
                    embed_t2 = embed_seq[:, 2:, :]
                    embed_t1 = embed_seq[:, 1:-1, :]
                    embed_t0 = embed_seq[:, :-2, :]
                    accel = embed_t2 - 2 * embed_t1 + embed_t0
                    loss_smooth = torch.sqrt(accel.float().pow(2).sum(dim=-1) + 1e-8).mean() * cfg.smooth_weight

                loss = recon_loss + loss_delta_seq + loss_smooth

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_recon += recon_loss.item()
            epoch_delta_seq += loss_delta_seq.item()
            epoch_smooth += loss_smooth.item()
            n_batches += 1

            if (step + 1) % cfg.log_every == 0:
                print(f"  [E{epoch:03d} S{step+1:04d}]  "
                      f"loss={loss.item():.6f}  "
                      f"recon={recon_loss.item():.6f}  "
                      f"dt={loss_delta_seq.item():.4f}  "
                      f"sm={loss_smooth.item():.4f}")
                if use_wandb:
                    global_step = epoch * len(train_loader) + step
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/recon_loss": recon_loss.item(),
                        "train/delta_seq": loss_delta_seq.item(),
                        "train/smooth": loss_smooth.item(),
                    }, step=global_step)

        avg_recon = epoch_recon / max(n_batches, 1)
        avg_dt    = epoch_delta_seq / max(n_batches, 1)
        avg_sm    = epoch_smooth / max(n_batches, 1)
        elapsed   = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_recon = 0.0
        val_dt    = 0.0
        val_sm    = 0.0
        val_n     = 0
        with torch.no_grad():
            for chunk in val_loader:
                chunk = chunk.to(device, non_blocking=True)
                B, S, C, H_, W_ = chunk.shape
                x_flat = chunk.view(B * S, C, H_, W_)

                with torch.amp.autocast("cuda", enabled=cfg.use_amp and device.type == "cuda"):
                    recon, embed = model(x_flat)
                    rl = LatentAE.loss_function(recon, x_flat)

                    embed_seq = embed.view(B, S, -1)
                    max_window = min(S - 1, cfg.delta_window)
                    lt = torch.tensor(0.0, device=device)
                    for k in range(1, max_window + 1):
                        embed_t_k = embed_seq[:, k:, :]
                        embed_prev_k = embed_seq[:, :-k, :]
                        lt += temporal_loss_fn(embed_t_k.reshape(-1, embed_t_k.shape[-1]),
                                               embed_prev_k.reshape(-1, embed_prev_k.shape[-1]))
                    if max_window > 0:
                        lt /= max_window

                    l_sm = torch.tensor(0.0, device=device)
                    if S >= 3 and cfg.smooth_weight > 0:
                        accel = embed_seq[:, 2:, :] - 2 * embed_seq[:, 1:-1, :] + embed_seq[:, :-2, :]
                        l_sm = torch.sqrt(accel.float().pow(2).sum(dim=-1) + 1e-8).mean() * cfg.smooth_weight

                val_recon += rl.item()
                val_dt    += lt.item()
                val_sm    += l_sm.item()
                val_n     += 1

        val_recon /= max(val_n, 1)
        val_dt    /= max(val_n, 1)
        val_sm    /= max(val_n, 1)
        val_total = val_recon + val_dt + val_sm

        print(f"Epoch {epoch:03d}  "
              f"tr_rec={avg_recon:.6f} tr_dt={avg_dt:.4f} tr_sm={avg_sm:.4f} | "
              f"val_rec={val_recon:.6f} val_dt={val_dt:.4f} val_sm={val_sm:.4f}  "
              f"time={elapsed:.1f}s")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch/train_recon": avg_recon,
                "epoch/train_delta_seq": avg_dt,
                "epoch/train_smooth": avg_sm,
                "epoch/train_total": avg_recon + avg_dt + avg_sm,
                "epoch/val_recon": val_recon,
                "epoch/val_delta_seq": val_dt,
                "epoch/val_smooth": val_sm,
                "epoch/val_total": val_total,
                "epoch/time_s": elapsed,
            }, step=(epoch + 1) * len(train_loader))

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_total < best_val_loss:
            best_val_loss = val_total
            ckpt_path = os.path.join(experiment_dir, "ae_epoch_best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": vars(cfg),
                "val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  → Saved NEW BEST checkpoint: {ckpt_path} (val_loss: {best_val_loss:.6f})")

        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            ckpt_path = os.path.join(experiment_dir, f"ae_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": vars(cfg),
            }, ckpt_path)
            print(f"  → Saved periodic checkpoint: {ckpt_path}")

    if use_wandb:
        wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
