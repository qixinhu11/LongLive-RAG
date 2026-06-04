"""Hyperparameter configuration for the Latent AE.

Supports three layers of configuration (later overrides earlier):
  1. Dataclass defaults
  2. YAML config file (--config path/to/config.yaml)
  3. CLI flags (--lr 1e-3 --epochs 500)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class AEConfig:
    # ── Experiment Tracking ───────────────────────────────────────────────────
    config: str = ""                # path to YAML config file
    tag: str = ""                   # human-readable experiment tag (e.g., "baseline", "high_lr")
    gpu: int = 0                    # CUDA device index

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = "datasets"
    val_split: float = 0.1          # fraction of files held out for validation
    num_workers: int = 4
    data_percentage: float = 1.0    # fraction of files to use for training/validation
    seq_len: int = 8                # number of continuous frames per dynamic chunk

    # ── Model ─────────────────────────────────────────────────────────────────
    in_channels: int = 16           # channel dim of each frame
    spatial_h: int = 60             # height of each frame
    spatial_w: int = 104            # width of each frame
    latent_dim: int = 1024          # target latent space dimensionality
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 128           # single-GPU batch
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 400


    # ── Window Temporal Delta + Smooth Loss ──────────────────────────────────
    delta_margin: float = 0.85      # m — cosine-similarity margin
    delta_weight: float = 1.0       # λ_Δ — Window Temporal Delta Loss weight
    delta_window: int = 3           # w — local window for temporal delta pairs
    smooth_weight: float = 1.0      # λ_smooth — trajectory-smoothing loss weight

    grad_clip: float = 1.0          # max gradient norm (0 = no clipping)
    use_amp: bool = True            # automatic mixed precision

    # ── Logging / Checkpointing ───────────────────────────────────────────────
    log_dir: str = "ae_runs"
    log_every: int = 20             # steps between console prints
    save_every: int = 50            # epochs between checkpoint saves
    resume: str = ""                # path to checkpoint to resume from

    # ── Wandb ─────────────────────────────────────────────────────────────────
    wandb_project: str = "latent-ae"        # wandb project name
    wandb_run_name: str = ""                # wandb run name (auto-generated if empty)
    wandb_entity: str = ""                  # wandb entity / team (default: personal)
    disable_wandb: bool = False             # set True to disable wandb logging

    # ── Seed ──────────────────────────────────────────────────────────────────
    seed: int = 42

    # ── Derived ───────────────────────────────────────────────────────────────
    def method_name(self) -> str:
        """Auto-derive the method name from the config."""
        name = "ae"
        if self.delta_weight > 0:
            name += "_delta"
        return name

    def experiment_name(self) -> str:
        """Build a descriptive experiment name for the directory and wandb.

        Format: {dataset}_{method}[_{tag}]
        Example: long_ae_delta_high_lr
        """
        data_name = os.path.basename(self.data_dir.strip('/'))
        name = f"{data_name}_{self.method_name()}"
        if self.tag:
            name += f"_{self.tag}"
        return name

    def key_params_str(self) -> str:
        """Compact string of the most important hyperparameters."""
        parts = [f"lr{self.lr}", f"bs{self.batch_size}", f"ld{self.latent_dim}"]
        if self.delta_weight > 0:
            parts.extend([f"dw{self.delta_weight}", f"m{self.delta_margin}",
                          f"win{self.delta_window}", f"sm{self.smooth_weight}"])
        return "_".join(parts)

    def to_yaml(self) -> str:
        """Serialise config to a YAML string (no dependency needed)."""
        import yaml
        return yaml.dump(vars(self), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "AEConfig":
        """Load config from a YAML file. Missing fields use dataclass defaults."""
        import yaml
        from dataclasses import fields as dc_fields
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        # Build a map of field name → expected Python type for coercion
        field_types = {f.name: f.type for f in dc_fields(cls)}
        for k, v in data.items():
            if hasattr(cfg, k):
                # Coerce to the declared type (handles e.g. "1e-4" → 0.0001)
                expected = field_types.get(k)
                if expected is float and not isinstance(v, float):
                    v = float(v)
                elif expected is int and not isinstance(v, (int, bool)):
                    v = int(v)
                setattr(cfg, k, v)
        return cfg

    def save(self, directory: str):
        """Save config as readable YAML + a copy of the source config (if any)."""
        import shutil
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "config.yaml"), "w") as f:
            f.write(self.to_yaml())
        # Copy the source YAML for exact reproducibility
        if self.config and os.path.isfile(self.config):
            shutil.copy2(self.config, os.path.join(directory, "source_config.yaml"))

    def summary(self) -> str:
        """Pretty-print for console output."""
        lines = [
            "=" * 60,
            f"  Experiment : {self.experiment_name()}",
            f"  Method     : {self.method_name()}",
            f"  Dataset    : {self.data_dir}",
            f"  Key params : {self.key_params_str()}",
            "-" * 60,
        ]
        for k, v in vars(self).items():
            lines.append(f"  {k:25s} = {v}")
        lines.append("=" * 60)
        return "\n".join(lines)
