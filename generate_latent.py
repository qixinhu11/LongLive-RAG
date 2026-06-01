# Generate the AE training latent corpus from prompts.
#
# Sharding model:
#   - The launcher (generate_latent.sh) starts one process per physical GPU
#     with CUDA_VISIBLE_DEVICES isolating that GPU, so each process sees cuda:0
#     and only needs to know its logical shard index (--gpu_id) and the total
#     shard count (--num_shards).
#   - All shards independently sample the SAME set of prompts (seeded by
#     config.seed) and then deterministically slice it: shard i takes
#     positions {i, i + N, i + 2N, ...} of the sorted sampled indices.
#   - Each prompt's position in the sorted sampled list IS its filename
#     (latent_{global_pos:06d}.pt), so the union of all shards' outputs is a
#     flat, gap-free, collision-free dataset directory regardless of NGPU.
#
# Usage:
#   bash generate_latent.sh                   # parallel, 1-8 GPUs
#   python generate_latent.py --config_path configs/generate_latent.yaml \
#       --gpu_id 0 --num_shards 1             # single-GPU debug run
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import random

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
from torch.utils.data import DataLoader, SequentialSampler, Subset

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

# ─────────────────────────────────────────────────────────────────────────────
# CLI + config
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--gpu_id", type=int, default=0, help="Logical shard index (0..num_shards-1)")
parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards (== NGPU)")
parser.add_argument("--reverse", action="store_true", help="Process this shard's items in reverse order")
parser.add_argument("--skip_existing", action="store_true",
                    help="Skip global positions whose latent_{pos:06d}.pt already exists")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)
gpu_id = args.gpu_id
num_shards = args.num_shards
assert 0 <= gpu_id < num_shards, f"gpu_id must be in [0, num_shards). Got {gpu_id} / {num_shards}"

sample_ratio = getattr(config, "sample_ratio", 0.1)
max_video_saves = getattr(config, "max_video_saves", 2)
num_samples = getattr(config, "num_samples", 1)

# CUDA_VISIBLE_DEVICES (set by the launcher) maps one physical GPU to cuda:0
device = torch.device("cuda:0")
# Same seed across shards for torch RNG would couple their per-prompt noise;
# offset by gpu_id so each shard's torch.randn sequence is independent.
set_seed(config.seed + gpu_id)
torch.set_grad_enabled(False)

print(f"[shard {gpu_id}/{num_shards}] device={device}  free VRAM={get_cuda_free_memory_gb(device):.1f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline construction + checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────
low_memory = True
pipeline = CausalInferencePipeline(config, device=device)

if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        cleaned = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[shard {gpu_id}] {len(missing)} missing params, e.g. {missing[:4]}")
        if unexpected:
            print(f"[shard {gpu_id}] {len(unexpected)} unexpected params, e.g. {unexpected[:4]}")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# ── Optional LoRA ───────────────────────────────────────────────────────────
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) is not None and configure_lora_for_model is not None:
    if gpu_id == 0:
        print(f"[shard 0] LoRA enabled with config: {config.adapter}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(gpu_id == 0),
    )
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if gpu_id == 0:
            print(f"[shard 0] Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
    pipeline.is_lora_enabled = True

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

if pipeline.is_lora_enabled:
    model_type = "lora"
elif getattr(config, "use_ema", False):
    model_type = "ema"
else:
    model_type = "regular"

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic sampling + sharding
# ─────────────────────────────────────────────────────────────────────────────
full_dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=config.data_path)
num_total = len(full_dataset)
num_sample = max(1, int(num_total * sample_ratio))

# Re-seed Python's `random` with config.seed (NOT seed+gpu_id) so every shard
# draws the exact same sample. This is independent of torch RNG, which stays
# offset by gpu_id and only governs the per-prompt diffusion noise.
random.seed(config.seed)
sampled_indices = sorted(random.sample(range(num_total), num_sample))

# Stripe sampled_indices across shards. `g` (the position in the sorted list)
# doubles as the global file id, so the union of all shards' outputs is a
# flat, gap-free dataset `latent_000000.pt ... latent_{N-1:06d}.pt`.
my_pairs = [(g, idx) for g, idx in enumerate(sampled_indices) if g % num_shards == gpu_id]
if args.reverse:
    my_pairs = my_pairs[::-1]

# Build a Subset over the prompt indices in shard order
shard_prompt_idxs = [idx for _, idx in my_pairs]
dataset = Subset(full_dataset, shard_prompt_idxs)
dataloader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                        num_workers=0, drop_last=False)

print(f"[shard {gpu_id}/{num_shards}] total prompts={num_total}  "
      f"sampled {sample_ratio*100:.0f}%={len(sampled_indices)}  "
      f"this shard={len(my_pairs)}")

# ─────────────────────────────────────────────────────────────────────────────
# Output layout — flat dataset of latent_{global_pos:06d}.pt, matching toydatasets/
# ─────────────────────────────────────────────────────────────────────────────
latent_folder = config.output_folder
video_folder = os.path.join(config.output_folder, "_verification_videos")
os.makedirs(latent_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)
print(f"[shard {gpu_id}] latents → {latent_folder}/latent_XXXXXX.pt")
if max_video_saves > 0 and gpu_id == 0:
    print(f"[shard 0]   verification videos (≤{max_video_saves}, shard 0 only) → {video_folder}/")


def latent_path_for(global_pos: int, seed_idx: int) -> str:
    if num_samples == 1:
        return os.path.join(latent_folder, f"latent_{global_pos:06d}.pt")
    return os.path.join(latent_folder, f"latent_{global_pos:06d}_s{seed_idx}.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Generation loop
# ─────────────────────────────────────────────────────────────────────────────
videos_saved = 0

for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader),
                          desc=f"shard{gpu_id}", position=gpu_id):
    global_pos, prompt_idx = my_pairs[i]

    if args.skip_existing:
        # If every seed's latent already exists for this global_pos, skip.
        if all(os.path.exists(latent_path_for(global_pos, s)) for s in range(num_samples)):
            continue

    batch = batch_data[0] if isinstance(batch_data, list) else batch_data
    prompt = batch["prompts"][0]
    extended_prompt = batch.get("extended_prompts", [None])[0]
    prompts = [extended_prompt if extended_prompt is not None else prompt] * num_samples

    sampled_noise = torch.randn(
        [num_samples, config.num_output_frames, 16, 60, 104],
        device=device, dtype=torch.bfloat16,
    )

    # Only shard 0 saves a small number of verification videos to keep the
    # output count == max_video_saves rather than max_video_saves * num_shards.
    save_video = (gpu_id == 0 and videos_saved < max_video_saves)

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
        skip_vae_decode=(not save_video),
    )
    if save_video:
        pipeline.vae.model.clear_cache()

    for seed_idx in range(num_samples):
        torch.save(latents[seed_idx].cpu(), latent_path_for(global_pos, seed_idx))

        if save_video and video is not None:
            current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
            vid_tensor = 255.0 * current_video
            video_path = os.path.join(video_folder, f"latent_{global_pos:06d}_s{seed_idx}.mp4")
            write_video(video_path, vid_tensor[seed_idx], fps=16)
            videos_saved += 1
            print(f"[shard 0] saved verification video {video_path}")

print(f"[shard {gpu_id}] done. processed {len(my_pairs)} prompts → {latent_folder}")
