# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import os
from tqdm import tqdm

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from utils.debug_option import DEBUG
import torch.distributed as dist

from ae.config import AEConfig
from ae.model import LatentAE


def avg_pool(latent_frame: torch.Tensor) -> torch.Tensor:
    """
    Compress a latent frame [B, C, H, W] into a descriptor [B, C] via spatial average pooling.
    
    This is the default compression method for computing latent descriptors
    used in memory token selection. Alternative methods can be swapped in
    by replacing this function.
    
    Args:
        latent_frame: Tensor of shape [B, C, H, W] (single frame latent)
    
    Returns:
        Tensor of shape [B, C] — the compressed descriptor
    """
    return latent_frame.mean(dim=(-2, -1))


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        if DEBUG:
            print(f"args.model_kwargs: {args.model_kwargs}")
        
        # Filter pipeline-specific settings out of model_kwargs so they don't reach the
        # WanDiffusionWrapper init.
        model_args_clean = dict(getattr(args, "model_kwargs", {}))
        for key in ["compression_method", "ae_ckpt", "recent_exclude"]:
            model_args_clean.pop(key, None)

        self.generator = WanDiffusionWrapper(
            **model_args_clean, is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # hard code for Wan2.1-T2V-1.3B
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        # Retrieval autoencoder (optional). compression_method ∈ {"avg_pool", "ae"}.
        self.compression_method = getattr(args.model_kwargs, "compression_method", "avg_pool")
        self.ae_model = None
        if self.compression_method == "ae":
            ae_ckpt = getattr(args.model_kwargs, "ae_ckpt", None)
            if ae_ckpt and os.path.exists(ae_ckpt):
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Loading LatentAE from {ae_ckpt} for compression...")
                import dataclasses
                ckpt = torch.load(ae_ckpt, map_location="cpu")
                ae_cfg_dict = ckpt["config"]

                # Sanitize old configs by keeping only fields that exist in the current AEConfig
                valid_keys = {f.name for f in dataclasses.fields(AEConfig)}
                ae_cfg_dict = {k: v for k, v in ae_cfg_dict.items() if k in valid_keys}

                ae_cfg = AEConfig(**ae_cfg_dict)
                self.ae_model = LatentAE(ae_cfg).to(device)
                self.ae_model.load_state_dict(ckpt["model"], strict=False)
                self.ae_model.eval()
            else:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Warning: ae_ckpt {ae_ckpt!r} not found; falling back to avg_pool.")
                self.compression_method = "avg_pool"

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        skip_vae_decode: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        memory_size_cfg = getattr(self.args.model_kwargs, "memory_size", 0)
        kv_policy = ""
        if memory_size_cfg > 0 and local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local+cpu_offload, size={local_attn_cfg} frames"
        elif local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # Initialize latent descriptor cache for memory token selection
        self.latent_descriptors = []  # List of [B, C] tensors (or [B, D] for AE)
        self.memory_indices_log = []  # Track memory selection
        # self.compression_method is already set in __init__
        sink_size = getattr(self.args.model_kwargs, "sink_size", 0)
        recent_exclude = getattr(self.args.model_kwargs, "recent_exclude", 0)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 2: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        pbar_blocks = tqdm(all_num_frames, desc=f"Generating blocks", disable=(dist.is_initialized() and dist.get_rank() != 0))
        for current_num_frames in pbar_blocks:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]

            # Step 2.0: Compute memory_indices from latent descriptors (shared across all layers)
            memory_indices = None
            if memory_size_cfg > 0:
                # Number of evicted frames in the CPU memory pool (same across all layers)
                num_evicted = len(self.kv_cache1[0].get("cpu_k_frames", []))
                # Exclude the `recent_exclude` most-recently-evicted frames from the
                # candidate pool (they sit right next to the local attention window).
                num_eligible = max(num_evicted - recent_exclude, 0)
                if num_eligible > 0 and len(self.latent_descriptors) > 0:
                    k_sel = min(memory_size_cfg, num_eligible)

                    evicted_descs = torch.stack([
                        self.latent_descriptors[sink_size + i]
                        for i in range(num_eligible)
                    ], dim=1)  # [B, num_eligible, C]

                    query_desc = self.latent_descriptors[-1].unsqueeze(1)  # [B, 1, C]

                    q_norm = query_desc / (query_desc.norm(dim=-1, keepdim=True) + 1e-8)
                    k_norm = evicted_descs / (evicted_descs.norm(dim=-1, keepdim=True) + 1e-8)
                    sims = torch.bmm(k_norm, q_norm.transpose(1, 2)).squeeze(-1)  # [B, num_eligible]

                    topk_sims, memory_indices = torch.topk(sims, k=k_sel, dim=-1)  # [B, k_sel]

                    global_frame_indices = memory_indices + sink_size
                    self.memory_indices_log.append({
                        "query_frame": current_start_frame,
                        "num_evicted": num_evicted,
                        "selected_pool_indices": memory_indices.cpu().tolist(),
                        "selected_global_frames": global_frame_indices.cpu().tolist(),
                        "selected_similarities": topk_sims.cpu().tolist(),
                        "compression_method": self.compression_method,
                    })

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")

                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        memory_indices=memory_indices
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        memory_indices=memory_indices
                    )
            # Step 2.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # Step 2.2.1: Store latent descriptors for newly denoised frames
            # denoised_pred shape: [B, current_num_frames, C, H, W]
            for f_idx in range(current_num_frames):
                frame = denoised_pred[:, f_idx]  # [B, C, H, W]
                if self.compression_method == "ae" and self.ae_model is not None:
                    desc = self.ae_model.encode(frame)  # [B, latent_dim]
                else:
                    desc = avg_pool(frame)              # [B, C]
                self.latent_descriptors.append(desc.detach())

            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * getattr(self.args, "context_noise", 0.0)
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                memory_indices=memory_indices
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 3: Decode the output
        if skip_vae_decode:
            video = None
            if profile:
                vae_end.record()
                torch.cuda.synchronize()
                vae_time = vae_start.elapsed_time(vae_end)
                total_time = init_time + diffusion_time + vae_time

                print("Profiling results:")
                print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
                print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
                for i, block_time in enumerate(block_times):
                    print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
                print(f"  - VAE decoding skipped")
                print(f"  - Total time: {total_time:.2f} ms")
        else:
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            if profile:
                # End VAE timing and synchronize CUDA
                vae_end.record()
                torch.cuda.synchronize()
                vae_time = vae_start.elapsed_time(vae_end)
                total_time = init_time + diffusion_time + vae_time

                print("Profiling results:")
                print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
                print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
                for i, block_time in enumerate(block_times):
                    print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
                print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
                print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "cpu_k_frames": [],
                "cpu_v_frames": []
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass