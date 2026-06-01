"""Dataset that serves individual frames from video latent .pt files."""

import os
import glob
import torch
from torch.utils.data import Dataset
from typing import Optional, List


import random

class LatentFrameDataset(Dataset):
    """
    Each .pt file has shape [T, C, H, W].
    This dataset lazily loads the .pt file based on the index to avoid OOM.
    It uses Dynamic Epoch Chunking: for each video accessed, it randomly
    slices a continuous chunk of `seq_len` frames.
    """

    def __init__(self, data_dir: str, file_list: Optional[List[str]] = None, data_percentage: float = 1.0, seq_len: int = 8):
        """
        Args:
            data_dir:  Directory containing .pt files.
            file_list: Optional explicit list of file paths. If None, all
                       .pt files in data_dir are used.
            data_percentage: Fraction of files to use (0.0 to 1.0]
            seq_len: Number of continuous frames per dynamic chunk
        """
        if file_list is not None:
            self.files = sorted(file_list)
        else:
            self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
            
        if data_percentage < 1.0:
            num_files = max(1, int(len(self.files) * data_percentage))
            self.files = self.files[:num_files]
            
        assert len(self.files) > 0, f"No .pt files found in {data_dir}"

        print(f"  Found {len(self.files)} video files for dynamic chunking (seq_len={seq_len}).")
        
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        tensor = torch.load(self.files[idx], map_location="cpu").float()

        T = tensor.shape[0]
        max_start = max(0, T - self.seq_len)
        start_idx = random.randint(0, max_start)

        chunk = tensor[start_idx : start_idx + self.seq_len]

        # Guard against exceptionally short videos by repeating the last frame
        if chunk.shape[0] < self.seq_len:
            pad = chunk[[-1]].repeat(self.seq_len - chunk.shape[0], 1, 1, 1)
            chunk = torch.cat([chunk, pad], dim=0)

        return chunk
