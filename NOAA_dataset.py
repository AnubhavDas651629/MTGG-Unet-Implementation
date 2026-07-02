"""
NOAA Tornado Dataset — Temporal Sliding Window

Reads the same per-day .npy files (CAPE, CIN, Geopotential Height, Tornado, SigTor)
from the Weather-Forecasting-Unet project, but creates temporal SEQUENCES of
consecutive days for the UNet-MTGNN hybrid model.

Data folder structure expected:
    data/
    ├── train/
    │   ├── cape/       ← 2014-01-01.npy, 2014-01-02.npy, ...
    │   ├── cin/        ← same filenames
    │   └── geo/        ← same filenames
    ├── train_masks/
    │   ├── tornado/    ← same filenames
    │   └── sigtor/     ← same filenames

Each .npy file is a 2D float32 array that gets resized to (256, 256) at load time.

Output per sample:
    x: (seq_in, 3, 256, 256)   — seq_in consecutive days of [CAPE, CIN, Geo]
    y: (seq_out, 2, 256, 256)  — next seq_out days of [tornado_prob, sigtor_prob]
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NOAATornadoTemporalDataset(Dataset):
    """
    Creates temporal sliding-window samples from daily NOAA .npy weather grids.

    Given a chronological list of daily files, builds (input_window, target_window)
    pairs where:
        - input_window  = seq_in  consecutive days of 3-channel weather maps
        - target_window = seq_out consecutive days of 2-channel tornado probability maps

    Args:
        root_path:  Path to the data directory (e.g., './data')
        seq_in:     Number of past days to use as input (default: 7)
        seq_out:    Number of future days to predict (default: 3)
        file_list:  Optional list of filenames to use (for train/val splitting).
                    If None, uses all files found in the cape folder.
        test:       If True, reads from 'manual_test' folders instead of 'train'.
        grid_size:  Target spatial resolution (default: 256). Must be divisible by 16.
    """

    def __init__(self, root_path, seq_in=7, seq_out=3, file_list=None,
                 test=False, grid_size=256):
        self.root_path = root_path
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.grid_size = grid_size

        folder_prefix = "manual_test" if test else "train"

        # Get sorted list of daily filenames (chronological order)
        if file_list is None:
            cape_dir = os.path.join(root_path, folder_prefix, "cape")
            self.file_ids = sorted(
                f for f in os.listdir(cape_dir) if f.endswith('.npy')
            )
        else:
            self.file_ids = sorted(file_list)

        # Build full paths for each channel
        self.cape_paths = [
            os.path.join(root_path, folder_prefix, "cape", f) for f in self.file_ids
        ]
        self.cin_paths = [
            os.path.join(root_path, folder_prefix, "cin", f) for f in self.file_ids
        ]
        self.geo_paths = [
            os.path.join(root_path, folder_prefix, "geo", f) for f in self.file_ids
        ]
        self.tor_paths = [
            os.path.join(root_path, f"{folder_prefix}_masks", "tornado", f)
            for f in self.file_ids
        ]
        self.sigtor_paths = [
            os.path.join(root_path, f"{folder_prefix}_masks", "sigtor", f)
            for f in self.file_ids
        ]

        # Total number of sliding window samples
        # We need seq_in days for input + seq_out days for target
        total_days = len(self.file_ids)
        self.num_samples = total_days - seq_in - seq_out + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough days ({total_days}) for seq_in={seq_in} + seq_out={seq_out}. "
                f"Need at least {seq_in + seq_out} days."
            )

        print(f"[Dataset] {total_days} days → {self.num_samples} sliding window samples "
              f"(seq_in={seq_in}, seq_out={seq_out})")

    def _load_and_resize(self, path):
        """Load a .npy file and resize to (grid_size, grid_size)."""
        arr = np.load(path).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Resize if needed
        if arr.shape[0] != self.grid_size or arr.shape[1] != self.grid_size:
            tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            tensor = F.interpolate(
                tensor, size=(self.grid_size, self.grid_size),
                mode='bilinear', align_corners=False
            )
            arr = tensor.squeeze().numpy()

        return arr

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            x: (seq_in, 3, grid_size, grid_size)  — input weather sequence
            y: (seq_out, 2, grid_size, grid_size)  — target tornado probability sequence
        """
        # Build input sequence: days [idx, idx+1, ..., idx+seq_in-1]
        x_frames = []
        for t in range(self.seq_in):
            day_idx = idx + t
            cape = self._load_and_resize(self.cape_paths[day_idx])
            cin = self._load_and_resize(self.cin_paths[day_idx])
            geo = self._load_and_resize(self.geo_paths[day_idx])

            # Stack 3 channels: (3, H, W)
            frame = np.stack([cape, cin, geo], axis=0)
            x_frames.append(frame)

        # Build target sequence: days [idx+seq_in, ..., idx+seq_in+seq_out-1]
        y_frames = []
        for t in range(self.seq_out):
            day_idx = idx + self.seq_in + t
            tor = self._load_and_resize(self.tor_paths[day_idx])
            sigtor = self._load_and_resize(self.sigtor_paths[day_idx])

            # Stack 2 channels: (2, H, W)
            frame = np.stack([tor, sigtor], axis=0)
            y_frames.append(frame)

        # Stack over time: (seq_in, 3, H, W) and (seq_out, 2, H, W)
        x = torch.tensor(np.stack(x_frames, axis=0), dtype=torch.float32)
        y = torch.tensor(np.stack(y_frames, axis=0), dtype=torch.float32)

        return x, y
