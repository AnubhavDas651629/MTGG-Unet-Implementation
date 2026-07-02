"""
Preprocessing script for NOAA NetCDF files.

Reads raw NetCDF (.nc) files, extracts daily slices, handles NOAA fill values,
resizes the grids to (256, 256) using bilinear interpolation, and saves them
as chronologically aligned .npy files.

Expected input files:
    - CAPE (.nc)
    - CIN (.nc)
    - Geopotential Height (.nc)
    - Target Tornado Probability (.nc)
    - Target Significant Tornado Probability (.nc)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from tqdm import tqdm


def create_folders(output_dir):
    """Create the directory structure expected by NOAA_dataset.py"""
    paths = [
        "train/cape",
        "train/cin",
        "train/geo",
        "train_masks/tornado",
        "train_masks/sigtor",
    ]
    for p in paths:
        os.makedirs(os.path.join(output_dir, p), exist_ok=True)
    print(f"Directory structure initialized under: {output_dir}")


def resize_grid(matrix_2d, target_size=256):
    """
    Resize a 2D weather matrix to exactly (target_size, target_size) using bilinear interpolation.
    Fills NaNs with 0.0.
    """
    arr = np.nan_to_num(matrix_2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    resized = F.interpolate(
        tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    return resized.squeeze().numpy()  # (target_size, target_size)


def get_variable(ds):
    """Auto-detect the main data variable in a NetCDF dataset."""
    candidates = list(ds.data_vars)
    if len(candidates) == 1:
        return candidates[0]
    for name in [
        "cape", "cin", "hgt", "prob",
        "CAPE", "CIN", "HGT", "PROB",
        "p_perfect_tor", "p_perfect_sigtor", "p_perfect_sig_tor",
    ]:
        if name in ds.data_vars:
            return name
    raise KeyError(
        f"Cannot auto-detect variable. Available: {candidates}. "
        "Please edit get_variable() or specify the name."
    )


def extract_2d_slice(da, date_val):
    """
    Safely extract a 2D (lat x lon) slice from a DataArray for a given date.
    Handles potential extra dimensions like 'level' or 'nbnds'.
    """
    # Select date
    da_slice = da.sel(time=date_val, method="nearest")

    # If there is a 'level' dimension, take the first level
    if "level" in da_slice.dims:
        da_slice = da_slice.isel(level=0)

    # If there is a 'nbnds' dimension, take the first index
    if "nbnds" in da_slice.dims:
        da_slice = da_slice.isel(nbnds=0)

    return da_slice.values


def main():
    parser = argparse.ArgumentParser(description="Preprocess NOAA NetCDF weather datasets into 2D .npy grids")
    parser.add_argument("--cape", type=str, required=True, help="Path to CAPE .nc file")
    parser.add_argument("--cin", type=str, required=True, help="Path to CIN .nc file")
    parser.add_argument("--hgt", type=str, required=True, help="Path to Geopotential Height .nc file")
    parser.add_argument("--tor", type=str, required=True, help="Path to Target Tornado Probability .nc file")
    parser.add_argument("--sigtor", type=str, required=True, help="Path to Target Sig-Tornado Probability .nc file")
    parser.add_argument("--output", type=str, default="./data", help="Output data directory")
    parser.add_argument("--grid_size", type=int, default=256, help="Target spatial grid resolution")
    args = parser.parse_args()

    # Step 1: Create folders
    create_folders(args.output)

    # Step 2: Open datasets using xarray
    print("\nOpening NetCDF files...")
    ds_cape = xr.open_dataset(args.cape)
    ds_cin = xr.open_dataset(args.cin)
    ds_hgt = xr.open_dataset(args.hgt)
    ds_tor = xr.open_dataset(args.tor)
    ds_sigtor = xr.open_dataset(args.sigtor)

    # Detect variable names
    var_cape = get_variable(ds_cape)
    var_cin = get_variable(ds_cin)
    var_hgt = get_variable(ds_hgt)
    var_tor = get_variable(ds_tor)
    var_sigtor = get_variable(ds_sigtor)

    da_cape = ds_cape[var_cape]
    da_cin = ds_cin[var_cin]
    da_hgt = ds_hgt[var_hgt]
    da_tor = ds_tor[var_tor]
    da_sigtor = ds_sigtor[var_sigtor]

    # Find the overlapping date range across all files
    # We convert to string date format YYYY-MM-DD to align them cleanly
    print("Finding overlapping dates...")
    dates_cape = set(np.datetime_as_string(da_cape.time.values, unit='D'))
    dates_cin = set(np.datetime_as_string(da_cin.time.values, unit='D'))
    dates_hgt = set(np.datetime_as_string(da_hgt.time.values, unit='D'))
    dates_tor = set(np.datetime_as_string(da_tor.time.values, unit='D'))
    dates_sigtor = set(np.datetime_as_string(da_sigtor.time.values, unit='D'))

    common_dates = sorted(list(
        dates_cape & dates_cin & dates_hgt & dates_tor & dates_sigtor
    ))

    if not common_dates:
        print("Error: No overlapping dates found between the datasets!")
        return

    print(f"Found {len(common_dates)} overlapping days to process ({common_dates[0]} to {common_dates[-1]}).")

    # Step 3: Process day by day
    print("\nProcessing days:")
    for date_str in tqdm(common_dates):
        # Extract 2D maps
        map_cape = extract_2d_slice(da_cape, date_str)
        map_cin = extract_2d_slice(da_cin, date_str)
        map_hgt = extract_2d_slice(da_hgt, date_str)
        map_tor = extract_2d_slice(da_tor, date_str)
        map_sigtor = extract_2d_slice(da_sigtor, date_str)

        # Resize to target resolution (256x256)
        resized_cape = resize_grid(map_cape, args.grid_size)
        resized_cin = resize_grid(map_cin, args.grid_size)
        resized_hgt = resize_grid(map_hgt, args.grid_size)
        resized_tor = resize_grid(map_tor, args.grid_size)
        resized_sigtor = resize_grid(map_sigtor, args.grid_size)

        # Save to corresponding folders
        filename = f"{date_str}.npy"
        np.save(os.path.join(args.output, "train/cape", filename), resized_cape)
        np.save(os.path.join(args.output, "train/cin", filename), resized_cin)
        np.save(os.path.join(args.output, "train/geo", filename), resized_hgt)
        np.save(os.path.join(args.output, "train_masks/tornado", filename), resized_tor)
        np.save(os.path.join(args.output, "train_masks/sigtor", filename), resized_sigtor)

    print(f"\nProcessing complete! All .npy maps saved under: {args.output}")


if __name__ == "__main__":
    main()
