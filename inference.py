"""
Inference & Visualization Script for UNet-MTGNN Tornado Forecasting.

Loads the trained hybrid model, reads a test sequence (7 days of history),
forecasts the future sequence (3 days of predictions), and saves a plot of
the predicted tornado probability maps.

Usage:
    python inference.py --model ./models/best_model.pth --data ./data --seq_in 7 --seq_out 3 --date 2014-06-15
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from NOAA_dataset import NOAATornadoTemporalDataset
from unet_mtgnn_net import UNetMTGNN


def plot_predictions(input_seq, target_seq, pred_probs, date_str, output_path):
    """
    Plots a grid comparing:
      - Last input day's weather indicators (CAPE, CIN)
      - Ground truth future tornado probabilities (over seq_out days)
      - Predicted future tornado probabilities (over seq_out days)
    """
    seq_in, _, H, W = input_seq.shape
    seq_out, _, _, _ = target_seq.shape

    fig, axes = plt.subplots(3, max(seq_out, 3), figsize=(15, 10))

    # --- Row 1: Last Input Day's Indicators ---
    # We display the last input frame (closest to the forecast window)
    last_input = input_seq[-1]  # (3, H, W)
    cape = last_input[0]
    cin = last_input[1]
    geo = last_input[2]

    axes[0, 0].imshow(cape, cmap='viridis')
    axes[0, 0].set_title(f"CAPE (Last Input Day)\n{date_str}")
    axes[0, 1].imshow(cin, cmap='plasma')
    axes[0, 1].set_title("CIN")
    axes[0, 2].imshow(geo, cmap='coolwarm')
    axes[0, 2].set_title("Geo Height")

    # Turn off unused axes in row 1
    for col in range(3, max(seq_out, 3)):
        axes[0, col].axis('off')

    # --- Row 2 & 3: Ground Truth vs Prediction ---
    for t in range(seq_out):
        # Ground Truth
        gt_tor = target_seq[t, 0]  # First channel is standard tornado prob
        axes[1, t].imshow(gt_tor, cmap='inferno', vmin=0, vmax=1)
        axes[1, t].set_title(f"GT Tornado Prob\n(Day t+{t+1})")

        # Prediction
        pred_tor = pred_probs[t, 0]
        axes[2, t].imshow(pred_tor, cmap='inferno', vmin=0, vmax=1)
        axes[2, t].set_title(f"Pred Tornado Prob\n(Day t+{t+1})")

    # Clean up display axes
    for r in range(3):
        for c in range(max(seq_out, 3)):
            if r == 0 and c >= 3:
                continue
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualized plot saved to: {output_path}")
    plt.close()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load file lists
    cape_dir = os.path.join(args.data, "train", "cape")
    all_files = sorted(f for f in os.listdir(cape_dir) if f.endswith('.npy'))

    # Locate the starting date in files
    start_date_filename = f"{args.date}.npy"
    if start_date_filename not in all_files:
        raise ValueError(
            f"Specified start date {args.date} not found in dataset! "
            f"Available range: {all_files[0][:-4]} to {all_files[-1][:-4]}"
        )

    start_idx = all_files.index(start_date_filename)

    # Check if there are enough subsequent days for sequence lengths
    if start_idx + args.seq_in + args.seq_out > len(all_files):
        raise ValueError(
            f"Not enough days remaining after {args.date} to build "
            f"seq_in={args.seq_in} + seq_out={args.seq_out} window."
        )

    # Instantiate single-sequence evaluation dataset
    target_files = all_files[start_idx : start_idx + args.seq_in + args.seq_out]
    dataset = NOAATornadoTemporalDataset(
        root_path=args.data,
        seq_in=args.seq_in,
        seq_out=args.seq_out,
        file_list=target_files,
        grid_size=args.grid_size
    )

    # Extract input sequence and ground truth target
    x, y = dataset[0]  # x: (seq_in, 3, H, W), y: (seq_out, 2, H, W)

    # Instantiate model
    model = UNetMTGNN(
        in_channels=3,
        out_channels=2,
        seq_in=args.seq_in,
        seq_out=args.seq_out,
        height=args.grid_size,
        width=args.grid_size,
        device=device,
        gcn_depth=args.gcn_depth,
        subgraph_size=args.subgraph_size,
        node_dim=args.node_dim,
        residual_channels=args.residual_channels,
        conv_channels=args.conv_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        mtgnn_layers=args.mtgnn_layers,
        dropout=0.0  # No dropout during inference
    )

    # Load weights
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model}")

    print(f"Loading weights from: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # Forward pass
    print("Running model prediction...")
    with torch.no_grad():
        x_in = x.unsqueeze(0).to(device)  # Add batch dimension: (1, seq_in, 3, H, W)
        logits = model(x_in)              # (1, seq_out, 2, H, W)
        pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # Probabilities: (seq_out, 2, H, W)

    # Save visualization plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_plot_path = os.path.join(args.output_dir, f"prediction_{args.date}.png")

    plot_predictions(
        input_seq=x.numpy(),
        target_seq=y.numpy(),
        pred_probs=pred_probs,
        date_str=args.date,
        output_path=output_plot_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and visualization for UNet-MTGNN")
    parser.add_argument('--model', type=str, default='./models/best_model.pth', help='Path to trained .pth model weights')
    parser.add_argument('--data', type=str, default='./data', help='Path to preprocessed NOAA .npy dataset directory')
    parser.add_argument('--date', type=str, required=True, help='Start date for prediction sequence (YYYY-MM-DD)')
    parser.add_argument('--grid_size', type=int, default=256, help='Spatial grid dimensions')

    parser.add_argument('--seq_in', type=int, default=7, help='Number of input sequence days')
    parser.add_argument('--seq_out', type=int, default=3, help='Number of forecasted sequence days')

    # Model parameters (must match training configuration)
    parser.add_argument('--gcn_depth', type=int, default=2)
    parser.add_argument('--subgraph_size', type=int, default=20)
    parser.add_argument('--node_dim', type=int, default=40)
    parser.add_argument('--residual_channels', type=int, default=32)
    parser.add_argument('--conv_channels', type=int, default=32)
    parser.add_argument('--skip_channels', type=int, default=64)
    parser.add_argument('--end_channels', type=int, default=128)
    parser.add_argument('--mtgnn_layers', type=int, default=3)

    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Directory to save output plots')

    args = parser.parse_args()
    main(args)
