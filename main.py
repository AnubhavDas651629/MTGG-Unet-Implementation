"""
Main entry point for training the UNet-MTGNN Tornado Forecasting Model.

This script imports and executes the training pipeline defined in train.py.
You can run training directly with:
    python main.py --epochs 50 --batch_size 4 --learning_rate 1e-4

Includes KL Divergence validation tracking, BCE loss logging, and best model checkpointing.
"""

from train import main, argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UNet-MTGNN hybrid model for tornado probability forecasting"
    )

    # Data
    parser.add_argument('--data', type=str, default='./data',
                        help='Path to data directory with train/ and train_masks/ folders')
    parser.add_argument('--grid_size', type=int, default=256,
                        help='Spatial grid size (must be divisible by 16)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of data for training (rest is validation)')

    # Sequence lengths
    parser.add_argument('--seq_in', type=int, default=7,
                        help='Number of input days (past weather history)')
    parser.add_argument('--seq_out', type=int, default=3,
                        help='Number of output days (future tornado prediction)')

    # Training & Optimization
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (reduce if GPU OOM)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Adam learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Adam weight decay (L2 regularization)')
    parser.add_argument('--clip', type=float, default=5.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--pos_weight', type=float, default=10.0,
                        help='Positive class weight for BCE loss (tornado events are rare)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate in MTGNN layers')

    # MTGNN architecture
    parser.add_argument('--gcn_depth', type=int, default=2)
    parser.add_argument('--subgraph_size', type=int, default=20)
    parser.add_argument('--node_dim', type=int, default=40)
    parser.add_argument('--residual_channels', type=int, default=32)
    parser.add_argument('--conv_channels', type=int, default=32)
    parser.add_argument('--skip_channels', type=int, default=64)
    parser.add_argument('--end_channels', type=int, default=128)
    parser.add_argument('--mtgnn_layers', type=int, default=3)
    parser.add_argument('--propalpha', type=float, default=0.05)
    parser.add_argument('--tanhalpha', type=float, default=3.0)
    parser.add_argument('--dilation_exponential', type=int, default=1)

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--save', type=str, default='./models/',
                        help='Directory to save model checkpoints')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--print_every', type=int, default=10)

    args = parser.parse_args()
    main(args)
