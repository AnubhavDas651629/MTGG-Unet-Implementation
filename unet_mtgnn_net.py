"""
UNet-MTGNN Hybrid Architecture for Spatio-Temporal Forecasting.

Data Flow (Method 2: MTGNN inside U-Net's Bottleneck):

    Input: A sequence of 2D spatial maps over time
           Shape: (Batch, seq_in, Channels, Height, Width)

    Step 1: U-Net ENCODER processes each time step independently.
            Compresses the full-resolution map (e.g., 128x128) down to a
            small bottleneck grid (e.g., 8x8). Each cell in this small grid
            becomes a "node" in the graph.

    Step 2: MTGNN BOTTLENECK replaces the standard U-Net DoubleConv bottleneck.
            It treats the compressed grid cells as graph nodes, automatically
            learns directed connections between them, and processes the temporal
            sequence to predict the future.

    Step 3: U-Net DECODER takes the MTGNN prediction and upsamples it back
            to full resolution (e.g., 128x128) using skip connections saved
            from the encoder.

    Output: Predicted future 2D spatial maps
            Shape: (Batch, seq_out, Channels, Height, Width)

Requirements:
    - Height and Width must be divisible by 16 (4 downsample layers, each halves)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import graph_constructor, mixprop, dilated_inception, LayerNorm
from unet_parts import DownSample, UpSample


class UNetMTGNN(nn.Module):
    def __init__(self, in_channels, out_channels, seq_in, seq_out,
                 height, width, device,
                 gcn_depth=2, subgraph_size=20, node_dim=40,
                 residual_channels=32, conv_channels=32,
                 skip_channels=64, end_channels=128,
                 mtgnn_layers=3, dropout=0.3,
                 propalpha=0.05, tanhalpha=3,
                 dilation_exponential=1):
        """
        Args:
            in_channels:    Number of input feature channels per map (e.g., 3 for temp/humidity/wind)
            out_channels:   Number of output feature channels per map (e.g., 1 for temperature only)
            seq_in:         Number of past time steps the model looks at (e.g., 12)
            seq_out:        Number of future time steps the model predicts (e.g., 12)
            height:         Spatial height of the input grid (must be divisible by 16)
            width:          Spatial width of the input grid (must be divisible by 16)
            device:         'cpu' or 'cuda'
            gcn_depth:      Number of hops in mix-hop graph propagation
            subgraph_size:  Top-k connections to keep per node in the learned graph
            node_dim:       Embedding dimension for graph learning
            residual_channels: Internal channel width for MTGNN residual stream
            conv_channels:  Internal channel width for MTGNN temporal convolutions
            skip_channels:  Channel width for MTGNN skip connections
            end_channels:   Channel width for MTGNN output projections
            mtgnn_layers:   Number of temporal+graph convolution layers in MTGNN
            dropout:        Dropout probability
            propalpha:      Retention ratio in mix-hop propagation (how much original signal to keep)
            tanhalpha:      Saturation parameter for graph constructor
            dilation_exponential: Dilation growth factor for temporal convolutions
        """
        super(UNetMTGNN, self).__init__()

        # Save key parameters
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.out_channels = out_channels
        self.dropout = dropout
        self.mtgnn_layers_count = mtgnn_layers

        # Bottleneck spatial dimensions after 4 downsamples (each halves H and W)
        assert height % 16 == 0 and width % 16 == 0, \
            f"Height ({height}) and Width ({width}) must be divisible by 16"
        self.h_b = height // 16
        self.w_b = width // 16
        self.num_nodes = self.h_b * self.w_b

        # ==================================================================
        # U-Net Encoder: Extracts spatial features from each input time step
        # ==================================================================
        self.down1 = DownSample(in_channels, 64)     # (H, W)       -> (H/2, W/2)
        self.down2 = DownSample(64, 128)              # (H/2, W/2)   -> (H/4, W/4)
        self.down3 = DownSample(128, 256)             # (H/4, W/4)   -> (H/8, W/8)
        self.down4 = DownSample(256, 512)             # (H/8, W/8)   -> (H/16, W/16)

        # ==================================================================
        # Bridge: Reduce channels from encoder (512) to MTGNN working width
        # This keeps the MTGNN lightweight and memory-efficient.
        # ==================================================================
        self.encoder_to_mtgnn = nn.Conv2d(512, residual_channels, kernel_size=1)

        # ==================================================================
        # MTGNN Bottleneck: Learns graph structure and temporal dynamics
        # This replaces the standard U-Net DoubleConv bottleneck.
        # ==================================================================

        # Graph constructor: learns which nodes influence which other nodes
        self.gc = graph_constructor(
            self.num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha
        )
        self.idx = torch.arange(self.num_nodes).to(device)

        # Start convolution: projects input into the residual stream
        self.start_conv = nn.Conv2d(
            in_channels=residual_channels,
            out_channels=residual_channels,
            kernel_size=(1, 1)
        )

        # Calculate how far back in time the dilated convolutions can see
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** mtgnn_layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = mtgnn_layers * (kernel_size - 1) + 1

        # Build the MTGNN layer stack (temporal conv + graph conv per layer)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        new_dilation = 1
        rf_size_i = 0

        for j in range(1, mtgnn_layers + 1):
            if dilation_exponential > 1:
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1)
                    * (dilation_exponential ** j - 1) / (dilation_exponential - 1)
                )
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            # Gated temporal convolutions (dilated inception with multiple kernel sizes)
            self.filter_convs.append(
                dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation)
            )
            self.gate_convs.append(
                dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation)
            )

            # Residual projection (brings conv_channels back to residual_channels)
            self.residual_convs.append(
                nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))
            )

            # Skip connection projection (collapses remaining time dimension)
            if seq_in > self.receptive_field:
                self.skip_convs.append(
                    nn.Conv2d(conv_channels, skip_channels,
                              kernel_size=(1, seq_in - rf_size_j + 1))
                )
            else:
                self.skip_convs.append(
                    nn.Conv2d(conv_channels, skip_channels,
                              kernel_size=(1, self.receptive_field - rf_size_j + 1))
                )

            # Bidirectional graph convolutions (mix-hop propagation)
            self.gconv1.append(
                mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
            )
            self.gconv2.append(
                mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
            )

            # Layer normalization
            if seq_in > self.receptive_field:
                self.norm.append(
                    LayerNorm((residual_channels, self.num_nodes,
                               seq_in - rf_size_j + 1), elementwise_affine=True)
                )
            else:
                self.norm.append(
                    LayerNorm((residual_channels, self.num_nodes,
                               self.receptive_field - rf_size_j + 1),
                              elementwise_affine=True)
                )

            new_dilation *= dilation_exponential

        # Skip connection entry (from raw input) and exit (from final layer)
        if seq_in > self.receptive_field:
            self.skip0 = nn.Conv2d(
                residual_channels, skip_channels,
                kernel_size=(1, seq_in), bias=True
            )
            self.skipE = nn.Conv2d(
                residual_channels, skip_channels,
                kernel_size=(1, seq_in - self.receptive_field + 1), bias=True
            )
        else:
            self.skip0 = nn.Conv2d(
                residual_channels, skip_channels,
                kernel_size=(1, self.receptive_field), bias=True
            )
            self.skipE = nn.Conv2d(
                residual_channels, skip_channels,
                kernel_size=(1, 1), bias=True
            )

        # MTGNN output projection layers
        self.end_conv_1 = nn.Conv2d(
            skip_channels, end_channels, kernel_size=(1, 1), bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            end_channels, end_channels, kernel_size=(1, 1), bias=True
        )

        # ==================================================================
        # Bridge: Expand channels from MTGNN output to decoder input (1024)
        # This matches the channel count the U-Net decoder expects.
        # ==================================================================
        self.mtgnn_to_decoder = nn.Conv2d(end_channels, 1024, kernel_size=1)

        # ==================================================================
        # U-Net Decoder: Upsamples predictions back to full resolution
        # Uses skip connections from the encoder to restore fine spatial detail.
        # ==================================================================
        self.up1 = UpSample(1024, 512)    # (H/16, W/16) -> (H/8, W/8)
        self.up2 = UpSample(512, 256)     # (H/8, W/8)   -> (H/4, W/4)
        self.up3 = UpSample(256, 128)     # (H/4, W/4)   -> (H/2, W/2)
        self.up4 = UpSample(128, 64)      # (H/2, W/2)   -> (H, W)

        # Final output: produces all future time steps and channels at once
        self.out_conv = nn.Conv2d(64, seq_out * out_channels, kernel_size=1)

    def forward(self, x):
        """
        Full forward pass: Encoder -> MTGNN Bottleneck -> Decoder

        Args:
            x: Input tensor of shape (B, seq_in, C_in, H, W)
               B       = batch size
               seq_in  = number of past time steps
               C_in    = input feature channels (e.g., temperature, humidity, wind)
               H, W    = spatial grid dimensions

        Returns:
            Prediction tensor of shape (B, seq_out, C_out, H, W)
               seq_out = number of predicted future time steps
               C_out   = output feature channels
        """
        B, T, C, H, W = x.shape

        # ==============================================================
        # STEP 1: U-Net Encoder
        # Process each time step through the shared encoder layers.
        # The encoder weights are shared across all time steps (like
        # applying the same camera lens to every frame of a video).
        # We save the skip connections from the LAST time step because
        # it is temporally closest to the future we are predicting.
        # ==============================================================
        bottleneck_list = []

        for t in range(T):
            frame = x[:, t]  # Extract one time step: (B, C_in, H, W)

            skip1, p1 = self.down1(frame)   # skip1: (B, 64, H/2, W/2)
            skip2, p2 = self.down2(p1)      # skip2: (B, 128, H/4, W/4)
            skip3, p3 = self.down3(p2)      # skip3: (B, 256, H/8, W/8)
            skip4, p4 = self.down4(p3)      # skip4: (B, 512, H/16, W/16)

            # Reduce 512 channels to residual_channels for MTGNN
            feat = self.encoder_to_mtgnn(p4)  # (B, residual_channels, H/16, W/16)
            bottleneck_list.append(feat)

        # Save skip connections from the last input time step for the decoder
        last_skip1 = skip1
        last_skip2 = skip2
        last_skip3 = skip3
        last_skip4 = skip4

        # ==============================================================
        # STEP 2: Reshape from Spatial Grid to Graph Nodes
        # Flatten each bottleneck map from (H/16, W/16) into a list of
        # N nodes, then stack all time steps along the time axis.
        # Result: (B, Channels, Nodes, Time) — the format MTGNN expects.
        # ==============================================================
        node_features = []
        for feat in bottleneck_list:
            # (B, C, H_b, W_b) -> (B, C, N) where N = H_b * W_b
            node_feat = feat.reshape(B, -1, self.num_nodes)
            node_features.append(node_feat)

        # Stack over time: (B, C, N, T)
        mtgnn_input = torch.stack(node_features, dim=3)

        # ==============================================================
        # STEP 3: MTGNN Bottleneck Processing
        # This is where the magic happens:
        #   a) The graph constructor learns which grid zones affect which
        #   b) Dilated inception convolutions capture multi-scale time patterns
        #   c) Mix-hop graph convolutions propagate information across zones
        # ==============================================================

        # Pad the time axis if the sequence is shorter than the receptive field
        if self.seq_in < self.receptive_field:
            mtgnn_input = F.pad(
                mtgnn_input, (self.receptive_field - self.seq_in, 0, 0, 0)
            )

        # Learn the adaptive directed adjacency matrix between all nodes
        adp = self.gc(self.idx)

        # Project into the residual stream and compute initial skip
        x_m = self.start_conv(mtgnn_input)
        skip = self.skip0(
            F.dropout(mtgnn_input, self.dropout, training=self.training)
        )

        # Iterate through MTGNN layers
        for i in range(self.mtgnn_layers_count):
            residual = x_m

            # --- Gated Temporal Convolution ---
            # tanh controls "what" information to pass
            # sigmoid controls "how much" to let through (the gate)
            filt = torch.tanh(self.filter_convs[i](x_m))
            gate = torch.sigmoid(self.gate_convs[i](x_m))
            x_m = filt * gate
            x_m = F.dropout(x_m, self.dropout, training=self.training)

            # Accumulate skip connection
            s = self.skip_convs[i](x_m)
            skip = s + skip

            # --- Bidirectional Graph Convolution ---
            # gconv1: propagates info along outgoing edges (A)
            # gconv2: propagates info along incoming edges (A transposed)
            x_m = self.gconv1[i](x_m, adp) + self.gconv2[i](x_m, adp.transpose(1, 0))

            # Residual connection (trim time dimension to match after convolutions)
            x_m = x_m + residual[:, :, :, -x_m.size(3):]

            # Layer normalization
            x_m = self.norm[i](x_m, self.idx)

        # Final skip aggregation and output projections
        skip = self.skipE(x_m) + skip
        x_m = F.relu(skip)
        x_m = F.relu(self.end_conv_1(x_m))
        x_m = self.end_conv_2(x_m)
        # x_m shape: (B, end_channels, N, 1)

        # ==============================================================
        # STEP 4: Reshape Graph Nodes Back to Spatial Grid
        # Convert from (B, C, N, 1) back to (B, C, H/16, W/16) so the
        # U-Net decoder can upsample it back to full resolution.
        # ==============================================================
        x_spatial = x_m.squeeze(3)  # Remove time dim: (B, end_channels, N)
        x_spatial = x_spatial.reshape(B, -1, self.h_b, self.w_b)  # (B, end_channels, H_b, W_b)
        x_spatial = self.mtgnn_to_decoder(x_spatial)  # (B, 1024, H_b, W_b)

        # ==============================================================
        # STEP 5: U-Net Decoder
        # Upsample back to full resolution using skip connections from
        # the encoder. The skip connections inject fine-grained spatial
        # details (terrain edges, city boundaries) that were lost during
        # downsampling.
        # ==============================================================
        up1 = self.up1(x_spatial, last_skip4)   # (B, 512, H/8, W/8)
        up2 = self.up2(up1, last_skip3)         # (B, 256, H/4, W/4)
        up3 = self.up3(up2, last_skip2)         # (B, 128, H/2, W/2)
        up4 = self.up4(up3, last_skip1)         # (B, 64, H, W)

        # Final 1x1 convolution: produces all future steps and channels at once
        out = self.out_conv(up4)  # (B, seq_out * out_channels, H, W)

        # Reshape to separate time and channel dimensions
        out = out.reshape(B, self.seq_out, self.out_channels, H, W)

        return out
