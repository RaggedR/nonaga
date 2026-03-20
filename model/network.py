"""
Neural network for Nonaga: ResNet with dual policy heads + value head.

Architecture (~200K params):
  Input: 6 × 7 × 7
  Conv(6→64, 3×3, pad=1) → BN → ReLU
  4 × ResBlock(64→64)
  Three heads:
    Piece policy: Conv(64→2) → BN → ReLU → FC → 294  (49 cells × 6 dirs)
    Tile policy:  Conv(64→2) → BN → ReLU → FC → 2401 (49 × 49 src×dst)
    Value:        Conv(64→1) → BN → ReLU → FC → 64 → ReLU → FC → 1 → Tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from game.nonaga import NonagaState

GRID_SIZE = 7
INPUT_CHANNELS = 6
HIDDEN_CHANNELS = 64
NUM_RES_BLOCKS = 4
PIECE_ACTION_SIZE = NonagaState.PIECE_ACTION_SIZE  # 294
TILE_ACTION_SIZE = NonagaState.TILE_ACTION_SIZE    # 2401


class ResBlock(nn.Module):
    """Standard residual block: conv → BN → ReLU → conv → BN + skip."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class NonagaNet(nn.Module):
    """
    AlphaZero-style network with two policy heads (piece and tile moves)
    and one value head.
    """

    def __init__(self):
        super().__init__()
        # Input conv
        self.conv_in = nn.Conv2d(INPUT_CHANNELS, HIDDEN_CHANNELS, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(HIDDEN_CHANNELS)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(HIDDEN_CHANNELS) for _ in range(NUM_RES_BLOCKS)
        ])

        # Piece policy head
        self.piece_conv = nn.Conv2d(HIDDEN_CHANNELS, 2, 1, bias=False)
        self.piece_bn = nn.BatchNorm2d(2)
        self.piece_fc = nn.Linear(2 * GRID_SIZE * GRID_SIZE, PIECE_ACTION_SIZE)

        # Tile policy head
        self.tile_conv = nn.Conv2d(HIDDEN_CHANNELS, 2, 1, bias=False)
        self.tile_bn = nn.BatchNorm2d(2)
        self.tile_fc = nn.Linear(2 * GRID_SIZE * GRID_SIZE, TILE_ACTION_SIZE)

        # Value head
        self.val_conv = nn.Conv2d(HIDDEN_CHANNELS, 1, 1, bias=False)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 6, 7, 7) board tensor

        Returns:
            piece_logits: (batch, 294) raw logits for piece moves
            tile_logits:  (batch, 2401) raw logits for tile moves
            value:        (batch, 1) value in [-1, 1]
        """
        # Shared trunk
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # Piece policy
        pp = F.relu(self.piece_bn(self.piece_conv(out)))
        pp = pp.view(pp.size(0), -1)
        piece_logits = self.piece_fc(pp)

        # Tile policy
        tp = F.relu(self.tile_bn(self.tile_conv(out)))
        tp = tp.view(tp.size(0), -1)
        tile_logits = self.tile_fc(tp)

        # Value
        v = F.relu(self.val_bn(self.val_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.val_fc1(v))
        value = torch.tanh(self.val_fc2(v))

        return piece_logits, tile_logits, value

    def predict(self, board_np):
        """
        Single-sample prediction from numpy array.
        Returns (piece_probs, tile_probs, value) as numpy arrays.
        """
        import numpy as np
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(board_np).unsqueeze(0)  # (1, 6, 7, 7)
            device = next(self.parameters()).device
            x = x.to(device)
            pp, tp, v = self(x)
            return (
                F.softmax(pp, dim=1).squeeze(0).cpu().numpy(),
                F.softmax(tp, dim=1).squeeze(0).cpu().numpy(),
                v.item(),
            )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
