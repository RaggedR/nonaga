# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nonaga board game with AlphaZero-trained AI. Python for the game engine + training pipeline, single-file HTML for browser play. The AI uses MCTS with neural network priors (not minimax) because Nonaga's compound turns create a branching factor of ~300.

## Commands

```bash
# Run tests
python tests/test_game.py

# Train (uses MPS on Apple Silicon, CUDA on Nvidia, else CPU)
python -m train.coach --iterations 100 --games 100 --sims 100

# Quick smoke test for training
python -m train.coach --iterations 2 --games 5 --sims 10

# Resume training from checkpoint
python -m train.coach --iterations 100 --resume 49

# Export trained weights for browser
python -m model.export_weights checkpoints/iteration_49.pt --output-dir web

# Export to ONNX (alternative, not used by current browser frontend)
python -m model.export_onnx checkpoints/iteration_49.pt --output web/model.onnx

# Serve browser game (must use HTTP server, not file://)
cd web && python -m http.server 8000
```

## Architecture

### Two-Ply Turn Decomposition (central design decision)

Each Nonaga turn is split into two plies in the game tree:
1. **Ply 1 (PIECE_MOVE)**: Slide a piece along a hex direction until it hits another piece or board edge (~18 possible actions: 3 pieces × 6 directions)
2. **Ply 2 (TILE_MOVE)**: Remove an unoccupied edge tile and place it at a valid edge position (~50-200 actions: source × dest pairs)

The `NonagaState` tracks `ply_type` and `current_player` — after a piece move, the same player does a tile move, then the turn passes. This reduces MCTS branching from ~300 compound to ~18 then ~100 at each node.

### Coordinate System

7×7 grid with axial hex coordinates. `idx = r * 7 + q`. Of 49 cells, only 37 are valid hex positions (side-4 hexagon). The 19 active game tiles are a moving subset. Board is canonicalized (centroid-shifted to center) before NN input.

### Neural Network (model/network.py, ~570K params)

ResNet with shared trunk → three heads:
- **Input**: 6 × 7 × 7 (tiles, current pieces, opponent pieces, ply type, last moved tile, removable edges)
- **Piece policy head**: 294 outputs (49 cells × 6 directions)
- **Tile policy head**: 2401 outputs (49 × 49 source × dest)
- **Value head**: scalar in [-1, 1]

Uses full 49-cell grid indexing (not just 37 valid cells) for simpler action encoding, with policy masks to exclude invalid actions.

### Browser Frontend (web/index.html)

Single-file: game engine (JS port), MCTS, and NN forward pass all in pure JS. Loads `weights.bin` + `manifest.json` via fetch. No ONNX runtime dependency. SVG hex board with click interactions. AI runs synchronously in the main thread with 30 MCTS simulations.

### Training Pipeline

`Coach` orchestrates: self-play → D6 augmentation (12× data) → train on replay buffer → arena (new vs old model) → accept/reject. Training data: `(board_6×7×7, ply_type, policy_target, value_target)`. Draws count as 0.5 in arena scoring.

### D6 Symmetry (game/symmetry.py)

Hex board has 12 symmetries (6 rotations × 2 reflections). Axial rotation: `(q,r) → (-r, q+r)`. Both board tensors and policy vectors are transformed. All transforms are relative to grid center (3,3).

## Key Constraints

- **Tile removal**: Only edge tiles with ≤4 tile neighbors can be removed (prevents interior holes). Must not be occupied, must not be the tile opponent just moved, and removal must keep board connected (BFS check).
- **Tile placement**: Prefer positions with 2+ adjacent tiles for compactness. Falls back to 1+ adjacent if no 2+ options exist. Must maintain board connectivity.
- **Piece sliding**: Pieces slide in a straight line until hitting another piece or running off tiles. Must move at least one cell.
- **Win condition**: Three pieces of the same player all mutually adjacent (triangle).

## Python ↔ JS Parity

The game engine exists in two implementations that must stay in sync:
- **Python**: `game/nonaga.py` (canonical, used for training)
- **JavaScript**: `web/index.html` GameState class (port, used for browser play)

Changes to game rules (especially `removableEdgeTiles`, `validPlacements`, `slide`) must be applied to both. After rule changes, existing trained models may not match the new rules — retraining may be needed.
