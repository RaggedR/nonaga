# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nonaga board game with multiple AI approaches. Python for the game engine + training pipeline, single-file HTML for browser play. The default browser AI is a GA-evolved 14-weight evaluation function (greedy 1-ply), which beats the 570K-parameter neural network 50-0. The NN AI is also available via toggle. See `DESIGN_DECISIONS.md` for the full training story (curriculum learning, draw shaping, why MCTS failed, endgame training, island-model GA, island-model AlphaZero with cross-play).

## Commands

```bash
# Run tests
python tests/test_game.py

# Train (uses MPS on Apple Silicon, CUDA on Nvidia, else CPU)
python -m train.coach --iterations 100 --games 100 --sims 100

# Quick smoke test for training
python -m train.coach --iterations 2 --games 5 --sims 10

# Train with curriculum pretraining (adjacency wins first, then triangle)
python -m train.coach --curriculum 20 --iterations 30 --games 100 --sims 50

# Resume training from checkpoint
python -m train.coach --iterations 100 --resume 49

# Fast endgame bootstrap training (generates random games, trains value head)
python -u fast_train.py

# Export trained weights for browser
python -m model.export_weights checkpoints/endgame_trained.pt --output-dir web

# Evaluate model vs random player
python eval_vs_random.py checkpoints/endgame_trained.pt --games 50 --sims 50

# Export to ONNX (alternative, not used by current browser frontend)
python -m model.export_onnx checkpoints/iteration_49.pt --output web/model.onnx

# Island-model GA evolution (ring topology, 14-weight evaluation function)
python ga_evolve.py
python ga_evolve.py --islands 5 --pop 16 --gens 200  # full run
python ga_evolve.py --islands 3 --pop 4 --gens 5 --tournament-games 1  # smoke test

# Island-model AlphaZero with cross-play (ring topology, multiple NNs)
python -m train.island_coach --islands 5 --iterations 100 --games 100 --sims 100
python -m train.island_coach --islands 2 --iterations 2 --games 5 --sims 10  # smoke test

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

Single-file: game engine (JS port), two AI players, SVG hex board with click interactions. Runs synchronously in the main thread.

- **GAPlayer** (default): 14-weight evaluation function evolved by island-model GA. Computes board features (adjacency, completing cells, mobility, distances) and picks the move with highest `tanh(features · weights)`. No NN needed — just 14 hardcoded floats.
- **GreedyAI** (toggle): Loads `weights.bin` + `manifest.json`, runs NN forward pass, picks the move with highest value head output. Pure JS, no ONNX runtime.
- Toggle button switches between GA and NN players.

### Training Pipeline

Four training approaches exist:

1. **Endgame bootstrap** (`fast_train.py`): Generate random-vs-random games, extract last 30 plies of decisive games, train value head on these. Fast and effective for bootstrapping.

2. **AlphaZero-style** (`train/coach.py`): `Coach` orchestrates: self-play → D6 augmentation (12× data) → train on replay buffer → arena (new vs old model) → accept/reject. Supports curriculum pretraining (adjacency win mode) and draw shaping.

3. **Island-model GA** (`ga_evolve.py`): Evolves 14-weight evaluation functions on a ring topology. No NN — pure feature-weighted greedy play. Currently the strongest AI (beats the NN 50-0).

4. **Island-model AlphaZero** (`train/island_coach.py`): N neural nets on a ring with cross-play between neighbors. Each iteration: 70% self-play + 30% cross-play → train → arena. Ring topology preserves diversity that standard AlphaZero loses.

### Configurable Win Condition

`NonagaState` carries a `win_mode` attribute (`'triangle'` or `'adjacency'`). In adjacency mode, any 2 same-color pieces touching = win. This is used for curriculum pretraining where games need to terminate quickly with clear signal. The mode propagates through `copy()` so MCTS automatically respects it.

### D6 Symmetry (game/symmetry.py)

Hex board has 12 symmetries (6 rotations × 2 reflections). Axial rotation: `(q,r) → (-r, q+r)`. Both board tensors and policy vectors are transformed. All transforms are relative to grid center (3,3).

## Key Constraints

- **Tile removal**: Only edge tiles with ≤4 tile neighbors can be removed (prevents interior holes). Must not be occupied, must not be the tile opponent just moved, and removal must keep board connected (BFS check).
- **Tile placement**: Prefer positions with 2+ adjacent tiles for compactness. Falls back to 1+ adjacent if no 2+ options exist. Must maintain board connectivity.
- **Piece sliding**: Pieces slide in a straight line until hitting another piece or running off tiles. Must move at least one cell.
- **Win condition**: Three pieces of the same player all mutually adjacent (triangle). Configurable: `win_mode='adjacency'` makes 2-adjacent a win (for curriculum pretraining).

## Python ↔ JS Parity

The game engine exists in two implementations that must stay in sync:
- **Python**: `game/nonaga.py` (canonical, used for training)
- **JavaScript**: `web/index.html` GameState class (port, used for browser play)

Changes to game rules (especially `removableEdgeTiles`, `validPlacements`, `slide`) must be applied to both. After rule changes, existing trained models may not match the new rules — retraining may be needed.
