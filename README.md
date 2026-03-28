# Nonaga

A browser-playable implementation of Nonaga. The goal is to train an AlphaZero-style AI opponent, but the AI currently plays no better than random — too many self-play games end in draws, starving the value head of learning signal.

**[Play it now](https://raggedr.github.io/nonaga/)** | [Rules (PDF)](https://www.steffen-spiele.de/fileadmin/media/Spiele/Nonaga/Nonaga_EN.pdf)

Or run locally:

```bash
cd web && python -m http.server 8000
```

Drag or click pieces to move them. The AI plays as Player 2 (teal).

## How it works

**Game engine** in Python (`game/`) and JavaScript (`web/index.html`) — two implementations that must stay in sync.

**AI** uses Monte Carlo Tree Search (MCTS) with neural network policy/value priors, the same approach as AlphaZero. We use MCTS instead of minimax because Nonaga's compound turns (move a piece, then relocate a tile) create a branching factor of ~300 — too wide for alpha-beta search.

Each turn is decomposed into two plies in the game tree:
1. **Piece move**: slide one of your 3 pieces along a hex direction (~18 possible actions)
2. **Tile move**: remove an unoccupied edge tile and place it elsewhere (~50-200 actions)

**Win condition**: get your three pieces into a triangle (all mutually adjacent).

## Training status

**The AI currently plays no better than random.** Training with AlphaZero-style self-play has not yet produced a competent player.

### The draw problem

The core issue is that too many self-play games end in draws, which means the value head gets no useful learning signal:

- **Games can cycle indefinitely** — unlike chess or Go, there's no natural game-ending mechanism. Pieces slide and tiles shift, but neither side is forced toward a conclusion
- **Degenerate equilibrium trap** — the model learns to *avoid* forming triangles entirely. Once defensive enough that games exceed the ply limit, all outcomes become draws, value targets become zeros, and the policy reinforces triangle-avoidance in a vicious cycle
- **Branching factor of ~300** per compound turn means MCTS needs many simulations to explore meaningfully, and even then most games don't reach decisive positions
- **Each iteration takes ~1-2 hours** on Apple Silicon (MPS), making experimentation slow

See `TRAINING_NOTES.md` for the full history of what we've tried.

### The GA alternative

We found the same pattern in **Blokus** — see [MCTS_Laboratory PR #107](https://github.com/TGALLOWAY1/MCTS_Laboratory/pull/107). An Island-Model Genetic Algorithm evolving 10 heuristic feature weights (zero lookahead) beat FastMCTS 8-to-1 in a 4-player Blokus arena. The insight is the same as Nonaga: **features matter more than search when rollout quality is poor.** In both games, high branching factors (~80-500 in Blokus, ~300 in Nonaga) make random rollouts nearly uninformative, so MCTS wastes compute on noise while a well-tuned evaluation function captures strategic knowledge directly.

### Train it yourself

```bash
pip install -r requirements.txt

# Quick smoke test (~2 min)
python -m train.coach --iterations 2 --games 5 --sims 10

# Full training (~days)
python -m train.coach --iterations 100 --games 100 --sims 100

# Export weights for browser play
python -m model.export_weights checkpoints/iteration_N.pt --output-dir web
```

Uses MPS on Apple Silicon, CUDA on Nvidia, CPU otherwise.

## Architecture

```
game/           Python game engine (canonical)
  nonaga.py     Game state, rules, move generation
  hex_grid.py   Axial hex coordinate system (7x7 grid, side-4 hexagon)
  symmetry.py   D6 symmetry group (12 transforms for data augmentation)

model/          Neural network (~570K params)
  network.py    ResNet: 6x7x7 input → piece policy + tile policy + value
  export_weights.py   Export to binary format for browser
  export_onnx.py      Export to ONNX (alternative, not currently used)

train/          AlphaZero training loop
  coach.py      Orchestrator: self-play → augment → train → arena
  mcts.py       Monte Carlo Tree Search with NN priors
  self_play.py  Parallel self-play game generation
  config.py     Hyperparameters

web/            Browser game (single-file HTML + exported weights)
  index.html    Complete game: JS engine + MCTS + NN forward pass + SVG UI

tests/          Game engine tests
```

## License

MIT
