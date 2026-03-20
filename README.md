# Nonaga

A browser-playable implementation of Nonaga with an AlphaZero-style AI opponent.

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

The AI is partially trained and plays at a basic level. Training is ongoing.

### Why training is slow

Nonaga is a surprisingly difficult game for AlphaZero-style training:

- **Branching factor of ~300** per compound turn means MCTS needs many simulations to explore meaningfully
- **Games can cycle indefinitely** — unlike chess or Go, there's no natural game-ending mechanism. Pieces slide and tiles shift, but neither side is forced toward a conclusion. This required careful tuning of ply limits (500) and temperature schedules to ensure games actually produce wins during self-play
- **Degenerate equilibrium trap** — our first 50-iteration training run produced a model that learned to *avoid* forming triangles entirely. Once the model got defensive enough that games exceeded the ply limit, all outcomes became draws, value targets became zeros, and the policy reinforced triangle-avoidance in a vicious cycle. We had to restart from scratch with weaker initial play (fewer MCTS sims) to ensure wins occur early and the value head gets learning signal
- **Each iteration takes ~1-2 hours** on Apple Silicon (MPS): 50 self-play games with 25 MCTS simulations each, D6 symmetry augmentation (12x data), neural network training, then arena evaluation

The current browser model is from an earlier training run (iteration 49). A fresh training run with corrected hyperparameters is in progress — see `TRAINING_NOTES.md` for the full history of what went wrong and what we learned.

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
