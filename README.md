# Nonaga

A browser-playable implementation of Nonaga with multiple AI approaches. We tried AlphaZero-style neural network training (self-play, MCTS, 570K parameters) but it struggled with Nonaga's draw-heavy dynamics. What actually worked was a genetic algorithm evolving just 14 real-valued weights for a board evaluation function, arranged in an island model with ring-topology migration to preserve strategic diversity. The GA's 14 weights beat the neural network 50-0.

**[Play it now](https://raggedr.github.io/nonaga/)** | [Rules (PDF)](https://www.steffen-spiele.de/fileadmin/media/Spiele/Nonaga/Nonaga_EN.pdf)

Or run locally:

```bash
cd web && python -m http.server 8000
```

Drag or click pieces to move them. The AI plays as Player 2 (teal). Toggle between GA and NN players in the browser.

## How it works

**Game engine** in Python (`game/`) and JavaScript (`web/index.html`) — two implementations that must stay in sync.

Each turn is decomposed into two plies:
1. **Piece move**: slide one of your 3 pieces along a hex direction (~18 possible actions)
2. **Tile move**: remove an unoccupied edge tile and place it elsewhere (~50-200 actions)

**Win condition**: get your three pieces into a triangle (all mutually adjacent).

## The AI story

### What failed: AlphaZero self-play

Standard AlphaZero training couldn't produce a competent player. The root cause chain:

1. Nonaga games cycle indefinitely under competent play
2. Self-play with an untrained model produces 80%+ draws
3. Draws give value=0, so the value head learns nothing
4. The policy reinforces triangle-avoidance in a vicious cycle
5. Random play (which stumbles into triangles accidentally) wins ~50% while the trained model wins 0%

We tried curriculum pretraining (adjacency wins), draw shaping, training against random opponents, and imitation learning from random games. None broke through. See `DESIGN_DECISIONS.md` for the full 12-attempt saga.

### The endgame bootstrap (Attempt 5)

The first breakthrough: generate random-vs-random games, keep only the last 30 plies of decisive games, and train the value head on those endgame positions. This gave the NN excellent position evaluation (near-triangle: +0.52, opponent near-triangle: -0.90) — but MCTS still got 0% wins because a sign bug in the two-ply backup was inverting values (Attempt 7 fixed this).

With both fixes, greedy 1-ply search (evaluate all legal moves one step ahead, pick the best) achieved 100% vs random. No tree search needed.

### What won: island-model GA (Attempt 10)

An island-model GA evolving 14-weight evaluation functions on a ring topology. No neural net, no MCTS — pure feature-weighted greedy 1-ply play at ~5ms/game.

- 5 islands x 16 individuals, 200 generations, ring migration
- **100% win rate vs random** (50 games, avg 21.5 plies)
- **100% win rate vs the 570K-parameter NN** (50 games)

The 14 features: adjacency pairs, triangle-completing cells (slide-into vs slide-past), backed pieces, mobility, and pairwise hex distances — computed for both players.

The dominant evolved weight: `own_completing_past = -6.3`. This penalises cells that *look like* triangle-completing positions but can't actually be reached by sliding a piece into them. This single feature, which distinguishes "slide-into" from "slide-past" cells, accounts for most of the GA's strength.

### Partial success: island-model AlphaZero with cross-play (Attempt 10, Phase 2)

The best NN approach: N neural networks on a ring, with 70% self-play and 30% cross-play between ring neighbors.

Cross-play was the key ingredient standard AlphaZero lacked. In normal self-play, the model plays itself, sees the same strategy on both sides, and converges (93% opening collapse by iteration 29). Cross-play between ring neighbors means different strategies *meet*, producing board states neither model encounters in self-play alone. Value disagreement between islands *increased* over training (0.137 to 0.280) — the opposite of standard AlphaZero's convergence.

**Results** (5 islands, 53 iterations, 50 MCTS sims, 30% cross-play):

| Iteration | Islands at 50/50 vs GA | Islands where GA dominates |
|-----------|----------------------|--------------------------|
| 20 | 1 (island 2) | 4 |
| 53 | 3 (islands 0, 1, 4) | 2 |

Island 2's strong strategy diffused outward along the ring to neighbors 0 and 4, but got diluted at the source (regressed from 50% to 0%). This is exactly the ring topology trade-off: slow diffusion preserves diversity but the originating island loses its edge.

Three of five islands matched the GA. None surpassed it. This is the only NN training approach that achieved any wins against the GA — direct training against the GA (Attempt 11) and GA self-play distillation (Attempt 12) both scored 0%.

### What failed: training the NN from the GA

Two attempts to transfer the GA's knowledge into the NN both failed:

**Attempt 11 (NN plays against GA)**: The NN plays the GA with policy imitation (learn what the GA would do) and game-length-shaped values. 0% eval win rate across 12 iterations. Three problems stacked: (1) policy imitation trains the policy head but move selection uses the value head — the knowledge never reaches decisions; (2) half the training positions come from the NN's own bad moves, which the GA would never encounter; (3) the shaped value signal was trivially easy to fit but didn't teach positional evaluation.

**Attempt 12 (NN watches GA self-play)**: Pure supervised learning from GA-vs-GA games. 0 decisive games out of 1,000 — two identical GAs draw every single game. Same draw problem that plagued AlphaZero from the start. Decisive outcomes require asymmetry in skill, which means self-play between equals produces no learning signal. This is a property of Nonaga, not the training method.

### Why the GA wins

The GA's search space (14 reals) is perfectly aligned with the game's strategic structure. The NN must discover the same features from raw 6x7x7 binary board tensors through noisy game outcomes — learning both features AND weights simultaneously. The dominant feature requires multi-hop spatial reasoning (find piece pairs, find completing vertices, check slide reachability) that a small ResNet is poorly suited to discover from self-play.

Branching factor was never the GA's problem. GAs with evaluation functions are inherently 1-ply — they evaluate positions, not sequences. The branching factor (~100-300 compound moves) only hurts tree search (MCTS), which the GA never uses. See `BRANCHING.md`.

### Same pattern in Blokus

We found the same result in **Blokus** — see [MCTS_Laboratory PR #107](https://github.com/TGALLOWAY1/MCTS_Laboratory/pull/107). An Island-Model Genetic Algorithm evolving 10 heuristic feature weights (zero lookahead) beat FastMCTS 8-to-1 in a 4-player Blokus arena. The insight is the same as Nonaga: **features matter more than search when rollout quality is poor.** In both games, high branching factors (~80-500 in Blokus, ~300 in Nonaga) make random rollouts nearly uninformative, so MCTS wastes compute on noise while a well-tuned evaluation function captures strategic knowledge directly.

## Commands

```bash
# Run tests
python tests/test_game.py

# Island-model GA evolution (the strongest AI)
python ga_evolve.py
python ga_evolve.py --islands 5 --pop 16 --gens 200              # full run
python ga_evolve.py --islands 3 --pop 4 --gens 5 --tournament-games 1  # smoke test

# Island-model AlphaZero with cross-play
python -m train.island_coach --islands 5 --iterations 100 --games 100 --sims 100
python -m train.island_coach --islands 2 --iterations 2 --games 5 --sims 10  # smoke test

# Single-model AlphaZero (historical — superseded by island model)
python -m train.coach --iterations 100 --games 100 --sims 100

# Endgame bootstrap training
python -u fast_train.py

# Export NN weights for browser
python -m model.export_weights checkpoints/endgame_trained.pt --output-dir web
```

Uses MPS on Apple Silicon, CUDA on Nvidia, CPU otherwise.

## Architecture

```
game/               Python game engine (canonical)
  nonaga.py         Game state, rules, move generation
  hex_grid.py       Axial hex coordinate system (7x7 grid, side-4 hexagon)
  symmetry.py       D6 symmetry group (12 transforms for data augmentation)

model/              Neural network (~570K params)
  network.py        ResNet: 6x7x7 input -> piece policy + tile policy + value
  export_weights.py Export to binary format for browser
  export_onnx.py    Export to ONNX (alternative, not currently used)

train/              AlphaZero training
  coach.py          Single-model: self-play -> augment -> train -> arena
  island_coach.py   Island-model: N models on ring with cross-play
  mcts.py           Monte Carlo Tree Search with NN priors
  self_play.py      Parallel self-play game generation
  config.py         Hyperparameters

ga_evolve.py        Island-model GA with ring topology (14-weight eval function)
fast_train.py       Endgame bootstrap training (random games -> value head)
train_vs_ga.py      Train NN against GA (Attempt 11 — failed)
train_distill.py    Distill GA self-play into NN (Attempt 12 — failed)

web/                Browser game (single-file HTML + exported weights)
  index.html        JS engine + GA player (default) + NN player (toggle) + SVG UI

tests/              Game engine tests
```

## Further reading

- `DESIGN_DECISIONS.md` — full training history (12 attempts, what worked and what didn't)
- `BRANCHING.md` — why branching factor is irrelevant to evaluation-based AI

## License

MIT
