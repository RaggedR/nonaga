# Design Decisions: Preliminary Training

## The Problem

The AlphaZero model was **worse than random**. It learned to avoid forming triangles entirely. Root cause chain:

1. Nonaga games cycle indefinitely under competent play
2. Self-play with untrained model produces mostly draws (80%+)
3. Draws give value=0, so the value head learns nothing
4. The policy reinforces triangle-avoidance in a vicious cycle
5. Random (which stumbles into triangles accidentally) wins ~50% while the model wins 0%

## What We Tried (and What Failed)

### Attempt 1: Curriculum Pretraining (adjacency wins)
- **Idea**: Pretrain on simpler objective (2-adjacent = win) to bootstrap value head
- **Result**: Curriculum phase worked perfectly (0 draws, 100% decisive, loss 11.2 -> 6.2)
- **But**: Didn't transfer to triangle wins. The model learned "adjacency good" but still couldn't form triangles in the full game. 0% win rate vs random.

### Attempt 2: Self-play with Draw Shaping
- **Idea**: Instead of value=0 for draws, compute shaped value based on positional quality (adjacency pairs, backed pieces, slide-into-able triangle-completing cells)
- **Result**: Draw shaping gave some signal (value loss rose from 0.03 to 0.11), a few wins per iteration appeared
- **But**: Too slow. Each iteration: 100 games x 400 plies x 50 MCTS sims. And the draw shaping signal (clamped ±0.3) was too weak to overcome the triangle-avoidance bias.

### Attempt 3: Training vs Random Opponents
- **Idea**: Random blunders into exploitable positions, giving clear loss signal
- **Result**: Model got strong "you're losing" signal (value loss 0.17), but still 0% win rate
- **Why**: Model only learned from its own moves. When random beats you, your moves are labeled as losing, but you never see what winning looks like.

### Attempt 4: Imitation Learning from Random Games (full game)
- **Idea**: Generate random-vs-random games, train on all moves with win/loss labels
- **Result**: 75% of random games are decisive — way more signal than self-play. But value loss dropped extremely slowly (0.998 -> 0.981 after 3 iterations)
- **Why**: Policy targets are uniform (random play), so policy head learns nothing. Value signal is noisy because random games are random. Optimizer reset per iteration killed Adam momentum.

### Attempt 5: Endgame-Focused Training (THE BREAKTHROUGH)
- **Idea**: Generate 1000 random games, keep only the last 30 plies of decisive games. Train a fresh network on these endgame positions for 20 epochs with persistent optimizer and higher LR (0.003).
- **Result**: Value loss dropped rapidly (0.80 -> 0.59). Value head learned excellent position evaluation:
  - Near-triangle (your move): +0.52
  - Near-triangle (opponent's move): -0.90
  - Random positions: wide range [-0.96, +0.87], std=0.45
- **But MCTS still got 0% wins** (at 25, 50, and 200 sims)

## The MCTS Problem

The value head was great. MCTS couldn't use it. Why?

MCTS allocates search budget proportionally to the **policy prior**. With a uniform prior (trained on random games), it spreads N sims evenly across ~18-100 legal moves. At 200 sims with 18 piece moves, each move gets ~11 visits — not bad. But each visit goes deeper into the tree, where the branching explodes again.

**Greedy 1-ply search** simply evaluates every legal move once and picks the best. This uses the value head directly: for ~18 piece moves, that's 18 forward passes. Result: **50/50 wins vs random, 100% win rate, average game length 20.6 plies.**

## Key Insight: MCTS Needs a Non-Uniform Prior

MCTS is designed for games where you have a **policy prior that concentrates search on promising moves**. AlphaZero's policy improves through the self-play loop: MCTS visit counts become the policy training targets, creating a virtuous cycle.

But bootstrapping this cycle requires either:
1. A policy that's already somewhat non-uniform (so MCTS can focus search)
2. Enough search budget to compensate for a uniform prior (impractical for Nonaga's branching factor)

Our greedy 1-ply approach sidesteps this: it doesn't need a policy prior at all. It just evaluates all children and picks the best.

## Current Architecture

### Training Pipeline
1. Generate 1000 random-vs-random games (fast, no NN)
2. Keep last 30 plies of decisive games only (endgame positions with clear signal)
3. Train fresh network on these ~21K examples with D6 augmentation (254K total)
4. 20 epochs, LR=0.003, persistent Adam optimizer, batch size 256

### Inference (Web)
- **Greedy 1-ply**: For each legal move, apply it, evaluate resulting position with NN, pick highest value
- No MCTS needed — simpler, faster, and currently stronger
- For piece moves (~18 options): 18 forward passes
- For tile moves (~50-200 options): evaluate all, pick best
- Immediate winning moves detected without NN (just check `is_terminal()`)

## Positional Evaluation (Draw Shaping)

The draw shaping function evaluates positions along three axes. While not used during endgame training (which uses binary win/loss), this analysis informed the value head's learning:

1. **Pair score**: For every pair of same-color pieces, find cells adjacent to both (triangle-completing vertices). Score by slide-into-ability:
   - Already touching: +1.0 (one move from triangle)
   - Completing cell is slide-into-able (blocker behind): +0.5
   - Completing cell exists but slide-past: +0.1
2. **Backed pieces**: Pieces against tile edges can't be slid away (+0.03 per backed piece)

### Slide-Into vs Slide-Past
A cell is "slide-into-able" if in some direction, the next cell beyond it is either off the tiles or occupied by another piece. This means a piece sliding toward that cell would stop there. A "slide-past" cell has empty tiles in all directions — pieces fly through it without stopping, making it hard to use for triangle completion.

## Numbers

| Metric | Value |
|--------|-------|
| Training data | 736 decisive random games / 1000 total |
| Endgame examples | 21,237 (last 30 plies of decisive games) |
| Augmented examples | 254,844 (12x D6 symmetry) |
| Training epochs | 20 |
| Final value loss (MSE) | 0.59 |
| vs Random win rate | **100%** (50/50) |
| Avg game length vs Random | 20.6 plies (~10 full turns) |
| Model params | ~570K |

## Attempt 6: GreedyAI Policy Bootstrap

- **Idea**: Use the good value head to generate non-uniform policy targets. At each position, evaluate all legal moves → softmax(values) → policy distribution. Play greedy-vs-random (100% decisive) and greedy self-play (with ε-greedy exploration). Train both policy and value heads. Iterate 3 rounds of 200 games each.
- **Result**: Greedy stayed at 100% vs random. MCTS improved from 0% to ~15-25% during training (small sample), settling at 7-13% in final eval (50-100 sims, 30 games).
- **Why it wasn't enough**: We were solving the wrong problem. See Attempt 7.

## Attempt 7: Fix MCTS Sign Bug (THE REAL BREAKTHROUGH)

- **Idea**: Noticed that `_backup` alternates value sign at every tree level, but Nonaga's two-ply system means piece→tile transitions are SAME player. The sign should only flip on actual player changes.
- **The bug**: In the tree `Root (A, PIECE) → Child (A, TILE) → Grandchild (B, PIECE)`, the alternating backup assigns the wrong sign at the root. It flips twice (even) across one player change (odd), putting the opponent's value at the root. MCTS was literally preferring moves good for the opponent.
- **The fix**: Two changes to `train/mcts.py`:
  1. `_backup`: Only flip `value = -value` when `current.state.current_player != current.parent.state.current_player`
  2. `_select_child`: Only negate `q = -child.q_value` when `child.state.current_player != node.state.current_player`
- **Result**: MCTS at 25 sims went from **0% to 100%** vs random (30-0-0, 30 games). The endgame-trained model's value head was good the whole time — MCTS just couldn't use it correctly.
- **Lesson**: Standard AlphaZero backup assumes every tree level is a player change. Games with compound turns break this silently — MCTS doesn't crash, it just plays terribly. The policy bootstrap (Attempt 6) was unnecessary; the value head alone is sufficient when the signs are correct.

### Both Problems Were Real

Two bugs were stacked:
1. **No learning signal** (draws in self-play) → solved by endgame training on random games (Attempt 5) → good value head
2. **MCTS sign error** (wrong values in tree search) → solved by player-aware backup → MCTS can use the value head

We discovered #1 first. Then spent Attempts 6+ trying to fix the policy prior, when #2 was the actual blocker.

## Attempt 8: Full AlphaZero Self-Play (30 iterations)

- **Setup**: Seeded with endgame_trained.pt (Attempt 5), MCTS with sign fix (Attempt 7). 100 games/iteration, 50 sims, temp_late=1.0, 8 workers.
- **Result**: Zero draws across 3000 self-play games. 22/30 iterations accepted by arena. Avg game length vs random dropped from 43 → 19 plies. 100% win rate throughout.
- **Diversity concern**: Opening diversity narrowed over training — 93% of games converged to the same first move by iteration 29. Piece policy entropy dropped 27% (2.04 → 1.50 bits). Replay buffer of 3 iterations too small to preserve older opening lines. Mitigable with larger buffer, spikier Dirichlet noise, or opening randomization.
- **Status**: Paused. Model is strong enough vs random; further gains require stronger opponents or diversity fixes.

| Metric | iter_0 (endgame) | iter_9 | iter_27 (web) |
|--------|-----------------|--------|---------------|
| vs Random win rate | 100% | 100% | 100% |
| Avg plies vs Random | 43 | 22 | 19.1 |
| Unique first moves (self-play) | 7 | — | 2 |
| Piece policy entropy | 2.04 bits | — | 1.50 bits |

## Current State

- **Web frontend**: GA player is the default AI (14 evolved weights, greedy 1-ply). NN player available via toggle button.
- **GA beats NN**: 50-0 in head-to-head (14 weights vs 570K parameters)
- **Best GA weights**: from island 0 of 200-generation ring-migration run
- **Training pipelines**: GA evolution, single-model AlphaZero, and island-model AlphaZero with cross-play all operational

## SAE Probe on Value Head (Attempt 9)

- **Idea**: Use a Sparse Autoencoder to decompose the NN's 64-dim value head bottleneck into interpretable features. Discover what the network actually learned, potentially extract features for a simpler GA-based evaluation function.
- **Setup**: 256-dim overcomplete SAE with L1 sparsity (λ=0.005), trained on ~115K activations from random game positions evaluated by iteration_39.
- **Result**: 89 active features (13% sparsity per position). Near-perfect reconstruction (MSE→0).
- **Key finding**: Known hand-designed features (adjacency, backed pieces, completing cells) correlate very weakly with SAE features (all r < 0.2). The NN learned fundamentally different representations — likely higher-level relational concepts (threat imminence, mobility squeeze, spatial spread asymmetry) rather than our static geometric features.
- **Caveat**: The model still loses to a human. The SAE faithfully decomposes an undertrained model's intuitions. Features should be re-extracted after stronger training to distinguish genuine strategic concepts from noise.

## Island-Model GA with Ring Topology (Planned — Attempt 10)

Inspired by Langer, Turing & Vega, "From Games to Graphs: Categorical Composition of Genetic Algorithms Across Domains" (ACT 2026, `~/git/lyra-paper/claudius-draft/paper/paper.tex`). Their key result: migration topology determines diversity dynamics *independently of fitness landscape*, with the ordering:

> none > ring > star > random > fully connected

holding with perfect rank correlation across 6 domains (Kendall's W = 1.0, p = 0.00008). Topology explains 28.7× more variance than domain. The mechanism: GA operators as Kleisli morphisms over a population monad; the island functor's laxator magnitude grows with algebraic connectivity λ₂(G), so smaller λ₂ = more diversity. Ring has the smallest λ₂ among connected topologies.

**Relevance to Nonaga**: Our AlphaZero training is effectively single-island fully connected — the worst topology for diversity preservation. The 93% opening collapse is exactly what the paper predicts. An island-model GA with ring migration should maintain strategic diversity by design.

**Planned design**:
- Evolve simple evaluation functions (weighted board features) instead of / alongside NN
- Island model: 5-7 islands × 16 individuals, ring topology
- Migration rate 0.1 every 5 generations (paper's parameters)
- Each individual plays using MCTS + its evaluation function; fitness = tournament win rate
- Feature set: start with known features + SAE-discovered features from stronger model
- Ring topology preserves distinct playing styles across islands

**Two applications of the island model insight**:
1. **GA + MCTS**: Evolve evaluation function weights with island-model GA
2. **AlphaZero restructuring**: Maintain N model checkpoints as islands, ring migration of weights/training data between adjacent islands to preserve strategic diversity

## Attempt 10: Island-Model GA + Island-Model AlphaZero (Implemented)

Two systems tackling the diversity problem from opposite ends.

### Phase 1: Island-Model GA (`ga_evolve.py`)

Evolves a 14-weight evaluation function with greedy 1-ply play. No neural net, no MCTS — pure feature-weighted search at ~5ms/game.

**Parameters** (defaults in `GAConfig`):
- 5 islands × 16 individuals = 80 total
- 200 generations
- Ring topology: island i migrates to island (i+1) % n
- Migration: 10% of population every 5 generations (1-2 individuals)
- Tournament selection (size 3), BLX-α crossover (α=0.5, rate 0.8), Gaussian mutation (σ=0.1, rate 0.15/gene)
- Elitism: best individual per island preserved each generation
- Fitness: intra-island round-robin tournament, 1 game per matchup (120 games/island/gen)

**14 board features** (genome):

| # | Feature | Description |
|---|---------|-------------|
| 0-1 | adj_pairs | Own/opponent adjacency pairs (pieces touching) |
| 2-3 | completing_into | Own/opponent triangle-completing cells reachable by slide |
| 4-5 | completing_past | Own/opponent completing cells NOT slide-reachable |
| 6-7 | backed | Own/opponent pieces backed against tile edge |
| 8-9 | mobility | Own/opponent total slide destinations |
| 10-11 | min_dist | Own/opponent min pairwise hex distance |
| 12-13 | max_dist | Own/opponent max pairwise hex distance |

Feature computation reuses patterns from `train/self_play.py:_shaped_draw_value()` and `sae_probe.py:compute_board_features()`.

**Smoke test**: `python ga_evolve.py --islands 3 --pop 4 --gens 5 --tournament-games 1 --eval-vs-random 10`

**Full run result** (200 gens, 5×16, ~3.5 hours):
- **100% win rate vs random** (50 games, avg 21.5 plies)
- **100% win rate vs the 570K-parameter neural network** (20 games) — 14 evolved weights beat the NN trained by AlphaZero
- Three islands tied at 0.833 fitness, inter-island divergence up to L2=5.83

**Best evolved weights** (island 0, fitness 0.833):

| Feature | Weight | Interpretation |
|---------|--------|----------------|
| `own_completing_past` | **-6.32** | Avoid false-hope positions (completing cells you can't land on) |
| `opp_backed` | **-2.92** | Good when opponent pieces are stuck against edges |
| `opp_completing_into` | **-2.22** | Block opponent's reachable triangle-completing cells |
| `own_adj_pairs` | **+1.48** | Keep your own pieces adjacent |
| `own_max_dist` | **-1.15** | Keep pieces compact (penalise spread) |
| `own_completing_into` | +0.17 | Slight preference for reachable completing cells |
| `own_mobility` | 0.00 | Mobility irrelevant (!) |
| `opp_mobility` | -0.09 | Opponent mobility also near-zero |

**Key insight: 14 weights > 570K parameters.** The GA found a more effective evaluation function in a vastly smaller search space. The NN has the capacity to represent this and more, but AlphaZero's self-play loop never discovered it — diversity collapse was the bottleneck, not model capacity. The dominant feature (`own_completing_past` at -6.3) encodes a subtle distinction the NN missed: slide-past completing cells are traps that look like near-wins but are unreachable.

**Diversity dynamics**: Ring migration created a wave of diversity propagating around the ring. Islands 2 and 3 converged (L2=0.40) while islands 1 and 4 maximally diverged (L2=5.83). Island 1 learned negative `opp_completing_past` (-0.88) while island 0 learned strong positive (+3.98) — different theories of the game coexisting, exactly as the ring topology theory predicts.

### Phase 2: Island-Model AlphaZero with Cross-Play (`train/island_coach.py`)

N neural network models on a ring. The key innovation over the original plan: **cross-play** between ring neighbors, not just training data migration.

**Why cross-play matters**: The original design was N independent self-play loops with occasional data sharing — essentially N separate AlphaZero runs. This isn't genuinely population-based because neural net training uses gradient descent, not crossover/mutation. There's no population within each island for ring topology to act on.

Cross-play fixes this: when island i plays island (i+1), the resulting positions come from *different strategies meeting* — board states that neither model would encounter in self-play. Both models train on these cross-games, breaking the self-play echo chamber. The ring controls who plays who: island 0 never directly faces island 3's strategy, preserving slow diversity diffusion.

**Parameters** (defaults in `train/config.py`):
- 5 islands, ring topology
- 30% cross-play rate (fraction of games per iteration that are cross-play with ring neighbor)
- 70% self-play rate (standard AlphaZero data generation)
- Training data migration: 10% of replay buffer every 5 iterations (supplements cross-play)
- Per-island: own model checkpoint, own replay buffer, own arena gating

**Each iteration**:
1. **Self-play**: each island plays itself (70% of games) — standard AlphaZero
2. **Cross-play**: each ring edge plays games (30% of games) — diversity engine
3. **Train**: each island trains on combined self-play + cross-play data
4. **Arena**: each island's new model vs its previous checkpoint

**Smoke test**: `python -m train.island_coach --islands 2 --iterations 1 --games 4 --sims 5 --cross-play-rate 0.5`

**Design rationale for 30% cross-play**: Enough foreign-strategy exposure to break the echo chamber, but 70% self-play preserves each island's own learning trajectory. Too much cross-play and islands can't develop distinct strategies; too little and they converge like standard AlphaZero.

**First run result** (3 islands, 10 iterations, 20 games/iter, 50 MCTS sims, 30% cross-play, ~3 hours):

| | Island 0 | Island 1 | Island 2 |
|---|---|---|---|
| Loss (iter 0 → 9) | 11.2 → 7.7 | 11.4 → 7.1 | 11.4 → 7.7 |
| Self-play avg plies (iter 0 → 9) | 75 → 34 | 72 → 38 | 54 → 44 |
| Arena accepts / total | 5/9 | 5/9 | 4/9 |

- **Value disagreement rose** from 0.137 (iter 0) to 0.280 (iter 5) — diversity *increasing*, opposite of standard AlphaZero's 93% opening convergence
- **Cross-play games were longer** than self-play (80 vs 34-38 avg plies at iter 9) — the islands developed complementary strategies that contest each other more than they contest themselves
- Island 2 rejected at iterations 1-3 but recovered, finishing with the strongest final iteration (100% arena, value loss 0.68 — lowest of all islands)
- Island trajectories diverged: island 0 learned to win fast (34 plies), island 1 played medium games (38), island 2 played longer (44)

**Full training run** (5 islands, 50 iterations, 50 games/iter, 50 MCTS sims, 30% cross-play, ~12-18 hours estimated):

Reduced from the original plan (100 iters × 100 games × 100 sims → ~4 days) to a feasible overnight run. The first test at 50 sims already showed good diversity dynamics, so halving search depth is acceptable. Key parameter choices:
- 50 sims (vs 100): Still enough tree search to produce non-uniform policy targets; the sign-fixed MCTS works well even at 25 sims
- 50 games/iter (vs 100): 35 self-play + 15 cross-play per island per iteration; fewer games but more iterations of the train loop
- 50 iterations (vs 100): Should be enough to see whether cross-play diversity translates to strength — the single-model AlphaZero peaked by iteration ~27

Results (53 iterations completed across multiple sessions):

- `max_game_plies` reduced from 500 → 200 mid-run (iteration 12) to cut outlier games
- Cross-play parallelized mid-run to speed up (multiprocessing pool, same as self-play)
- Iteration times varied: 12 min (iter 0) → 25-40 min (typical) → 200 min (outliers from long cross-play games)
- Loss dropped from ~11 (iter 0) to ~6.2 (iter 53), plateauing around iter 30
- Diversity held at 0.24-0.35 throughout — cross-play successfully prevented the 93% opening convergence seen in standard AlphaZero

**Head-to-head vs GA at iteration 20 (greedy 1-ply NN vs greedy 1-ply GA, 20 games):**

| Island | GA win rate | Notes |
|--------|-----------|-------|
| 0 | 100% | GA dominates |
| 1 | 100% | GA dominates |
| **2** | **50%** | Competitive — confirmed at 50 games |
| 3 | 100% | GA dominates |
| 4 | 100% | GA dominates |

**Head-to-head vs GA at iteration 53 (50 games each):**

| Island | GA win rate | Change from iter 20 |
|--------|-----------|-------------------|
| **0** | **50%** | Improved (was 100%) |
| **1** | **50%** | Improved (was 100%) |
| 2 | 100% | Regressed (was 50%) |
| 3 | 100% | No change |
| **4** | **50%** | Improved (was 100%) |

**Key findings**:

1. **Ring migration diffuses strategies**: Island 2 peaked early (50/50 at iter 20) but regressed to 100% GA wins by iter 53. Meanwhile its ring neighbors (islands 0 and 4, both adjacent on the ring) improved from 100% to 50%. The strong strategy diffused outward but got diluted at the source — exactly the ring topology trade-off.

2. **3 of 5 islands reached parity, but none surpassed the GA**: The best any island achieved was 50/50 against 14 evolved weights. The NN's 570K parameters can match but not exceed what the GA found.

3. **Loss plateaued while play strength improved**: Island 2's loss barely moved from iter 23-53 (6.5→6.2) yet islands 0, 1, 4 went from losing to tying. Training loss is a poor proxy for play strength in this regime.

**GA wins overall**: The 14-weight evolved evaluation function remains the strongest Nonaga AI. The NN's advantage (capacity to represent complex patterns) is offset by the difficulty of discovering the right patterns through self-play. The GA's search space is tiny (14 reals) but perfectly aligned with the game's strategic structure.

### Why Greedy 1-Ply Beats MCTS (and Why This Matters)

Both the GA player and the NN evaluation use **greedy 1-ply search**: evaluate all legal moves one step ahead, pick the best. No tree search. This sidesteps Nonaga's high branching factor (~18 piece moves × ~100+ tile placements) which made MCTS ineffective throughout the project:

- **Attempt 5**: Endgame-trained value head was excellent (near-triangle: +0.52, opponent's near-triangle: -0.90) but MCTS at 25, 50, and 200 sims still got **0% wins vs random**. The uniform policy prior spread search budget evenly across 100+ moves — each move got ~2 visits at depth, not enough to find winning lines.
- **Attempt 7**: Fixed the MCTS sign bug (two-ply turns broke alternating backup). MCTS at 25 sims jumped to **100% vs random** — but only because the value head was already good enough to guide shallow search. The policy head never became strong enough to focus deep search.
- **Greedy 1-ply** with the same value head also got 100% vs random, at a fraction of the compute. It evaluates ~18-100 positions (one per legal move) instead of building a tree of thousands of nodes.

**The implication for the GA comparison**: AlphaZero trains with MCTS (building policy targets from visit counts) but we evaluate with greedy 1-ply (value head only, ignoring the policy). The NN was optimized for a different use case than how we tested it. A fairer comparison would be NN+MCTS vs GA — the policy head might direct search toward the right moves even if the value head alone isn't as sharp as the GA's 14 weights. This comparison has not been run.

### Why the NN Can't Find What the GA Found

The GA's dominant feature (`own_completing_past = -6.3`) is a subtle spatial relationship: cells that *look like* triangle-completing positions but can't actually be reached by sliding a piece into them. This requires:

1. Identifying pairs of same-color pieces
2. Finding cells adjacent to both (completing vertices)
3. Checking whether any piece can slide *into* (not past) each completing cell

The GA is *given* these 14 precomputed features and only needs to find weights — a 14-dimensional optimization. The NN must discover these features from raw 6×7×7 binary board tensors through self-play game outcomes. It's learning both the features AND the weights simultaneously, from a noisy signal (win/loss/draw) that doesn't indicate *which* feature mattered. Self-play also acts as a local optimizer — it finds strategies that beat the previous version, not globally optimal strategies — so if early training doesn't stumble onto the "completing_past" concept, later training may never escape that basin.

## Possible Future Work

1. ~~**Export to web**~~: Done — greedy 1-ply in JS, updated to iteration_27
2. ~~**Policy bootstrap**~~: Unnecessary — sign fix makes MCTS work with uniform prior
3. ~~**Self-play loop**~~: Done — 30 iterations, zero draws, model wins in ~19 plies
4. **Web upgrade to MCTS**: Update browser frontend from greedy 1-ply to MCTS for stronger play (requires JS MCTS + NN inference)
5. ~~**Diversity fixes**~~: Partial — increased replay buffer (3→8), spikier Dirichlet noise (0.3→0.15), 200 sims. Deeper fix: ~~island-model topology (Attempt 10)~~ — implemented as cross-play island AlphaZero
6. **Stronger opponents**: Evaluate model-vs-model across checkpoints, or find/build a stronger Nonaga AI
7. ~~**Island-model GA**~~: Done — `ga_evolve.py` with ring topology (Attempt 10, Phase 1)
8. **SAE re-probe**: Re-run SAE on stronger model after extended training to extract reliable features for GA
9. ~~**GA→web export**~~: Done — GAPlayer is the default browser AI with hardcoded evolved weights
10. **League training**: Keep historical checkpoints as sparring partners — model occasionally plays against older versions of itself for additional diversity
