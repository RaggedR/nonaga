# Training Notes

See [ALPHA_ZERO.md](ALPHA_ZERO.md) for a plain-English explanation of how AlphaZero works and our network architecture.

## Training Run 1 (Original, iterations 0-49)
- **Date**: March 18, 2025
- **Config**: 100 games/iter, 100 MCTS sims, max_plies=100, temp_threshold=15
- **Device**: MPS (Apple Silicon)
- **Self-play**: Sequential (single process)
- **Duration**: ~2 hours/iteration, ran overnight
- **Result**: 50 iterations completed. Model learned basic play but not strong.
- **Issue**: max_plies=100 was likely too low even here — unclear how many games produced actual wins vs draws.

## Training Run 2 (Accidental restart, iterations 0-9)
- Accidentally restarted from scratch instead of resuming. Produced weak checkpoints that overwrote originals for iterations 0-9.
- Resumed from iteration 49 weights partway through.

## Training Run 3 (Sequential resume from 49, iterations 1-5)
- **Date**: March 19, 2025
- **Config**: Same as Run 1
- **Resumed from**: iteration_49.pt
- **Issue**: Checkpoint numbering restarted from 1, overwriting Run 2's checkpoints.

## Training Run 4 (Parallel, iterations 6-15)
- **Date**: March 19-20, 2025
- **Config**: 100 games/iter, 100 sims, max_plies=100, temp_threshold=15
- **Self-play**: Parallel (8 workers on CPU via multiprocessing)
- **Speedup**: ~3x (40 min/iter vs 2 hours sequential)
- **Critical issue**: ALL games ended in draws (D=100, avg_plies=100.0).
  - Value head received only zero targets — learned nothing.
  - Root cause: max_plies=100 too short (median random game ~171 plies).

## Training Run 4b (max_plies bump, iteration 15)
- **Config change**: max_plies=100 → 300, temp_threshold=15 → 30
- **Result**: Still all draws! avg_plies=300.0. MCTS play is too defensive.
  - With temp=0 (greedy) after threshold, both players enter defensive loops.
  - Random play produces wins (median 171 plies), but MCTS never does.

## Training Run 5 (temp_late=0.3, from iteration 15) — FAILED
- **Date**: March 20, 2025
- **Config changes**: max_plies=300, temp_threshold=30, temp_late=0.3, 50 games, 8 workers
- **Result**: Still all draws (D=50, avg_plies=298.8). temp=0.3 not enough.

## Post-mortem: Why all runs after iteration 49 failed

**The iteration_49 model has a degenerate policy that actively avoids forming triangles.**

Verification test:
- No network (random policy), 10 sims: **5/20 games produce wins** in 300 plies
- Iteration 49 model, 10 sims, temp=0.3: **0/10 wins**
- Iteration 49 model, 25 sims, temp=0.3: **0/10 wins**
- Iteration 49 model, 50 sims, temp=0.3: **0/10 wins**

The model's policy priors guide moves away from winning positions at every sim level.
This likely happened during Run 1: once the model became defensive enough to prevent
wins within 100 plies, all games became draws, value targets became all zeros, and the
policy was trained only on draw-producing move distributions — reinforcing the avoidance
of triangles in a vicious cycle.

**Root cause chain:**
1. max_plies=100 was too short for Nonaga (median random game ~171 plies)
2. As the model improved, games exceeded 100 plies → all draws
3. Value head got zero targets → stopped learning
4. Policy learned from draw-only games → learned to avoid triangles
5. This made games even longer → more draws → positive feedback loop

**No amount of tuning (sims, temperature, ply limit) can fix a degenerate policy.**
The only option is to start fresh with correct settings.

## Training Run 6 (Fresh start)
- **Date**: March 20, 2025
- **Config**: 50 games/iter, 25 MCTS sims (weak play → games terminate),
  max_plies=500, temp_threshold=30, temp_late=0.5, 8 workers
- **Starting from**: Random initialization (no checkpoint)
- **Rationale**: Fresh network has uniform policy → games play ~randomly → wins occur →
  value head gets signal from day 1. Lower sims keeps play weak enough to produce wins
  in early iterations. Can increase sims later once model is strong.

## Memory Constraints (16GB Mac)

Parallel self-play with 8 workers is memory-intensive: each worker holds a full copy of the neural network plus its own MCTS tree. On a 16GB Mac, this left little headroom for the training phase.

Config adjustments made to stay within RAM:
- **Replay buffer**: 5 → 3 iterations (each iteration produces ~60K augmented examples)
- **Training epochs**: 5 → 2 per iteration (fewer passes over the buffer)
- **Batch size**: 64 → 128 (fewer batches = less overhead, faster epochs)

The replay buffer reduction is the most impactful — 5 iterations × ~60K examples × the tensor sizes was pushing memory limits when combined with the parallel workers. Shrinking to 3 keeps the training data diverse enough while fitting in RAM.

## Key Lessons
1. **max_game_plies must exceed median game length** — Nonaga random games take ~171 plies median, 621 at 90th percentile.
2. **Temperature must not drop to zero** if the game can cycle. MCTS with temp=0 produces deterministic defensive loops that never terminate.
3. **A degenerate model cannot be rescued by tuning hyperparameters.** If the policy has learned to avoid winning, it poisons all future self-play regardless of sims/temperature.
4. **The original AlphaZero worked because games (chess/Go) have natural endings.** Nonaga can cycle indefinitely, so ply limits and temperature management are critical from the start.
5. **Parallel self-play on CPU gives ~3x speedup** with 8 workers vs single-process MPS.
6. **12x D6 augmentation** amplifies data heavily — 50 games → ~60K examples.
7. **Checkpoint numbering** must account for resume offset to avoid overwriting.

## Web Deployment
- Current browser game uses iteration_49 weights (from Run 1)
- Export: `python -m model.export_weights checkpoints/iteration_N.pt --output-dir web`
- Browser runs 30 MCTS sims in pure JS (no ONNX runtime)
