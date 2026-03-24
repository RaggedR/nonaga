# Lessons Learned

## MCTS sign handling in multi-ply games

Standard AlphaZero MCTS alternates the value sign at every tree level (`value = -value`). This assumes every move switches the player. In games with compound turns (like Nonaga's piece move + tile move by the same player), this is WRONG. The sign must only flip when `current_player` actually changes between parent and child nodes.

Symptom: MCTS plays worse than random despite a good value head. The search actively prefers opponent-favourable moves.

Fix: Compare `node.state.current_player` between parent and child in both `_backup` and `_select_child`.

## Diagnose infrastructure before algorithms

When a trained model works with greedy evaluation but fails with MCTS, the problem is likely in MCTS itself — not in the policy prior, branching factor, or training data. Check the basics (sign handling, value conventions) before building elaborate workarounds.

## Endgame-focused training for games with many draws

For games where competent play cycles indefinitely (draws), training only on the last N plies of decisive random games gives clean win/loss signal without the noise of early/mid-game positions.

## Migration topology governs diversity, not fitness landscape

When evolving populations (GA or self-play), the topology of how subpopulations exchange information determines diversity dynamics 28.7× more than the problem domain. The ordering: none > ring > star > random > fully connected. Ring is optimal among connected topologies (smallest algebraic connectivity λ₂). Single-population training (e.g., standard AlphaZero self-play) is effectively fully connected — the worst for diversity.

Reference: Langer, Turing & Vega, "From Games to Graphs" (ACT 2026). The mechanism: island functor's laxator magnitude grows with λ₂(G).

## SAE on weak models finds weak features

Don't run interpretability analysis (SAE, probing) on undertrained models expecting to find deep strategic concepts. The SAE faithfully decomposes whatever the network has learned — if the model can't beat a human, its internal representations are incomplete. Train first, interpret second.
