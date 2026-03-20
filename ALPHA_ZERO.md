# How AlphaZero works (plain English)

Traditional game AIs use handwritten rules: "control the centre", "protect your king", etc. AlphaZero takes a different approach — it starts knowing nothing except the rules and teaches itself to play by playing millions of games against itself.

Two components work together:

## The neural network (the "intuition")

The neural network looks at a board position and instantly outputs two things:

1. **Policy** — "which moves look promising?" A probability for every legal move. High probability = the network thinks this move is worth exploring. Think of this as the AI's gut feeling about what to try.
2. **Value** — "who's winning?" A single number from -1 (I'm losing) to +1 (I'm winning). Think of this as the AI glancing at the board and making a snap judgement.

The network is *not* searching ahead — it's pattern-matching on the current board position, like a human player who sees a position and "just knows" it looks good or bad. It's fast but imprecise.

## MCTS (the "thinking ahead")

Monte Carlo Tree Search is where the actual thinking happens. Starting from the current position, MCTS builds a tree of possibilities:

1. **Select** — Starting from the current position, walk down the tree, choosing moves that balance "the network thinks this is good" (policy) with "we haven't explored this enough yet" (visit count). This is the explore-vs-exploit tradeoff.
2. **Expand** — When you reach a position you haven't seen before, ask the neural network: "what do you think of this position?" The network gives a value estimate and policy suggestions.
3. **Backup** — Walk back up the tree, updating every position along the way with what you learned. If the leaf looked bad for the opponent, that makes the parent position look good for us.
4. **Repeat** — Do this 25-100 times (each called a "simulation"). Positions that look promising get explored deeper. Dead ends get abandoned quickly.

After all simulations, MCTS picks the move that got the most visits — not the highest value, but the most visits, because a move that kept drawing attention across many simulations is more reliable than one that looked great once.

**The key insight**: MCTS uses the neural network's intuition to guide which branches to explore, but it doesn't blindly trust it — it verifies by actually searching. The neural network says "try this", MCTS says "let me check".

## Training (the feedback loop)

This is where it gets clever. After each self-play game:

1. At every position during the game, we recorded what MCTS actually chose to do (after careful search). These MCTS visit counts become the **training target for the policy head** — "next time you see a position like this, these are the moves worth considering."
2. We also recorded who won the game. Every position gets labelled +1 (this player won) or -1 (this player lost). This becomes the **training target for the value head** — "positions like this tend to lead to wins/losses."
3. The neural network trains on these examples, updating its weights to better match MCTS's conclusions.
4. Next iteration, the improved network guides MCTS better, which produces better training data, which improves the network further.

This is why it's called "self-play reinforcement learning" — the system generates its own training data by playing itself, and uses the outcome to improve. There's no human games database, no handwritten evaluation function.

## Our network (569,920 parameters)

The network is a small ResNet (residual network), the same architecture used in image recognition. It treats the 7×7 hex grid like a tiny image with 6 "colour channels":

| Channel | What it encodes |
|---------|----------------|
| 0 | Which cells have tiles (the shifting board) |
| 1 | Where my pieces are |
| 2 | Where opponent's pieces are |
| 3 | Whether this is a piece-move or tile-move ply |
| 4 | Which tile was last placed (can't remove it) |
| 5 | Which tiles are removable (edge tiles only) |

The network structure:

```
Input (6×7×7 = 294 values)
  ↓
Conv 3×3 → 64 channels → BatchNorm → ReLU           3,584 params
  ↓
ResBlock × 4 (each: two 3×3 convs + skip connection) 295,424 params
  ↓
  ├─→ Piece policy head → 294 outputs (49 cells × 6 directions)    29,238 params
  ├─→ Tile policy head → 2,401 outputs (49 × 49 source × dest)   237,833 params
  └─→ Value head → 1 output (who's winning, -1 to +1)              3,841 params
                                                    ─────────────
                                              Total: 569,920 params
```

This is tiny by modern standards (GPT-4 has ~1.8 trillion parameters). The tile policy head alone accounts for 42% of all parameters because it has 2,401 possible outputs — every combination of "remove tile X, place it at Y". This is the cost of Nonaga's compound turn structure.

The entire network fits in a 2.2MB file and runs in the browser via pure JavaScript — no GPU needed for inference.
