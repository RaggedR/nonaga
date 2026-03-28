# Branching Factor: Why It Didn't Matter

## The mistaken assumption

We originally thought GAs wouldn't work for Nonaga because of the high branching factor (~18 piece moves x 50-200 tile placements = 100-300 compound moves per turn). This concern was misplaced — it conflated two unrelated things:

1. **Tree search** (MCTS, minimax): branching factor is the central constraint. At depth d with branching factor b, you explore b^d nodes. Nonaga's b~100+ makes even shallow trees impractical at realistic sim counts.

2. **Evaluation functions** (GA, hand-crafted heuristics): always 1-ply. You evaluate each legal move's resulting position and pick the best. The cost is O(b), not O(b^d). Whether b is 18 or 200, you just do 18 or 200 evaluations — there's no exponential blowup.

## The real insight

GAs don't "sidestep" or "dodge" the branching problem. The branching problem simply doesn't exist for them. A GA-evolved evaluation function was never going to build a game tree. It evaluates positions, not sequences.

The actual mistake was assuming we needed tree search (AlphaZero/MCTS) to play Nonaga well. We didn't. A 14-weight evaluation function with greedy 1-ply move selection beats the 570K-parameter neural network 50-0. The game's strategic complexity fits entirely in position evaluation — no lookahead required.

## Where branching *did* hurt

Branching factor was devastating for the MCTS-based approaches:

- **Attempt 5**: Excellent value head (near-triangle: +0.52, opponent near-triangle: -0.90) but MCTS at 200 sims got 0% vs random. With a uniform policy prior and 100+ legal moves, each move got ~2 visits at depth.
- **Attempt 7**: Even after fixing the MCTS sign bug, MCTS added nothing over greedy 1-ply with the same value head. Both got 100% vs random, but greedy does 18-100 evaluations while MCTS builds thousands of nodes.
- **AlphaZero training loop**: MCTS visit counts become policy training targets. But when visits are spread thin across 100+ moves, the policy targets are near-uniform — the loop can't bootstrap.

## The branching-search connection

| Approach | Cost per move | Branching matters? | Result |
|----------|--------------|-------------------|--------|
| GA (greedy 1-ply) | O(b) evaluations | No | 100% vs random, beats NN 50-0 |
| MCTS (50 sims) | O(sims) node expansions | Yes — sims/b visits per move | 0-100% depending on value head quality |
| MCTS (200 sims) | O(sims) node expansions | Yes — still too thin | No improvement over greedy |

The lesson: branching factor determines whether *search* is feasible, not whether *evaluation* is feasible. For Nonaga, evaluation alone is sufficient.
