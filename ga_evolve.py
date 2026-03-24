"""
Island-model GA evolving evaluation function weights for Nonaga.

Each individual is a vector of 14 real-valued weights. The evaluation function
computes board features from a NonagaState and returns tanh(dot(features, weights)).

Islands are arranged in a ring topology. Migration sends top individuals from
island i to island (i+1) % n every migration_freq generations.

Game playing uses greedy 1-ply: for each legal move, apply it, evaluate the
resulting position with the weighted feature function, pick the highest.

Usage:
    # Full run (default: 5 islands x 16 individuals, 200 generations)
    python ga_evolve.py

    # Quick smoke test
    python ga_evolve.py --islands 5 --pop 4 --gens 10 --tournament-games 2

    # Custom parameters
    python ga_evolve.py --islands 5 --pop 16 --gens 200 --eval-vs-random 100
"""

import argparse
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from game.nonaga import NonagaState, PlyType, Player
from game.hex_grid import NEIGHBORS, DIRECTIONS, VALID_SET, idx_to_qr, qr_to_idx


# ── Feature computation ──────────────────────────────────────────────

NUM_FEATURES = 14

FEATURE_NAMES = [
    "own_adj_pairs",        # 0
    "opp_adj_pairs",        # 1
    "own_completing_into",  # 2
    "own_completing_past",  # 3
    "opp_completing_into",  # 4
    "opp_completing_past",  # 5
    "own_backed",           # 6
    "opp_backed",           # 7
    "own_mobility",         # 8
    "opp_mobility",         # 9
    "own_min_dist",         # 10
    "opp_min_dist",         # 11
    "own_max_dist",         # 12
    "opp_max_dist",         # 13
]


def _is_slide_into(cell, state):
    """Check if a piece could stop at this cell (blocker or tile-edge behind it)."""
    occ = state.occupied
    q, r = idx_to_qr(cell)
    for dq, dr in DIRECTIONS:
        nq, nr = q + dq, r + dr
        if (nq, nr) not in VALID_SET or qr_to_idx(nq, nr) not in state.tiles:
            return True
        if qr_to_idx(nq, nr) in occ:
            return True
    return False


def compute_features(state):
    """
    Compute 14 board features for the current player's perspective.

    Returns a numpy array of shape (14,) with raw feature values.
    Features are from the current player's viewpoint (own = current player).
    """
    cur = state.current_player
    opp = Player(1 - cur)
    occ = state.occupied
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    for p_idx, p in enumerate([cur, opp]):
        pcs = list(state.pieces[p])
        nbrs_cache = {pc: set(NEIGHBORS.get(pc, ())) for pc in pcs}

        # Adjacency pairs
        adj_pairs = 0
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                if pcs[j] in nbrs_cache[pcs[i]]:
                    adj_pairs += 1

        # Completing cells (triangle-completing vertices)
        completing_into = 0
        completing_past = 0
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                cells = (nbrs_cache[pcs[i]] & nbrs_cache[pcs[j]]) - set(pcs)
                for cell in cells:
                    if cell not in state.tiles or cell in occ:
                        continue
                    if _is_slide_into(cell, state):
                        completing_into += 1
                    else:
                        completing_past += 1

        # Backed pieces
        backed = 0
        for piece in pcs:
            q, r = idx_to_qr(piece)
            for dq, dr in DIRECTIONS:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in VALID_SET or qr_to_idx(nq, nr) not in state.tiles:
                    backed += 1
                    break

        # Mobility (total slide destinations)
        mobility = 0
        for piece in pcs:
            q, r = idx_to_qr(piece)
            for dq, dr in DIRECTIONS:
                nq, nr = q + dq, r + dr
                while True:
                    if (nq, nr) not in VALID_SET or qr_to_idx(nq, nr) not in state.tiles:
                        break
                    if qr_to_idx(nq, nr) in occ:
                        break
                    mobility += 1
                    nq, nr = nq + dq, nr + dr

        # Pairwise hex distances
        min_dist = 10.0  # large default
        max_dist = 0.0
        if len(pcs) >= 2:
            for i in range(len(pcs)):
                qi, ri = idx_to_qr(pcs[i])
                for j in range(i + 1, len(pcs)):
                    qj, rj = idx_to_qr(pcs[j])
                    d = (abs(qi - qj) + abs(ri - rj) + abs((qi + ri) - (qj + rj))) // 2
                    min_dist = min(min_dist, d)
                    max_dist = max(max_dist, d)

        if p_idx == 0:  # own
            features[0] = adj_pairs
            features[2] = completing_into
            features[3] = completing_past
            features[6] = backed
            features[8] = mobility
            features[10] = min_dist
            features[12] = max_dist
        else:  # opponent
            features[1] = adj_pairs
            features[4] = completing_into
            features[5] = completing_past
            features[7] = backed
            features[9] = mobility
            features[11] = min_dist
            features[13] = max_dist

    return features


def evaluate_position(state, weights):
    """Evaluate a position: tanh(features · weights) from current player's view."""
    features = compute_features(state)
    return math.tanh(np.dot(features, weights))


# ── Greedy 1-ply player ──────────────────────────────────────────────

def greedy_move(state, weights):
    """Pick the legal move with highest evaluation after applying it."""
    moves = state.get_legal_moves()
    if not moves:
        return None

    best_move = None
    best_val = -float('inf')

    for move in moves:
        next_state = state.apply_move(move)
        # After applying, the perspective may have changed.
        # If ply_type was PIECE_MOVE, next state is still same player (TILE_MOVE).
        # If ply_type was TILE_MOVE, next state is opponent's turn.
        if state.ply_type == PlyType.TILE_MOVE:
            # Next state is opponent's turn — negate value
            val = -evaluate_position(next_state, weights)
        else:
            # Same player continues (piece → tile)
            val = evaluate_position(next_state, weights)
        if val > best_val:
            best_val = val
            best_move = move

    return best_move


def play_game_ga(weights_p1, weights_p2, max_plies=300):
    """
    Play a game between two weight vectors. Returns winner (0, 1, or None for draw).
    """
    state = NonagaState()
    ply_count = 0
    weights = {Player.ONE: weights_p1, Player.TWO: weights_p2}

    while not state.is_terminal() and ply_count < max_plies:
        w = weights[state.current_player]
        move = greedy_move(state, w)
        if move is None:
            if state.ply_type == PlyType.TILE_MOVE:
                state = state.copy()
                state.current_player = Player(1 - state.current_player)
                state.ply_type = PlyType.PIECE_MOVE
                ply_count += 1
                continue
            else:
                break
        state = state.apply_move(move)
        ply_count += 1

    if state.winner is not None:
        return int(state.winner), ply_count
    return None, ply_count


# ── GA operators ─────────────────────────────────────────────────────

@dataclass
class GAConfig:
    num_islands: int = 5
    pop_per_island: int = 16
    num_generations: int = 200
    tournament_size: int = 3
    crossover_rate: float = 0.8
    blx_alpha: float = 0.5
    mutation_rate: float = 0.15  # per gene
    mutation_sigma: float = 0.1
    migration_freq: int = 5
    migration_rate: float = 0.1  # fraction of pop to migrate
    max_game_plies: int = 300
    tournament_games: int = 1  # games per matchup in round-robin
    eval_vs_random_games: int = 50
    seed: int = 42


def init_population(rng, pop_size):
    """Initialize population with small random weights."""
    return rng.standard_normal((pop_size, NUM_FEATURES)).astype(np.float32) * 0.5


def tournament_select(rng, pop, fitnesses, tournament_size):
    """Tournament selection with replacement. Returns new population."""
    n = len(pop)
    selected = np.empty_like(pop)
    for i in range(n):
        candidates = rng.integers(0, n, size=tournament_size)
        winner = candidates[np.argmax(fitnesses[candidates])]
        selected[i] = pop[winner]
    return selected


def blx_alpha_crossover(rng, pop, crossover_rate, alpha=0.5):
    """BLX-α blend crossover for continuous genomes."""
    n = len(pop)
    children = pop.copy()
    indices = rng.permutation(n)

    for k in range(0, n - 1, 2):
        i, j = indices[k], indices[k + 1]
        if rng.random() < crossover_rate:
            p1, p2 = pop[i], pop[j]
            lo = np.minimum(p1, p2)
            hi = np.maximum(p1, p2)
            span = hi - lo
            children[i] = rng.uniform(lo - alpha * span, hi + alpha * span).astype(np.float32)
            children[j] = rng.uniform(lo - alpha * span, hi + alpha * span).astype(np.float32)

    return children


def gaussian_mutate(rng, pop, mutation_rate, sigma):
    """Gaussian mutation: add N(0, sigma) noise per gene with probability mutation_rate."""
    mask = rng.random(pop.shape) < mutation_rate
    noise = rng.standard_normal(pop.shape).astype(np.float32) * sigma
    return pop + mask * noise


def evaluate_island(pop, config):
    """
    Round-robin tournament within island. Returns fitness array.
    Fitness = (wins + 0.5 * draws) / games_played.
    """
    n = len(pop)
    scores = np.zeros(n, dtype=np.float32)
    games_played = np.zeros(n, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            for _ in range(config.tournament_games):
                winner, _ = play_game_ga(pop[i], pop[j], max_plies=config.max_game_plies)
                games_played[i] += 1
                games_played[j] += 1
                if winner == 0:
                    scores[i] += 1.0
                elif winner == 1:
                    scores[j] += 1.0
                else:
                    scores[i] += 0.5
                    scores[j] += 0.5

    # Fitness = win rate
    fitnesses = np.where(games_played > 0, scores / games_played, 0.0)
    return fitnesses


def ring_migrate(rng, islands, fitnesses_list, migration_rate):
    """
    Ring topology migration: island i sends top individuals to island (i+1) % n.
    Migrants replace worst individuals in destination.
    """
    n = len(islands)
    if n < 2:
        return islands, fitnesses_list

    num_migrants = max(1, int(migration_rate * len(islands[0])))
    new_islands = [isl.copy() for isl in islands]
    new_fitnesses = [f.copy() for f in fitnesses_list]

    for i in range(n):
        dest = (i + 1) % n
        # Select top migrants from source
        src_order = np.argsort(fitnesses_list[i])[::-1]
        migrant_idx = src_order[:num_migrants]
        # Replace worst in destination
        dest_order = np.argsort(new_fitnesses[dest])
        worst_idx = dest_order[:num_migrants]

        for m, w in zip(migrant_idx, worst_idx):
            new_islands[dest][w] = islands[i][m].copy()
            new_fitnesses[dest][w] = fitnesses_list[i][m]

    return new_islands, new_fitnesses


def weight_diversity(pop):
    """Mean pairwise L2 distance between individuals, normalized by genome length."""
    n = len(pop)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(pop[i] - pop[j])
            count += 1
    return total / (count * np.sqrt(NUM_FEATURES))


# ── Evaluation helpers ───────────────────────────────────────────────

def play_vs_random(weights, n_games=50, max_plies=300):
    """Evaluate weights against a random player. Returns (win_rate, avg_plies)."""
    wins = 0
    total_plies = 0

    for g in range(n_games):
        state = NonagaState()
        ply_count = 0
        # Alternate sides
        evolved_player = Player(g % 2)

        while not state.is_terminal() and ply_count < max_plies:
            if state.current_player == evolved_player:
                move = greedy_move(state, weights)
            else:
                moves = state.get_legal_moves()
                move = random.choice(moves) if moves else None

            if move is None:
                if state.ply_type == PlyType.TILE_MOVE:
                    state = state.copy()
                    state.current_player = Player(1 - state.current_player)
                    state.ply_type = PlyType.PIECE_MOVE
                    ply_count += 1
                    continue
                else:
                    break
            state = state.apply_move(move)
            ply_count += 1

        total_plies += ply_count
        if state.winner is not None and int(state.winner) == evolved_player:
            wins += 1

    return wins / n_games, total_plies / n_games


def play_vs_nn_greedy(weights, checkpoint_path, n_games=20):
    """Evaluate weights against NN greedy player. Returns evolved win rate."""
    try:
        import torch
        from model.network import NonagaNet
    except ImportError:
        print("  (skipping NN comparison — torch not available)")
        return None

    network = NonagaNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    wins = 0
    for g in range(n_games):
        state = NonagaState()
        ply_count = 0
        evolved_player = Player(g % 2)

        while not state.is_terminal() and ply_count < 300:
            if state.current_player == evolved_player:
                move = greedy_move(state, weights)
            else:
                # NN greedy: evaluate all moves, pick best by value head
                moves = state.get_legal_moves()
                if not moves:
                    move = None
                else:
                    best_move = None
                    best_val = -float('inf')
                    for m in moves:
                        ns = state.apply_move(m)
                        board = ns.encode()
                        _, _, val = network.predict(board)
                        v = float(val)
                        # If it's now opponent's turn, negate
                        if state.ply_type == PlyType.TILE_MOVE:
                            v = -v
                        if v > best_val:
                            best_val = v
                            best_move = m
                    move = best_move

            if move is None:
                if state.ply_type == PlyType.TILE_MOVE:
                    state = state.copy()
                    state.current_player = Player(1 - state.current_player)
                    state.ply_type = PlyType.PIECE_MOVE
                    ply_count += 1
                    continue
                else:
                    break
            state = state.apply_move(move)
            ply_count += 1

        if state.winner is not None and int(state.winner) == evolved_player:
            wins += 1

    return wins / n_games


# ── Main evolution loop ──────────────────────────────────────────────

def evolve(config=None):
    """Run island-model GA evolution."""
    if config is None:
        config = GAConfig()

    rng = np.random.default_rng(config.seed)
    random.seed(config.seed)

    print(f"Island-Model GA for Nonaga")
    print(f"  {config.num_islands} islands × {config.pop_per_island} individuals")
    print(f"  {config.num_generations} generations, migration every {config.migration_freq}")
    print(f"  Ring topology, migration rate {config.migration_rate}")
    print(f"  {NUM_FEATURES} features, BLX-α crossover (α={config.blx_alpha})")
    print()

    # Initialize islands
    islands = [init_population(rng, config.pop_per_island)
               for _ in range(config.num_islands)]
    fitnesses_list = [np.zeros(config.pop_per_island) for _ in range(config.num_islands)]

    best_overall = None
    best_overall_fitness = -1
    stats_log = []

    for gen in range(config.num_generations):
        gen_start = time.time()

        # Evaluate each island
        for isl_idx in range(config.num_islands):
            fitnesses_list[isl_idx] = evaluate_island(islands[isl_idx], config)

            # Track best
            best_idx = np.argmax(fitnesses_list[isl_idx])
            best_fit = fitnesses_list[isl_idx][best_idx]
            if best_fit > best_overall_fitness:
                best_overall_fitness = best_fit
                best_overall = islands[isl_idx][best_idx].copy()

        # Log stats
        diversities = [weight_diversity(isl) for isl in islands]
        mean_fits = [f.mean() for f in fitnesses_list]
        best_fits = [f.max() for f in fitnesses_list]
        gen_time = time.time() - gen_start

        stats_log.append({
            'gen': gen,
            'best_fits': best_fits,
            'mean_fits': mean_fits,
            'diversities': diversities,
            'time': gen_time,
        })

        if gen % 10 == 0 or gen == config.num_generations - 1:
            div_str = " ".join(f"{d:.2f}" for d in diversities)
            fit_str = " ".join(f"{f:.3f}" for f in best_fits)
            print(f"Gen {gen:3d} | best: [{fit_str}] | div: [{div_str}] | {gen_time:.1f}s")

        # Migration
        if gen > 0 and gen % config.migration_freq == 0:
            islands, fitnesses_list = ring_migrate(
                rng, islands, fitnesses_list, config.migration_rate)
            if gen % 10 == 0:
                print(f"  → Ring migration ({max(1, int(config.migration_rate * config.pop_per_island))} individuals)")

        # Selection → crossover → mutation (preserve elite)
        for isl_idx in range(config.num_islands):
            pop = islands[isl_idx]
            fit = fitnesses_list[isl_idx]

            # Elitism: save best individual
            elite_idx = np.argmax(fit)
            elite = pop[elite_idx].copy()

            # GA operators
            selected = tournament_select(rng, pop, fit, config.tournament_size)
            crossed = blx_alpha_crossover(rng, selected, config.crossover_rate, config.blx_alpha)
            mutated = gaussian_mutate(rng, crossed, config.mutation_rate, config.mutation_sigma)

            # Restore elite
            mutated[0] = elite
            islands[isl_idx] = mutated

    print(f"\n{'='*60}")
    print(f"Evolution complete — {config.num_generations} generations")
    print(f"{'='*60}")

    # Report best per island
    print("\nBest individual per island:")
    for isl_idx in range(config.num_islands):
        fitnesses_list[isl_idx] = evaluate_island(islands[isl_idx], config)
        best_idx = np.argmax(fitnesses_list[isl_idx])
        best_fit = fitnesses_list[isl_idx][best_idx]
        w = islands[isl_idx][best_idx]
        w_str = " ".join(f"{v:+.3f}" for v in w)
        print(f"  Island {isl_idx}: fitness={best_fit:.3f} weights=[{w_str}]")

    # Find overall best
    all_best = []
    for isl_idx in range(config.num_islands):
        best_idx = np.argmax(fitnesses_list[isl_idx])
        all_best.append((fitnesses_list[isl_idx][best_idx], islands[isl_idx][best_idx]))
    all_best.sort(key=lambda x: x[0], reverse=True)
    best_weights = all_best[0][1]

    print(f"\nBest overall weights:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:>25s}: {best_weights[i]:+.4f}")

    # Evaluate vs random
    print(f"\nEvaluating best vs random ({config.eval_vs_random_games} games)...")
    win_rate, avg_plies = play_vs_random(best_weights, n_games=config.eval_vs_random_games)
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Avg plies: {avg_plies:.1f}")

    # Try to evaluate vs NN
    import glob
    checkpoints = glob.glob('checkpoints/endgame_trained.pt')
    if not checkpoints:
        checkpoints = glob.glob('checkpoints/iteration_*.pt')
    if checkpoints:
        ckpt = checkpoints[0]
        print(f"\nEvaluating best vs NN greedy ({ckpt})...")
        nn_wr = play_vs_nn_greedy(best_weights, ckpt, n_games=20)
        if nn_wr is not None:
            print(f"  Win rate vs NN: {nn_wr:.1%}")

    # Final diversity report
    print(f"\nFinal island diversities:")
    for isl_idx in range(config.num_islands):
        d = weight_diversity(islands[isl_idx])
        print(f"  Island {isl_idx}: {d:.3f}")

    # Inter-island divergence
    if config.num_islands >= 2:
        print(f"\nInter-island divergence (mean L2 between island centroids):")
        centroids = [isl.mean(axis=0) for isl in islands]
        for i in range(config.num_islands):
            for j in range(i + 1, config.num_islands):
                d = np.linalg.norm(centroids[i] - centroids[j])
                print(f"  Island {i} ↔ {j}: {d:.3f}")

    return best_weights, islands, stats_log


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Island-model GA for Nonaga")
    parser.add_argument("--islands", type=int, default=5)
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--mutation-sigma", type=float, default=0.1)
    parser.add_argument("--migration-freq", type=int, default=5)
    parser.add_argument("--migration-rate", type=float, default=0.1)
    parser.add_argument("--tournament-games", type=int, default=1,
                        help="Games per matchup in round-robin (default: 1)")
    parser.add_argument("--eval-vs-random", type=int, default=50,
                        help="Games to play vs random at the end")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = GAConfig(
        num_islands=args.islands,
        pop_per_island=args.pop,
        num_generations=args.gens,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        migration_freq=args.migration_freq,
        migration_rate=args.migration_rate,
        tournament_games=args.tournament_games,
        eval_vs_random_games=args.eval_vs_random,
        seed=args.seed,
    )

    evolve(config)


if __name__ == "__main__":
    main()
