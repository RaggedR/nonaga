"""
Self-play game generation for AlphaZero training.

Each game produces training examples:
  (board_encoding, ply_type, policy_target, value_target)

The value target is set retrospectively: +1 for the winner's positions,
-1 for the loser's, 0 for draws.

Supports parallel self-play using multiprocessing for ~8x speedup.
"""

import numpy as np
import random
import multiprocessing as mp
from game.nonaga import NonagaState, PlyType, Player
from game.hex_grid import NEIGHBORS, DIRECTIONS, VALID_SET, idx_to_qr, qr_to_idx
from train.mcts import MCTS


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


def _shaped_draw_value(state, player):
    """
    Small value signal for draws based on positional quality.

    For every pair of same-color pieces, finds cells adjacent to both
    (triangle-completing vertices) and scores them by slide-into-ability.
    Adjacent pairs get a bonus on top. Also rewards backed pieces.
    """
    def pair_score(p):
        """Score all pairs: slide-into-able completing cells + adjacency bonus."""
        occ = state.occupied
        pcs = list(state.pieces[p])
        score = 0.0
        for i in range(len(pcs)):
            nbrs_i = set(NEIGHBORS.get(pcs[i], ()))
            for j in range(i + 1, len(pcs)):
                nbrs_j = set(NEIGHBORS.get(pcs[j], ()))
                adjacent = pcs[j] in nbrs_i

                # Bonus for pieces already touching
                if adjacent:
                    score += 1.0

                # Find cells that would complete a triangle with this pair
                completing = (nbrs_i & nbrs_j) - set(pcs)
                for cell in completing:
                    if cell not in state.tiles or cell in occ:
                        continue
                    if _is_slide_into(cell, state):
                        score += 0.5  # reachable completing cell
                    else:
                        score += 0.1  # exists but hard to reach
        return score

    def backed_count(p):
        """Count pieces with at least one direction blocked by tile edge."""
        count = 0
        for piece in state.pieces[p]:
            q, r = idx_to_qr(piece)
            for dq, dr in DIRECTIONS:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in VALID_SET or qr_to_idx(nq, nr) not in state.tiles:
                    count += 1
                    break
        return count

    opp = Player(1 - player)
    ps = (pair_score(Player(player)) - pair_score(opp)) * 0.08
    bs = (backed_count(Player(player)) - backed_count(opp)) * 0.03
    return max(-0.3, min(0.3, ps + bs))


def play_game(mcts, config, win_mode='triangle'):
    """
    Play one self-play game.

    Returns list of (board_6x7x7, ply_type, policy, current_player)
    tuples. Value targets are filled in after the game ends.
    """
    state = NonagaState()
    state.win_mode = win_mode
    examples = []
    ply_count = 0

    while not state.is_terminal() and ply_count < config.max_game_plies:
        # Temperature: explore early, lower (but nonzero) later
        temp_late = getattr(config, 'temp_late', 0.0)
        temp = 1.0 if ply_count < config.temp_threshold else temp_late

        move, policy = mcts.get_action_with_temp(state, temperature=temp)

        if move is None:
            # No legal moves (degenerate board) — skip this ply
            if state.ply_type == PlyType.TILE_MOVE:
                state = state.copy()
                state.current_player = Player(1 - state.current_player)
                state.ply_type = PlyType.PIECE_MOVE
                ply_count += 1
                continue
            else:
                break  # No piece moves = stuck, end game

        # Record training example
        board = state.encode()
        examples.append({
            'board': board,
            'ply_type': int(state.ply_type),
            'policy': policy,
            'player': int(state.current_player),
        })

        state = state.apply_move(move)
        ply_count += 1

    # Assign value targets
    if state.winner is not None:
        winner = int(state.winner)
    else:
        winner = -1  # draw

    # Draw shaping only in full-rules mode (Phase 2) — curriculum games
    # should produce clean binary signal
    use_draw_shaping = (winner == -1 and win_mode == 'triangle')

    training_examples = []
    for ex in examples:
        if winner == -1:
            if use_draw_shaping:
                value = _shaped_draw_value(state, ex['player'])
            else:
                value = 0.0
        elif ex['player'] == winner:
            value = 1.0
        else:
            value = -1.0

        training_examples.append((
            ex['board'],
            ex['ply_type'],
            ex['policy'],
            np.float32(value),
        ))

    return training_examples, state.winner, ply_count


def generate_self_play_data(network, config, win_mode='triangle', verbose=True):
    """
    Generate training data from self-play games.

    Returns:
        examples: list of (board, ply_type, policy, value)
        stats: dict with game statistics
    """
    mcts = MCTS(network, config)
    all_examples = []
    wins = {0: 0, 1: 0}
    draws = 0
    total_plies = 0

    for game_idx in range(config.num_self_play_games):
        examples, winner, plies = play_game(mcts, config, win_mode=win_mode)
        all_examples.extend(examples)
        total_plies += plies

        if winner is not None:
            wins[int(winner)] += 1
        else:
            draws += 1

        if verbose and (game_idx + 1) % 10 == 0:
            print(f"  Game {game_idx + 1}/{config.num_self_play_games}: "
                  f"{len(all_examples)} examples, "
                  f"P1={wins[0]} P2={wins[1]} D={draws}")

    stats = {
        'num_games': config.num_self_play_games,
        'num_examples': len(all_examples),
        'wins_p1': wins[0],
        'wins_p2': wins[1],
        'draws': draws,
        'avg_plies': total_plies / config.num_self_play_games,
    }
    return all_examples, stats


def _worker_play_games(args):
    """Worker function for parallel self-play. Runs in a separate process."""
    checkpoint_path, num_games, seed, config_vals = args
    import torch
    from model.network import NonagaNet
    from train.config import Config

    random.seed(seed)
    np.random.seed(seed)

    # Load model on CPU (each worker gets its own copy)
    network = NonagaNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    config = Config()
    for k, v in config_vals.items():
        setattr(config, k, v)

    win_mode = config_vals.get('win_mode', 'triangle')
    mcts = MCTS(network, config)

    results = []
    for _ in range(num_games):
        examples, winner, plies = play_game(mcts, config, win_mode=win_mode)
        winner_int = int(winner) if winner is not None else None
        results.append((examples, winner_int, plies))

    return results


def generate_self_play_data_parallel(checkpoint_path, config, win_mode='triangle', verbose=True):
    """
    Generate training data using parallel self-play across multiple workers.
    Each worker loads the model on CPU and plays games independently.
    """
    num_workers = getattr(config, 'num_workers', 8)
    total_games = config.num_self_play_games
    games_per_worker = total_games // num_workers
    remainder = total_games % num_workers

    # Config values needed by workers
    config_vals = {
        'num_mcts_sims': config.num_mcts_sims,
        'cpuct': config.cpuct,
        'dirichlet_alpha': config.dirichlet_alpha,
        'dirichlet_epsilon': config.dirichlet_epsilon,
        'temp_threshold': config.temp_threshold,
        'temp_late': getattr(config, 'temp_late', 0.0),
        'max_game_plies': config.max_game_plies,
        'win_mode': win_mode,
    }

    args_list = []
    for i in range(num_workers):
        n = games_per_worker + (1 if i < remainder else 0)
        if n > 0:
            args_list.append((checkpoint_path, n, i * 10000 + 42, config_vals))

    if verbose:
        print(f"  Dispatching {total_games} games across {len(args_list)} workers...")

    ctx = mp.get_context('spawn')
    with ctx.Pool(len(args_list)) as pool:
        all_worker_results = pool.map(_worker_play_games, args_list)

    # Aggregate results
    all_examples = []
    wins = {0: 0, 1: 0}
    draws = 0
    total_plies = 0

    for worker_results in all_worker_results:
        for examples, winner, plies in worker_results:
            all_examples.extend(examples)
            total_plies += plies
            if winner is not None:
                wins[winner] += 1
            else:
                draws += 1

    stats = {
        'num_games': total_games,
        'num_examples': len(all_examples),
        'wins_p1': wins[0],
        'wins_p2': wins[1],
        'draws': draws,
        'avg_plies': total_plies / max(total_games, 1),
    }

    if verbose:
        print(f"  All workers done: {stats['num_examples']} examples, "
              f"P1={wins[0]} P2={wins[1]} D={draws}")

    return all_examples, stats
