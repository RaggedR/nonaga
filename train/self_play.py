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
from train.mcts import MCTS


def play_game(mcts, config):
    """
    Play one self-play game.

    Returns list of (board_6x7x7, ply_type, policy, current_player)
    tuples. Value targets are filled in after the game ends.
    """
    state = NonagaState()
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

    training_examples = []
    for ex in examples:
        if winner == -1:
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


def generate_self_play_data(network, config, verbose=True):
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
        examples, winner, plies = play_game(mcts, config)
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

    mcts = MCTS(network, config)

    results = []
    for _ in range(num_games):
        examples, winner, plies = play_game(mcts, config)
        winner_int = int(winner) if winner is not None else None
        results.append((examples, winner_int, plies))

    return results


def generate_self_play_data_parallel(checkpoint_path, config, verbose=True):
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
