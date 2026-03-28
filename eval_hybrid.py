"""Evaluate fixed MCTS and hybrid MCTS vs random."""

import torch
import random
import numpy as np
from model.network import NonagaNet
from train.mcts import MCTS
from train.config import Config
from game.nonaga import NonagaState, PlyType, Player
import time


def flush(*args):
    print(*args, flush=True)


def play_vs_random(mcts, n_games=30):
    """Play model vs random using the given MCTS instance."""
    model_wins = random_wins = draws = 0
    total_plies = 0

    for g in range(n_games):
        state = NonagaState()
        model_is_p1 = (g % 2 == 0)
        model_player = 0 if model_is_p1 else 1
        ply_count = 0

        while not state.is_terminal() and ply_count < 500:
            is_model = (state.current_player == 0) == model_is_p1

            if is_model:
                move, _ = mcts.get_action_with_temp(state, temperature=0, add_noise=False)
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
        if state.winner is not None:
            if int(state.winner) == model_player:
                model_wins += 1
            else:
                random_wins += 1
        else:
            draws += 1

    return model_wins, random_wins, draws, total_plies / max(n_games, 1)


# Setup
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
flush(f'Device: {device}')

config = Config()
config.max_game_plies = 500
N_GAMES = 30

# Test with endgame_trained model
for ckpt_name in ['endgame_trained', 'bootstrap_final']:
    path = f'checkpoints/{ckpt_name}.pt'
    try:
        net = NonagaNet()
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device)
        net.eval()
    except FileNotFoundError:
        flush(f'\nSkipping {ckpt_name} (not found)')
        continue

    flush(f'\n{"="*60}')
    flush(f'Model: {ckpt_name}')
    flush(f'{"="*60}')

    for n_sims in [25, 50, 100]:
        config.num_mcts_sims = n_sims

        # Standard MCTS (with sign fix)
        mcts_std = MCTS(net, config, greedy_tile=False)
        t0 = time.time()
        m, r, d, avg_plies = play_vs_random(mcts_std, N_GAMES)
        elapsed = time.time() - t0
        flush(f'  Fixed MCTS ({n_sims} sims): {m}-{r}-{d} ({m/(m+r+d):.0%}) '
              f'avg_plies={avg_plies:.0f} time={elapsed:.1f}s')

        # Hybrid MCTS (greedy tiles)
        mcts_hyb = MCTS(net, config, greedy_tile=True)
        t0 = time.time()
        m, r, d, avg_plies = play_vs_random(mcts_hyb, N_GAMES)
        elapsed = time.time() - t0
        flush(f'  Hybrid MCTS ({n_sims} sims): {m}-{r}-{d} ({m/(m+r+d):.0%}) '
              f'avg_plies={avg_plies:.0f} time={elapsed:.1f}s')
