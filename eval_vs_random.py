"""Evaluate a trained model against a random player."""

import argparse
import random
import torch
import numpy as np
from game.nonaga import NonagaState, PlyType, Player
from model.network import NonagaNet
from train.mcts import MCTS
from train.config import Config


def play_game(mcts, model_is_p1, config):
    """Play one game: model vs random. Returns winner (0, 1, or None)."""
    state = NonagaState()
    ply_count = 0

    while not state.is_terminal() and ply_count < config.max_game_plies:
        is_model_turn = (state.current_player == 0) == model_is_p1

        if is_model_turn:
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

    return int(state.winner) if state.winner is not None else None, ply_count


def main():
    parser = argparse.ArgumentParser(description="Evaluate model vs random")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--games", type=int, default=50, help="Games to play")
    parser.add_argument("--sims", type=int, default=50, help="MCTS sims per move")
    args = parser.parse_args()

    config = Config()
    config.num_mcts_sims = args.sims

    network = NonagaNet()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    mcts = MCTS(network, config)

    model_wins = 0
    random_wins = 0
    draws = 0
    total_plies = 0

    for i in range(args.games):
        model_is_p1 = (i % 2 == 0)
        winner, plies = play_game(mcts, model_is_p1, config)
        total_plies += plies

        model_player = 0 if model_is_p1 else 1
        if winner == model_player:
            model_wins += 1
        elif winner is not None:
            random_wins += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            total = i + 1
            print(f"  [{total}/{args.games}] Model={model_wins} Random={random_wins} "
                  f"Draws={draws} ({model_wins/total:.0%} model win rate)")

    total = args.games
    print(f"\nFinal: Model={model_wins} Random={random_wins} Draws={draws}")
    print(f"Model win rate: {model_wins/total:.1%}")
    print(f"Avg plies: {total_plies/total:.1f}")


if __name__ == "__main__":
    main()
