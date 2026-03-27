"""
Train NN by playing against the GA with dense signal.

Two innovations over standard AlphaZero self-play:
1. Game-length-shaped values: surviving longer against GA = less negative value.
   A position from a 20-ply loss is worse than one from a 100-ply loss.
2. GA policy imitation: softmax over GA's evaluations as policy target.
   Directly transfers the GA's strategic knowledge into the NN.

Data generation is fast (~5ms/game) because the GA needs no MCTS —
just 14 dot products per move. We can generate 10,000 training games
in under a minute, vs hours for MCTS self-play.

Usage:
    # Full run (starts from best island checkpoint)
    python -u train_vs_ga.py

    # From scratch
    python -u train_vs_ga.py --from-scratch

    # Custom parameters
    python -u train_vs_ga.py --iterations 30 --games 500 --checkpoint checkpoints/island_2/iteration_20.pt
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.nonaga import NonagaState, PlyType, Player, GRID_SIZE, NUM_DIRS
from game.symmetry import augment_example
from model.network import NonagaNet
from ga_evolve import (
    compute_features, evaluate_position, greedy_move,
    play_vs_nn_greedy as ga_play_vs_nn,
)

# Best GA weights from island-model evolution
GA_WEIGHTS = np.array([
    1.4832, -0.3479, 0.1681, -6.3177, -2.2217, 3.9770,
    -0.0497, -2.9234, 0.0024, -0.0922, 0.2998, 0.4741,
    -1.1508, 0.5765
], dtype=np.float32)


def ga_policy_distribution(state, weights, temperature=0.5):
    """
    Compute softmax probability distribution over legal moves from GA evaluation.

    Returns (moves, policy_vector) where policy_vector is in the NN's action
    encoding format (294 for piece moves, 2401 for tile moves).
    """
    moves = state.get_legal_moves()
    if not moves:
        return moves, None

    # Check for immediate wins — one-hot on winning move
    for move in moves:
        ns = state.apply_move(move)
        if ns.winner is not None and int(ns.winner) == state.current_player:
            if state.ply_type == PlyType.PIECE_MOVE:
                policy = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
                action = NonagaState.piece_move_to_action(move[0], move[1])
                policy[action] = 1.0
            else:
                policy = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
                action = NonagaState.tile_move_to_action(move[0], move[1])
                policy[action] = 1.0
            return moves, policy

    # Evaluate all moves
    values = []
    for move in moves:
        ns = state.apply_move(move)
        val = evaluate_position(ns, weights)
        if state.ply_type == PlyType.TILE_MOVE:
            val = -val  # Negate for tile moves (opponent's turn next)
        values.append(val)

    values = np.array(values, dtype=np.float32)

    # Softmax with temperature
    values = values / max(temperature, 1e-8)
    values = values - values.max()  # Numerical stability
    exp_values = np.exp(values)
    probs = exp_values / exp_values.sum()

    # Encode into NN policy vector format
    if state.ply_type == PlyType.PIECE_MOVE:
        policy = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
        for move, prob in zip(moves, probs):
            action = NonagaState.piece_move_to_action(move[0], move[1])
            policy[action] = prob
    else:
        policy = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
        for move, prob in zip(moves, probs):
            action = NonagaState.tile_move_to_action(move[0], move[1])
            policy[action] = prob

    return moves, policy


def nn_pick_move(state, network, device, epsilon=0.1):
    """Pick a move using NN value head with ε-greedy exploration."""
    moves = state.get_legal_moves()
    if not moves:
        return None

    # ε-greedy: random move with probability epsilon
    if random.random() < epsilon:
        return random.choice(moves)

    # Check immediate wins
    for move in moves:
        ns = state.apply_move(move)
        if ns.winner is not None and int(ns.winner) == state.current_player:
            return move

    # Greedy: evaluate all moves with NN value head
    best_move = None
    best_val = -float('inf')

    for move in moves:
        ns = state.apply_move(move)
        board = torch.FloatTensor(ns.encode()).unsqueeze(0).to(device)
        ply_type = torch.LongTensor([int(ns.ply_type)]).to(device)
        with torch.no_grad():
            _, _, value = network(board)
        val = value.item()

        # Negate if tile move (next state is opponent's perspective)
        if state.ply_type == PlyType.TILE_MOVE:
            val = -val

        if val > best_val:
            best_val = val
            best_move = move

    return best_move


def play_game_vs_ga(network, device, ga_weights, max_plies=200, epsilon=0.1):
    """
    Play one game: NN vs GA. Returns training examples.

    At every position (both NN's and GA's turns), we record:
    - Board state (input)
    - GA policy distribution (what GA thinks is best — imitation target)
    - Player identity (for value assignment)

    Alternates who plays as P1 across games via the caller.
    """
    state = NonagaState()
    raw_examples = []
    ply_count = 0

    while not state.is_terminal() and ply_count < max_plies:
        is_nn_turn = (int(state.current_player) == 0)  # NN always plays as P1 in this game

        # Record board state and GA policy at this position
        board = state.encode()
        _, ga_policy = ga_policy_distribution(state, ga_weights)
        if ga_policy is not None:
            # Store separate piece/tile policies for D6 augmentation
            if state.ply_type == PlyType.PIECE_MOVE:
                piece_policy = ga_policy
                tile_policy = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
            else:
                piece_policy = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
                tile_policy = ga_policy
            raw_examples.append({
                'board': board,
                'ply_type': int(state.ply_type),
                'piece_policy': piece_policy,
                'tile_policy': tile_policy,
                'player': int(state.current_player),
            })

        # Pick move
        if is_nn_turn:
            move = nn_pick_move(state, network, device, epsilon=epsilon)
        else:
            move = greedy_move(state, ga_weights)

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

    # Assign shaped values based on game outcome and length
    winner = int(state.winner) if state.winner is not None else -1

    training_examples = []
    for ex in raw_examples:
        if winner == -1:
            # Draw — small positive for NN (surviving against GA is good)
            value = 0.1 if ex['player'] == 0 else -0.1
        elif ex['player'] == winner:
            # This player won
            value = 1.0
        else:
            # This player lost — but surviving longer is better
            # Scale from -1.0 (instant loss) to -0.2 (survived to max_plies)
            value = -1.0 + 0.8 * (ply_count / max_plies)

        training_examples.append((
            ex['board'],
            ex['ply_type'],
            ex['piece_policy'],
            ex['tile_policy'],
            np.float32(value),
        ))

    return training_examples, winner, ply_count


def augment_all(examples):
    """Apply D6 symmetry augmentation (12× data)."""
    augmented = []
    for board, ply_type, piece_policy, tile_policy, value in examples:
        for aug_board, aug_pp, aug_tp, aug_val in augment_example(board, piece_policy, tile_policy, value):
            augmented.append((aug_board, ply_type, aug_pp, aug_tp, aug_val))
    return augmented


def train_network(network, optimizer, data, device, batch_size=256, epochs=5):
    """Train network on collected data."""
    network.train()
    random.shuffle(data)

    total_loss = 0
    total_p_loss = 0
    total_v_loss = 0
    n_batches = 0

    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            boards, ply_types, piece_policies, tile_policies, values = zip(*batch)

            boards = torch.FloatTensor(np.array(boards)).to(device)
            ply_types_t = torch.LongTensor(np.array(ply_types)).to(device)
            values = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device)

            piece_logits, tile_logits, pred_values = network(boards)

            # Value loss
            v_loss = nn.MSELoss()(pred_values, values)

            # Policy loss — cross-entropy with GA's softmax targets, split by ply type
            p_loss = torch.tensor(0.0, device=device)
            piece_mask = (ply_types_t == 0)
            tile_mask = (ply_types_t == 1)

            if piece_mask.any():
                piece_targets = torch.FloatTensor(
                    np.array([pp for pp, pt in zip(piece_policies, ply_types) if pt == 0])
                ).to(device)
                piece_log_probs = torch.log_softmax(piece_logits[piece_mask], dim=1)
                p_loss = p_loss - (piece_targets * piece_log_probs).sum(dim=1).mean()

            if tile_mask.any():
                tile_targets = torch.FloatTensor(
                    np.array([tp for tp, pt in zip(tile_policies, ply_types) if pt == 1])
                ).to(device)
                tile_log_probs = torch.log_softmax(tile_logits[tile_mask], dim=1)
                p_loss = p_loss - (tile_targets * tile_log_probs).sum(dim=1).mean()

            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            n_batches += 1

    return {
        'total': total_loss / max(n_batches, 1),
        'policy': total_p_loss / max(n_batches, 1),
        'value': total_v_loss / max(n_batches, 1),
    }


def evaluate_vs_ga(network, device, ga_weights, n_games=50):
    """Play NN vs GA and return win rates."""
    nn_wins = ga_wins = draws = 0
    total_plies = 0

    for g in range(n_games):
        state = NonagaState()
        nn_player = g % 2  # Alternate sides
        ply_count = 0

        while not state.is_terminal() and ply_count < 200:
            if int(state.current_player) == nn_player:
                move = nn_pick_move(state, network, device, epsilon=0)
            else:
                move = greedy_move(state, ga_weights)

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
            if int(state.winner) == nn_player:
                nn_wins += 1
            else:
                ga_wins += 1
        else:
            draws += 1

    return {
        'nn_wins': nn_wins, 'ga_wins': ga_wins, 'draws': draws,
        'nn_win_rate': nn_wins / n_games,
        'avg_plies': total_plies / max(n_games, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Train NN by playing against GA')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--games', type=int, default=500,
                        help='Games per iteration (fast — ~5ms each)')
    parser.add_argument('--eval-games', type=int, default=50,
                        help='Evaluation games per checkpoint')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs per iteration')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=0.15,
                        help='Exploration rate for NN moves')
    parser.add_argument('--policy-temp', type=float, default=0.5,
                        help='Temperature for GA policy softmax')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Starting checkpoint (default: best island)')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Start from random initialization')
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Device: {device}")
    print(f"Training NN against GA ({args.games} games/iter, {args.iterations} iterations)")
    print(f"  epsilon={args.epsilon}, policy_temp={args.policy_temp}, lr={args.lr}")

    # Load or create network
    network = NonagaNet().to(device)
    if not args.from_scratch:
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            # Find the best island checkpoint
            best_path = None
            for island in range(5):
                for it in range(100, -1, -1):
                    p = f'checkpoints/island_{island}/iteration_{it}.pt'
                    if os.path.exists(p):
                        if best_path is None:
                            best_path = p
                        break
            ckpt_path = best_path

        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            network.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded: {ckpt_path}")
        else:
            print("  No checkpoint found, starting from scratch")

    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4)

    # Initial evaluation
    print("\n--- Initial evaluation ---")
    result = evaluate_vs_ga(network, device, GA_WEIGHTS, n_games=args.eval_games)
    print(f"  NN {result['nn_wins']}-{result['ga_wins']}-{result['draws']} "
          f"(NN wins {result['nn_win_rate']:.0%}, avg {result['avg_plies']:.0f} plies)")

    os.makedirs('checkpoints/vs_ga', exist_ok=True)
    best_nn_win_rate = result['nn_win_rate']

    for iteration in range(args.iterations):
        t0 = time.time()

        # Generate training data
        all_examples = []
        nn_wins = ga_wins = draws = 0
        total_plies = 0

        for g in range(args.games):
            examples, winner, plies = play_game_vs_ga(
                network, device, GA_WEIGHTS,
                max_plies=200, epsilon=args.epsilon)
            all_examples.extend(examples)
            total_plies += plies
            if winner == 0:
                nn_wins += 1
            elif winner == 1:
                ga_wins += 1
            else:
                draws += 1

        avg_plies = total_plies / max(args.games, 1)

        # Augment
        augmented = augment_all(all_examples)

        # Train
        loss = train_network(network, optimizer, augmented, device,
                             epochs=args.epochs)

        elapsed = time.time() - t0

        print(f"\nIteration {iteration} ({elapsed:.0f}s):")
        print(f"  Games: NN={nn_wins} GA={ga_wins} D={draws} "
              f"(avg {avg_plies:.0f} plies)")
        print(f"  Training: {len(augmented)} examples, "
              f"loss={loss['total']:.4f} (p={loss['policy']:.4f} v={loss['value']:.4f})")

        # Evaluate
        result = evaluate_vs_ga(network, device, GA_WEIGHTS, n_games=args.eval_games)
        print(f"  Eval: NN {result['nn_wins']}-{result['ga_wins']}-{result['draws']} "
              f"({result['nn_win_rate']:.0%}, avg {result['avg_plies']:.0f} plies)")

        # Save checkpoint
        torch.save({
            'iteration': iteration,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoints/vs_ga/iteration_{iteration}.pt')

        if result['nn_win_rate'] > best_nn_win_rate:
            best_nn_win_rate = result['nn_win_rate']
            torch.save({
                'iteration': iteration,
                'model_state_dict': network.state_dict(),
            }, 'checkpoints/vs_ga/best.pt')
            print(f"  *** New best: {best_nn_win_rate:.0%} ***")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final evaluation (100 games)")
    result = evaluate_vs_ga(network, device, GA_WEIGHTS, n_games=100)
    print(f"  NN {result['nn_wins']}-{result['ga_wins']}-{result['draws']} "
          f"(NN wins {result['nn_win_rate']:.0%}, avg {result['avg_plies']:.0f} plies)")


if __name__ == '__main__':
    main()
