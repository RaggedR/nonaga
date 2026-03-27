"""
Train NN by watching GA play against itself (knowledge distillation).

The NN observes GA-vs-GA games and learns:
1. Policy: what the GA would do at each position (softmax over GA evals)
2. Value: who won the game (binary signal from decisive games between equals)

Unlike train_vs_ga.py, ALL positions come from strong play on both sides.
No RL, no exploration, no self-play — pure supervised learning from expert data.

Data generation is instant (~5ms/game, no NN forward passes needed).
10,000 games = ~50 seconds, producing ~500K training positions.

Usage:
    python -u train_distill.py                          # default: 10K games, 20 epochs
    python -u train_distill.py --games 50000 --epochs 50  # more data
    python -u train_distill.py --from-scratch           # don't load checkpoint
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

from game.nonaga import NonagaState, PlyType, Player
from game.symmetry import augment_example
from model.network import NonagaNet
from ga_evolve import compute_features, evaluate_position, greedy_move

# Best GA weights from island-model evolution
GA_WEIGHTS = np.array([
    1.4832, -0.3479, 0.1681, -6.3177, -2.2217, 3.9770,
    -0.0497, -2.9234, 0.0024, -0.0922, 0.2998, 0.4741,
    -1.1508, 0.5765
], dtype=np.float32)


def ga_policy_vector(state, weights, temperature=0.5):
    """Compute softmax policy over legal moves from GA evaluation."""
    moves = state.get_legal_moves()
    if not moves:
        return None, None

    # Immediate win → one-hot
    for move in moves:
        ns = state.apply_move(move)
        if ns.winner is not None and int(ns.winner) == state.current_player:
            if state.ply_type == PlyType.PIECE_MOVE:
                pp = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
                pp[NonagaState.piece_move_to_action(move[0], move[1])] = 1.0
                return pp, np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
            else:
                tp = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
                tp[NonagaState.tile_move_to_action(move[0], move[1])] = 1.0
                return np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32), tp

    # Evaluate all moves
    values = []
    for move in moves:
        ns = state.apply_move(move)
        val = evaluate_position(ns, weights)
        if state.ply_type == PlyType.TILE_MOVE:
            val = -val
        values.append(val)

    values = np.array(values, dtype=np.float32)
    values = values / max(temperature, 1e-8)
    values = values - values.max()
    exp_values = np.exp(values)
    probs = exp_values / exp_values.sum()

    if state.ply_type == PlyType.PIECE_MOVE:
        pp = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
        for move, prob in zip(moves, probs):
            pp[NonagaState.piece_move_to_action(move[0], move[1])] = prob
        return pp, np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
    else:
        tp = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
        for move, prob in zip(moves, probs):
            tp[NonagaState.tile_move_to_action(move[0], move[1])] = prob
        return np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32), tp


def play_ga_vs_ga(weights, max_plies=200):
    """Play one GA-vs-GA game. Returns training examples."""
    state = NonagaState()
    raw_examples = []
    ply_count = 0

    while not state.is_terminal() and ply_count < max_plies:
        board = state.encode()
        pp, tp = ga_policy_vector(state, weights)
        if pp is not None:
            raw_examples.append({
                'board': board,
                'ply_type': int(state.ply_type),
                'piece_policy': pp,
                'tile_policy': tp,
                'player': int(state.current_player),
            })

        move = greedy_move(state, weights)
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

    # Value targets: +1 for winner's positions, -1 for loser's
    winner = int(state.winner) if state.winner is not None else -1

    examples = []
    for ex in raw_examples:
        if winner == -1:
            value = 0.0  # Draw — no signal
        elif ex['player'] == winner:
            value = 1.0
        else:
            value = -1.0

        examples.append((
            ex['board'],
            ex['ply_type'],
            ex['piece_policy'],
            ex['tile_policy'],
            np.float32(value),
        ))

    return examples, winner, ply_count


def generate_dataset(n_games, weights):
    """Generate training data from GA-vs-GA games."""
    all_examples = []
    wins = {0: 0, 1: 0}
    draws = 0
    total_plies = 0

    for _ in range(n_games):
        examples, winner, plies = play_ga_vs_ga(weights)
        all_examples.extend(examples)
        total_plies += plies
        if winner >= 0:
            wins[winner] += 1
        else:
            draws += 1

    return all_examples, wins, draws, total_plies / max(n_games, 1)


def augment_all(examples):
    """Apply D6 symmetry augmentation (12× data)."""
    augmented = []
    for board, ply_type, pp, tp, value in examples:
        for aug_board, aug_pp, aug_tp, aug_val in augment_example(board, pp, tp, value):
            augmented.append((aug_board, ply_type, aug_pp, aug_tp, aug_val))
    return augmented


def train_network(network, optimizer, data, device, batch_size=256):
    """Train one epoch on the data."""
    network.train()
    random.shuffle(data)

    total_loss = total_p = total_v = 0
    n = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        boards, ply_types, piece_pols, tile_pols, values = zip(*batch)

        boards = torch.FloatTensor(np.array(boards)).to(device)
        ply_types_t = torch.LongTensor(np.array(ply_types)).to(device)
        values = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device)

        piece_logits, tile_logits, pred_values = network(boards)

        v_loss = nn.MSELoss()(pred_values, values)

        p_loss = torch.tensor(0.0, device=device)
        piece_mask = (ply_types_t == 0)
        tile_mask = (ply_types_t == 1)

        if piece_mask.any():
            targets = torch.FloatTensor(
                np.array([p for p, pt in zip(piece_pols, ply_types) if pt == 0])
            ).to(device)
            log_probs = torch.log_softmax(piece_logits[piece_mask], dim=1)
            p_loss = p_loss - (targets * log_probs).sum(dim=1).mean()

        if tile_mask.any():
            targets = torch.FloatTensor(
                np.array([t for t, pt in zip(tile_pols, ply_types) if pt == 1])
            ).to(device)
            log_probs = torch.log_softmax(tile_logits[tile_mask], dim=1)
            p_loss = p_loss - (targets * log_probs).sum(dim=1).mean()

        loss = p_loss + v_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_p += p_loss.item()
        total_v += v_loss.item()
        n += 1

    return total_loss / max(n, 1), total_p / max(n, 1), total_v / max(n, 1)


def evaluate_vs_ga(network, device, weights, n_games=50):
    """Evaluate NN (greedy value head) vs GA."""
    nn_wins = ga_wins = draws = 0

    for g in range(n_games):
        state = NonagaState()
        nn_player = g % 2
        ply = 0

        while not state.is_terminal() and ply < 200:
            if int(state.current_player) == nn_player:
                # NN greedy value head
                moves = state.get_legal_moves()
                if not moves:
                    move = None
                else:
                    # Immediate win check
                    move = None
                    for m in moves:
                        ns = state.apply_move(m)
                        if ns.winner is not None and int(ns.winner) == state.current_player:
                            move = m
                            break
                    if move is None:
                        best_v, best_m = -9999, moves[0]
                        for m in moves:
                            ns = state.apply_move(m)
                            b = torch.FloatTensor(ns.encode()).unsqueeze(0).to(device)
                            with torch.no_grad():
                                _, _, v = network(b)
                            val = -v.item() if state.ply_type == PlyType.TILE_MOVE else v.item()
                            if val > best_v:
                                best_v, best_m = val, m
                        move = best_m
            else:
                move = greedy_move(state, weights)

            if move is None:
                if state.ply_type == PlyType.TILE_MOVE:
                    state = state.copy()
                    state.current_player = Player(1 - state.current_player)
                    state.ply_type = PlyType.PIECE_MOVE
                    ply += 1
                    continue
                else:
                    break
            state = state.apply_move(move)
            ply += 1

        if state.winner is not None:
            if int(state.winner) == nn_player:
                nn_wins += 1
            else:
                ga_wins += 1
        else:
            draws += 1

    return nn_wins, ga_wins, draws


def main():
    parser = argparse.ArgumentParser(description='Distill GA knowledge into NN')
    parser.add_argument('--games', type=int, default=10000,
                        help='Number of GA-vs-GA games to generate')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval-every', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--eval-games', type=int, default=50)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Generate dataset
    print(f"\nGenerating {args.games} GA-vs-GA games...")
    t0 = time.time()
    raw_examples, wins, draws, avg_plies = generate_dataset(args.games, GA_WEIGHTS)
    gen_time = time.time() - t0
    decisive = wins[0] + wins[1]
    print(f"  {gen_time:.1f}s — {len(raw_examples)} examples "
          f"(P1={wins[0]} P2={wins[1]} D={draws}, avg {avg_plies:.0f} plies)")
    print(f"  {decisive}/{args.games} decisive ({decisive/args.games:.0%})")

    # Filter: only keep examples from decisive games
    # (draws give value=0 which adds noise)
    decisive_examples = [ex for ex in raw_examples if ex[4] != 0.0]
    print(f"  {len(decisive_examples)} decisive examples (filtered out draws)")

    # Augment
    print("Augmenting with D6 symmetry (12×)...")
    t0 = time.time()
    augmented = augment_all(decisive_examples)
    print(f"  {time.time()-t0:.1f}s — {len(augmented)} augmented examples")

    # Load or create network
    network = NonagaNet().to(device)
    if not args.from_scratch:
        ckpt_path = args.checkpoint or 'checkpoints/island_2/iteration_20.pt'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            network.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded: {ckpt_path}")
        else:
            print(f"Checkpoint not found: {ckpt_path}, starting from scratch")

    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4)
    os.makedirs('checkpoints/distill', exist_ok=True)

    # Initial eval
    network.eval()
    nw, gw, dr = evaluate_vs_ga(network, device, GA_WEIGHTS, args.eval_games)
    print(f"\nInitial: NN {nw}-{gw}-{dr} (NN wins {nw/args.eval_games:.0%})")

    best_win_rate = nw / args.eval_games

    # Train
    print(f"\nTraining for {args.epochs} epochs on {len(augmented)} examples...")
    for epoch in range(args.epochs):
        t0 = time.time()
        loss, p_loss, v_loss = train_network(network, optimizer, augmented, device)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch}: loss={loss:.4f} (p={p_loss:.4f} v={v_loss:.4f}) [{elapsed:.0f}s]")

        if (epoch + 1) % args.eval_every == 0:
            network.eval()
            nw, gw, dr = evaluate_vs_ga(network, device, GA_WEIGHTS, args.eval_games)
            wr = nw / args.eval_games
            print(f"    Eval: NN {nw}-{gw}-{dr} ({wr:.0%})")

            torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/distill/epoch_{epoch}.pt')

            if wr > best_win_rate:
                best_win_rate = wr
                torch.save({'model_state_dict': network.state_dict()},
                           'checkpoints/distill/best.pt')
                print(f"    *** New best: {wr:.0%} ***")

    # Final eval
    network.eval()
    nw, gw, dr = evaluate_vs_ga(network, device, GA_WEIGHTS, 100)
    print(f"\nFinal (100 games): NN {nw}-{gw}-{dr} (NN wins {nw/100:.0%})")


if __name__ == '__main__':
    main()
