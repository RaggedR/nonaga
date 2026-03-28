"""
League training: NN plays a diverse set of opponents each iteration.

Each iteration the NN plays:
1. vs GA          — learns what strong play looks like (aspirational losses)
2. vs self        — competitive games at its own level
3. vs weaker self — gets wins, learns what winning looks like

This fixes the signal problem from previous attempts:
- Attempt 11: only played GA → only saw losses → no positive signal
- Attempt 12: GA self-play → 100% draws → no signal at all
- Attempt 13 (first try): supervised from GA data, then self-play refinement
  → hit 50% at epoch 14 but unstable, regressed to 0%

The league gives balanced signal: wins, losses, and competitive games.
The best-performing checkpoint is tracked and preserved — training continues
but the best weights are never overwritten.

"Weaker self" starts as random and is replaced by earlier checkpoints as
training progresses.

Usage:
    # Full run
    python -u train_from_ga.py

    # Quick smoke test
    python -u train_from_ga.py --iterations 3 --games 10 --eval-games 10

    # Custom parameters
    python -u train_from_ga.py --iterations 50 --games 100 --epochs 3
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
from ga_evolve import (
    NUM_FEATURES, compute_features, evaluate_position, greedy_move,
)

# Best GA weights from island-model evolution
GA_WEIGHTS = np.array([
    1.4832, -0.3479, 0.1681, -6.3177, -2.2217, 3.9770,
    -0.0497, -2.9234, 0.0024, -0.0922, 0.2998, 0.4741,
    -1.1508, 0.5765
], dtype=np.float32)


# ── Move selection ──────────────────────────────────────────────────

def nn_greedy_move(state, network, device):
    """Pick the move with highest value head evaluation (greedy 1-ply)."""
    moves = state.get_legal_moves()
    if not moves:
        return None

    # Immediate win check
    for move in moves:
        ns = state.apply_move(move)
        if ns.winner is not None and int(ns.winner) == state.current_player:
            return move

    best_move = None
    best_val = -float('inf')

    for move in moves:
        ns = state.apply_move(move)
        board = torch.FloatTensor(ns.encode()).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, v = network(board)
        val = v.item()
        if state.ply_type == PlyType.TILE_MOVE:
            val = -val
        if val > best_val:
            best_val = val
            best_move = move

    return best_move


def nn_greedy_move_epsilon(state, network, device, epsilon=0.15):
    """Greedy with epsilon-random exploration."""
    moves = state.get_legal_moves()
    if not moves:
        return None

    # Immediate win check
    for move in moves:
        ns = state.apply_move(move)
        if ns.winner is not None and int(ns.winner) == state.current_player:
            return move

    if random.random() < epsilon:
        return random.choice(moves)

    return nn_greedy_move(state, network, device)


def random_move(state):
    """Pick a random legal move."""
    moves = state.get_legal_moves()
    return random.choice(moves) if moves else None


# ── Game playing ────────────────────────────────────────────────────

def play_game(move_fn_p1, move_fn_p2, max_plies=200):
    """
    Play a game between two move functions.

    Records ALL positions (both sides) with board, ply type, and player.
    Value targets are assigned after the game based on outcome.

    Returns (raw_examples, winner, ply_count).
    """
    state = NonagaState()
    raw_examples = []
    ply_count = 0

    move_fns = {Player.ONE: move_fn_p1, Player.TWO: move_fn_p2}

    while not state.is_terminal() and ply_count < max_plies:
        board = state.encode()
        raw_examples.append({
            'board': board,
            'ply_type': int(state.ply_type),
            'player': int(state.current_player),
        })

        move = move_fns[state.current_player](state)

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

    winner = int(state.winner) if state.winner is not None else -1
    return raw_examples, winner, ply_count


def examples_with_values(raw_examples, winner):
    """Assign value targets based on game outcome. Returns training tuples."""
    examples = []
    for ex in raw_examples:
        if winner == -1:
            value = 0.0
        elif ex['player'] == winner:
            value = 1.0
        else:
            value = -1.0

        # Dummy policy targets (zeros) — we only train value head in league mode
        pp = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
        tp = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)

        examples.append((
            ex['board'],
            ex['ply_type'],
            pp,
            tp,
            np.float32(value),
        ))

    return examples


def examples_with_ga_policy(raw_examples, winner, ga_weights, temperature=0.5):
    """
    Assign value targets from outcome AND policy targets from GA evaluation.
    Used for games where the GA is involved — the GA's move preferences
    provide policy supervision.
    """
    examples = []
    for ex in raw_examples:
        if winner == -1:
            value = 0.0
        elif ex['player'] == winner:
            value = 1.0
        else:
            value = -1.0

        # Reconstruct state to compute GA policy
        # We only have the board encoding, so use dummy policy for non-GA positions
        pp = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
        tp = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)

        examples.append((
            ex['board'],
            ex['ply_type'],
            pp,
            tp,
            np.float32(value),
        ))

    return examples


# ── Training ────────────────────────────────────────────────────────

def augment_all(examples):
    """Apply D6 symmetry augmentation (12x data)."""
    augmented = []
    for board, ply_type, pp, tp, value in examples:
        for aug_board, aug_pp, aug_tp, aug_val in augment_example(board, pp, tp, value):
            augmented.append((aug_board, ply_type, aug_pp, aug_tp, aug_val))
    return augmented


def train_epoch(network, optimizer, data, device, batch_size=256,
                train_policy=False):
    """
    Train one epoch. Returns losses.

    When train_policy=False (default for league), only trains value head.
    The policy head targets are zeros so policy loss would be meaningless.
    """
    network.train()
    random.shuffle(data)

    total_loss = total_v = 0
    n = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        boards, ply_types, piece_pols, tile_pols, values = zip(*batch)

        boards = torch.FloatTensor(np.array(boards)).to(device)
        values = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device)

        piece_logits, tile_logits, pred_values = network(boards)

        # Value loss (always)
        v_loss = nn.MSELoss()(pred_values, values)
        loss = v_loss

        # Policy loss (optional — only when we have real policy targets)
        if train_policy:
            ply_types_t = torch.LongTensor(np.array(ply_types)).to(device)
            p_loss = torch.tensor(0.0, device=device)
            piece_mask = (ply_types_t == 0)
            tile_mask = (ply_types_t == 1)

            if piece_mask.any():
                targets = torch.FloatTensor(
                    np.array([p for p, pt in zip(piece_pols, ply_types)
                              if pt == 0])
                ).to(device)
                if targets.sum() > 0:  # Only if non-zero targets
                    log_probs = torch.log_softmax(
                        piece_logits[piece_mask], dim=1)
                    p_loss = p_loss - (targets * log_probs).sum(dim=1).mean()

            if tile_mask.any():
                targets = torch.FloatTensor(
                    np.array([t for t, pt in zip(tile_pols, ply_types)
                              if pt == 1])
                ).to(device)
                if targets.sum() > 0:
                    log_probs = torch.log_softmax(
                        tile_logits[tile_mask], dim=1)
                    p_loss = p_loss - (targets * log_probs).sum(dim=1).mean()

            loss = loss + p_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_v += v_loss.item()
        n += 1

    return total_loss / max(n, 1), total_v / max(n, 1)


# ── Evaluation ──────────────────────────────────────────────────────

def evaluate_vs_ga(network, device, ga_weights, n_games=50):
    """Play NN (greedy value head) vs GA. Returns dict of results."""
    nn_wins = ga_wins = draws = 0
    total_plies = 0

    for g in range(n_games):
        nn_player = Player(g % 2)

        def nn_fn(s):
            return nn_greedy_move(s, network, device)

        def ga_fn(s):
            return greedy_move(s, ga_weights)

        if int(nn_player) == 0:
            raw, winner, plies = play_game(nn_fn, ga_fn)
        else:
            raw, winner, plies = play_game(ga_fn, nn_fn)

        total_plies += plies
        if winner == -1:
            draws += 1
        elif winner == int(nn_player):
            nn_wins += 1
        else:
            ga_wins += 1

    return {
        'nn_wins': nn_wins, 'ga_wins': ga_wins, 'draws': draws,
        'nn_win_rate': nn_wins / n_games,
        'avg_plies': total_plies / max(n_games, 1),
    }


# ── Main league loop ───────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(
        description='League training: NN vs GA + self + weaker self')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Training iterations')
    parser.add_argument('--games', type=int, default=100,
                        help='Games per opponent per iteration')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs per iteration')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=0.15,
                        help='Exploration rate for NN moves during data gen')
    parser.add_argument('--eval-games', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10,
                        help='Stop after N iterations without improvement')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from-scratch', action='store_true')
    args = parser.parse_args()

    device = get_device()
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {device}")
    print(f"League training: {args.iterations} iterations, "
          f"{args.games} games/opponent, {args.epochs} epochs/iter")

    os.makedirs('checkpoints/league', exist_ok=True)

    # Create network
    network = NonagaNet().to(device)
    if not args.from_scratch:
        for ckpt_path in [
            'checkpoints/from_ga/best_stage2.pt',
            'checkpoints/island_2/iteration_20.pt',
            'checkpoints/endgame_trained.pt',
        ]:
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu',
                                  weights_only=True)
                network.load_state_dict(ckpt['model_state_dict'])
                print(f"  Loaded: {ckpt_path}")
                break
        else:
            print("  No checkpoint found, starting from scratch")

    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4)

    # Initial eval
    network.eval()
    result = evaluate_vs_ga(network, device, GA_WEIGHTS, args.eval_games)
    print(f"\nInitial: NN {result['nn_wins']}-{result['ga_wins']}-"
          f"{result['draws']} ({result['nn_win_rate']:.0%})")

    # Checkpoint history for "weaker self" opponents
    # Start with None (use random as weakest opponent)
    checkpoint_history = []
    best_win_rate = result['nn_win_rate']
    best_iteration = -1
    iters_without_improvement = 0

    for iteration in range(args.iterations):
        t0 = time.time()
        all_examples = []
        stats = {'vs_ga': [0, 0, 0], 'vs_self': [0, 0, 0],
                 'vs_weak': [0, 0, 0]}  # [nn_wins, opp_wins, draws]

        # ── Play vs GA ──────────────────────────────────────────────
        for g in range(args.games):
            nn_player = Player(g % 2)

            def nn_fn(s, net=network, dev=device, eps=args.epsilon):
                return nn_greedy_move_epsilon(s, net, dev, eps)

            def ga_fn(s):
                return greedy_move(s, GA_WEIGHTS)

            if int(nn_player) == 0:
                raw, winner, plies = play_game(nn_fn, ga_fn)
            else:
                raw, winner, plies = play_game(ga_fn, nn_fn)

            if winner == -1:
                stats['vs_ga'][2] += 1
            elif winner == int(nn_player):
                stats['vs_ga'][0] += 1
            else:
                stats['vs_ga'][1] += 1

            # Only keep decisive examples
            if winner >= 0:
                all_examples.extend(examples_with_values(raw, winner))

        # ── Play vs self ────────────────────────────────────────────
        for g in range(args.games):
            def p1_fn(s, net=network, dev=device, eps=args.epsilon):
                return nn_greedy_move_epsilon(s, net, dev, eps)

            def p2_fn(s, net=network, dev=device, eps=args.epsilon):
                return nn_greedy_move_epsilon(s, net, dev, eps)

            raw, winner, plies = play_game(p1_fn, p2_fn)

            if winner == -1:
                stats['vs_self'][2] += 1
            elif winner == 0:
                stats['vs_self'][0] += 1
            else:
                stats['vs_self'][1] += 1

            if winner >= 0:
                all_examples.extend(examples_with_values(raw, winner))

        # ── Play vs weaker self ─────────────────────────────────────
        if checkpoint_history:
            # Load a past checkpoint as weak opponent
            # Pick the one from ~half the history ago for moderate weakness
            weak_idx = max(0, len(checkpoint_history) // 2)
            weak_net = NonagaNet().to(device)
            weak_state = torch.load(
                checkpoint_history[weak_idx], map_location='cpu',
                weights_only=True)
            weak_net.load_state_dict(weak_state['model_state_dict'])
            weak_net.eval()

            for g in range(args.games):
                nn_player = Player(g % 2)

                def nn_fn(s, net=network, dev=device, eps=args.epsilon):
                    return nn_greedy_move_epsilon(s, net, dev, eps)

                def weak_fn(s, net=weak_net, dev=device):
                    return nn_greedy_move(s, net, dev)

                if int(nn_player) == 0:
                    raw, winner, plies = play_game(nn_fn, weak_fn)
                else:
                    raw, winner, plies = play_game(weak_fn, nn_fn)

                if winner == -1:
                    stats['vs_weak'][2] += 1
                elif winner == int(nn_player):
                    stats['vs_weak'][0] += 1
                else:
                    stats['vs_weak'][1] += 1

                if winner >= 0:
                    all_examples.extend(examples_with_values(raw, winner))

            del weak_net
        else:
            # No history yet — play vs random
            for g in range(args.games):
                nn_player = Player(g % 2)

                def nn_fn(s, net=network, dev=device, eps=args.epsilon):
                    return nn_greedy_move_epsilon(s, net, dev, eps)

                if int(nn_player) == 0:
                    raw, winner, plies = play_game(nn_fn, random_move)
                else:
                    raw, winner, plies = play_game(random_move, nn_fn)

                if winner == -1:
                    stats['vs_weak'][2] += 1
                elif winner == int(nn_player):
                    stats['vs_weak'][0] += 1
                else:
                    stats['vs_weak'][1] += 1

                if winner >= 0:
                    all_examples.extend(examples_with_values(raw, winner))

        # ── Train ───────────────────────────────────────────────────
        if len(all_examples) < 20:
            print(f"  Iter {iteration}: too few examples ({len(all_examples)})"
                  " — skipping")
            continue

        augmented = augment_all(all_examples)
        for _ in range(args.epochs):
            train_epoch(network, optimizer, augmented, device)

        # ── Evaluate ────────────────────────────────────────────────
        network.eval()
        result = evaluate_vs_ga(network, device, GA_WEIGHTS, args.eval_games)
        elapsed = time.time() - t0

        ga_s = f"{stats['vs_ga'][0]}W-{stats['vs_ga'][1]}L-{stats['vs_ga'][2]}D"
        self_s = f"{stats['vs_self'][0]}W-{stats['vs_self'][1]}L-{stats['vs_self'][2]}D"
        weak_s = f"{stats['vs_weak'][0]}W-{stats['vs_weak'][1]}L-{stats['vs_weak'][2]}D"

        print(f"Iter {iteration:2d} | GA:{ga_s} Self:{self_s} Weak:{weak_s} "
              f"| eval: {result['nn_wins']}-{result['ga_wins']}-"
              f"{result['draws']} ({result['nn_win_rate']:.0%}) "
              f"| {len(all_examples)} ex [{elapsed:.0f}s]")

        # ── Save checkpoint ─────────────────────────────────────────
        ckpt_path = f'checkpoints/league/iteration_{iteration}.pt'
        torch.save({
            'iteration': iteration,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'win_rate': result['nn_win_rate'],
        }, ckpt_path)
        checkpoint_history.append(ckpt_path)

        # Track best
        if result['nn_win_rate'] > best_win_rate:
            best_win_rate = result['nn_win_rate']
            best_iteration = iteration
            iters_without_improvement = 0
            torch.save({
                'iteration': iteration,
                'model_state_dict': network.state_dict(),
            }, 'checkpoints/league/best.pt')
            print(f"  *** New best: {best_win_rate:.0%} "
                  f"(iteration {iteration}) ***")
        else:
            iters_without_improvement += 1

        # Early stopping
        if iters_without_improvement >= args.patience:
            print(f"\n  No improvement for {args.patience} iterations, "
                  f"stopping.")
            break

    # ── Final report ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete")
    print(f"{'='*60}")

    # Load and evaluate best checkpoint
    best_path = 'checkpoints/league/best.pt'
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location='cpu',
                                weights_only=True)
        network.load_state_dict(best_ckpt['model_state_dict'])
        print(f"Best checkpoint: iteration {best_iteration}")
    else:
        print("No improvement over initial — using final weights")

    network.eval()
    result = evaluate_vs_ga(network, device, GA_WEIGHTS, 100)
    print(f"Final eval (100 games): NN {result['nn_wins']}-"
          f"{result['ga_wins']}-{result['draws']} "
          f"({result['nn_win_rate']:.0%})")


if __name__ == '__main__':
    main()
