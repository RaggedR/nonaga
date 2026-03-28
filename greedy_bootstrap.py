"""
Bootstrap AlphaZero: use GreedyAI's value head to generate non-uniform policy targets.

The endgame-trained model has an excellent value head but a near-uniform policy head.
MCTS can't work with a uniform prior. This script:
1. Loads the endgame-trained model
2. Plays greedy-vs-random and greedy self-play games
3. At each position, evaluates all legal moves -> softmax(values) -> policy target
4. Trains both policy and value heads
5. Tests MCTS to see if the non-uniform policy enables focused search

The key insight: we're generating policy targets from the VALUE head's evaluations,
not from actual MCTS visit counts. This bootstraps the policy without needing MCTS.
"""

import torch
import random
import numpy as np
from model.network import NonagaNet
from train.mcts import MCTS
from train.config import Config
from game.nonaga import NonagaState, PlyType, Player
from game.symmetry import augment_example
import torch.nn as nn
import torch.optim as optim
import time


def flush(*args):
    print(*args, flush=True)


def greedy_evaluate_moves(net, device, state):
    """
    Evaluate all legal moves from this position using batched NN forward pass.

    Returns (moves, values) where values[i] is the value of move i
    from the CURRENT player's perspective.
    Returns (moves, None) if an immediate win is available.
    """
    moves = state.get_legal_moves()
    if not moves:
        return [], np.array([])

    # Check for immediate wins and collect next states
    next_states = []
    for m in moves:
        ns = state.apply_move(m)
        if ns.is_terminal() and ns.winner == state.current_player:
            return moves, None  # Signal: immediate win
        next_states.append(ns)

    # Batch evaluate all next states
    boards = np.array([ns.encode() for ns in next_states])
    boards_t = torch.tensor(boards, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, _, values_t = net(boards_t)

    raw_values = values_t.squeeze(1).cpu().numpy()

    # Flip sign when next state's current player != our player.
    # After piece move: same player does tile move -> no flip
    # After tile move: opponent's turn -> flip
    values = np.empty(len(moves), dtype=np.float32)
    for i, ns in enumerate(next_states):
        v = raw_values[i]
        if ns.current_player != state.current_player:
            v = -v
        values[i] = v

    return moves, values


def make_policy_target(state, moves, values, temperature=1.0):
    """Convert move values to a policy distribution via softmax."""
    if state.ply_type == PlyType.PIECE_MOVE:
        policy = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
    else:
        policy = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)

    if values is None:
        # Immediate win -- concentrate on winning moves
        for m in moves:
            ns = state.apply_move(m)
            if ns.is_terminal() and ns.winner == state.current_player:
                if state.ply_type == PlyType.PIECE_MOVE:
                    idx = NonagaState.piece_move_to_action(m[0], m[1])
                else:
                    idx = NonagaState.tile_move_to_action(m[0], m[1])
                policy[idx] = 1.0
                return policy
        return policy  # fallback

    # Softmax over legal move values
    v = values.astype(np.float64) / max(temperature, 1e-8)
    v -= v.max()
    exp_v = np.exp(v)
    probs = exp_v / exp_v.sum()

    for i, m in enumerate(moves):
        if state.ply_type == PlyType.PIECE_MOVE:
            idx = NonagaState.piece_move_to_action(m[0], m[1])
        else:
            idx = NonagaState.tile_move_to_action(m[0], m[1])
        policy[idx] = probs[i].astype(np.float32)

    return policy


def play_greedy_vs_random_game(net, device, policy_temp=1.0, max_plies=500):
    """
    Play one game: greedy model vs random.

    At EVERY position (both model and random turns), compute the greedy
    policy target from the value head. This means even the random player's
    positions get informative policy targets.
    """
    state = NonagaState()
    history = []
    ply_count = 0
    model_is_p1 = random.choice([True, False])

    while not state.is_terminal() and ply_count < max_plies:
        is_model = (state.current_player == 0) == model_is_p1

        moves, values = greedy_evaluate_moves(net, device, state)

        if not moves:
            if state.ply_type == PlyType.TILE_MOVE:
                state = state.copy()
                state.current_player = Player(1 - state.current_player)
                state.ply_type = PlyType.PIECE_MOVE
                ply_count += 1
                continue
            else:
                break

        # Policy target from value head (for both sides)
        policy = make_policy_target(state, moves, values, temperature=policy_temp)

        # Choose move
        if values is None:
            # Immediate win
            move = moves[0]  # placeholder
            for m in moves:
                ns = state.apply_move(m)
                if ns.is_terminal() and ns.winner == state.current_player:
                    move = m
                    break
        elif is_model:
            move = moves[np.argmax(values)]
        else:
            move = random.choice(moves)

        history.append({
            'board': state.encode(),
            'ply_type': int(state.ply_type),
            'policy': policy,
            'player': int(state.current_player),
        })

        state = state.apply_move(move)
        ply_count += 1

    return history, state.winner, ply_count


def play_greedy_selfplay_game(net, device, policy_temp=1.0, explore_eps=0.15, max_plies=500):
    """Play a greedy self-play game with epsilon-greedy exploration."""
    state = NonagaState()
    history = []
    ply_count = 0

    while not state.is_terminal() and ply_count < max_plies:
        moves, values = greedy_evaluate_moves(net, device, state)

        if not moves:
            if state.ply_type == PlyType.TILE_MOVE:
                state = state.copy()
                state.current_player = Player(1 - state.current_player)
                state.ply_type = PlyType.PIECE_MOVE
                ply_count += 1
                continue
            else:
                break

        policy = make_policy_target(state, moves, values, temperature=policy_temp)

        if values is None:
            move = moves[0]
            for m in moves:
                ns = state.apply_move(m)
                if ns.is_terminal() and ns.winner == state.current_player:
                    move = m
                    break
        elif random.random() < explore_eps:
            move = random.choice(moves)
        else:
            move = moves[np.argmax(values)]

        history.append({
            'board': state.encode(),
            'ply_type': int(state.ply_type),
            'policy': policy,
            'player': int(state.current_player),
        })

        state = state.apply_move(move)
        ply_count += 1

    return history, state.winner, ply_count


def eval_greedy_vs_random(net, device, n_games=30):
    """Quick eval: greedy model vs random."""
    net.eval()
    model_wins = random_wins = draws = 0

    for g in range(n_games):
        state = NonagaState()
        model_is_p1 = (g % 2 == 0)
        model_player = 0 if model_is_p1 else 1
        ply_count = 0

        while not state.is_terminal() and ply_count < 500:
            is_model = (state.current_player == 0) == model_is_p1

            if is_model:
                moves, values = greedy_evaluate_moves(net, device, state)
                if not moves:
                    move = None
                elif values is None:
                    move = moves[0]
                    for m in moves:
                        ns = state.apply_move(m)
                        if ns.is_terminal() and ns.winner == state.current_player:
                            move = m
                            break
                else:
                    move = moves[np.argmax(values)]
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

        if state.winner is not None:
            if int(state.winner) == model_player:
                model_wins += 1
            else:
                random_wins += 1
        else:
            draws += 1

    return model_wins, random_wins, draws


def eval_mcts_vs_random(net, device, n_games=20, n_sims=50):
    """Eval with MCTS to check if the policy prior is good enough."""
    net.eval()
    config = Config()
    config.num_mcts_sims = n_sims
    config.max_game_plies = 500
    mcts = MCTS(net, config)

    model_wins = random_wins = draws = 0
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

        if state.winner is not None:
            if int(state.winner) == model_player:
                model_wins += 1
            else:
                random_wins += 1
        else:
            draws += 1
    return model_wins, random_wins, draws


def augment(examples):
    """Apply D6 symmetry augmentation."""
    augmented = []
    for board, ply_type, policy, value in examples:
        if ply_type == int(PlyType.PIECE_MOVE):
            pp, tp = policy, np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
        else:
            pp, tp = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32), policy
        for ab, app, atp, av in augment_example(board, pp, tp, value):
            augmented.append((ab, ply_type, app if ply_type == int(PlyType.PIECE_MOVE) else atp, av))
    return augmented


# ============================================================
# MAIN
# ============================================================

flush("=" * 60)
flush("GreedyAI Bootstrap: training non-uniform policy from value head")
flush("=" * 60)

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
flush(f'Device: {device}')

# Load the endgame-trained model
net = NonagaNet()
checkpoint = torch.load('checkpoints/endgame_trained.pt', map_location='cpu', weights_only=True)
net.load_state_dict(checkpoint['model_state_dict'])
net = net.to(device)
flush('Loaded endgame_trained.pt')

# Verify greedy still works
flush('\nVerifying greedy baseline...')
net.eval()
m, r, d = eval_greedy_vs_random(net, device, n_games=20)
flush(f'  Greedy vs Random: {m}-{r}-{d} ({m/(m+r+d):.0%} model)')

flush('\nBaseline MCTS (50 sims) vs Random:')
m, r, d = eval_mcts_vs_random(net, device, n_games=10, n_sims=50)
flush(f'  MCTS vs Random: {m}-{r}-{d} ({m/(m+r+d):.0%} model)')

# Iterative bootstrap
NUM_ROUNDS = 3
GAMES_PER_ROUND = 200
POLICY_TEMP = 1.0
EPOCHS = 15
LR = 0.002

optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)

for round_idx in range(NUM_ROUNDS):
    flush(f'\n{"="*60}')
    flush(f'ROUND {round_idx + 1}/{NUM_ROUNDS}')
    flush(f'{"="*60}')

    # Generate games
    net.eval()
    all_examples = []
    wins = 0
    t0 = time.time()

    # 2/3 greedy-vs-random (always decisive), 1/3 greedy self-play (diverse positions)
    n_vs_random = GAMES_PER_ROUND * 2 // 3
    n_selfplay = GAMES_PER_ROUND - n_vs_random

    for g in range(n_vs_random):
        history, winner, plies = play_greedy_vs_random_game(net, device, policy_temp=POLICY_TEMP)
        if winner is None:
            continue
        wins += 1
        winner_int = int(winner)
        for ex in history:
            value = 1.0 if ex['player'] == winner_int else -1.0
            all_examples.append((ex['board'], ex['ply_type'], ex['policy'], np.float32(value)))

        if (g + 1) % 50 == 0:
            flush(f'  vs-random: {g+1}/{n_vs_random}, {wins} decisive, {len(all_examples)} examples')

    vs_random_decisive = wins

    for g in range(n_selfplay):
        history, winner, plies = play_greedy_selfplay_game(
            net, device, policy_temp=POLICY_TEMP, explore_eps=0.15)
        if winner is None:
            continue
        wins += 1
        winner_int = int(winner)
        for ex in history:
            value = 1.0 if ex['player'] == winner_int else -1.0
            all_examples.append((ex['board'], ex['ply_type'], ex['policy'], np.float32(value)))

    selfplay_decisive = wins - vs_random_decisive
    elapsed = time.time() - t0
    flush(f'  Generated {len(all_examples)} examples from {wins} decisive games '
          f'({vs_random_decisive} vs-random, {selfplay_decisive} self-play) '
          f'in {elapsed:.1f}s')

    if len(all_examples) < 100:
        flush('  Too few examples, skipping training')
        continue

    # Augment
    augmented = augment(all_examples)
    flush(f'  Augmented: {len(all_examples)} -> {len(augmented)}')

    # Train
    net.train()
    for epoch in range(EPOCHS):
        random.shuffle(augmented)
        tl = pl = til = vl = 0
        nb = 0
        for i in range(0, len(augmented), 256):
            batch = augmented[i:i + 256]
            if len(batch) < 2:
                continue
            boards = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
            ply_types = [x[1] for x in batch]
            policies = [x[2] for x in batch]
            values = torch.tensor([x[3] for x in batch], dtype=torch.float32).unsqueeze(1).to(device)

            piece_logits, tile_logits, pred_values = net(boards)
            value_loss = nn.functional.mse_loss(pred_values, values)

            piece_loss = torch.tensor(0.0, device=device)
            tile_loss = torch.tensor(0.0, device=device)
            pc = tc = 0
            for j, (pt, pol) in enumerate(zip(ply_types, policies)):
                target = torch.tensor(pol, dtype=torch.float32, device=device)
                if pt == int(PlyType.PIECE_MOVE):
                    lp = nn.functional.log_softmax(piece_logits[j], dim=0)
                    piece_loss = piece_loss - (target * lp).sum()
                    pc += 1
                else:
                    lp = nn.functional.log_softmax(tile_logits[j], dim=0)
                    tile_loss = tile_loss - (target * lp).sum()
                    tc += 1
            if pc > 0: piece_loss = piece_loss / pc
            if tc > 0: tile_loss = tile_loss / tc

            total_loss = piece_loss + tile_loss + value_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            tl += total_loss.item(); pl += piece_loss.item()
            til += tile_loss.item(); vl += value_loss.item()
            nb += 1

        if nb > 0 and (epoch + 1) % 5 == 0:
            flush(f'  Epoch {epoch}: total={tl/nb:.4f} piece={pl/nb:.4f} '
                  f'tile={til/nb:.4f} value={vl/nb:.4f}')

    # Evaluate
    flush('\n  Evaluating...')
    net.eval()

    m, r, d = eval_greedy_vs_random(net, device, n_games=30)
    flush(f'  Greedy vs Random: {m}-{r}-{d} ({m/(m+r+d):.0%})')

    flush(f'  Testing MCTS (50 sims)...')
    m, r, d = eval_mcts_vs_random(net, device, n_games=20, n_sims=50)
    flush(f'  MCTS vs Random: {m}-{r}-{d} ({m/(m+r+d):.0%})')

    if m > 0:
        flush(f'  ** MCTS is winning! Policy bootstrap is working!')

    torch.save({'model_state_dict': net.state_dict()}, f'checkpoints/bootstrap_round{round_idx+1}.pt')
    flush(f'  Saved bootstrap_round{round_idx+1}.pt')

# Final comprehensive eval
flush('\n' + '=' * 60)
flush('FINAL EVALUATION')
flush('=' * 60)

net.eval()
flush('\nGreedy vs Random (50 games):')
m, r, d = eval_greedy_vs_random(net, device, n_games=50)
flush(f'  {m}-{r}-{d} ({m/(m+r+d):.0%} model)')

flush('\nMCTS (50 sims) vs Random (30 games):')
m, r, d = eval_mcts_vs_random(net, device, n_games=30, n_sims=50)
flush(f'  {m}-{r}-{d} ({m/(m+r+d):.0%} model)')

flush('\nMCTS (100 sims) vs Random (30 games):')
m, r, d = eval_mcts_vs_random(net, device, n_games=30, n_sims=100)
flush(f'  {m}-{r}-{d} ({m/(m+r+d):.0%} model)')

torch.save({'model_state_dict': net.state_dict()}, 'checkpoints/bootstrap_final.pt')
flush('\nSaved bootstrap_final.pt')
