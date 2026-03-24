"""Bootstrap: train value head on endgame positions from random games."""

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

def flush(*args):
    print(*args, flush=True)


def play_random_game(max_plies=500):
    """Play a fully random game. Return positions + outcome."""
    state = NonagaState()
    history = []
    ply_count = 0

    while not state.is_terminal() and ply_count < max_plies:
        moves = state.get_legal_moves()
        if not moves:
            if state.ply_type == PlyType.TILE_MOVE:
                state = state.copy()
                state.current_player = Player(1 - state.current_player)
                state.ply_type = PlyType.PIECE_MOVE
                ply_count += 1
                continue
            else:
                break

        move = random.choice(moves)

        # Build policy over legal moves
        if state.ply_type == PlyType.PIECE_MOVE:
            policy = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
            for p, d, l in state.get_piece_moves():
                policy[NonagaState.piece_move_to_action(p, d)] = 1.0
        else:
            policy = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
            for s, d in state.get_tile_moves():
                policy[NonagaState.tile_move_to_action(s, d)] = 1.0
        policy /= policy.sum()

        history.append({
            'board': state.encode(),
            'ply_type': int(state.ply_type),
            'policy': policy,
            'player': int(state.current_player),
        })

        state = state.apply_move(move)
        ply_count += 1

    return history, state.winner, ply_count


def augment(examples):
    augmented = []
    for board, ply_type, policy, value in examples:
        if ply_type == int(PlyType.PIECE_MOVE):
            pp, tp = policy, np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
        else:
            pp, tp = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32), policy
        for ab, app, atp, av in augment_example(board, pp, tp, value):
            augmented.append((ab, ply_type, app if ply_type == int(PlyType.PIECE_MOVE) else atp, av))
    return augmented


def eval_vs_random(net, device, n_games=30, n_sims=50):
    config = Config()
    config.num_mcts_sims = n_sims
    config.max_game_plies = 500
    net.eval()
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


# ============================================================
flush("Generating random games for endgame training data...")
flush("=" * 60)

net = NonagaNet()
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
net = net.to(device)
flush(f'Device: {device}')

# Generate a BIG batch of random games once
all_examples = []
wins_count = 0
ENDGAME_PLIES = 30  # Only keep last N plies before a win

for g in range(1000):
    history, winner, plies = play_random_game(max_plies=500)
    if winner is None:
        continue  # Skip draws entirely — only learn from decisive games

    wins_count += 1
    winner_int = int(winner)

    # Keep last ENDGAME_PLIES positions (most informative)
    endgame = history[-ENDGAME_PLIES:] if len(history) > ENDGAME_PLIES else history

    for ex in endgame:
        if ex['player'] == winner_int:
            value = 1.0
        else:
            value = -1.0
        all_examples.append((ex['board'], ex['ply_type'], ex['policy'], np.float32(value)))

flush(f'Generated {wins_count} decisive games out of 1000')
flush(f'Endgame training examples: {len(all_examples)}')

# Augment
flush('Augmenting...')
augmented = augment(all_examples)
flush(f'Augmented: {len(all_examples)} -> {len(augmented)}')

# Train for many epochs with persistent optimizer
flush('\nTraining...')
net.train()
optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-4)

for epoch in range(20):
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

    flush(f'  Epoch {epoch}: total={tl/nb:.4f} piece={pl/nb:.4f} tile={til/nb:.4f} value={vl/nb:.4f}')

    # Eval every 5 epochs
    if (epoch + 1) % 5 == 0:
        m, r, d = eval_vs_random(net, device, n_games=20, n_sims=25)
        flush(f'  >> EVAL: Model={m} Random={r} Draws={d} ({m/(m+r+d):.0%})')
        net.train()

torch.save({'model_state_dict': net.state_dict()}, 'checkpoints/endgame_trained.pt')
flush('\nSaved endgame_trained.pt')

# Final eval
flush('\n' + '=' * 60)
flush('FINAL EVAL (50 games, 50 sims)')
flush('=' * 60)
m, r, d = eval_vs_random(net, device, n_games=50, n_sims=50)
flush(f'Model={m} Random={r} Draws={d} ({m/(m+r+d):.0%} model win rate)')
