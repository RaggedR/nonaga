"""
Sparse Autoencoder probe on the Nonaga neural network's value head.

Discovers what features the NN has learned by decomposing the 64-dim
value bottleneck into sparse, interpretable components.

Pipeline:
1. Generate diverse positions (random games + self-play)
2. Collect 64-dim activations from value head hidden layer
3. Train SAE (64 → 256 → 64) with L1 sparsity
4. Analyze: for each SAE feature, find max-activating positions
5. Visualize board states to interpret what each feature means
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from model.network import NonagaNet
from game.nonaga import NonagaState, PlyType, Player
from game.hex_grid import (
    NEIGHBORS, DIRECTIONS, VALID_SET, VALID_CELLS,
    idx_to_qr, qr_to_idx,
)


# ── Position generation ─────────────────────────────────────────────

def generate_random_positions(n_games=500, max_plies=200):
    """Play random games, collect all intermediate positions."""
    positions = []
    for _ in range(n_games):
        state = NonagaState()
        ply = 0
        while not state.is_terminal() and ply < max_plies:
            moves = state.get_legal_moves()
            if not moves:
                break
            state = state.apply_move(random.choice(moves))
            positions.append(state)
            ply += 1
    return positions


def generate_structured_positions(positions):
    """Add edge cases: near-wins, opening, backed positions."""
    extras = []
    for pos in positions[:2000]:
        # Already have these, just tag them
        pass
    return positions


# ── Activation collection ────────────────────────────────────────────

def collect_activations(net, positions, device, batch_size=512):
    """Run positions through the network, capture value head hidden layer."""
    net.eval()
    all_activations = []
    all_values = []
    all_boards = []

    # Hook to capture val_fc1 output (after ReLU)
    activations_buf = []

    def hook_fn(module, input, output):
        activations_buf.append(output.detach())

    handle = net.val_fc1.register_forward_hook(hook_fn)

    for i in range(0, len(positions), batch_size):
        batch_states = positions[i:i + batch_size]
        boards = np.array([s.encode() for s in batch_states])
        boards_t = torch.tensor(boards, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, _, values = net(boards_t)

        # The hook captured pre-ReLU output; apply ReLU to match forward pass
        raw = activations_buf[-1]
        acts = F.relu(raw)  # Match the forward pass: F.relu(self.val_fc1(v))
        all_activations.append(acts.cpu())
        all_values.append(values.squeeze(1).cpu())
        all_boards.append(boards)

    handle.remove()

    return (
        torch.cat(all_activations, dim=0),  # (N, 64)
        torch.cat(all_values, dim=0),        # (N,)
        positions,
    )


# ── Sparse Autoencoder ──────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """
    Overcomplete autoencoder with L1 sparsity on hidden activations.
    Decomposes 64 entangled neurons into 256 sparse features.
    """

    def __init__(self, input_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        # Tie decoder bias to act as data centering
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # Center
        x_centered = x - self.bias
        # Encode → sparse hidden
        hidden = F.relu(self.encoder(x_centered))
        # Decode
        x_hat = self.decoder(hidden) + self.bias
        return x_hat, hidden

    def loss(self, x, l1_coeff=5e-3):
        x_hat, hidden = self.forward(x)
        mse = F.mse_loss(x_hat, x)
        l1 = hidden.abs().mean()
        return mse + l1_coeff * l1, mse, l1, hidden


def train_sae(activations, input_dim=64, hidden_dim=256,
              epochs=200, batch_size=1024, lr=1e-3, l1_coeff=5e-3):
    """Train the SAE on collected activations."""
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    sae = SparseAutoencoder(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = activations.to(device)
    n = len(dataset)

    print(f"\nTraining SAE: {input_dim} → {hidden_dim} → {input_dim}")
    print(f"  {n} samples, {epochs} epochs, L1={l1_coeff}")

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = total_mse = total_l1 = 0
        total_active = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = dataset[idx]

            loss, mse, l1, hidden = sae.loss(batch, l1_coeff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_l1 += l1.item()
            total_active += (hidden > 0).float().mean().item()
            n_batches += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            avg_active = total_active / n_batches
            print(f"  Epoch {epoch:3d}: loss={total_loss/n_batches:.4f} "
                  f"mse={total_mse/n_batches:.4f} l1={total_l1/n_batches:.4f} "
                  f"active={avg_active:.1%}")

    return sae


# ── Feature analysis ─────────────────────────────────────────────────

def analyze_features(sae, activations, values, positions, device, top_k=10):
    """For each SAE feature, find max-activating positions and statistics."""
    sae.eval()
    dataset = activations.to(device)

    with torch.no_grad():
        _, all_hidden = sae(dataset)
    all_hidden = all_hidden.cpu().numpy()  # (N, 256)
    values_np = values.numpy()

    n_features = all_hidden.shape[1]

    # Feature statistics
    features = []
    for f in range(n_features):
        feat_acts = all_hidden[:, f]
        n_active = (feat_acts > 0).sum()
        if n_active < 5:
            continue  # Skip dead features

        mean_act = feat_acts[feat_acts > 0].mean()
        # Correlation with value
        mask = feat_acts > 0
        if mask.sum() > 10:
            corr = np.corrcoef(feat_acts[mask], values_np[mask])[0, 1]
        else:
            corr = 0.0

        # Top activating positions
        top_idx = np.argsort(feat_acts)[-top_k:][::-1]

        features.append({
            'idx': f,
            'n_active': int(n_active),
            'frac_active': n_active / len(feat_acts),
            'mean_activation': float(mean_act),
            'max_activation': float(feat_acts.max()),
            'value_corr': float(corr),
            'top_positions': top_idx,
            'top_activations': feat_acts[top_idx],
            'top_values': values_np[top_idx],
        })

    # Sort by max activation (most distinctive features first)
    features.sort(key=lambda f: f['max_activation'], reverse=True)
    return features


# ── Board visualization ──────────────────────────────────────────────

def render_board(state):
    """Render a hex board as ASCII art."""
    # Build grid using idx_to_qr for tile/piece indices
    grid = {}
    for idx in state.tiles:
        q, r = idx_to_qr(idx)
        if idx in state.pieces[Player.ONE]:
            grid[(q, r)] = 'A'
        elif idx in state.pieces[Player.TWO]:
            grid[(q, r)] = 'B'
        else:
            grid[(q, r)] = '·'

    lines = []
    for r in range(7):
        indent = '  ' * (3 - r) if r <= 3 else '  ' * (r - 3)
        cells = []
        for q in range(7):
            if (q, r) in grid:
                cells.append(grid[(q, r)])
        if cells:
            lines.append(indent + '  '.join(cells))
    return '\n'.join(lines)


def compute_board_features(state):
    """Compute interpretable features for a position."""
    features = {}

    for p in [Player.ONE, Player.TWO]:
        label = 'P1' if p == Player.ONE else 'P2'
        pcs = list(state.pieces[p])

        # Adjacency pairs
        adj = 0
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                if pcs[j] in set(NEIGHBORS.get(pcs[i], ())):
                    adj += 1
        features[f'{label}_adj'] = adj

        # Backed pieces
        backed = 0
        for piece in pcs:
            q, r = idx_to_qr(piece)
            for dq, dr in DIRECTIONS:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in VALID_SET or qr_to_idx(nq, nr) not in state.tiles:
                    backed += 1
                    break
        features[f'{label}_backed'] = backed

        # Mobility (number of valid slide destinations)
        mobility = 0
        for piece in pcs:
            q, r = idx_to_qr(piece)
            for dq, dr in DIRECTIONS:
                nq, nr = q + dq, r + dr
                while True:
                    if (nq, nr) not in VALID_SET or qr_to_idx(nq, nr) not in state.tiles:
                        break
                    if qr_to_idx(nq, nr) in state.occupied:
                        break
                    mobility += 1
                    nq, nr = nq + dq, nr + dr
        features[f'{label}_mobility'] = mobility

        # Min pairwise distance
        if len(pcs) >= 2:
            min_dist = float('inf')
            max_dist = 0
            for i in range(len(pcs)):
                qi, ri = idx_to_qr(pcs[i])
                for j in range(i + 1, len(pcs)):
                    qj, rj = idx_to_qr(pcs[j])
                    # Hex distance
                    d = (abs(qi - qj) + abs(ri - rj) + abs((qi + ri) - (qj + rj))) // 2
                    min_dist = min(min_dist, d)
                    max_dist = max(max_dist, d)
            features[f'{label}_min_dist'] = min_dist
            features[f'{label}_max_dist'] = max_dist

        # Triangle-completing cells
        completing = 0
        for i in range(len(pcs)):
            nbrs_i = set(NEIGHBORS.get(pcs[i], ()))
            for j in range(i + 1, len(pcs)):
                nbrs_j = set(NEIGHBORS.get(pcs[j], ()))
                cells = (nbrs_i & nbrs_j) - set(pcs)
                for c in cells:
                    if c in state.tiles and c not in state.occupied:
                        completing += 1
        features[f'{label}_completing'] = completing

    features['ply_type'] = 'PIECE' if state.ply_type == PlyType.PIECE_MOVE else 'TILE'
    features['current'] = 'P1' if state.current_player == Player.ONE else 'P2'
    return features


def print_feature_report(features, positions, values, top_n_features=20, top_n_positions=5):
    """Print detailed report of discovered features."""
    print(f"\n{'='*70}")
    print(f"SAE FEATURE ANALYSIS — Top {top_n_features} features")
    print(f"{'='*70}")

    for rank, feat in enumerate(features[:top_n_features]):
        corr_str = f"{feat['value_corr']:+.3f}"
        direction = "→ GOOD" if feat['value_corr'] > 0.2 else "→ BAD" if feat['value_corr'] < -0.2 else "→ neutral"

        print(f"\n{'─'*70}")
        print(f"Feature #{feat['idx']} (rank {rank+1})")
        print(f"  Active: {feat['frac_active']:.1%} of positions | "
              f"Max activation: {feat['max_activation']:.2f} | "
              f"Value correlation: {corr_str} {direction}")
        print()

        for i in range(min(top_n_positions, len(feat['top_positions']))):
            pos_idx = feat['top_positions'][i]
            state = positions[pos_idx]
            val = feat['top_values'][i]
            act = feat['top_activations'][i]

            bf = compute_board_features(state)
            print(f"  Position {i+1} (activation={act:.2f}, value={val:+.2f}):")
            print(f"    Turn: {bf['current']} {bf['ply_type']} | "
                  f"P1: adj={bf['P1_adj']} backed={bf['P1_backed']} mob={bf['P1_mobility']} "
                  f"dist={bf.get('P1_min_dist','?')}-{bf.get('P1_max_dist','?')} "
                  f"completing={bf['P1_completing']}")
            print(f"    P2: adj={bf['P2_adj']} backed={bf['P2_backed']} mob={bf['P2_mobility']} "
                  f"dist={bf.get('P2_min_dist','?')}-{bf.get('P2_max_dist','?')} "
                  f"completing={bf['P2_completing']}")

            board_str = render_board(state)
            for line in board_str.split('\n'):
                print(f"      {line}")
            print()

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    active_features = [f for f in features if f['frac_active'] > 0.01]
    positive = [f for f in active_features if f['value_corr'] > 0.2]
    negative = [f for f in active_features if f['value_corr'] < -0.2]
    neutral = [f for f in active_features if -0.2 <= f['value_corr'] <= 0.2]

    print(f"Total active features: {len(active_features)}")
    print(f"  Positive (good for current player): {len(positive)}")
    print(f"  Negative (bad for current player):  {len(negative)}")
    print(f"  Neutral:                            {len(neutral)}")

    # Feature co-occurrence with known board features
    print(f"\nTop 10 features correlated with VALUE (good positions):")
    by_corr = sorted(active_features, key=lambda f: f['value_corr'], reverse=True)
    for f in by_corr[:10]:
        print(f"  Feature #{f['idx']:3d}: corr={f['value_corr']:+.3f} "
              f"active={f['frac_active']:.1%} max={f['max_activation']:.2f}")

    print(f"\nTop 10 features anti-correlated with VALUE (bad positions):")
    for f in by_corr[-10:]:
        print(f"  Feature #{f['idx']:3d}: corr={f['value_corr']:+.3f} "
              f"active={f['frac_active']:.1%} max={f['max_activation']:.2f}")


# ── Correlation with known features ──────────────────────────────────

def correlate_with_known_features(sae, activations, positions, device):
    """Check which SAE features correlate with known hand-designed features."""
    sae.eval()
    dataset = activations.to(device)

    with torch.no_grad():
        _, all_hidden = sae(dataset)
    all_hidden = all_hidden.cpu().numpy()

    # Compute known features for all positions
    print(f"\nComputing board features for {len(positions)} positions...")
    known = defaultdict(list)
    for state in positions:
        bf = compute_board_features(state)
        for k, v in bf.items():
            if isinstance(v, (int, float)):
                known[k].append(v)

    known_arr = {k: np.array(v) for k, v in known.items()}

    print(f"\n{'='*70}")
    print("SAE FEATURES vs KNOWN BOARD FEATURES")
    print(f"{'='*70}")

    for feat_name, feat_vals in sorted(known_arr.items()):
        # Find SAE features most correlated with this known feature
        correlations = []
        for f in range(all_hidden.shape[1]):
            sae_acts = all_hidden[:, f]
            if (sae_acts > 0).sum() < 20:
                continue
            corr = np.corrcoef(sae_acts, feat_vals)[0, 1]
            if not np.isnan(corr):
                correlations.append((f, corr))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        top3 = correlations[:3]

        print(f"\n  {feat_name}:")
        for f_idx, corr in top3:
            print(f"    SAE #{f_idx:3d}: r={corr:+.3f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Load best model
    import glob
    checkpoints = glob.glob('checkpoints/iteration_*.pt')
    if not checkpoints:
        print("No checkpoints found!")
        return

    # Find highest iteration number
    def iter_num(path):
        import re
        m = re.search(r'iteration_(\d+)', path)
        return int(m.group(1)) if m else -1

    best_ckpt = max(checkpoints, key=iter_num)
    print(f"Loading: {best_ckpt}")

    net = NonagaNet()
    ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=True)
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(device)
    net.eval()

    # Generate positions
    print("\nGenerating positions from random games...")
    positions = generate_random_positions(n_games=1000, max_plies=150)
    print(f"  Collected {len(positions)} positions")

    # Collect activations
    print("\nCollecting value head activations...")
    activations, values, positions = collect_activations(net, positions, device)
    print(f"  Activations shape: {activations.shape}")
    print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"  Value std: {values.std():.3f}")

    # Train SAE
    sae = train_sae(
        activations,
        input_dim=64,
        hidden_dim=256,
        epochs=300,
        batch_size=1024,
        lr=1e-3,
        l1_coeff=5e-3,
    )

    # Analyze features
    features = analyze_features(sae, activations, values, positions, device)
    print_feature_report(features, positions, values, top_n_features=20, top_n_positions=3)

    # Correlate with known features
    correlate_with_known_features(sae, activations, positions, device)

    # Save SAE
    torch.save({
        'sae_state_dict': sae.state_dict(),
        'feature_stats': [{k: v for k, v in f.items()
                          if k not in ('top_positions', 'top_activations', 'top_values')}
                         for f in features],
    }, 'checkpoints/sae_probe.pt')
    print(f"\nSaved SAE to checkpoints/sae_probe.pt")


if __name__ == '__main__':
    main()
