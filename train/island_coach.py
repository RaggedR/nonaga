"""
Island-model AlphaZero training with ring topology and cross-play.

N models arranged on a ring. Each iteration has two phases:
1. Self-play: each island plays itself (standard AlphaZero)
2. Cross-play: each island plays its ring neighbor (diversity engine)

Cross-play is what makes this genuinely population-based rather than
just N independent training runs. When island i plays island (i+1),
the resulting positions come from different strategies meeting — board
states that neither model would encounter in self-play. Both islands
train on these cross-games, learning from the collision of strategies.

Ring topology controls who sees who: island 0 never directly encounters
island 3's strategy — it diffuses through 1 and 2 first. Same slow
mixing that preserves diversity in the GA.

Usage:
    # Full run (5 islands, 100 iterations)
    python -m train.island_coach --islands 5 --iterations 100 --games 100 --sims 100

    # Smoke test
    python -m train.island_coach --islands 3 --iterations 2 --games 5 --sims 10

    # Resume from iteration
    python -m train.island_coach --islands 5 --iterations 100 --resume 20
"""

import copy
import os
import random
import time
from collections import deque

import numpy as np
import torch

from game.nonaga import NonagaState, PlyType, Player
from game.symmetry import augment_example
from model.network import NonagaNet
from train.mcts import MCTS
from train.self_play import generate_self_play_data_parallel, play_game, _shaped_draw_value
from train.config import Config
from train.coach import get_device, Coach


class IslandCoach:
    """
    Orchestrates N AlphaZero models on a ring with cross-play.

    Each island has its own model, replay buffer, and checkpoint directory.
    Every iteration:
    1. Self-play: each island plays itself (standard AlphaZero data)
    2. Cross-play: each island plays its ring neighbor — the diversity engine
    3. Train: each island trains on its combined self-play + cross-play data
    4. Arena: each island's new model vs its previous checkpoint

    Cross-play frequency is controlled by cross_play_rate (fraction of
    total games per iteration that are cross-games with ring neighbor).
    """

    def __init__(self, config=None):
        self.config = config or Config()
        self.device = get_device()
        self.num_islands = self.config.num_islands
        self.cross_play_rate = getattr(self.config, 'island_cross_play_rate', 0.3)

        # Per-island state
        self.networks = []
        self.optimizers = []
        self.replay_buffers = []
        self.checkpoint_dirs = []

        for i in range(self.num_islands):
            net = NonagaNet().to(self.device)
            self.networks.append(net)
            self.optimizers.append(torch.optim.Adam(
                net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            ))
            self.replay_buffers.append(deque(maxlen=self.config.replay_buffer_size))
            ckpt_dir = os.path.join(self.config.checkpoint_dir, f"island_{i}")
            os.makedirs(ckpt_dir, exist_ok=True)
            self.checkpoint_dirs.append(ckpt_dir)

        self.iteration = 0
        print(f"Island-Model AlphaZero")
        print(f"  {self.num_islands} islands, ring topology with cross-play")
        print(f"  Cross-play rate: {self.cross_play_rate:.0%} of games")
        print(f"  Migration: every {self.config.island_migration_freq} iters, "
              f"rate {self.config.island_migration_rate}")
        print(f"  Device: {self.device}")

    def _init_from_checkpoint(self, checkpoint_path):
        """Initialize all islands from the same checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        for net in self.networks:
            net.load_state_dict(checkpoint['model_state_dict'])
        print(f"All islands initialized from {checkpoint_path}")

    def train(self):
        """Run the full island-model training loop."""
        start_iter = self.iteration
        end_iter = start_iter + self.config.num_iterations

        for iteration in range(start_iter, end_iter):
            self.iteration = iteration
            iter_start = time.time()

            print(f"\n{'='*60}")
            print(f"Iteration {iteration} (#{iteration - start_iter + 1}/{self.config.num_iterations})")
            print(f"{'='*60}")

            # Phase 1: Self-play — each island plays itself
            for isl in range(self.num_islands):
                self._island_self_play(isl)

            # Phase 2: Cross-play — ring neighbors play each other
            if self.num_islands >= 2 and self.cross_play_rate > 0:
                self._cross_play_round()

            # Phase 3: Train each island on its combined data
            for isl in range(self.num_islands):
                self._island_train(isl, iteration)

            # Data migration (supplement to cross-play)
            if (iteration > 0 and
                    iteration % self.config.island_migration_freq == 0):
                self._ring_migrate()

            # Cross-island diversity check
            if iteration % 5 == 0:
                self._log_diversity(iteration)

            iter_time = time.time() - iter_start
            print(f"\nIteration {iteration} complete in {iter_time:.1f}s")

    def _island_self_play(self, island_idx):
        """Generate self-play data for one island (model plays itself)."""
        net = self.networks[island_idx]
        ckpt_dir = self.checkpoint_dirs[island_idx]

        # Reduce self-play games to make room for cross-play
        total_games = self.config.num_self_play_games
        self_play_games = max(1, int(total_games * (1 - self.cross_play_rate)))

        # Use a shallow copy of config to avoid mutating the shared instance
        sp_config = copy.copy(self.config)
        sp_config.num_self_play_games = self_play_games

        print(f"\n--- Island {island_idx}: self-play ({self_play_games} games) ---")
        temp_ckpt = os.path.join(ckpt_dir, "_temp_selfplay.pt")
        torch.save({'model_state_dict': net.state_dict()}, temp_ckpt)
        examples, stats = generate_self_play_data_parallel(
            temp_ckpt, sp_config, win_mode='triangle')
        os.remove(temp_ckpt)

        augmented = self._augment(examples)
        self.replay_buffers[island_idx].append(augmented)
        print(f"  {stats['num_examples']} examples "
              f"(P1={stats['wins_p1']} P2={stats['wins_p2']} D={stats['draws']} "
              f"avg={stats['avg_plies']:.0f})")

    def _cross_play_round(self):
        """
        Cross-play on ring edges: island i plays island (i+1) % n.

        Both islands get the training examples from these games, producing
        positions from different strategies meeting.
        """
        total_games = self.config.num_self_play_games
        cross_games = max(1, int(total_games * self.cross_play_rate))
        n = self.num_islands

        print(f"\n--- Cross-play round ({cross_games} games per edge) ---")

        for i in range(n):
            j = (i + 1) % n
            examples_i, examples_j, stats = self._play_cross_games(i, j, cross_games)

            # Augment and add to both islands' buffers
            aug_i = self._augment(examples_i)
            aug_j = self._augment(examples_j)
            self.replay_buffers[i].append(aug_i)
            self.replay_buffers[j].append(aug_j)

            print(f"  Island {i} vs {j}: {stats['num_examples']} examples "
                  f"(P1={stats['wins_p1']} P2={stats['wins_p2']} D={stats['draws']} "
                  f"avg={stats['avg_plies']:.0f})")

    def _play_cross_games(self, island_a, island_b, num_games):
        """
        Play games between two islands' models. Returns training examples
        for each island separately (with correct value targets).

        Alternates who plays as Player ONE to avoid first-move bias.
        """
        net_a = self.networks[island_a]
        net_b = self.networks[island_b]
        net_a.eval()
        net_b.eval()
        mcts_a = MCTS(net_a, self.config)
        mcts_b = MCTS(net_b, self.config)

        all_examples_a = []  # examples from island_a's perspective
        all_examples_b = []  # examples from island_b's perspective
        wins = {0: 0, 1: 0}
        draws = 0
        total_plies = 0

        for g in range(num_games):
            # Alternate sides to avoid first-mover bias
            if g % 2 == 0:
                mcts_p1, mcts_p2 = mcts_a, mcts_b
            else:
                mcts_p1, mcts_p2 = mcts_b, mcts_a

            examples, winner, ply_count = self._play_one_cross_game(
                mcts_p1, mcts_p2)
            total_plies += ply_count

            if winner is not None:
                wins[int(winner)] += 1
            else:
                draws += 1

            # Split examples by which island generated them
            for ex in examples:
                # Both islands get all examples — the positions are
                # informative for both. Value targets are already correct
                # (relative to the player at that position).
                all_examples_a.append(ex)
                all_examples_b.append(ex)

        stats = {
            'num_examples': len(all_examples_a),
            'wins_p1': wins[0],
            'wins_p2': wins[1],
            'draws': draws,
            'avg_plies': total_plies / max(num_games, 1),
        }
        return all_examples_a, all_examples_b, stats

    def _play_one_cross_game(self, mcts_p1, mcts_p2):
        """
        Play one game with two different MCTS instances.
        Returns training examples in the same format as self_play.play_game.
        """
        state = NonagaState()
        raw_examples = []
        ply_count = 0

        while not state.is_terminal() and ply_count < self.config.max_game_plies:
            mcts = mcts_p1 if state.current_player == Player.ONE else mcts_p2

            temp_late = getattr(self.config, 'temp_late', 0.0)
            temp = 1.0 if ply_count < self.config.temp_threshold else temp_late

            move, policy = mcts.get_action_with_temp(state, temperature=temp)

            if move is None:
                if state.ply_type == PlyType.TILE_MOVE:
                    state = state.copy()
                    state.current_player = Player(1 - state.current_player)
                    state.ply_type = PlyType.PIECE_MOVE
                    ply_count += 1
                    continue
                else:
                    break

            board = state.encode()
            raw_examples.append({
                'board': board,
                'ply_type': int(state.ply_type),
                'policy': policy,
                'player': int(state.current_player),
            })

            state = state.apply_move(move)
            ply_count += 1

        # Assign value targets (same logic as self_play.play_game)
        # TODO: draw shaping uses final state for all positions — ideally each
        # position should use its own state for shaped draw values. This is a
        # known limitation shared with self_play.play_game.
        winner = int(state.winner) if state.winner is not None else -1
        use_draw_shaping = (winner == -1)

        training_examples = []
        for ex in raw_examples:
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

    def _island_train(self, island_idx, iteration):
        """Train one island on its accumulated replay buffer, run arena."""
        net = self.networks[island_idx]
        buf = self.replay_buffers[island_idx]
        ckpt_dir = self.checkpoint_dirs[island_idx]

        print(f"\n--- Island {island_idx}: train ---")

        all_data = []
        for b in buf:
            all_data.extend(b)
        random.shuffle(all_data)
        loss_info = self._train_network(net, all_data, self.optimizers[island_idx])
        print(f"  Loss={loss_info['total']:.4f} "
              f"(p={loss_info['piece']:.4f} t={loss_info['tile']:.4f} "
              f"v={loss_info['value']:.4f}) on {len(all_data)} examples")

        # Arena (skip first iteration)
        if iteration > 0:
            prev_path = os.path.join(ckpt_dir, f"iteration_{iteration - 1}.pt")
            if os.path.exists(prev_path):
                win_rate = self._arena(net, prev_path)
                print(f"  Arena: {win_rate:.1%}", end="")
                if win_rate < self.config.arena_threshold:
                    print(" → rejected")
                    checkpoint = torch.load(prev_path, map_location='cpu',
                                            weights_only=True)
                    net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    print(" → accepted")

        # Save checkpoint (includes optimizer state for proper resume)
        path = os.path.join(ckpt_dir, f"iteration_{iteration}.pt")
        torch.save({
            'iteration': iteration,
            'island': island_idx,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': self.optimizers[island_idx].state_dict(),
        }, path)

    def _ring_migrate(self):
        """
        Ring migration of training data.

        Island i sends migration_rate fraction of its replay buffer
        to island (i+1) % n. Data is appended to the receiving island's
        buffer (may temporarily exceed max size; oldest data will be
        naturally evicted on next append).
        """
        rate = self.config.island_migration_rate
        n = self.num_islands
        if n < 2:
            return

        # Collect migrant data from each island before modifying any
        migrants = []
        for isl in range(n):
            buf = self.replay_buffers[isl]
            if not buf:
                migrants.append([])
                continue
            # Flatten buffer, sample fraction
            all_examples = []
            for batch in buf:
                all_examples.extend(batch)
            num_to_send = max(1, int(len(all_examples) * rate))
            sampled = random.sample(all_examples, min(num_to_send, len(all_examples)))
            migrants.append(sampled)

        # Send to ring neighbor
        total_migrated = 0
        for isl in range(n):
            dest = (isl + 1) % n
            if migrants[isl]:
                # Append as a new "batch" in the destination's buffer
                self.replay_buffers[dest].append(migrants[isl])
                total_migrated += len(migrants[isl])

        print(f"\n  Ring migration: {total_migrated} total examples migrated")

    def _log_diversity(self, iteration):
        """Log cross-island diversity metrics."""
        # Compare model outputs on a fixed set of random positions
        positions = []
        state = NonagaState()
        positions.append(state)
        # Generate a few positions by random play
        rng_state = random.getstate()
        random.seed(iteration * 1000)
        for _ in range(20):
            s = NonagaState()
            for _ in range(random.randint(4, 30)):
                moves = s.get_legal_moves()
                if not moves or s.is_terminal():
                    break
                s = s.apply_move(random.choice(moves))
            if not s.is_terminal():
                positions.append(s)
        random.setstate(rng_state)

        if not positions:
            return

        # Evaluate each position with each island's network
        boards = np.array([s.encode() for s in positions])
        boards_t = torch.tensor(boards, dtype=torch.float32).to(self.device)

        values_per_island = []
        for net in self.networks:
            net.eval()
            with torch.no_grad():
                _, _, vals = net(boards_t)
            values_per_island.append(vals.squeeze(1).cpu().numpy())

        # Compute pairwise value disagreement
        n = len(values_per_island)
        if n < 2:
            return

        disagreements = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = np.mean(np.abs(values_per_island[i] - values_per_island[j]))
                disagreements.append(diff)

        mean_disagree = np.mean(disagreements)
        print(f"\n  Diversity: mean value disagreement = {mean_disagree:.4f} "
              f"(across {len(positions)} positions)")

    def _augment(self, examples):
        """Augment examples with D6 symmetries (same as Coach._augment)."""
        augmented = []
        for board, ply_type, policy, value in examples:
            if ply_type == int(PlyType.PIECE_MOVE):
                piece_policy = policy
                tile_policy = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
            else:
                piece_policy = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
                tile_policy = policy

            aug_list = augment_example(board, piece_policy, tile_policy, value)
            for aug_board, aug_pp, aug_tp, aug_val in aug_list:
                if ply_type == int(PlyType.PIECE_MOVE):
                    augmented.append((aug_board, ply_type, aug_pp, aug_val))
                else:
                    augmented.append((aug_board, ply_type, aug_tp, aug_val))
        return augmented

    def _train_network(self, network, data, optimizer):
        """Train a single network on data using a persistent optimizer."""
        import torch.nn as nn

        network.train()

        total_loss_sum = 0
        piece_loss_sum = 0
        tile_loss_sum = 0
        value_loss_sum = 0
        num_batches = 0

        for epoch in range(self.config.num_epochs):
            random.shuffle(data)
            for i in range(0, len(data), self.config.batch_size):
                batch = data[i:i + self.config.batch_size]
                if len(batch) < 2:
                    continue

                boards = torch.tensor(
                    np.array([x[0] for x in batch]),
                    dtype=torch.float32
                ).to(self.device)
                ply_types = [x[1] for x in batch]
                policies = [x[2] for x in batch]
                values = torch.tensor(
                    [x[3] for x in batch],
                    dtype=torch.float32
                ).unsqueeze(1).to(self.device)

                piece_logits, tile_logits, pred_values = network(boards)
                value_loss = nn.functional.mse_loss(pred_values, values)

                piece_loss = torch.tensor(0.0, device=self.device)
                tile_loss = torch.tensor(0.0, device=self.device)
                piece_count = 0
                tile_count = 0

                for j, (ply_type, policy) in enumerate(zip(ply_types, policies)):
                    target = torch.tensor(policy, dtype=torch.float32,
                                          device=self.device)
                    if ply_type == int(PlyType.PIECE_MOVE):
                        log_probs = nn.functional.log_softmax(piece_logits[j], dim=0)
                        piece_loss = piece_loss - (target * log_probs).sum()
                        piece_count += 1
                    else:
                        log_probs = nn.functional.log_softmax(tile_logits[j], dim=0)
                        tile_loss = tile_loss - (target * log_probs).sum()
                        tile_count += 1

                if piece_count > 0:
                    piece_loss = piece_loss / piece_count
                if tile_count > 0:
                    tile_loss = tile_loss / tile_count

                total_loss = piece_loss + tile_loss + value_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_loss_sum += total_loss.item()
                piece_loss_sum += piece_loss.item()
                tile_loss_sum += tile_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1

        if num_batches == 0:
            return {'total': 0, 'piece': 0, 'tile': 0, 'value': 0}

        return {
            'total': total_loss_sum / num_batches,
            'piece': piece_loss_sum / num_batches,
            'tile': tile_loss_sum / num_batches,
            'value': value_loss_sum / num_batches,
        }

    def _arena(self, network, prev_checkpoint_path):
        """Pit current network against previous checkpoint. Returns win rate."""
        prev_network = NonagaNet().to(self.device)
        checkpoint = torch.load(prev_checkpoint_path, map_location='cpu',
                                weights_only=True)
        prev_network.load_state_dict(checkpoint['model_state_dict'])
        prev_network.to(self.device)
        prev_network.eval()

        new_mcts = MCTS(network, self.config)
        old_mcts = MCTS(prev_network, self.config)

        new_score = 0
        for game_idx in range(self.config.arena_games):
            if game_idx % 2 == 0:
                winner = self._play_arena_game(new_mcts, old_mcts)
                if winner == 0:
                    new_score += 1
                elif winner is None:
                    new_score += 0.5
            else:
                winner = self._play_arena_game(old_mcts, new_mcts)
                if winner == 1:
                    new_score += 1
                elif winner is None:
                    new_score += 0.5

        return new_score / self.config.arena_games

    def _play_arena_game(self, mcts_p1, mcts_p2):
        """Play one arena game. Returns winner (0, 1, or None)."""
        state = NonagaState()
        ply_count = 0

        while not state.is_terminal() and ply_count < self.config.max_game_plies:
            mcts = mcts_p1 if state.current_player == 0 else mcts_p2
            move, _ = mcts.get_action_with_temp(state, temperature=0, add_noise=False)
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

        return int(state.winner) if state.winner is not None else None

    def _save_all(self, iteration):
        """Save all island checkpoints."""
        for isl in range(self.num_islands):
            path = os.path.join(self.checkpoint_dirs[isl], f"iteration_{iteration}.pt")
            torch.save({
                'iteration': iteration,
                'island': isl,
                'model_state_dict': self.networks[isl].state_dict(),
                'optimizer_state_dict': self.optimizers[isl].state_dict(),
            }, path)

    def _load_all(self, iteration):
        """Load all island checkpoints from a given iteration."""
        for isl in range(self.num_islands):
            path = os.path.join(self.checkpoint_dirs[isl], f"iteration_{iteration}.pt")
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                self.networks[isl].load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizers[isl].load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  Island {isl}: loaded iteration {iteration}")
            else:
                print(f"  Island {isl}: no checkpoint at {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Island-model AlphaZero for Nonaga")
    parser.add_argument("--islands", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--migration-freq", type=int, default=None)
    parser.add_argument("--migration-rate", type=float, default=None)
    parser.add_argument("--cross-play-rate", type=float, default=None,
                        help="Fraction of games that are cross-play (default: 0.3)")
    parser.add_argument("--resume", type=int, default=None,
                        help="Resume from iteration")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Initialize all islands from this checkpoint")
    args = parser.parse_args()

    config = Config()
    if args.islands is not None:
        config.num_islands = args.islands
    if args.iterations is not None:
        config.num_iterations = args.iterations
    if args.games is not None:
        config.num_self_play_games = args.games
    if args.sims is not None:
        config.num_mcts_sims = args.sims
    if args.migration_freq is not None:
        config.island_migration_freq = args.migration_freq
    if args.migration_rate is not None:
        config.island_migration_rate = args.migration_rate
    if args.cross_play_rate is not None:
        config.island_cross_play_rate = args.cross_play_rate

    coach = IslandCoach(config)

    if args.init_checkpoint:
        coach._init_from_checkpoint(args.init_checkpoint)

    if args.resume is not None:
        print(f"Resuming from iteration {args.resume}")
        coach._load_all(args.resume)
        coach.iteration = args.resume

    coach.train()


if __name__ == "__main__":
    main()
