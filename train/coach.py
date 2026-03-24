"""
Training coach: orchestrates the AlphaZero training loop.

Each iteration:
1. Self-play with current model → generate training examples
2. Augment with D6 symmetries (12× data)
3. Train on replay buffer (last N iterations)
4. Arena: new model vs current model
5. If new model wins > threshold, accept it
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from game.nonaga import NonagaState, PlyType, Player
from game.symmetry import augment_example
from model.network import NonagaNet
from train.mcts import MCTS
from train.self_play import generate_self_play_data, generate_self_play_data_parallel, play_game
from train.config import Config


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Coach:
    """Orchestrates AlphaZero training."""

    def __init__(self, config=None):
        self.config = config or Config()
        self.device = get_device()
        self.network = NonagaNet().to(self.device)
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)
        self.iteration = 0
        print(f"Using device: {self.device}")

    def train(self):
        """Run the full training loop with optional curriculum pretraining."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Phase 1: Curriculum pretraining (adjacency wins)
        if self.config.curriculum_pretrain_iters > 0 and self.iteration == 0:
            self._run_curriculum_phase()

        # Phase 2: Full AlphaZero training (triangle wins, draw shaping)
        self._run_full_training()

    def _run_curriculum_phase(self):
        """Phase 1: Pretrain with simpler win condition (2-adjacent = win)."""
        n_iters = self.config.curriculum_pretrain_iters
        win_mode = self.config.curriculum_win_mode
        print(f"\n{'='*60}")
        print(f"PHASE 1: Curriculum pretraining ({win_mode} wins, {n_iters} iters)")
        print(f"{'='*60}")

        # Temporarily swap config for curriculum
        saved_sims = self.config.num_mcts_sims
        saved_games = self.config.num_self_play_games
        self.config.num_mcts_sims = self.config.curriculum_num_mcts_sims
        self.config.num_self_play_games = self.config.curriculum_num_games

        curriculum_buffer = deque(maxlen=self.config.replay_buffer_size)

        for i in range(n_iters):
            print(f"\n--- Curriculum iter {i+1}/{n_iters} ---")

            # Self-play with curriculum win mode
            temp_ckpt = os.path.join(self.config.checkpoint_dir, "_temp_selfplay.pt")
            torch.save({'model_state_dict': self.network.state_dict()}, temp_ckpt)
            examples, stats = generate_self_play_data_parallel(
                temp_ckpt, self.config, win_mode=win_mode)
            os.remove(temp_ckpt)
            print(f"  {stats['num_examples']} examples "
                  f"(P1={stats['wins_p1']} P2={stats['wins_p2']} D={stats['draws']} "
                  f"avg_plies={stats['avg_plies']:.1f})")

            # Augment and train
            augmented = self._augment(examples)
            curriculum_buffer.append(augmented)
            all_data = []
            for buf in curriculum_buffer:
                all_data.extend(buf)
            random.shuffle(all_data)
            loss_info = self._train_network(all_data)
            print(f"  Loss: total={loss_info['total']:.4f} "
                  f"piece={loss_info['piece']:.4f} "
                  f"tile={loss_info['tile']:.4f} "
                  f"value={loss_info['value']:.4f}")

            # Save curriculum checkpoint
            self._save_checkpoint(i, prefix="curriculum_")

        # Restore config
        self.config.num_mcts_sims = saved_sims
        self.config.num_self_play_games = saved_games
        print(f"\nCurriculum pretraining complete. Weights carry over to Phase 2.")

    def _run_full_training(self):
        """Phase 2: Full AlphaZero training with triangle wins and draw shaping."""
        start = self.iteration + 1 if self.iteration > 0 else 0
        end = start + self.config.num_iterations

        if self.config.curriculum_pretrain_iters > 0 and start == 0:
            print(f"\n{'='*60}")
            print(f"PHASE 2: Full training (triangle wins, draw shaping)")
            print(f"{'='*60}")

        for iteration in range(start, end):
            self.iteration = iteration
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} (#{iteration - start + 1}/{self.config.num_iterations})")
            print(f"{'='*60}")

            # 1. Self-play (parallel, triangle mode with draw shaping)
            print("\n--- Self-play ---")
            temp_ckpt = os.path.join(self.config.checkpoint_dir, "_temp_selfplay.pt")
            torch.save({'model_state_dict': self.network.state_dict()}, temp_ckpt)
            examples, stats = generate_self_play_data_parallel(
                temp_ckpt, self.config, win_mode='triangle')
            os.remove(temp_ckpt)
            print(f"Generated {stats['num_examples']} examples "
                  f"(P1={stats['wins_p1']} P2={stats['wins_p2']} D={stats['draws']} "
                  f"avg_plies={stats['avg_plies']:.1f})")

            # 2. Augment
            print("\n--- Augmentation ---")
            augmented = self._augment(examples)
            print(f"Augmented: {len(examples)} → {len(augmented)} examples")
            self.replay_buffer.append(augmented)

            # 3. Train
            print("\n--- Training ---")
            all_data = []
            for buf in self.replay_buffer:
                all_data.extend(buf)
            random.shuffle(all_data)
            loss_info = self._train_network(all_data)
            print(f"Loss: total={loss_info['total']:.4f} "
                  f"piece={loss_info['piece']:.4f} "
                  f"tile={loss_info['tile']:.4f} "
                  f"value={loss_info['value']:.4f}")

            # 4. Arena
            if iteration > 0:
                print("\n--- Arena ---")
                win_rate = self._arena()
                print(f"New model win rate: {win_rate:.1%}")
                if win_rate < self.config.arena_threshold:
                    print("Rejected: loading previous model")
                    self._load_checkpoint(iteration - 1)
                    continue
                else:
                    print("Accepted!")

            # 5. Save checkpoint
            self._save_checkpoint(iteration)
            print(f"Saved checkpoint iteration_{iteration}.pt")

    def _augment(self, examples):
        """Augment examples with D6 symmetries."""
        augmented = []
        for board, ply_type, policy, value in examples:
            # Create dummy policies for the other ply type
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

    def _train_network(self, data):
        """Train the network on collected data."""
        self.network.train()
        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

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

                boards = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(self.device)
                ply_types = [x[1] for x in batch]
                policies = [x[2] for x in batch]
                values = torch.tensor([x[3] for x in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

                # Forward pass
                piece_logits, tile_logits, pred_values = self.network(boards)

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(pred_values, values)

                # Policy loss (cross-entropy with soft targets)
                piece_loss = torch.tensor(0.0, device=self.device)
                tile_loss = torch.tensor(0.0, device=self.device)
                piece_count = 0
                tile_count = 0

                for j, (ply_type, policy) in enumerate(zip(ply_types, policies)):
                    target = torch.tensor(policy, dtype=torch.float32, device=self.device)
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

    def _arena(self):
        """
        Pit new model against previous checkpoint.
        Returns win rate of new model.
        """
        # Load previous model
        prev_network = NonagaNet().to(self.device)
        prev_path = os.path.join(
            self.config.checkpoint_dir,
            f"iteration_{self.iteration - 1}.pt"
        )
        if not os.path.exists(prev_path):
            return 1.0  # No previous model, accept by default

        checkpoint = torch.load(prev_path, map_location="cpu", weights_only=True)
        prev_network.load_state_dict(checkpoint["model_state_dict"])
        prev_network.to(self.device)
        prev_network.eval()

        new_mcts = MCTS(self.network, self.config)
        old_mcts = MCTS(prev_network, self.config)

        new_score = 0
        for game_idx in range(self.config.arena_games):
            # Alternate who plays first
            if game_idx % 2 == 0:
                winner = self._play_arena_game(new_mcts, old_mcts)
                if winner == 0:  # Player ONE = new model
                    new_score += 1
                elif winner is None:  # Draw = 0.5 each
                    new_score += 0.5
            else:
                winner = self._play_arena_game(old_mcts, new_mcts)
                if winner == 1:  # Player TWO = new model
                    new_score += 1
                elif winner is None:  # Draw = 0.5 each
                    new_score += 0.5

        return new_score / self.config.arena_games

    def _play_arena_game(self, mcts_p1, mcts_p2):
        """Play one arena game. Returns winner (0, 1, or None for draw)."""
        state = NonagaState()
        ply_count = 0

        while not state.is_terminal() and ply_count < self.config.max_game_plies:
            if state.current_player == 0:
                mcts = mcts_p1
            else:
                mcts = mcts_p2

            move, _ = mcts.get_action_with_temp(state, temperature=0, add_noise=False)
            if move is None:
                # No legal moves — skip tile ply
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

    def _save_checkpoint(self, iteration, prefix="iteration_"):
        path = os.path.join(self.config.checkpoint_dir, f"{prefix}{iteration}.pt")
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
        }, path)

    def _load_checkpoint(self, iteration, prefix="iteration_"):
        path = os.path.join(self.config.checkpoint_dir, f"{prefix}{iteration}.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.network.load_state_dict(checkpoint["model_state_dict"])


def main():
    """Run training."""
    import argparse
    parser = argparse.ArgumentParser(description="Train Nonaga AlphaZero")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--resume", type=int, default=None, help="Resume from iteration")
    parser.add_argument("--curriculum", type=int, default=None,
                        help="Curriculum pretraining iterations (0 to skip)")
    args = parser.parse_args()

    config = Config()
    if args.iterations:
        config.num_iterations = args.iterations
    if args.games:
        config.num_self_play_games = args.games
    if args.sims:
        config.num_mcts_sims = args.sims
    if args.curriculum is not None:
        config.curriculum_pretrain_iters = args.curriculum

    coach = Coach(config)

    if args.resume is not None:
        print(f"Resuming from iteration {args.resume}")
        coach._load_checkpoint(args.resume)
        coach.iteration = args.resume

    coach.train()


if __name__ == "__main__":
    main()
