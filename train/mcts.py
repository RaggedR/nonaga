"""
AlphaZero-style Monte Carlo Tree Search.

Each node represents a game state. The tree is built incrementally:
1. SELECT: traverse tree using PUCT formula
2. EXPAND: create leaf node, evaluate with neural network
3. BACKUP: propagate value up the path

Key differences from vanilla MCTS:
- Prior probabilities from neural network guide exploration
- Value estimate from neural network replaces random rollouts
- Dirichlet noise at root for exploration during self-play

IMPORTANT: Nonaga uses two-ply turns (piece move + tile move by the same player).
The backup and selection must only flip value sign when the PLAYER changes between
nodes, not at every tree level. Standard alternating backup is WRONG here because
piece->tile transitions don't change the player.

Hybrid mode (greedy_tile=True): Instead of branching over ~100 tile moves in the
tree, resolve tile moves greedily using the NN value head. This collapses effective
branching from ~18*100=1800 to ~18, making low-sim MCTS viable.
"""

import math
import numpy as np
import torch
from game.nonaga import NonagaState, PlyType, Player


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = [
        'state', 'parent', 'action', 'children',
        'visit_count', 'value_sum', 'prior',
        '_legal_moves', '_expanded',
    ]

    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self._legal_moves = None
        self._expanded = False

    @property
    def q_value(self):
        """Mean value estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return self._expanded

    def legal_moves(self):
        if self._legal_moves is None:
            self._legal_moves = self.state.get_legal_moves()
        return self._legal_moves


class MCTS:
    """AlphaZero MCTS with neural network evaluation."""

    def __init__(self, network, config, greedy_tile=False):
        """
        Args:
            network: NonagaNet instance (or None for random play)
            config: Config with cpuct, num_mcts_sims, etc.
            greedy_tile: If True, resolve tile moves greedily instead of
                         branching in the search tree. Dramatically reduces
                         effective branching factor.
        """
        self.network = network
        self.config = config
        self.greedy_tile = greedy_tile

    def search(self, state, add_noise=True):
        """
        Run MCTS from the given state.
        Returns: policy vector (visit count distribution over actions).
        """
        root = MCTSNode(state)
        self._expand(root)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.children:
            noise = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(root.children)
            )
            eps = self.config.dirichlet_epsilon
            for child, n in zip(root.children, noise):
                child.prior = (1 - eps) * child.prior + eps * n

        # Run simulations
        for _ in range(self.config.num_mcts_sims):
            node = root
            # SELECT
            while node.is_expanded() and node.children:
                node = self._select_child(node)
            # EXPAND & EVALUATE
            if not node.state.is_terminal():
                value = self._expand(node)
            else:
                # Terminal node: value from current_player's perspective
                if node.state.winner is None:
                    value = 0.0  # draw
                elif node.state.winner == node.state.current_player:
                    value = 1.0
                else:
                    value = -1.0
            # BACKUP
            self._backup(node, value)

        return self._get_policy(root, state)

    def _select_child(self, node):
        """Select child with highest PUCT score.

        Only negates q_value when the child's current_player differs from
        the parent's (i.e., a real player change). For same-player transitions
        (piece -> tile), the child's value is already from the right perspective.
        """
        best_score = -float('inf')
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count)

        for child in node.children:
            # Negate only when player changes between parent and child
            if child.state.current_player != node.state.current_player:
                q = -child.q_value
            else:
                q = child.q_value
            u = self.config.cpuct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node):
        """
        Expand a node: create children, evaluate with NN.
        Returns the value estimate for this node's state.

        In hybrid mode (greedy_tile), children of PIECE_MOVE nodes skip through
        the tile move phase: each child represents the state AFTER the best greedy
        tile move has been applied.
        """
        state = node.state
        moves = node.legal_moves()

        if not moves:
            node._expanded = True
            return 0.0  # No moves available

        # Get NN evaluation
        if self.network is not None:
            board = state.encode()
            piece_probs, tile_probs, value = self.network.predict(board)
        else:
            # Random policy for testing
            piece_probs = np.ones(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
            tile_probs = np.ones(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
            value = 0.0

        # Get the appropriate policy based on ply type
        if state.ply_type == PlyType.PIECE_MOVE:
            policy = piece_probs
            action_fn = lambda m: NonagaState.piece_move_to_action(m[0], m[1])
        else:
            policy = tile_probs
            action_fn = lambda m: NonagaState.tile_move_to_action(m[0], m[1])

        # Mask and normalize
        mask = state.get_policy_mask()
        masked_policy = policy * mask
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            # Uniform over legal moves
            masked_policy = mask / mask.sum()

        # Create children
        for move in moves:
            action = action_fn(move)
            prior = masked_policy[action]
            child_state = state.apply_move(move)

            # Hybrid mode: collapse tile moves for children of piece-move nodes
            if (self.greedy_tile and
                    child_state.ply_type == PlyType.TILE_MOVE and
                    not child_state.is_terminal()):
                child_state = self._greedy_tile_step(child_state)

            child = MCTSNode(child_state, parent=node, action=move, prior=prior)
            node.children.append(child)

        node._expanded = True
        return value

    def _greedy_tile_step(self, state):
        """
        Greedily pick the best tile move using the NN value head.
        Returns the resulting state (should be opponent's PIECE_MOVE).
        """
        tile_moves = state.get_legal_moves()
        if not tile_moves:
            # No tile moves — skip to next player
            s = state.copy()
            s.current_player = Player(1 - s.current_player)
            s.ply_type = PlyType.PIECE_MOVE
            return s

        # Check for immediate wins
        next_states = []
        for tm in tile_moves:
            ns = state.apply_move(tm)
            if ns.is_terminal() and ns.winner == state.current_player:
                return ns
            next_states.append(ns)

        if self.network is None:
            # Random tile move when no network
            idx = np.random.randint(len(tile_moves))
            return next_states[idx]

        # Batch evaluate all tile moves
        boards = np.array([ns.encode() for ns in next_states])
        device = next(self.network.parameters()).device
        boards_t = torch.tensor(boards, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, _, values_t = self.network(boards_t)
        values = values_t.squeeze(1).cpu().numpy()

        # Flip values when next state's player differs (opponent's perspective)
        for i, ns in enumerate(next_states):
            if ns.current_player != state.current_player:
                values[i] = -values[i]

        best_idx = np.argmax(values)
        return next_states[best_idx]

    def _backup(self, node, value):
        """Propagate value up the tree.

        Flips the value sign only when the player changes between a node
        and its parent. In Nonaga's two-ply system, piece->tile transitions
        are same-player and should NOT flip.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            if current.parent is not None:
                if current.state.current_player != current.parent.state.current_player:
                    value = -value
                # Same player: don't flip
            current = current.parent

    def _get_policy(self, root, state):
        """
        Extract policy from root visit counts.
        Returns (policy_vector, moves_list) where policy is over the action space.
        """
        if state.ply_type == PlyType.PIECE_MOVE:
            policy = np.zeros(NonagaState.PIECE_ACTION_SIZE, dtype=np.float32)
            action_fn = lambda m: NonagaState.piece_move_to_action(m[0], m[1])
        else:
            policy = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
            action_fn = lambda m: NonagaState.tile_move_to_action(m[0], m[1])

        for child in root.children:
            action = action_fn(child.action)
            policy[action] = child.visit_count

        # Normalize
        total = policy.sum()
        if total > 0:
            policy /= total

        return policy

    def get_action_with_temp(self, state, temperature=1.0, add_noise=True):
        """
        Run MCTS and select an action using temperature.

        temperature=1.0: proportional to visit counts (exploration)
        temperature->0:   greedy (exploitation)

        In hybrid mode with a TILE_MOVE state, bypasses MCTS entirely
        and returns the greedy-best tile move.

        Returns: (action_move, policy_vector)
        """
        moves = state.get_legal_moves()
        if not moves:
            return None, np.zeros(1)

        # Hybrid mode: handle tile moves greedily, skip tree search
        if self.greedy_tile and state.ply_type == PlyType.TILE_MOVE:
            return self._greedy_tile_action(state, moves)

        policy = self.search(state, add_noise=add_noise)

        if temperature == 0:
            action_idx = np.argmax(policy)
        else:
            if temperature != 1.0:
                adjusted = policy ** (1.0 / temperature)
                total = adjusted.sum()
                if total > 0:
                    adjusted /= total
                else:
                    adjusted = policy
            else:
                adjusted = policy

            total = adjusted.sum()
            if total > 0:
                action_idx = np.random.choice(len(adjusted), p=adjusted)
            else:
                action_idx = np.argmax(policy) if policy.sum() > 0 else 0

        # Find the matching legal move
        if state.ply_type == PlyType.PIECE_MOVE:
            for move in moves:
                if NonagaState.piece_move_to_action(move[0], move[1]) == action_idx:
                    return move, policy
        else:
            for move in moves:
                if NonagaState.tile_move_to_action(move[0], move[1]) == action_idx:
                    return move, policy

        # Fallback: random legal move
        return moves[np.random.randint(len(moves))], policy

    def _greedy_tile_action(self, state, moves):
        """
        Greedy tile move selection for hybrid mode.
        Returns (best_move, policy_vector).
        """
        policy = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)

        # Check for immediate wins
        next_states = []
        for m in moves:
            ns = state.apply_move(m)
            if ns.is_terminal() and ns.winner == state.current_player:
                idx = NonagaState.tile_move_to_action(m[0], m[1])
                policy[idx] = 1.0
                return m, policy
            next_states.append(ns)

        if self.network is None:
            # Random
            choice = np.random.randint(len(moves))
            for i, m in enumerate(moves):
                policy[NonagaState.tile_move_to_action(m[0], m[1])] = 1.0 / len(moves)
            return moves[choice], policy

        # Batch evaluate
        boards = np.array([ns.encode() for ns in next_states])
        device = next(self.network.parameters()).device
        boards_t = torch.tensor(boards, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, _, values_t = self.network(boards_t)
        values = values_t.squeeze(1).cpu().numpy()

        for i, ns in enumerate(next_states):
            if ns.current_player != state.current_player:
                values[i] = -values[i]

        # Softmax for policy target
        v = values.astype(np.float64)
        v -= v.max()
        exp_v = np.exp(v)
        probs = exp_v / exp_v.sum()

        for i, m in enumerate(moves):
            idx = NonagaState.tile_move_to_action(m[0], m[1])
            policy[idx] = probs[i].astype(np.float32)

        best_idx = np.argmax(values)
        return moves[best_idx], policy
