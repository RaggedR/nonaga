"""
Nonaga game engine with two-ply turn decomposition.

Each full turn consists of:
  Ply 1 (PIECE_MOVE): Move one of your pieces by sliding
  Ply 2 (TILE_MOVE): Relocate an unoccupied edge tile

The game tracks an intermediate state between plies.
"""

from enum import IntEnum
from copy import deepcopy
from game.hex_grid import (
    GRID_SIZE, VALID_INDEX_SET, VALID_SET, VALID_INDICES, DIRECTIONS, NUM_DIRS,
    qr_to_idx, idx_to_qr, neighbors, slide, direction_neighbors,
    canonicalize_tiles, apply_shift, NEIGHBORS,
)


class PlyType(IntEnum):
    PIECE_MOVE = 0
    TILE_MOVE = 1


class Player(IntEnum):
    ONE = 0
    TWO = 1


# Starting positions: 6 alternating corners of the side-3 hex
# The side-3 hex centered at (3,3) has corners at distance 2 from center
# In axial coords: (3±2, 3), (3, 3±2), (3+2, 3-2), (3-2, 3+2)...
# Actually the 6 corners of the initial hex (side 3 = radius 2):
_INITIAL_HEX_CORNERS = [
    (5, 1),  # NE corner
    (5, 3),  # E corner
    (3, 5),  # SE corner
    (1, 5),  # SW corner
    (1, 3),  # W corner
    (3, 1),  # NW corner
]

# Initial 19-tile hex (side 3, radius 2 from center)
INITIAL_TILES = frozenset(
    qr_to_idx(q, r)
    for r in range(GRID_SIZE)
    for q in range(GRID_SIZE)
    if (q, r) in VALID_SET and max(abs(q - 3), abs(r - 3), abs((q - 3) + (r - 3))) <= 2
)

# Players start at alternating corners
INITIAL_PIECES = {
    Player.ONE: frozenset(qr_to_idx(*_INITIAL_HEX_CORNERS[i]) for i in [0, 2, 4]),
    Player.TWO: frozenset(qr_to_idx(*_INITIAL_HEX_CORNERS[i]) for i in [1, 3, 5]),
}


class NonagaState:
    """
    Complete game state for Nonaga.

    Attributes:
        tiles: frozenset of tile indices (19 tiles)
        pieces: dict {Player: frozenset of piece indices} (3 each)
        current_player: Player whose turn it is
        ply_type: PlyType.PIECE_MOVE or PlyType.TILE_MOVE
        last_moved_tile: idx of tile opponent relocated (can't be moved this turn), or None
        piece_moved_from: where the current player's piece came from this turn (for tile restriction)
        winner: Player or None
    """

    __slots__ = [
        'tiles', 'pieces', 'current_player', 'ply_type',
        'last_moved_tile', 'piece_moved_from', 'winner',
        '_piece_move_cache', '_tile_move_cache',
    ]

    def __init__(self):
        self.tiles = INITIAL_TILES
        self.pieces = {
            Player.ONE: INITIAL_PIECES[Player.ONE],
            Player.TWO: INITIAL_PIECES[Player.TWO],
        }
        self.current_player = Player.ONE
        self.ply_type = PlyType.PIECE_MOVE
        self.last_moved_tile = None
        self.piece_moved_from = None
        self.winner = None
        self._piece_move_cache = None
        self._tile_move_cache = None

    def copy(self):
        s = NonagaState.__new__(NonagaState)
        s.tiles = self.tiles
        s.pieces = {p: self.pieces[p] for p in Player}
        s.current_player = self.current_player
        s.ply_type = self.ply_type
        s.last_moved_tile = self.last_moved_tile
        s.piece_moved_from = self.piece_moved_from
        s.winner = self.winner
        s._piece_move_cache = None
        s._tile_move_cache = None
        return s

    @property
    def occupied(self):
        """Set of all cells occupied by any piece."""
        return self.pieces[Player.ONE] | self.pieces[Player.TWO]

    def is_terminal(self):
        return self.winner is not None

    def _check_winner(self):
        """Check if either player has won (3 pieces mutually adjacent)."""
        for player in Player:
            pcs = list(self.pieces[player])
            if len(pcs) != 3:
                continue
            # Three pieces are mutually adjacent if each pair shares a hex edge
            a, b, c = pcs
            nbrs_a = set(NEIGHBORS.get(a, ()))
            nbrs_b = set(NEIGHBORS.get(b, ()))
            if b in nbrs_a and c in nbrs_a and c in nbrs_b:
                return player
        return None

    # --- Edge tiles ---

    def _edge_tiles(self):
        """
        Return set of tile indices that are on the edge of the board.
        An edge tile has at least one neighbor (in the valid hex grid)
        that is NOT a tile.
        """
        edge = set()
        for idx in self.tiles:
            for nbr in NEIGHBORS.get(idx, ()):
                if nbr not in self.tiles:
                    edge.add(idx)
                    break
        return edge

    def _edge_positions(self):
        """
        Return set of valid hex positions adjacent to current tiles
        but not currently tiles themselves. These are positions where
        a tile could be placed.
        """
        positions = set()
        for idx in self.tiles:
            for nbr in NEIGHBORS.get(idx, ()):
                if nbr not in self.tiles:
                    positions.add(nbr)
        return positions

    # --- Connectivity check ---

    def _is_connected_without(self, tile_idx):
        """
        Check if removing tile_idx keeps the remaining tiles connected.
        Uses BFS from an arbitrary remaining tile.
        """
        remaining = self.tiles - {tile_idx}
        if not remaining:
            return False
        start = next(iter(remaining))
        visited = {start}
        queue = [start]
        while queue:
            current = queue.pop()
            for nbr in NEIGHBORS.get(current, ()):
                if nbr in remaining and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return len(visited) == len(remaining)

    def _removable_edge_tiles(self):
        """
        Return set of edge tiles that can be removed without
        disconnecting the board and are not occupied.
        Only tiles with ≤3 tile neighbors (truly on the fringe) can be removed.
        """
        occ = self.occupied
        removable = set()
        for idx in self._edge_tiles():
            if idx in occ:
                continue
            # Don't allow moving the tile the opponent just relocated
            if idx == self.last_moved_tile:
                continue
            # Only allow removing tiles with ≤4 tile neighbors to prevent holes
            tile_nbr_count = sum(1 for nbr in NEIGHBORS.get(idx, ()) if nbr in self.tiles)
            if tile_nbr_count > 4:
                continue
            if self._is_connected_without(idx):
                removable.add(idx)
        return removable

    def _valid_placements(self, removed_tile):
        """
        Return set of valid positions to place a tile, given that
        removed_tile has been taken. The placed tile must be adjacent
        to 2-4 remaining tiles (per rules), and the new board must be connected.
        """
        remaining = self.tiles - {removed_tile}
        # Candidate positions: adjacent to remaining tiles, not already a tile
        candidates = set()
        for idx in remaining:
            for nbr in NEIGHBORS.get(idx, ()):
                if nbr not in remaining:
                    candidates.add(nbr)
        # Don't place back where we removed from (that would be a no-op)
        candidates.discard(removed_tile)
        # Per rules: tile must be placed adjacent to 2-4 existing tiles
        valid = set()
        for pos in candidates:
            adj_count = sum(1 for nbr in NEIGHBORS.get(pos, ()) if nbr in remaining)
            if adj_count < 2 or adj_count >= 5:
                continue  # must touch 2-4 tiles
            new_tiles = remaining | {pos}
            # BFS connectivity check
            start = next(iter(new_tiles))
            visited = {start}
            queue = [start]
            while queue:
                current = queue.pop()
                for nbr in NEIGHBORS.get(current, ()):
                    if nbr in new_tiles and nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            if len(visited) == len(new_tiles):
                valid.add(pos)
        return valid

    # --- Move generation ---

    def get_piece_moves(self):
        """
        Generate all piece moves for current player.
        Returns list of (piece_idx, direction, landing_idx).
        """
        if self._piece_move_cache is not None:
            return self._piece_move_cache
        moves = []
        occ = self.occupied
        for piece in self.pieces[self.current_player]:
            q, r = idx_to_qr(piece)
            for d, (dq, dr) in enumerate(DIRECTIONS):
                landing = slide(piece, d, self.tiles, occ)
                if landing is not None:
                    moves.append((piece, d, landing))
        self._piece_move_cache = moves
        return moves

    def get_tile_moves(self):
        """
        Generate all tile moves.
        Returns list of (source_tile_idx, dest_position_idx).
        """
        if self._tile_move_cache is not None:
            return self._tile_move_cache
        moves = []
        removable = self._removable_edge_tiles()
        for src in removable:
            for dst in self._valid_placements(src):
                moves.append((src, dst))
        self._tile_move_cache = moves
        return moves

    def get_legal_moves(self):
        """Return legal moves for the current ply type."""
        if self.winner is not None:
            return []
        if self.ply_type == PlyType.PIECE_MOVE:
            return self.get_piece_moves()
        else:
            return self.get_tile_moves()

    # --- Apply moves ---

    def apply_piece_move(self, piece_idx, direction, landing_idx):
        """
        Apply a piece move. Returns new state (ply type switches to TILE_MOVE).
        """
        assert self.ply_type == PlyType.PIECE_MOVE
        s = self.copy()
        player = s.current_player

        # Move the piece
        s.pieces[player] = (s.pieces[player] - {piece_idx}) | {landing_idx}
        s.piece_moved_from = piece_idx
        s.ply_type = PlyType.TILE_MOVE

        # Check if this piece move created a win
        w = s._check_winner()
        if w is not None:
            s.winner = w

        s._piece_move_cache = None
        s._tile_move_cache = None
        return s

    def apply_tile_move(self, source_idx, dest_idx):
        """
        Apply a tile move. Returns new state (turn switches to opponent,
        ply type resets to PIECE_MOVE).
        """
        assert self.ply_type == PlyType.TILE_MOVE
        s = self.copy()

        # Move the tile
        s.tiles = (s.tiles - {source_idx}) | {dest_idx}
        s.last_moved_tile = dest_idx
        s.piece_moved_from = None

        # Switch to opponent's turn
        s.current_player = Player(1 - s.current_player)
        s.ply_type = PlyType.PIECE_MOVE

        # Check if tile move created a win (pieces may now be adjacent on new board)
        w = s._check_winner()
        if w is not None:
            s.winner = w

        s._piece_move_cache = None
        s._tile_move_cache = None
        return s

    def apply_move(self, move):
        """
        Apply a move (either ply type). Returns new state.
        """
        if self.ply_type == PlyType.PIECE_MOVE:
            piece_idx, direction, landing_idx = move
            return self.apply_piece_move(piece_idx, direction, landing_idx)
        else:
            source_idx, dest_idx = move
            return self.apply_tile_move(source_idx, dest_idx)

    # --- Encoding for neural network ---

    def encode(self):
        """
        Encode state as 6×7×7 numpy array for neural network input.

        Channels:
          0: Tile present
          1: Current player's pieces
          2: Opponent's pieces
          3: Ply type (all 1s = piece-move, all 0s = tile-move)
          4: Last moved tile (opponent's)
          5: Removable edge tiles

        Board is canonicalized (centered) before encoding.
        """
        import numpy as np

        # Canonicalize
        new_tiles, sq, sr = canonicalize_tiles(self.tiles)
        if sq != 0 or sr != 0:
            # Apply shift to pieces too
            cur_pieces = frozenset(apply_shift(p, sq, sr) for p in self.pieces[self.current_player])
            opp = Player(1 - self.current_player)
            opp_pieces = frozenset(apply_shift(p, sq, sr) for p in self.pieces[opp])
            last_tile = apply_shift(self.last_moved_tile, sq, sr) if self.last_moved_tile is not None else None
        else:
            new_tiles = self.tiles
            cur_pieces = self.pieces[self.current_player]
            opp = Player(1 - self.current_player)
            opp_pieces = self.pieces[opp]
            last_tile = self.last_moved_tile

        board = np.zeros((6, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for idx in new_tiles:
            q, r = idx_to_qr(idx)
            board[0, r, q] = 1.0

        for idx in cur_pieces:
            q, r = idx_to_qr(idx)
            board[1, r, q] = 1.0

        for idx in opp_pieces:
            q, r = idx_to_qr(idx)
            board[2, r, q] = 1.0

        if self.ply_type == PlyType.PIECE_MOVE:
            # Only set on valid cells so symmetry transforms preserve it
            for idx in VALID_INDICES:
                q2, r2 = idx_to_qr(idx)
                board[3, r2, q2] = 1.0

        if last_tile is not None:
            q, r = idx_to_qr(last_tile)
            board[4, r, q] = 1.0

        # Removable edge tiles (useful info for tile-move ply)
        if self.ply_type == PlyType.TILE_MOVE:
            for idx in self._removable_edge_tiles():
                if sq != 0 or sr != 0:
                    shifted = apply_shift(idx, sq, sr)
                else:
                    shifted = idx
                q, r = idx_to_qr(shifted)
                if 0 <= q < GRID_SIZE and 0 <= r < GRID_SIZE:
                    board[5, r, q] = 1.0

        return board

    # --- Action encoding/decoding ---

    @staticmethod
    def piece_move_to_action(piece_idx, direction):
        """Encode piece move as action index: cell * 6 + direction. Max: 37*6=222."""
        # Map piece_idx (0-48) to a cell index within VALID_CELLS
        q, r = idx_to_qr(piece_idx)
        cell = r * GRID_SIZE + q  # Use flat grid index directly
        return cell * NUM_DIRS + direction

    @staticmethod
    def action_to_piece_move(action):
        """Decode piece move action index."""
        cell = action // NUM_DIRS
        direction = action % NUM_DIRS
        # cell is flat grid index
        return cell, direction

    @staticmethod
    def tile_move_to_action(source_idx, dest_idx):
        """Encode tile move as action index: source * 49 + dest. Max: 49*49=2401 but we use 37*37=1369."""
        return source_idx * (GRID_SIZE * GRID_SIZE) + dest_idx

    @staticmethod
    def action_to_tile_move(action):
        """Decode tile move action index."""
        n = GRID_SIZE * GRID_SIZE
        source = action // n
        dest = action % n
        return source, dest

    # For NN output sizes
    PIECE_ACTION_SIZE = GRID_SIZE * GRID_SIZE * NUM_DIRS  # 49 * 6 = 294
    TILE_ACTION_SIZE = (GRID_SIZE * GRID_SIZE) ** 2  # 49 * 49 = 2401

    def get_policy_mask(self):
        """
        Return a binary mask over the action space for the current ply type.
        """
        import numpy as np

        if self.ply_type == PlyType.PIECE_MOVE:
            mask = np.zeros(self.PIECE_ACTION_SIZE, dtype=np.float32)
            for piece_idx, direction, landing_idx in self.get_piece_moves():
                action = self.piece_move_to_action(piece_idx, direction)
                mask[action] = 1.0
        else:
            mask = np.zeros(self.TILE_ACTION_SIZE, dtype=np.float32)
            for source_idx, dest_idx in self.get_tile_moves():
                action = self.tile_move_to_action(source_idx, dest_idx)
                mask[action] = 1.0
        return mask

    def __repr__(self):
        return (f"NonagaState(player={self.current_player.name}, "
                f"ply={self.ply_type.name}, "
                f"tiles={len(self.tiles)}, "
                f"winner={self.winner})")
