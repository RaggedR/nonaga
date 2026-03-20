"""Tests for the Nonaga game engine."""

import sys
import os
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.hex_grid import (
    VALID_CELLS, VALID_SET, VALID_INDICES, NEIGHBORS,
    qr_to_idx, idx_to_qr, slide, neighbors, canonicalize_tiles,
    GRID_SIZE,
)
from game.nonaga import (
    NonagaState, PlyType, Player, INITIAL_TILES, INITIAL_PIECES,
)
from game.symmetry import (
    transform_idx, transform_board, transform_direction,
    all_symmetries, augment_example,
)


# --- Hex Grid Tests ---

def test_valid_cells():
    """37 valid cells in a side-4 hex."""
    assert len(VALID_CELLS) == 37, f"Expected 37 valid cells, got {len(VALID_CELLS)}"
    # Center should be valid
    assert (3, 3) in VALID_SET
    # Corners of grid should not be valid
    assert (0, 0) not in VALID_SET
    assert (6, 6) not in VALID_SET
    print("  valid_cells: OK")


def test_neighbors():
    """Center cell should have 6 neighbors."""
    center = qr_to_idx(3, 3)
    nbrs = neighbors(center)
    assert len(nbrs) == 6, f"Center has {len(nbrs)} neighbors, expected 6"
    # Corner cells should have fewer
    corner = qr_to_idx(0, 3)
    assert len(neighbors(corner)) == 3, f"Corner (0,3) has {len(neighbors(corner))} neighbors"
    print("  neighbors: OK")


def test_slide():
    """Test piece sliding mechanics."""
    tiles = INITIAL_TILES
    center = qr_to_idx(3, 3)
    # Place a single piece at center, slide it east — should go to edge of tiles
    occupied = {center}
    landing = slide(center, 0, tiles, occupied)
    assert landing is not None, "Should be able to slide east from center"
    lq, lr = idx_to_qr(landing)
    assert lr == 3, f"Sliding east shouldn't change row, got r={lr}"
    assert lq > 3, f"Should slide to right, got q={lq}"

    # Slide into a blocker — should stop one cell before
    blocker = qr_to_idx(5, 3)
    occupied2 = {center, blocker}
    landing2 = slide(center, 0, tiles, occupied2)
    assert landing2 is not None, "Should be able to slide east into blocker"
    lq2, lr2 = idx_to_qr(landing2)
    assert (lq2, lr2) == (4, 3), f"Should land at (4,3), got ({lq2},{lr2})"

    # Slide into adjacent piece — can't move
    adjacent = qr_to_idx(4, 3)
    occupied3 = {center, adjacent}
    landing3 = slide(center, 0, tiles, occupied3)
    assert landing3 is None, "Can't slide if immediately blocked"
    print("  slide: OK")


def test_canonicalize():
    """Canonicalization should center tiles."""
    tiles = INITIAL_TILES
    new_tiles, sq, sr = canonicalize_tiles(tiles)
    # Initial tiles are already centered, shift should be 0
    assert sq == 0 and sr == 0, f"Initial tiles already centered, got shift ({sq},{sr})"
    print("  canonicalize: OK")


# --- Game State Tests ---

def test_initial_state():
    """Initial state should have 19 tiles and 6 pieces."""
    state = NonagaState()
    assert len(state.tiles) == 19, f"Expected 19 tiles, got {len(state.tiles)}"
    assert len(state.pieces[Player.ONE]) == 3
    assert len(state.pieces[Player.TWO]) == 3
    assert state.current_player == Player.ONE
    assert state.ply_type == PlyType.PIECE_MOVE
    assert state.winner is None
    # All pieces should be on tiles
    for p in Player:
        for piece in state.pieces[p]:
            assert piece in state.tiles, f"Piece {piece} not on a tile"
    print("  initial_state: OK")


def test_piece_moves():
    """Initial state should have piece moves available."""
    state = NonagaState()
    moves = state.get_piece_moves()
    assert len(moves) > 0, "Should have piece moves at start"
    # Each move is (piece_idx, direction, landing_idx)
    for piece, d, landing in moves:
        assert piece in state.pieces[Player.ONE], "Move should be for current player's piece"
        assert landing in state.tiles, "Landing should be on a tile"
        assert landing not in state.occupied, "Landing should not be occupied"
    print(f"  piece_moves: {len(moves)} moves, OK")


def test_apply_piece_move():
    """Applying a piece move should switch to tile-move ply."""
    state = NonagaState()
    moves = state.get_piece_moves()
    move = moves[0]
    new_state = state.apply_piece_move(*move)
    assert new_state.ply_type == PlyType.TILE_MOVE
    assert new_state.current_player == Player.ONE  # Same player, second ply
    assert move[2] in new_state.pieces[Player.ONE]  # Piece is at landing
    assert move[0] not in new_state.pieces[Player.ONE]  # Piece left origin
    print("  apply_piece_move: OK")


def test_tile_moves():
    """After piece move, should have tile moves."""
    state = NonagaState()
    moves = state.get_piece_moves()
    state2 = state.apply_piece_move(*moves[0])
    tile_moves = state2.get_tile_moves()
    assert len(tile_moves) > 0, "Should have tile moves after piece move"
    for src, dst in tile_moves:
        assert src in state2.tiles, "Source should be a current tile"
        assert dst not in state2.tiles, "Dest should not be a current tile"
        assert src not in state2.occupied, "Source tile should not be occupied"
    print(f"  tile_moves: {len(tile_moves)} moves, OK")


def test_apply_tile_move():
    """Applying a tile move should switch to opponent's turn."""
    state = NonagaState()
    pm = state.get_piece_moves()
    state2 = state.apply_piece_move(*pm[0])
    tm = state2.get_tile_moves()
    state3 = state2.apply_tile_move(*tm[0])
    assert state3.current_player == Player.TWO
    assert state3.ply_type == PlyType.PIECE_MOVE
    assert len(state3.tiles) == 19  # Same number of tiles
    print("  apply_tile_move: OK")


def test_win_detection():
    """Test that mutual adjacency is correctly detected."""
    state = NonagaState()
    # Manually place 3 pieces of P1 in a triangle
    center = qr_to_idx(3, 3)
    nbrs = list(neighbors(center))
    # Find two neighbors that are also adjacent to each other
    for i in range(len(nbrs)):
        for j in range(i + 1, len(nbrs)):
            if nbrs[j] in NEIGHBORS.get(nbrs[i], ()):
                # Found a triangle: center, nbrs[i], nbrs[j]
                state.pieces[Player.ONE] = frozenset([center, nbrs[i], nbrs[j]])
                w = state._check_winner()
                assert w == Player.ONE, f"Should detect P1 win, got {w}"
                print(f"  win_detection: triangle at {center},{nbrs[i]},{nbrs[j]}, OK")
                return
    assert False, "Should have found a triangle"


def test_encode():
    """Test board encoding produces correct shape."""
    state = NonagaState()
    board = state.encode()
    assert board.shape == (6, 7, 7), f"Expected (6,7,7), got {board.shape}"
    # Channel 0: 19 tiles
    assert board[0].sum() == 19, f"Expected 19 tiles in channel 0, got {board[0].sum()}"
    # Channel 1: 3 pieces for current player
    assert board[1].sum() == 3, f"Expected 3 P1 pieces, got {board[1].sum()}"
    # Channel 2: 3 pieces for opponent
    assert board[2].sum() == 3, f"Expected 3 P2 pieces, got {board[2].sum()}"
    # Channel 3: 1s on valid cells for piece-move ply
    assert board[3].sum() == 37, f"Expected 37 for ply marker (valid cells), got {board[3].sum()}"
    print("  encode: OK")


def test_policy_mask():
    """Test policy mask has correct entries."""
    state = NonagaState()
    mask = state.get_policy_mask()
    moves = state.get_piece_moves()
    assert mask.sum() == len(moves), f"Mask sum {mask.sum()} != {len(moves)} moves"
    print("  policy_mask: OK")


# --- Symmetry Tests ---

def test_symmetry_roundtrip():
    """Applying all 6 rotations should return to identity."""
    idx = qr_to_idx(4, 2)
    current = idx
    for _ in range(6):
        current = transform_idx(current, 1, False)
        assert current is not None, "Rotation should keep cell in valid set"
    assert current == idx, "6 rotations of 60° should return to start"
    print("  symmetry_roundtrip: OK")


def test_symmetry_direction():
    """Direction transforms should be consistent."""
    # Rotating direction 0 (East) by 60° should give a different direction
    d = transform_direction(0, 1, False)
    assert 0 <= d < 6 and d != 0, f"Rotated East should not be East, got {d}"
    # 6 rotations should return to original
    current = 0
    for _ in range(6):
        current = transform_direction(current, 1, False)
    assert current == 0, "6 direction rotations should return to start"
    print("  symmetry_direction: OK")


def test_symmetry_board_augment():
    """Board augmentation should produce 12 variants."""
    state = NonagaState()
    board = state.encode()
    pp = state.get_policy_mask()
    # For augmentation test, create a dummy tile policy
    tp = np.zeros(NonagaState.TILE_ACTION_SIZE, dtype=np.float32)
    examples = augment_example(board, pp, tp, 0.5)
    assert len(examples) == 12, f"Expected 12 augmented examples, got {len(examples)}"
    # Identity transform should preserve the board
    assert np.allclose(examples[0][0], board), "Identity should preserve board"
    print("  symmetry_augment: OK")


# --- Random play test ---

def test_random_games(n_games=20):
    """Play random games to verify no crashes."""
    wins = {Player.ONE: 0, Player.TWO: 0}
    total_plies = 0
    max_plies = 500  # Safety limit

    for game_num in range(n_games):
        state = NonagaState()
        plies = 0
        while not state.is_terminal() and plies < max_plies:
            moves = state.get_legal_moves()
            if not moves:
                break  # Stalemate (shouldn't happen in Nonaga normally)
            move = random.choice(moves)
            state = state.apply_move(move)
            plies += 1
            total_plies += 1

        if state.winner is not None:
            wins[state.winner] += 1

    avg_plies = total_plies / n_games
    print(f"  random_games: {n_games} games, P1={wins[Player.ONE]}, P2={wins[Player.TWO]}, "
          f"draws={n_games - wins[Player.ONE] - wins[Player.TWO]}, "
          f"avg_plies={avg_plies:.1f}, OK")


if __name__ == "__main__":
    random.seed(42)
    print("=== Hex Grid Tests ===")
    test_valid_cells()
    test_neighbors()
    test_slide()
    test_canonicalize()

    print("\n=== Game State Tests ===")
    test_initial_state()
    test_piece_moves()
    test_apply_piece_move()
    test_tile_moves()
    test_apply_tile_move()
    test_win_detection()
    test_encode()
    test_policy_mask()

    print("\n=== Symmetry Tests ===")
    test_symmetry_roundtrip()
    test_symmetry_direction()
    test_symmetry_board_augment()

    print("\n=== Random Play ===")
    test_random_games()

    print("\nAll tests passed!")
