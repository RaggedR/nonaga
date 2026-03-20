"""
D6 symmetry group for hexagonal boards.

The hex board has 12 symmetries: 6 rotations × 2 (with/without reflection).
In axial coordinates centered at origin, rotation by 60° is:
  (q, r) → (-r, q + r)

Reflection across the q-axis:
  (q, r) → (q + r, -r)

We apply these relative to the grid center (3, 3).
"""

import numpy as np
from game.hex_grid import GRID_SIZE, qr_to_idx, idx_to_qr, VALID_SET, NUM_DIRS, DIRECTIONS

CENTER_Q, CENTER_R = 3, 3


def _rot60(dq, dr):
    """Rotate (dq, dr) by 60° clockwise in axial coordinates."""
    return -dr, dq + dr


def _reflect(dq, dr):
    """Reflect across q-axis: (q, r) → (q+r, -r)."""
    return dq + dr, -dr


def _apply_transform(q, r, rotation, reflected):
    """
    Apply a D6 transformation to axial coordinates.
    rotation: 0-5 (multiples of 60°)
    reflected: bool
    """
    # Center on origin
    dq, dr = q - CENTER_Q, r - CENTER_R
    # Apply reflection first (if any)
    if reflected:
        dq, dr = _reflect(dq, dr)
    # Apply rotation
    for _ in range(rotation):
        dq, dr = _rot60(dq, dr)
    # Translate back
    return dq + CENTER_Q, dr + CENTER_R


def transform_idx(idx, rotation, reflected):
    """Transform a cell index by a D6 symmetry element."""
    q, r = idx_to_qr(idx)
    nq, nr = _apply_transform(q, r, rotation, reflected)
    if (nq, nr) not in VALID_SET:
        return None
    return qr_to_idx(nq, nr)


def transform_board(board_array, rotation, reflected):
    """
    Transform a 6×7×7 board encoding by a D6 symmetry element.
    Returns a new 6×7×7 array.
    """
    new_board = np.zeros_like(board_array)
    for r in range(GRID_SIZE):
        for q in range(GRID_SIZE):
            if (q, r) not in VALID_SET:
                continue
            nq, nr = _apply_transform(q, r, rotation, reflected)
            if (nq, nr) not in VALID_SET:
                continue
            if 0 <= nq < GRID_SIZE and 0 <= nr < GRID_SIZE:
                # Copy all channels
                new_board[:, nr, nq] = board_array[:, r, q]
    # Channel 3 (ply type) is uniform, already handled
    return new_board


def transform_direction(direction, rotation, reflected):
    """
    Transform a direction index (0-5) by a D6 symmetry element.
    """
    dq, dr = DIRECTIONS[direction]
    if reflected:
        dq, dr = _reflect(dq, dr)
    for _ in range(rotation):
        dq, dr = _rot60(dq, dr)
    # Find which direction this maps to
    for i, (ddq, ddr) in enumerate(DIRECTIONS):
        if (dq, dr) == (ddq, ddr):
            return i
    raise ValueError(f"Transformed direction ({dq},{dr}) not found in DIRECTIONS")


def transform_piece_policy(policy, rotation, reflected):
    """
    Transform a piece-move policy vector.
    policy: array of shape (49*6,) = (PIECE_ACTION_SIZE,)
    """
    from game.nonaga import NonagaState
    new_policy = np.zeros_like(policy)
    for action in range(len(policy)):
        if policy[action] == 0:
            continue
        cell, direction = NonagaState.action_to_piece_move(action)
        q, r = idx_to_qr(cell)
        if (q, r) not in VALID_SET:
            continue
        nq, nr = _apply_transform(q, r, rotation, reflected)
        if (nq, nr) not in VALID_SET:
            continue
        new_dir = transform_direction(direction, rotation, reflected)
        new_cell = qr_to_idx(nq, nr)
        new_action = NonagaState.piece_move_to_action(new_cell, new_dir)
        new_policy[new_action] = policy[action]
    return new_policy


def transform_tile_policy(policy, rotation, reflected):
    """
    Transform a tile-move policy vector.
    policy: array of shape (49*49,) = (TILE_ACTION_SIZE,)
    """
    from game.nonaga import NonagaState
    new_policy = np.zeros_like(policy)
    for action in range(len(policy)):
        if policy[action] == 0:
            continue
        src, dst = NonagaState.action_to_tile_move(action)
        sq, sr = idx_to_qr(src)
        dq, dr = idx_to_qr(dst)
        if (sq, sr) not in VALID_SET or (dq, dr) not in VALID_SET:
            continue
        nsq, nsr = _apply_transform(sq, sr, rotation, reflected)
        ndq, ndr = _apply_transform(dq, dr, rotation, reflected)
        if (nsq, nsr) not in VALID_SET or (ndq, ndr) not in VALID_SET:
            continue
        new_src = qr_to_idx(nsq, nsr)
        new_dst = qr_to_idx(ndq, ndr)
        new_action = NonagaState.tile_move_to_action(new_src, new_dst)
        new_policy[new_action] = policy[action]
    return new_policy


def all_symmetries():
    """Yield all 12 D6 symmetry elements as (rotation, reflected) tuples."""
    for rot in range(6):
        yield rot, False
        yield rot, True


def augment_example(board, piece_policy, tile_policy, value):
    """
    Augment a training example with all 12 D6 symmetries.
    Returns list of 12 (board, piece_policy, tile_policy, value) tuples.
    """
    examples = []
    for rot, ref in all_symmetries():
        new_board = transform_board(board, rot, ref)
        new_pp = transform_piece_policy(piece_policy, rot, ref)
        new_tp = transform_tile_policy(tile_policy, rot, ref)
        examples.append((new_board, new_pp, new_tp, value))
    return examples
