"""
Axial hex coordinate system on a 7×7 grid.

Axial coordinates (q, r) where the six hex directions are:
  E=(+1,0), W=(-1,0), NE=(+1,-1), SW=(-1,+1), NW=(0,-1), SE=(0,+1)

The 7×7 grid has 49 cells, but only 37 form a valid hexagon of side 4
(the largest hex that fits). The 19-tile Nonaga board is a subset of
these 37 positions that moves during play.

We index cells as idx = r * 7 + q, so a flat array of length 49 can
represent the board. Only indices in VALID_CELLS are usable hex positions.
"""

import numpy as np

GRID_SIZE = 7
NUM_CELLS = GRID_SIZE * GRID_SIZE  # 49
CENTER = (3, 3)  # center of the 7×7 grid in axial coords

# Six hex directions in axial coordinates: E, NE, NW, W, SW, SE
DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
NUM_DIRS = 6

# --- Precompute valid hex cells ---
# A hex of side S centered at (3,3) contains cells where |q-3|+|r-3|+|q+r-6| <= 2*(S-1)
# Side 4 gives all 37 cells within the hex grid.

def _hex_dist_from_center(q, r):
    """Cube distance from center (3,3) in axial coords."""
    cq, cr = CENTER
    dq = q - cq
    dr = r - cr
    return max(abs(dq), abs(dr), abs(dq + dr))


# All 37 valid hex positions (side-4 hexagon)
VALID_CELLS = []
VALID_SET = set()
for _r in range(GRID_SIZE):
    for _q in range(GRID_SIZE):
        if _hex_dist_from_center(_q, _r) <= 3:
            VALID_CELLS.append((_q, _r))
            VALID_SET.add((_q, _r))
VALID_CELLS = tuple(VALID_CELLS)

# Mapping between (q,r) and flat index
def qr_to_idx(q, r):
    return r * GRID_SIZE + q

def idx_to_qr(idx):
    return idx % GRID_SIZE, idx // GRID_SIZE

# Precompute valid cell indices
VALID_INDICES = tuple(qr_to_idx(q, r) for q, r in VALID_CELLS)
VALID_INDEX_SET = frozenset(VALID_INDICES)

# Precompute neighbor lookup: neighbors[idx] = list of neighbor indices
NEIGHBORS = {}
for q, r in VALID_CELLS:
    idx = qr_to_idx(q, r)
    nbrs = []
    for dq, dr in DIRECTIONS:
        nq, nr = q + dq, r + dr
        if (nq, nr) in VALID_SET:
            nbrs.append(qr_to_idx(nq, nr))
    NEIGHBORS[idx] = tuple(nbrs)


def neighbors(idx):
    """Return tuple of neighbor indices for a valid hex cell."""
    return NEIGHBORS.get(idx, ())


def direction_neighbors(idx):
    """Return list of (direction_index, neighbor_idx) for a valid hex cell."""
    q, r = idx_to_qr(idx)
    result = []
    for d, (dq, dr) in enumerate(DIRECTIONS):
        nq, nr = q + dq, r + dr
        if (nq, nr) in VALID_SET:
            result.append((d, qr_to_idx(nq, nr)))
    return result


def ray(idx, direction, tile_set):
    """
    Cast a ray from idx in the given direction (0-5).
    Returns list of cells along the ray that are in tile_set,
    stopping at the first cell NOT in tile_set.
    """
    q, r = idx_to_qr(idx)
    dq, dr = DIRECTIONS[direction]
    cells = []
    while True:
        q, r = q + dq, r + dr
        cell = qr_to_idx(q, r)
        if (q, r) not in VALID_SET or cell not in tile_set:
            break
        cells.append(cell)
    return cells


def slide(idx, direction, tile_set, occupied_set):
    """
    Slide a piece from idx in direction until it hits another piece or
    runs off the tiles. Returns the landing cell, or None if can't move.

    The piece slides over tiles (must be in tile_set) and stops when:
    - Next cell is occupied (stop at current cell)
    - Next cell is not a tile (stop at current cell)
    The piece must move at least one cell.
    """
    q, r = idx_to_qr(idx)
    dq, dr = DIRECTIONS[direction]
    landing = None
    cq, cr = q, r
    while True:
        nq, nr = cq + dq, cr + dr
        if (nq, nr) not in VALID_SET:
            break
        ncell = qr_to_idx(nq, nr)
        if ncell not in tile_set:
            break
        if ncell in occupied_set:
            break
        cq, cr = nq, nr
        landing = ncell
    return landing


def compute_centroid(tile_set):
    """Compute the centroid of tile positions in axial coordinates."""
    if not tile_set:
        return CENTER
    sum_q, sum_r = 0, 0
    for idx in tile_set:
        q, r = idx_to_qr(idx)
        sum_q += q
        sum_r += r
    n = len(tile_set)
    return sum_q / n, sum_r / n


def canonicalize_tiles(tile_set):
    """
    Shift all tiles so their centroid is as close as possible to (3,3).
    Returns (new_tile_set, shift_q, shift_r) where shift is the integer
    offset applied.

    This keeps the board centered in the 7×7 grid for NN input.
    """
    cq, cr = compute_centroid(tile_set)
    # Round to nearest integer shift
    shift_q = round(CENTER[0] - cq)
    shift_r = round(CENTER[1] - cr)

    if shift_q == 0 and shift_r == 0:
        return tile_set, 0, 0

    new_tiles = set()
    for idx in tile_set:
        q, r = idx_to_qr(idx)
        nq, nr = q + shift_q, r + shift_r
        if (nq, nr) in VALID_SET:
            new_tiles.add(qr_to_idx(nq, nr))
        else:
            # Shift would push tiles off grid — don't shift
            return tile_set, 0, 0

    return frozenset(new_tiles), shift_q, shift_r


def apply_shift(idx, shift_q, shift_r):
    """Apply a coordinate shift to an index."""
    q, r = idx_to_qr(idx)
    return qr_to_idx(q + shift_q, r + shift_r)


def board_to_grid(tile_set, piece_positions_1, piece_positions_2):
    """
    Convert game state to a 7×7 grid representation.
    Returns a dict mapping idx -> content for visualization/debugging.
    """
    grid = {}
    for idx in tile_set:
        if idx in piece_positions_1:
            grid[idx] = 'P1'
        elif idx in piece_positions_2:
            grid[idx] = 'P2'
        else:
            grid[idx] = '.'
    return grid


def print_hex_board(tile_set, piece_positions_1, piece_positions_2):
    """Print a hex board to console for debugging."""
    grid = board_to_grid(tile_set, piece_positions_1, piece_positions_2)
    for r in range(GRID_SIZE):
        indent = abs(r - 3)
        cells = []
        for q in range(GRID_SIZE):
            idx = qr_to_idx(q, r)
            if idx in grid:
                cells.append(f" {grid[idx]:>2}")
            elif (q, r) in VALID_SET:
                cells.append("  _")
            else:
                cells.append("   ")
        print(" " * indent + "".join(cells))
