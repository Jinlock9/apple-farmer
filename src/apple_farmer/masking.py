"""Action enumeration, prefix sums, and vectorized action masks.

A rectangle is encoded by inclusive corners (r1, c1, r2, c2) with
0 <= r1 <= r2 < ROWS and 0 <= c1 <= c2 < COLS.

Number of rectangles:
    C(ROWS+1, 2) * C(COLS+1, 2) = 55 * 153 = 8415.
"""

from __future__ import annotations

import numpy as np

ROWS = 10
COLS = 17
TARGET_SUM = 10
N_ACTIONS = 8415  # C(11,2) * C(18,2) = 55 * 153


def build_action_table() -> np.ndarray:
    """Return shape (N_ACTIONS, 4) int array of (r1, c1, r2, c2) tuples."""
    rs = np.arange(ROWS)
    cs = np.arange(COLS)
    r1, r2 = np.meshgrid(rs, rs, indexing="ij")
    row_pairs = np.stack([r1[r1 <= r2], r2[r1 <= r2]], axis=1)  # (55, 2)
    c1, c2 = np.meshgrid(cs, cs, indexing="ij")
    col_pairs = np.stack([c1[c1 <= c2], c2[c1 <= c2]], axis=1)  # (153, 2)

    n_rp = row_pairs.shape[0]
    n_cp = col_pairs.shape[0]
    rp = np.repeat(row_pairs, n_cp, axis=0)         # (N, 2)
    cp = np.tile(col_pairs, (n_rp, 1))              # (N, 2)
    table = np.empty((n_rp * n_cp, 4), dtype=np.int32)
    table[:, 0] = rp[:, 0]  # r1
    table[:, 1] = cp[:, 0]  # c1
    table[:, 2] = rp[:, 1]  # r2
    table[:, 3] = cp[:, 1]  # c2
    return table


_ACTION_TABLE = build_action_table()
assert _ACTION_TABLE.shape == (N_ACTIONS, 4)

# Pre-split arrays for vectorized indexing into the prefix-sum table.
_R1 = _ACTION_TABLE[:, 0]
_C1 = _ACTION_TABLE[:, 1]
_R2 = _ACTION_TABLE[:, 2]
_C2 = _ACTION_TABLE[:, 3]
_R2P1 = _R2 + 1
_C2P1 = _C2 + 1


def action_to_rect(action: int) -> tuple[int, int, int, int]:
    """Look up (r1, c1, r2, c2) for an action index."""
    r1, c1, r2, c2 = _ACTION_TABLE[action]
    return int(r1), int(c1), int(r2), int(c2)


def compute_prefix_sum(grid: np.ndarray) -> np.ndarray:
    """2D inclusive prefix sum, padded with a zero row/column.

    P has shape (ROWS+1, COLS+1) and P[i+1, j+1] = grid[:i+1, :j+1].sum().
    Use int32 to avoid overflow on int8 inputs.
    """
    g = grid.astype(np.int32, copy=False)
    p = np.zeros((g.shape[0] + 1, g.shape[1] + 1), dtype=np.int32)
    p[1:, 1:] = g.cumsum(axis=0).cumsum(axis=1)
    return p


def compute_action_mask(grid: np.ndarray) -> np.ndarray:
    """Boolean mask of shape (N_ACTIONS,); True iff rectangle sum == TARGET_SUM."""
    p = compute_prefix_sum(grid)
    sums = (
        p[_R2P1, _C2P1]
        - p[_R1, _C2P1]
        - p[_R2P1, _C1]
        + p[_R1, _C1]
    )
    return sums == TARGET_SUM


def apply_action(grid: np.ndarray, action: int) -> int:
    """Zero the rectangle and return the count of non-zero cells cleared.

    Mutates ``grid`` in place. Asserts the rectangle sum was exactly TARGET_SUM.
    """
    r1, c1, r2, c2 = action_to_rect(action)
    block = grid[r1 : r2 + 1, c1 : c2 + 1]
    s = int(block.sum())
    assert s == TARGET_SUM, f"action {action} sum={s} != {TARGET_SUM}"
    cleared = int(np.count_nonzero(block))
    block[...] = 0
    return cleared
