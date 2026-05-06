"""Tests for masking.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apple_farmer.masking import (  # noqa: E402
    COLS,
    N_ACTIONS,
    ROWS,
    TARGET_SUM,
    action_to_rect,
    apply_action,
    build_action_table,
    compute_action_mask,
    compute_prefix_sum,
)


def brute_force_mask(grid: np.ndarray) -> np.ndarray:
    """O(R^2 * C^2) reference for compute_action_mask."""
    table = build_action_table()
    mask = np.zeros(table.shape[0], dtype=bool)
    for i, (r1, c1, r2, c2) in enumerate(table):
        s = int(grid[r1 : r2 + 1, c1 : c2 + 1].sum())
        mask[i] = s == TARGET_SUM
    return mask


def test_action_table_unique_and_size():
    table = build_action_table()
    assert table.shape == (N_ACTIONS, 4)
    # All within bounds and r1 <= r2, c1 <= c2.
    assert (table[:, 0] >= 0).all() and (table[:, 2] < ROWS).all()
    assert (table[:, 1] >= 0).all() and (table[:, 3] < COLS).all()
    assert (table[:, 0] <= table[:, 2]).all()
    assert (table[:, 1] <= table[:, 3]).all()
    # Uniqueness.
    rows_as_tuples = {tuple(row) for row in table.tolist()}
    assert len(rows_as_tuples) == N_ACTIONS


def test_action_to_rect_round_trip():
    table = build_action_table()
    for idx in [0, 1, 100, 4000, N_ACTIONS - 1]:
        assert action_to_rect(idx) == tuple(int(v) for v in table[idx])


def test_compute_prefix_sum_small_grid():
    g = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.int8,
    )
    p = compute_prefix_sum(g)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 3, 6],
            [0, 5, 12, 21],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(p, expected)
    # Cross-check rectangle sum formula:
    # sum of full grid = 21 = p[2,3] - p[0,3] - p[2,0] + p[0,0]
    assert p[2, 3] - p[0, 3] - p[2, 0] + p[0, 0] == g.sum()


def test_compute_action_mask_matches_brute_force():
    rng = np.random.default_rng(123)
    for trial in range(10):
        grid = rng.integers(1, 10, size=(ROWS, COLS), dtype=np.int8)
        # Sprinkle some zeros to exercise the empty-cell path.
        if trial % 2 == 0:
            zmask = rng.random(grid.shape) < 0.1
            grid[zmask] = 0
        fast = compute_action_mask(grid)
        slow = brute_force_mask(grid)
        np.testing.assert_array_equal(fast, slow)
        assert fast.dtype == np.bool_
        assert fast.shape == (N_ACTIONS,)


def test_apply_action_clears_and_returns_count():
    grid = np.zeros((ROWS, COLS), dtype=np.int8)
    grid[3, 4] = 4
    grid[3, 5] = 6
    mask = compute_action_mask(grid)
    valid = np.flatnonzero(mask)
    # Find the (3,4,3,5) action.
    target = None
    table = build_action_table()
    for idx in valid:
        if tuple(table[idx]) == (3, 4, 3, 5):
            target = int(idx)
            break
    assert target is not None
    cleared = apply_action(grid, target)
    assert cleared == 2
    assert grid[3, 4] == 0
    assert grid[3, 5] == 0


def test_apply_action_counts_only_nonzero_cells():
    grid = np.zeros((ROWS, COLS), dtype=np.int8)
    # Rectangle (0,0)-(0,2): values 10, 0, 0 — sum 10, only one non-zero cell.
    grid[0, 0] = 10  # not a real game value but legal for this assertion
    table = build_action_table()
    for idx, row in enumerate(table.tolist()):
        if tuple(row) == (0, 0, 0, 2):
            target = idx
            break
    cleared = apply_action(grid, target)
    assert cleared == 1


def test_apply_action_rejects_wrong_sum():
    grid = np.zeros((ROWS, COLS), dtype=np.int8)
    grid[0, 0] = 5
    grid[0, 1] = 3
    table = build_action_table()
    for idx, row in enumerate(table.tolist()):
        if tuple(row) == (0, 0, 0, 1):
            target = idx
            break
    with pytest.raises(AssertionError):
        apply_action(grid, target)
