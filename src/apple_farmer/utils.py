"""Shared helpers."""

from __future__ import annotations

import numpy as np


def grid_to_ascii(grid: np.ndarray) -> str:
    """Render a 2D int grid as ASCII; '.' for zero, digits otherwise."""
    rows = []
    for r in range(grid.shape[0]):
        chars = []
        for c in range(grid.shape[1]):
            v = int(grid[r, c])
            chars.append("." if v == 0 else str(v))
        rows.append(" ".join(chars))
    return "\n".join(rows)
