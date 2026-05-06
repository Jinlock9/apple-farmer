"""Minimal Fruit Box environment.

No gymnasium dependency — exposes a small, gym-style API:
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)
    mask = env.action_masks()
    s = env.render()
"""

from __future__ import annotations

import numpy as np

from apple_farmer.masking import (
    COLS,
    N_ACTIONS,
    ROWS,
    apply_action,
    compute_action_mask,
)
from apple_farmer.utils import grid_to_ascii


class FruitBoxEnv:
    ROWS = ROWS
    COLS = COLS
    MAX_STEPS = 100
    N_ACTIONS = N_ACTIONS

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.grid = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self._steps = 0
        self._last_mask: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.grid = self._rng.integers(1, 10, size=(self.ROWS, self.COLS), dtype=np.int8)
        self._steps = 0
        mask = compute_action_mask(self.grid)
        self._last_mask = mask
        info = {"action_mask": mask, "valid_actions": int(mask.sum())}
        return self.grid.copy(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = apply_action(self.grid, action)
        self._steps += 1
        mask = compute_action_mask(self.grid)
        self._last_mask = mask
        terminated = not bool(mask.any())
        truncated = self._steps >= self.MAX_STEPS
        info = {"action_mask": mask, "valid_actions": int(mask.sum())}
        return self.grid.copy(), float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if self._last_mask is None:
            self._last_mask = compute_action_mask(self.grid)
        return self._last_mask

    def render(self) -> str:
        return grid_to_ascii(self.grid)
