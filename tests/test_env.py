"""Tests for env.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apple_farmer.env import FruitBoxEnv  # noqa: E402
from apple_farmer.masking import (  # noqa: E402
    N_ACTIONS,
    build_action_table,
    compute_action_mask,
)


def _action_for(r1: int, c1: int, r2: int, c2: int) -> int:
    table = build_action_table()
    matches = np.where(
        (table[:, 0] == r1)
        & (table[:, 1] == c1)
        & (table[:, 2] == r2)
        & (table[:, 3] == c2)
    )[0]
    assert matches.size == 1
    return int(matches[0])


def test_reset_returns_valid_grid():
    env = FruitBoxEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (env.ROWS, env.COLS)
    assert obs.dtype == np.int8
    assert (obs >= 1).all() and (obs <= 9).all()
    assert info["action_mask"].shape == (N_ACTIONS,)
    assert info["action_mask"].dtype == np.bool_
    assert info["valid_actions"] == int(info["action_mask"].sum())


def test_reset_seed_is_reproducible():
    env1 = FruitBoxEnv(seed=7)
    env2 = FruitBoxEnv(seed=7)
    o1, _ = env1.reset()
    o2, _ = env2.reset()
    np.testing.assert_array_equal(o1, o2)


def test_step_clears_known_rectangle():
    env = FruitBoxEnv()
    env.reset()
    # Force a known board state.
    env.grid[:] = 1  # all 1s; many sum-to-10 rectangles are 1x10 or 2x5 etc.
    env._steps = 0
    env._last_mask = compute_action_mask(env.grid)

    action = _action_for(0, 0, 0, 9)  # ten 1s in a row.
    obs, reward, terminated, truncated, info = env.step(action)

    assert reward == 10.0
    assert obs[0, :10].sum() == 0
    assert (obs[0, 10:] == 1).all()
    assert (obs[1:, :] == 1).all()
    assert not terminated
    assert not truncated
    assert info["valid_actions"] == int(info["action_mask"].sum())


def test_termination_when_no_valid_action():
    env = FruitBoxEnv()
    env.reset()
    # Fill with 9s (every rectangle sum is a multiple of 9, never 10).
    env.grid[:] = 9
    env._steps = 0
    env._last_mask = compute_action_mask(env.grid)
    assert not env._last_mask.any()

    # Manually take a step path: there are no valid actions, so we just check
    # via env.action_masks() and step semantics on the next state. We craft a
    # trivial reachable terminal: clear a single rectangle that leaves a
    # 9-only board.
    env.grid[0, 0] = 1
    env.grid[0, 1] = 9  # rectangle (0,0)-(0,1) sums to 10? 1+9=10. Yes.
    env._last_mask = compute_action_mask(env.grid)
    action = _action_for(0, 0, 0, 1)
    _, reward, terminated, truncated, info = env.step(action)
    assert reward == 2.0
    assert terminated
    assert not truncated
    assert info["valid_actions"] == 0


def test_truncation_at_max_steps():
    env = FruitBoxEnv()
    env.reset()
    env.grid[:] = 1
    env._steps = 0
    env._last_mask = compute_action_mask(env.grid)

    # Repeatedly clear ten 1s. After the grid is fully zero we'd terminate, so
    # we manually re-fill row 0 each step to keep going up to MAX_STEPS.
    action = _action_for(0, 0, 0, 9)
    for i in range(env.MAX_STEPS - 1):
        _, _, terminated, truncated, _ = env.step(action)
        assert not truncated
        # Reset the relevant cells back to 1 to keep the action valid.
        env.grid[0, :10] = 1
        env._last_mask = compute_action_mask(env.grid)
        if terminated:
            # Shouldn't happen since we keep refilling, but break defensively.
            break
    # One more step should trigger truncation.
    _, _, _, truncated, _ = env.step(action)
    assert truncated


def test_action_masks_shape_and_dtype():
    env = FruitBoxEnv(seed=1)
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (N_ACTIONS,)
    assert mask.dtype == np.bool_


def test_render_returns_string_with_dots_for_zeros():
    env = FruitBoxEnv()
    env.reset()
    env.grid[:] = 0
    env.grid[0, 0] = 5
    s = env.render()
    assert isinstance(s, str)
    lines = s.split("\n")
    assert len(lines) == env.ROWS
    assert lines[0].startswith("5")
    # Most cells should be '.'.
    assert s.count(".") >= env.ROWS * env.COLS - 1
