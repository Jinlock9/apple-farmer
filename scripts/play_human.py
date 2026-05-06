"""Interactive CLI to play one game by hand."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apple_farmer.env import FruitBoxEnv  # noqa: E402
from apple_farmer.masking import _ACTION_TABLE  # noqa: E402


def find_action(r1: int, c1: int, r2: int, c2: int) -> int | None:
    matches = np.where(
        (_ACTION_TABLE[:, 0] == r1)
        & (_ACTION_TABLE[:, 1] == c1)
        & (_ACTION_TABLE[:, 2] == r2)
        & (_ACTION_TABLE[:, 3] == c2)
    )[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def main() -> None:
    env = FruitBoxEnv(seed=None)
    _, info = env.reset()
    score = 0
    print("Enter rectangles as 'r1 c1 r2 c2' (inclusive). Type 'q' to quit.\n")
    while True:
        print(env.render())
        print(f"score={score}  valid_actions={info['valid_actions']}  steps={env._steps}")
        if info["valid_actions"] == 0:
            print("No valid actions remain. Game over.")
            return
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.lower() in {"q", "quit", "exit"}:
            return
        parts = line.split()
        if len(parts) != 4:
            print("Need 4 integers: r1 c1 r2 c2")
            continue
        try:
            r1, c1, r2, c2 = (int(x) for x in parts)
        except ValueError:
            print("Could not parse integers.")
            continue
        if not (0 <= r1 <= r2 < env.ROWS and 0 <= c1 <= c2 < env.COLS):
            print(f"Out of bounds (rows 0..{env.ROWS - 1}, cols 0..{env.COLS - 1}).")
            continue
        action = find_action(r1, c1, r2, c2)
        if action is None:
            print("No matching action index (shouldn't happen).")
            continue
        if not env.action_masks()[action]:
            block_sum = int(env.grid[r1 : r2 + 1, c1 : c2 + 1].sum())
            print(f"Invalid: rectangle sum is {block_sum}, need 10.")
            continue
        _, reward, terminated, truncated, info = env.step(action)
        score += int(reward)
        if terminated:
            print(env.render())
            print(f"score={score}  game over (no valid actions).")
            return
        if truncated:
            print(env.render())
            print(f"score={score}  truncated at MAX_STEPS={env.MAX_STEPS}.")
            return


if __name__ == "__main__":
    main()
