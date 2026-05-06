"""Random-policy baseline over 1000 episodes."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apple_farmer.env import FruitBoxEnv  # noqa: E402
from apple_farmer.masking import compute_action_mask  # noqa: E402


def run_episode(env: FruitBoxEnv, rng: np.random.Generator) -> tuple[int, int]:
    _, info = env.reset()
    total_reward = 0
    length = 0
    mask = info["action_mask"]
    while True:
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            break
        action = int(rng.choice(valid))
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += int(reward)
        length += 1
        mask = info["action_mask"]
        if terminated or truncated:
            break
    return total_reward, length


def benchmark_mask(n: int = 1000) -> float:
    rng = np.random.default_rng(0)
    grids = [rng.integers(1, 10, size=(10, 17), dtype=np.int8) for _ in range(n)]
    t0 = time.perf_counter()
    for g in grids:
        compute_action_mask(g)
    dt = time.perf_counter() - t0
    return dt / n


def main() -> None:
    n_episodes = 1000
    seed = 42
    env = FruitBoxEnv(seed=seed)
    rng = np.random.default_rng(seed + 1)

    rewards: list[int] = []
    lengths: list[int] = []

    t0 = time.perf_counter()
    for _ in range(n_episodes):
        r, L = run_episode(env, rng)
        rewards.append(r)
        lengths.append(L)
    elapsed = time.perf_counter() - t0

    rewards_arr = np.asarray(rewards)
    lengths_arr = np.asarray(lengths)

    print(f"random baseline over {n_episodes} episodes ({elapsed:.2f}s total)")
    print(f"reward  mean={rewards_arr.mean():.2f}  std={rewards_arr.std():.2f}  "
          f"min={rewards_arr.min()}  max={rewards_arr.max()}  "
          f"median={statistics.median(rewards)}")
    print(f"length  mean={lengths_arr.mean():.2f}  std={lengths_arr.std():.2f}  "
          f"min={lengths_arr.min()}  max={lengths_arr.max()}  "
          f"median={statistics.median(lengths)}")

    avg_us = benchmark_mask() * 1e6
    print(f"compute_action_mask: {avg_us:.1f} us / call (avg over 1000 random grids)")


if __name__ == "__main__":
    main()
