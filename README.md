# apple-farmer

Reinforcement learning agent for the Fruit Box / Apple Game.

## Game

A 10×17 grid is filled with digits 1–9. The player selects an axis-aligned
rectangle. If the digits inside it sum to exactly 10, those cells become 0
(empty). Cleared cells do not collapse — the grid stays in place. The goal is
to remove as many apples as possible.

## Layout

```
src/apple_farmer/
  env.py       # FruitBoxEnv (custom minimal interface, no gymnasium)
  masking.py   # action table, prefix sum, vectorized action mask
  config.py    # hyperparameters / shared constants
  utils.py     # helpers
scripts/
  play_random.py  # random-policy baseline over 1000 episodes
  play_human.py   # interactive CLI to play by hand
tests/
  test_env.py
  test_masking.py
```

## Setup

```bash
pip install -e ".[dev]"
pytest tests/
python scripts/play_random.py
```

Python 3.11+, NumPy, PyTorch (with MPS support on macOS).
