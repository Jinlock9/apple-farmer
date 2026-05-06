"""apple-farmer: RL agent for the Fruit Box / Apple Game."""

from apple_farmer.env import FruitBoxEnv
from apple_farmer.masking import (
    COLS,
    N_ACTIONS,
    ROWS,
    action_to_rect,
    apply_action,
    build_action_table,
    compute_action_mask,
    compute_prefix_sum,
)

__all__ = [
    "FruitBoxEnv",
    "ROWS",
    "COLS",
    "N_ACTIONS",
    "action_to_rect",
    "apply_action",
    "build_action_table",
    "compute_action_mask",
    "compute_prefix_sum",
]
