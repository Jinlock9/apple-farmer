"""Shared configuration constants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    rows: int = 10
    cols: int = 17
    target_sum: int = 10
    max_steps: int = 100


GAME = GameConfig()
