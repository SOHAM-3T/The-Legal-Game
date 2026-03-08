from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DebateRound:
    round_index: int
    prosecutor_argument: str
    defense_argument: str
    winner: str
    prosecutor_score: float
    defense_score: float
