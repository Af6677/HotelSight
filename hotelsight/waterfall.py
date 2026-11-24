"""The steps of a SHAP waterfall for one booking: from the model's average score to this booking's score.

Scores are log-odds. Each step starts where the previous one ended, so the last step ends exactly at
base + the sum of all contributions, which equals the model's output for the booking.
"""
from __future__ import annotations

import math

START, STEP, FINAL = "start", "step", "final"


def probability(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-log_odds))


def waterfall_steps(base: float, items: list[tuple[str, float]], start_label: str = "Average prediction",
                    final_label: str = "This booking") -> list[dict]:
    """items = [(label, contribution)] in the order they should be drawn (largest first).

    Returns one row per bar: kind, label, start, end, value (the contribution, or the score for the two totals)."""
    rows = [{"kind": START, "label": start_label, "start": 0.0, "end": base, "value": base, "score": base}]
    running = base
    for label, value in items:
        rows.append({"kind": STEP, "label": label, "start": running, "end": running + value, "value": value,
                     "score": running + value})
        running += value
    rows.append({"kind": FINAL, "label": final_label, "start": 0.0, "end": running, "value": running, "score": running})
    return rows
