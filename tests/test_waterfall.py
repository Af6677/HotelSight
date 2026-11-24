"""The waterfall steps add up exactly to the model's score for the booking."""
import pytest

from hotelsight.waterfall import FINAL, START, STEP, probability, waterfall_steps


def test_steps_chain_from_the_average_to_the_final_score():
    rows = waterfall_steps(-0.005, [("Country", 0.68), ("Deposit", -0.32), ("Other 14 inputs combined", 0.41)])
    assert [r["kind"] for r in rows] == [START, STEP, STEP, STEP, FINAL]
    for before, after in zip(rows[1:-1], rows[2:-1]):
        assert after["start"] == pytest.approx(before["end"])           # each step starts where the last one ended
    assert rows[1]["start"] == pytest.approx(-0.005)                    # the first step starts at the average score
    assert rows[-1]["end"] == pytest.approx(-0.005 + 0.68 - 0.32 + 0.41)


def test_totals_are_measured_from_zero_and_steps_carry_their_own_size():
    rows = waterfall_steps(0.2, [("A", 0.5), ("B", -0.1)])
    assert rows[0]["start"] == 0.0 and rows[0]["end"] == 0.2
    assert rows[-1]["start"] == 0.0 and rows[-1]["end"] == pytest.approx(0.6)
    assert [r["value"] for r in rows[1:-1]] == [0.5, -0.1]
    assert rows[-1]["score"] == pytest.approx(0.6)


def test_no_items_gives_only_the_two_totals():
    rows = waterfall_steps(0.3, [])
    assert [r["kind"] for r in rows] == [START, FINAL] and rows[-1]["end"] == 0.3


def test_probability_of_known_log_odds():
    assert probability(0.0) == 0.5
    assert probability(1.764) == pytest.approx(0.8537, abs=1e-3)
