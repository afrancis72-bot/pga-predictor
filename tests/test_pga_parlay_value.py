"""Regression tests for price-aware PGA 3-ball and 4-of-6 optimizer."""
from pathlib import Path
import sys
import unittest
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pga_engine" if (ROOT / "pga_engine").is_dir() else ROOT
sys.path.insert(0, str(ENGINE))

from parlay_value_optimizer import (
    american_decimal, evaluate_prices, optimize_round_prices, _ticket_metrics,
)


def prob_table():
    return pd.DataFrame([
        {"group": i, "player": f"P{i}", "win_pct": 45.,
         "push_pct": 5., "loss_pct": 50., "status": "OK"}
        for i in range(1, 8)
    ])


class PGAParlayPriceOptimization(unittest.TestCase):
    def test_american_odds_and_invalid_price(self):
        self.assertAlmostEqual(american_decimal(150), 2.5)
        self.assertAlmostEqual(american_decimal(-110), 1 + 100 / 110)
        self.assertAlmostEqual(american_decimal(-100), 2.)
        for bad in [0, 90, -90, float("nan")]:
            with self.assertRaises(ValueError):
                american_decimal(bad)

    def test_push_aware_ev_and_stress(self):
        p = pd.DataFrame([{"group": 1, "player": "A", "win_pct": 50.,
                           "push_pct": 10., "loss_pct": 40., "status": "OK"}])
        v = evaluate_prices(p, pd.DataFrame([{"group": 1, "player": "A", "odds": 100}]),
                            round_no=3, win_haircut_pp=4).iloc[0]
        self.assertAlmostEqual(v.raw_roi, 0.10, places=8)
        self.assertAlmostEqual(v.stressed_roi, 0.02, places=8)
        self.assertAlmostEqual(v.break_even_nonpush, .5)

    def test_never_invent_missing_odds_or_match_other_rounds(self):
        p = prob_table()
        self.assertTrue(evaluate_prices(p, pd.DataFrame(columns=["group","player","odds"])).empty)
        with self.assertRaises(ValueError):
            evaluate_prices(p, pd.DataFrame([{"group": 1, "player": "Unknown", "odds": 200}]))
        with self.assertRaises(ValueError):
            evaluate_prices(p, pd.DataFrame([{"group": 1, "player": "P1", "odds": 200,
                                              "round": 2}]), round_no=3)
        with self.assertRaises(ValueError):
            evaluate_prices(p, pd.DataFrame([{"group": 1, "player": "P1", "odds": 200,
                                              "market_type": "2ball"}]))

    def test_exact_six_leg_and_all_fifteen_four_folds(self):
        six = pd.DataFrame([{"group": i, "player": f"P{i}", "decimal_odds": 2.,
                             "p_win": 1., "p_push": 0., "stress_win": 1.}
                            for i in range(1, 7)])
        m, combos = _ticket_metrics(six)
        self.assertEqual(m["round_robin_bets"], 15)
        self.assertEqual(len(combos), 15)
        self.assertAlmostEqual(m["all_six_win_pct"], 100)
        self.assertAlmostEqual(m["at_least_four_win_pct"], 100)
        self.assertAlmostEqual(m["six_leg_roi"], 63.)
        self.assertAlmostEqual(m["round_robin_4of6_roi"], 15.)

    def test_push_reduces_paid_legs_instead_of_auto_loss(self):
        six = pd.DataFrame([{"group": i, "player": f"P{i}", "decimal_odds": 2.5,
                             "p_win": 0., "p_push": 1., "stress_win": 0.}
                            for i in range(1, 7)])
        m, combos = _ticket_metrics(six)
        self.assertAlmostEqual(m["six_leg_roi"], 0.)
        self.assertAlmostEqual(m["round_robin_4of6_roi"], 0.)
        self.assertAlmostEqual(m["at_least_four_win_pct"], 0.)

    def test_no_bet_and_uncalibrated_research_gate(self):
        priced = evaluate_prices(prob_table(),
                pd.DataFrame([{"group": i, "player": f"P{i}", "odds": 400} for i in range(1, 8)]))
        research = optimize_round_prices(priced, ticket_count=2, max_shared_players=5)
        self.assertIn("NO BET RECOMMENDED", research["decision"])
        self.assertGreaterEqual(len(research["tickets"]), 1)
        self.assertTrue(research["tickets"].decision.str.contains("UNCALIBRATED").all())
        self.assertEqual(len(research["four_leg_bets"]), 15 * len(research["tickets"]))
        self.assertEqual(len(research["qualified_legs"].group.unique()), 7)
        failed = optimize_round_prices(priced, min_stressed_leg_roi=1.0)
        self.assertIn("NO BET", failed["decision"])
        self.assertTrue(failed["tickets"].empty)

    def test_six_distinct_groups_required(self):
        priced = evaluate_prices(prob_table(),
                pd.DataFrame([{"group": i, "player": f"P{i}", "odds": 400} for i in range(1, 6)]))
        o = optimize_round_prices(priced)
        self.assertIn("NO BET", o["decision"])
        self.assertTrue(o["tickets"].empty)


if __name__ == "__main__":
    unittest.main()
