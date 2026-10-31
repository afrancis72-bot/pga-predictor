"""PGA regression tests: score/cut identity, dispersion, and joint six-golfer DFS."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pga_engine" if (ROOT / "pga_engine").is_dir() else ROOT
sys.path.insert(0, str(ENGINE))

from golf_first_v11 import simulate_golf_first
from pga_joint_portfolio import joint_candidate_metrics
from portfolio_optimizer import PortfolioSettings, optimize_portfolio


class PGATournamentIntegrity(unittest.TestCase):
    def test_score_and_cut_matrix_reconcile_to_each_golfer(self):
        golfers = [f"Golfer {i}" for i in range(8)]
        p = pd.DataFrame({
            "player": golfers, "salary": [8000]*8,
            "expected_round": [-0.5, 0, 0.2, -0.8, 0.3, 0.1, -0.2, 0.4],
            "round_sd": [2.0, 2.0, 2.0, 2.0, 3.7, 3.7, 3.7, 3.7],
        })
        # 4 par-3 + 11 par-4 + 3 par-5, an explicit par-71 routing.
        hole_pars = [3]*4 + [4]*11 + [5]*3
        result, draws, diag, cuts = simulate_golf_first(
            p, sims=400, seed=144, course_par=71, cut_size=4,
            no_cut=False, hole_pars=hole_pars, return_cut_draws=True)
        self.assertEqual(draws.shape, (400, 8))
        self.assertEqual(cuts.shape, (400, 8))
        self.assertEqual(result.player.tolist(), golfers)
        np.testing.assert_allclose(result.v11_mean_dk, draws.mean(axis=0), atol=0.001)
        np.testing.assert_allclose(result.v11_p90_dk, np.percentile(draws, 90, axis=0), atol=0.001)
        np.testing.assert_allclose(result.v11_make_cut_pct, cuts.mean(axis=0)*100, atol=0.001)
        self.assertEqual(diag["hole_pars_source"], "provided course hole pars")
        self.assertTrue(diag["golfer_sd_used"])
        self.assertTrue(result["v11_top5_pct"].between(0, 100).all())
        self.assertTrue(result["v11_top20_pct"].between(0, 100).all())
        self.assertTrue(np.isfinite(result["v11_expected_finish"]).all())

    def test_scenario_reordering_does_not_change_lineup_metrics(self):
        golfers = ["A", "B", "C", "D", "E", "F", "G", "H"]
        rng = np.random.default_rng(144)
        draws = rng.normal(60, 15, size=(400, 8))
        cuts = rng.random((400, 8)) < np.arange(0.45, 0.85, 0.05)[None, :]
        lineup = frozenset(golfers[:6])
        prior = joint_candidate_metrics([lineup], golfers, draws, cuts, "Cut Equity")[lineup]
        order = [7, 2, 4, 1, 5, 0, 6, 3]
        reshuffled = joint_candidate_metrics([lineup], [golfers[i] for i in order],
                                             draws[:, order], cuts[:, order],
                                             "Cut Equity")[lineup]
        for key in prior:
            self.assertAlmostEqual(prior[key], reshuffled[key], places=9)
        with self.assertRaises(ValueError):
            joint_candidate_metrics([lineup], golfers, draws[:, :-1], cuts, "Cut Equity")

    def test_optimizer_uses_joint_cut_probability_and_preserves_constraints(self):
        names = [f"Golfer {i}" for i in range(9)]
        predictions = pd.DataFrame({
            "player": names, "salary": [8000,7900,7800,7700,7600,7500,7400,7300,7200],
            "make_cut_pct": [.8,.76,.72,.70,.68,.65,.60,.56,.52],
            "top10_pct": [.25,.23,.21,.20,.17,.15,.14,.13,.11],
            "win_pct": [.05,.045,.04,.035,.03,.025,.020,.015,.01],
            "dk_points_proxy": [80,79,78,77,76,75,74,73,72],
            "v11_mean_dk": [80,79,78,77,76,75,74,73,72],
            "v11_p75_dk": [105,104,103,102,101,100,99,98,97],
            "v11_p90_dk": [120,119,118,117,116,115,114,113,112],
        })
        rng = np.random.default_rng(47)
        draws = rng.uniform(30, 130, size=(300, 9)).astype("float32")
        cuts = rng.random((300, 9)) < np.array(predictions.make_cut_pct)[None, :]
        settings = PortfolioSettings(lineup_count=2, salary_floor=40000,
                                     min_player_salary=6500, max_exposure=1.0,
                                     min_unique=1, seed=144, target_candidates=35,
                                     candidate_samples=900)
        portfolio, summary, exposure = optimize_portfolio(
            predictions, settings, scenario_scores=draws,
            scenario_cuts=cuts, scenario_players=names)
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(portfolio), 12)
        self.assertTrue(summary["salary"].between(settings.salary_floor, 50000).all())
        self.assertTrue(summary["simulated_six_of_six_pct"].between(0, 100).all())
        self.assertIn("simulated_lineup_p99", summary.columns)
        for row in summary.itertuples():
            selected = portfolio.loc[portfolio.lineup == row.lineup, "player"].tolist()
            ix = [names.index(n) for n in selected]
            self.assertEqual(len(set(selected)), 6)
            self.assertAlmostEqual(row.simulated_six_of_six_pct,
                                   round(100*cuts[:, ix].all(axis=1).mean(), 2), places=2)
            self.assertAlmostEqual(row.simulated_lineup_mean,
                                   round(float(draws[:, ix].sum(axis=1).mean()), 2), places=2)
        with self.assertRaises(ValueError):
            optimize_portfolio(predictions, settings)


if __name__ == "__main__":
    unittest.main()
