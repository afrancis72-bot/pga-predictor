"""Regression checks for PGA V11.1 modules. Run: python -m unittest test_v11_1.py"""
import unittest
import numpy as np
import pandas as pd
from venue_history_v11_1 import venue_history_features
from wind_scenarios_v11_1 import shared_weather_shocks
from golf_first_v11 import simulate_golf_first
from lineup_simulation_v11_1 import evaluate_lineup_candidates, validate_eight_portfolio, select_eight_from_candidates, generate_candidate_bank

class TestV111(unittest.TestCase):
    def test_candidate_bank_salary_and_locks(self):
        p=pd.DataFrame({"player":[f"P{i}" for i in range(20)],
                        "salary":[8000]*20,"v11_mean_dk":np.arange(20)+60,
                        "v11_p90_dk":np.arange(20)+90})
        bank=generate_candidate_bank(p,locks=["P0"],target=30,draws=9000,seed=3)
        self.assertGreaterEqual(bank.lineup.nunique(),8)
        self.assertTrue(all("P0" in set(g.player) for _,g in bank.groupby("lineup")))
        self.assertTrue(all(len(g)==6 for _,g in bank.groupby("lineup")))
    def test_lineup_joint_scores(self):
        p=pd.DataFrame({"player":["A","B","C"]})
        x=np.array([[10.,20.,30.],[30.,10.,20.],[5.,5.,5.]])
        lu=pd.DataFrame({"lineup":[1,1,2,2],"player":["A","B","B","C"]})
        ev=evaluate_lineup_candidates(p,x,lu)
        self.assertAlmostEqual(float(ev.iloc[0].mean_dk),80/3)
        self.assertAlmostEqual(float(ev.iloc[1].mean_dk),90/3)
    def test_portfolio_rejects_excess_exposure(self):
        p=pd.DataFrame({"player":[f"P{i}" for i in range(48)],"salary":[8000]*48})
        lu=pd.DataFrame({"lineup":np.repeat(np.arange(8),6),
                         "player":[f"P{i}" for i in range(48)]})
        self.assertTrue(validate_eight_portfolio(lu,p,max_exposure=.35)["valid"])
        lu.loc[6,"player"]="P0"
        lu.loc[12,"player"]="P0"
        with self.assertRaises(ValueError):
            validate_eight_portfolio(lu,p,max_exposure=.35,min_unique=3)
    def test_candidate_selector_distinct(self):
        p=pd.DataFrame({"player":[f"P{i}" for i in range(48)],"salary":[8000]*48})
        lu=pd.DataFrame({"lineup":np.repeat(np.arange(8),6),
                         "player":[f"P{i}" for i in range(48)]})
        rng=np.random.default_rng(12)
        x=rng.normal(65,10,size=(100,48))
        chosen,report=select_eight_from_candidates(p,x,lu,max_exposure=.35)
        self.assertEqual(len(chosen),48)
        self.assertEqual((report.portfolio_role=="balanced").sum(),6)
    def test_history_shrinkage_and_no_future_leakage(self):
        players=pd.DataFrame({"player":["A","B"]})
        h=pd.DataFrame({"player":["A","A"],"course":["Port Royal","Port Royal"],
                        "year":[2025,2026],"strokes_gained_per_round":[2.,100.],"rounds":[4,4]})
        d=venue_history_features(players,h,"Port Royal",{},2026)
        self.assertAlmostEqual(float(d.loc[0,"venue_fit"]),0.5)
        self.assertEqual(float(d.loc[1,"venue_fit"]),0.0)
    def test_shared_wave_shocks(self):
        w=shared_weather_shocks(100,["AM","AM","PM"],[10,20,25,15])
        self.assertEqual(w.shape,(100,3,4))
        np.testing.assert_array_equal(w[:,0,:],w[:,1,:])
        self.assertTrue(np.isfinite(w).all())
    def test_weather_shape_guard(self):
        d=pd.DataFrame({"player":["A","B"],"salary":[8000,9000],
                        "expected_round":[0.,0.],"round_sd":[2.7,2.7]})
        with self.assertRaises(ValueError):
            simulate_golf_first(d,sims=100,weather_shifts=np.zeros((100,3,4)))
    def test_weather_changes_scoring(self):
        d=pd.DataFrame({"player":["A","B"],"salary":[8000,9000],
                        "expected_round":[0.,0.],"round_sd":[2.7,2.7]})
        base,_,_=simulate_golf_first(d,sims=100,seed=7)
        wind,_,diag=simulate_golf_first(d,sims=100,seed=7,
                              weather_shifts=np.full((100,2,4),0.6))
        self.assertTrue(diag["weather_shifts_applied"])
        self.assertGreater(float(wind.v11_avg_strokes.mean()),float(base.v11_avg_strokes.mean()))

if __name__=="__main__":
    unittest.main()
