"""Regression checks for PGA V11.1 modules. Run: python -m unittest test_v11_1.py"""
import unittest
import numpy as np
import pandas as pd
from venue_history_v11_1 import venue_history_features
from wind_scenarios_v11_1 import shared_weather_shocks
from golf_first_v11 import simulate_golf_first

class TestV111(unittest.TestCase):
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
