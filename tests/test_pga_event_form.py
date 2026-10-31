"""PGA rolling multi-event SG tests, including weather-disrupted shortened events."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/"pga_engine" if (ROOT/"pga_engine").is_dir() else ROOT
sys.path.insert(0,str(ENGINE))

from reliability_features import build_pre_event_evidence, apply_reliability_research
from pga_event_form import attach_rolling_sg, rolling_sg_evidence

class RollingFormTests(unittest.TestCase):
    def _history(self):
        return pd.DataFrame([
            # Most recent tournament shortened due to abnormal weather. The
            # signal should remain visible but not wash out the two clean weeks.
            {"player":"Leader","date":"2026-10-18","event_id":"Bermuda",
             "sg_ott":-2.,"sg_approach":-2.,"sg_arg":-2.,"sg_putting":-2.,
             "abnormal_conditions":True,"scheduled_rounds":4,"completed_rounds":3},
            {"player":"Leader","date":"2026-10-11","event_id":"Event B",
             "sg_ott":.4,"sg_approach":.3,"sg_arg":.2,"sg_putting":.1,
             "abnormal_conditions":False,"scheduled_rounds":4,"completed_rounds":4},
            {"player":"Leader","date":"2026-10-04","event_id":"Event A",
             "sg_ott":.5,"sg_approach":.7,"sg_arg":.1,"sg_putting":.4,
             "abnormal_conditions":False,"scheduled_rounds":4,"completed_rounds":4},
            {"player":"Leader","date":"2026-11-01","event_id":"future",
             "sg_ott":-10,"sg_approach":-10,"sg_arg":-10,"sg_putting":-10,
             "abnormal_conditions":False,"scheduled_rounds":4,"completed_rounds":4},
            {"player":"One event","date":"2026-10-11","event_id":"Event B",
             "sg_ott":.4,"sg_approach":.3,"sg_arg":.2,"sg_putting":.1},
            {"player":"Incomplete","date":"2026-10-11","event_id":"Event B",
             "sg_ott":.4,"sg_approach":np.nan,"sg_arg":.2,"sg_putting":.1},
            {"player":"Incomplete","date":"2026-10-04","event_id":"Event A",
             "sg_ott":.4,"sg_approach":.3,"sg_arg":.2,"sg_putting":.1},
        ])

    def test_weather_excluded_but_two_earlier_events_count(self):
        r=rolling_sg_evidence(self._history().assign(key=lambda x:x.player.str.lower()),
                              "2026-10-29").set_index("key")
        leader=r.loc["leader"]
        self.assertEqual(leader.recent_disrupted_events,1)
        self.assertEqual(leader.recent_complete_sg_events,2)
        self.assertEqual(leader.recent_rolling_positive_categories,4)
        self.assertTrue(leader.recent_all_four_positive)
        self.assertGreater(leader.recent_min_component_sg,0)
        self.assertEqual(r.loc["one event","recent_complete_sg_events"],1)
        self.assertTrue(pd.isna(r.loc["one event","recent_all_four_positive"]))
        self.assertTrue(pd.isna(r.loc["incomplete","recent_all_four_positive"]))

    def test_forecast_does_not_overreact_to_one_tournament(self):
        history=self._history()
        ps=pd.DataFrame({"player":["Leader","One event","Incomplete"]})
        base=build_pre_event_evidence(ps,event_results=history,tournament_start="2026-10-29")
        evidence=attach_rolling_sg(base,history,"2026-10-29")
        pred=ps.copy()
        pred["expected_round"]=[0.,0.,0.]
        adj=apply_reliability_research(pred,evidence)
        self.assertLess(adj.loc[0,"four_category_strokes_adjustment"],0)
        self.assertEqual(adj.loc[1,"four_category_strokes_adjustment"],0.)
        self.assertEqual(adj.loc[2,"four_category_strokes_adjustment"],0.)
        self.assertLessEqual(adj["four_category_strokes_adjustment"].abs().max(),.0751)

    def test_round_only_rows_cannot_qualify(self):
        h=pd.DataFrame([{"key":"leader","player":"Leader","date":"2026-10-11","event_id":"B",
                         "round":1,"sg_ott":1,"sg_approach":1,"sg_arg":1,"sg_putting":1},
                        {"key":"leader","player":"Leader","date":"2026-10-04","event_id":"A",
                         "round":2,"sg_ott":1,"sg_approach":1,"sg_arg":1,"sg_putting":1}])
        r=rolling_sg_evidence(h,"2026-10-29")
        self.assertTrue(r.empty)

if __name__=="__main__":
    unittest.main()
