"""PGA reliability evidence tests: no lookahead, missingness, SG breadth and score inputs."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pga_engine" if (ROOT / "pga_engine").is_dir() else ROOT
sys.path.insert(0, str(ENGINE))

from reliability_features import build_pre_event_evidence, apply_reliability_research, snapshot_last_events

class ReliabilityResearchTests(unittest.TestCase):
    def test_keefer_pre_event_snapshot_not_future_cut(self):
        players=pd.DataFrame({"player":["Johnny Keefer","Player B"]})
        snapshots=pd.DataFrame([
            {"player":"Johnny Keefer","as_of":"2026-10-14","events_played":26,"cuts_made":18},
            {"player":"Johnny Keefer","as_of":"2026-10-31","events_played":27,"cuts_made":18},
            {"player":"Player B","as_of":"2026-11-02","events_played":20,"cuts_made":20}
        ])
        e=build_pre_event_evidence(players,snapshots,tournament_start="2026-10-29")
        k=e.set_index("player").loc["Johnny Keefer"]
        self.assertEqual(k.season_starts,26)
        self.assertEqual(k.season_cuts_made,18)
        self.assertAlmostEqual(k.historical_cut_pct,100*18/26,places=8)
        self.assertTrue(pd.isna(e.set_index("player").loc["Player B","season_starts"]))

    def test_all_four_sg_only_if_previous_event_complete_and_before_lock(self):
        players=pd.DataFrame({"player":["A","B","C"]})
        logs=pd.DataFrame([
            {"player":"A","date":"2026-10-20","event_id":"prior",
             "SG_off_the_tee":.1,"SG_approach":.2,"SG_around_green":.3,"SG_putting":.4},
            {"player":"A","date":"2026-11-01","event_id":"future",
             "SG_off_the_tee":10,"SG_approach":10,"SG_around_green":10,"SG_putting":10},
            {"player":"B","date":"2026-10-20","event_id":"prior",
             "SG_off_the_tee":.1,"SG_approach":.2,"SG_around_green":-.3,"SG_putting":.4},
            {"player":"C","date":"2026-10-20","event_id":"prior",
             "SG_off_the_tee":.1,"SG_approach":.2,"SG_around_green":np.nan,"SG_putting":.4},
        ])
        e=build_pre_event_evidence(players,event_results=logs,tournament_start="2026-10-29").set_index("player")
        self.assertTrue(e.loc["A","last_event_all_four_positive"])
        self.assertEqual(e.loc["A","last_event_positive_categories"],4)
        self.assertEqual(e.loc["A","last_event_date"],"2026-10-20")
        self.assertFalse(e.loc["B","last_event_all_four_positive"])
        self.assertEqual(e.loc["B","last_event_positive_categories"],3)
        self.assertTrue(pd.isna(e.loc["C","last_event_all_four_positive"]))

    def test_single_snapshot_splits_into_previous_event_records(self):
        s=pd.DataFrame([{"player":"A","as_of":"2026-10-22","events_played":21,
                         "cuts_made":13,"last_event_name":"First Event",
                         "last_event_date":"2026-10-19","last_event_sg_ott":.3,
                         "last_event_sg_approach":.2,"last_event_sg_arg":.1,
                         "last_event_sg_putting":.4}])
        e=build_pre_event_evidence(["A"],season_stats=s,
                                   event_results=snapshot_last_events(s),
                                   tournament_start="2026-10-29")
        self.assertTrue(e.loc[0,"last_event_all_four_positive"])
        self.assertEqual(e.loc[0,"season_starts"],21)

    def test_uncertain_missing_data_cannot_create_bonus(self):
        players=pd.DataFrame({"player":["A","B"]})
        e=build_pre_event_evidence(players,tournament_start="2026-10-29")
        p=players.copy()
        p["expected_round"]=[-.2,.1]
        q=apply_reliability_research(p,e)
        np.testing.assert_allclose(q.expected_round,p.expected_round)
        self.assertEqual(float(q.reliability_strokes_adjustment.sum()),0.)

    def test_cut_impact_requires_coverage_and_breadth_is_bounded(self):
        names=[f"Golfer {i}" for i in range(12)]
        starts=pd.DataFrame({"player":names,"as_of":["2026-10-20"]*12,
                             "events_played":[20]*12,
                             "cuts_made":[15]*6+[8]*6})
        e=build_pre_event_evidence(names,season_stats=starts,
                                   tournament_start="2026-10-29")
        p=pd.DataFrame({"player":names,"expected_round":[0.]*12})
        q=apply_reliability_research(p,e)
        self.assertTrue(q.loc[:5,"cut_history_strokes_adjustment"].lt(0).all())
        self.assertTrue(q.loc[6:,"cut_history_strokes_adjustment"].gt(0).all())
        self.assertTrue(q["cut_history_strokes_adjustment"].abs().le(.18).all())
        nc=apply_reliability_research(p,e,no_cut=True)
        self.assertTrue(nc["cut_history_strokes_adjustment"].eq(0).all())

if __name__=="__main__":
    unittest.main()
