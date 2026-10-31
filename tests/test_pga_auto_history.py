"""PGA auto-provider fixture tests (no live network dependency)."""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/"pga_engine" if (ROOT/"pga_engine").is_dir() else ROOT
sys.path.insert(0,str(ENGINE))
from pga_auto_history import (validate_completed_events,normalize_leaderboard,
                               normalize_event_sg,collect_pre_event_history)

class FakeProvider:
    calls=[]
    @staticmethod
    def pga_schedule(year,tour):
        return pd.DataFrame([
          {"tournament_id":"R2026031","tournament_name":"Event A",
           "display_date":"Oct 15 - 18","year":2026,"status":"COMPLETED"},
          {"tournament_id":"R2026032","tournament_name":"Event B",
           "display_date":"Oct 22 - 25","year":2026,"status":"COMPLETED"},
          {"tournament_id":"R2026033","tournament_name":"Future",
           "display_date":"Oct 29 - Nov 1","year":2026,"status":"IN_PROGRESS"},
        ]) if year==2026 else pd.DataFrame()
    @staticmethod
    def pga_leaderboard(event_id):
        # 70 finishers + 10 missed-cuts
        rows=[{"player_id":str(i),"display_name":f"Other {i}",
               "position":str(i+1) if i<70 else "CUT",
               "round_1":70,"round_2":70,"round_3":70,"round_4":70}
              for i in range(80)]
        rows[0].update({"player_id":"111","display_name":"Golfer A","position":"T1"})
        lb=pd.DataFrame(rows)
        lb.attrs["tournament_status"]="COMPLETED"
        return lb
    @classmethod
    def pga_scorecard_stats(cls,event_id,pid,round="-1"):
        cls.calls.append((event_id,pid))
        return pd.DataFrame([
            {"round":"-1","section":"strokesGained","stat_id":sid,"total":4}
            for sid in ("02567","02568","02569","02564")
        ])

class AutoPGA(unittest.TestCase):
    def test_completed_schedule_cutoff(self):
        e,ignored=validate_completed_events(FakeProvider.pga_schedule(2026,"R"),"2026-10-29")
        self.assertEqual(len(e),2)
        self.assertNotIn("R2026033",e.event_id.tolist())
    def test_true_cut_not_inferred_in_no_cut_event(self):
        lb=FakeProvider.pga_leaderboard("R2026031")
        event={"event_id":"R2026031","end_date":pd.Timestamp("2026-10-18"),"name":"Event A"}
        accepted,status=normalize_leaderboard(lb,event)
        self.assertEqual(status,"accepted")
        self.assertTrue(bool(accepted.loc[accepted.player.eq("Golfer A"),"made_cut"].iloc[0]))
        lb["position"]=[str(i+1) for i in range(len(lb))]
        empty,status=normalize_leaderboard(lb,event)
        self.assertTrue(empty.empty)
    def test_sg_totals_not_ytd_and_incomplete_rejected(self):
        raw=FakeProvider.pga_scorecard_stats("R2026031","111")
        self.assertEqual(normalize_event_sg(raw,completed_rounds=4)["sg_ott"],1.)
        raw.loc[0,"total"]=np.nan
        self.assertIsNone(normalize_event_sg(raw,completed_rounds=4))
        raw.loc[0,"total"]=4
        raw.loc[0,"round"]="1"
        self.assertIsNone(normalize_event_sg(raw,completed_rounds=4))
    def test_auto_collect_no_future_results(self):
        FakeProvider.calls=[]
        snap,sg,logs=collect_pre_event_history(["Golfer A"],"2026-10-29",
                         provider=FakeProvider,max_workers=1,min_field=60)
        self.assertEqual(int(snap.iloc[0].events_played),2)
        self.assertEqual(int(snap.iloc[0].cuts_made),2)
        self.assertEqual(len(sg),2)
        self.assertTrue(pd.to_datetime(sg.date).lt(pd.Timestamp("2026-10-29")).all())
        self.assertEqual(len(FakeProvider.calls),2)
        self.assertTrue(logs.stage.eq("summary").any())
    def test_future_schedule_rejected_even_marked_completed(self):
        d=FakeProvider.pga_schedule(2026,"R")
        d.loc[d.tournament_id.eq("R2026033"),"status"]="COMPLETED"
        e,ignored=validate_completed_events(d,"2026-10-29")
        self.assertEqual(len(e),2)

class TwoTourCase(unittest.TestCase):
    def test_korn_ferry_form_is_visible_but_not_pga_cut_record(self):
        class DualProvider:
            @staticmethod
            def pga_schedule(year,tour):
                if year!=2026:return pd.DataFrame()
                schedules={"R":[("R2026031","Oct 15 - 18")],
                           "H":[("H2026094","Oct 8 - 11"),
                                ("H2026093","Oct 1 - 4"),
                                ("H2026092","Sep 17 - 20")]}
                return pd.DataFrame([
                    {"tournament_id":tid,"tournament_name":tid,
                     "display_date":dt,"status":"COMPLETED"}
                    for tid,dt in schedules[tour]])
            @staticmethod
            def pga_leaderboard(event_id):
                rows=[{"player_id":str(i+10000),"display_name":f"Other {i}",
                       "position":str(i+1) if i<70 else "CUT",
                       "round_1":70,"round_2":70,"round_3":70,"round_4":70}
                      for i in range(80)]
                if event_id.startswith("H"):
                    rows[0].update({"player_id":"65001","display_name":"Alvaro Ortiz",
                                    "position":("T9" if event_id.endswith("94") else
                                                "T24" if event_id.endswith("93") else "1")})
                else:
                    rows[0].update({"player_id":"65002","display_name":"William Mouw",
                                    "position":"T32"})
                lb=pd.DataFrame(rows)
                lb.attrs["tournament_status"]="COMPLETED"
                return lb
            @staticmethod
            def pga_scorecard_stats(event_id,pid,round="-1"):
                return pd.DataFrame([{"round":"-1","section":"strokesGained",
                                     "stat_id":sid,"total":4}
                                     for sid in ("02567","02568","02569","02564")])
        cut,sg,logs=collect_pre_event_history(
            ["Alvaro Ortiz","William Mouw"],"2026-10-29",
            provider=DualProvider,max_workers=1,lookback=25)
        row=cut.set_index("player").loc["Alvaro Ortiz"]
        self.assertTrue(pd.isna(row["events_played"]))
        self.assertEqual(int(row["kft_starts"]),3)
        self.assertEqual(int(row["kft_last5_top25"]),3)
        self.assertTrue(bool(row["kft_review_candidate"]))
        self.assertEqual(int(cut.set_index("player").loc["William Mouw","events_played"]),1)
        self.assertTrue(sg.loc[sg.player.eq("Alvaro Ortiz"),"source_tour"].eq("H").all())
        self.assertTrue(logs.stage.eq("summary").any())


if __name__=="__main__":
    unittest.main()
