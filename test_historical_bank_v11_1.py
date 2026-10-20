import unittest
import pandas as pd
from historical_bank_v11_1 import normalize_logs, merge_logs, query_logs

class HistoricalBankTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame([
            dict(date="2025-07-20", tour="PGA", golfer="Cam Davis",
                 tournament="Barracuda Championship", course="Old Greenwood",
                 position="CUT", ott="—", app="—", arg="—", putt="—", tot="-1.93"),
            dict(date="2025-01-12", tour="PGA", golfer="Mark Hubbard",
                 tournament="Sony Open in Hawaii", course="Waialae CC",
                 position="T21", ott="2.54", app="2.67", arg="0.16", putt="-0.67", tot="4.70"),
        ])
    def test_missing_shotlink_is_not_zero(self):
        d=normalize_logs(self.raw)
        self.assertTrue(pd.isna(d.iloc[0].ott))
        self.assertFalse(bool(d.iloc[0].shotlink_complete))
        self.assertEqual(d.iloc[0].tot,-1.93)
    def test_filter_year_tournament_category(self):
        d=query_logs(self.raw,years=2025,tournaments="Sony Open in Hawaii",categories=["app"])
        self.assertEqual(len(d),1)
        self.assertEqual(d.iloc[0].app,2.67)
        self.assertNotIn("ott",d.columns)
    def test_idempotent_merge(self):
        first, report=merge_logs(None,self.raw)
        self.assertEqual(report["added"],2)
        second, report=merge_logs(first,self.raw)
        self.assertEqual(len(second),2)
        self.assertEqual(report["unchanged"],2)
    def test_missing_stat_can_be_filled(self):
        first,_=merge_logs(None,self.raw)
        more=self.raw.iloc[[0]].copy()
        more.loc[more.index[0],"ott"]=1.0
        second,report=merge_logs(first,more)
        self.assertEqual(report["updated"],1)
        self.assertEqual(float(second.loc[second.golfer=="Cam Davis","ott"].iloc[0]),1.0)
    def test_conflicting_stats_not_silently_replaced(self):
        first,_=merge_logs(None,self.raw)
        more=self.raw.iloc[[1]].copy()
        more.loc[more.index[0],"app"]=99.0
        second,report=merge_logs(first,more)
        self.assertGreater(report["conflicts"],0)
        self.assertEqual(float(second.loc[second.golfer=="Mark Hubbard","app"].iloc[0]),2.67)

if __name__=="__main__":
    unittest.main()
