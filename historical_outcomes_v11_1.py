"""Validate and normalize externally acquired historical golfer outcomes.

This module never invents historical pre-event snapshots. It only prepares
verified tournament outcomes for later joins to frozen projections.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED=("event_id","event_date","player","rounds_played","total_strokes","course_par")
def normalize_historical_outcomes(raw):
    """One golfer/event: actual average strokes relative to par per played round.

    course_par is per-round par. For courses with variable round pars, callers
    must provide total_par_played instead (the sum of pars of completed rounds).
    No made-cut or fantasy-point proxies are substituted for actual strokes.
    """
    if raw is None or raw.empty: raise ValueError("Historical outcomes file is empty")
    d=raw.copy()
    missing=[c for c in REQUIRED if c not in d.columns and not (c=="course_par" and "total_par_played" in d)]
    if missing: raise ValueError("Missing outcome fields: "+", ".join(missing))
    d["event_date"]=pd.to_datetime(d.event_date,errors="coerce")
    for c in ("rounds_played","total_strokes"):
        d[c]=pd.to_numeric(d[c],errors="coerce")
    if "total_par_played" in d:
        par=pd.to_numeric(d.total_par_played,errors="coerce")
    else:
        par=pd.to_numeric(d.course_par,errors="coerce")*d.rounds_played
    if d[["event_id","event_date","player","rounds_played","total_strokes"]].isna().any().any() or par.isna().any():
        raise ValueError("Missing required golfer/event outcome values")
    if (d.rounds_played<=0).any() or (d.rounds_played>4).any():
        raise ValueError("Rounds played must be between 1 and 4")
    if (d.total_strokes<=0).any() or (par<=0).any():
        raise ValueError("Strokes and total par must be positive")
    if d.duplicated(["event_id","player"]).any(): raise ValueError("Duplicate golfer/event outcome")
    if d.groupby("event_id").event_date.nunique().max()>1:
        raise ValueError("Conflicting event start dates")
    d["actual_round_strokes"]=(d.total_strokes-par)/d.rounds_played
    return d[["event_id","event_date","player","actual_round_strokes","rounds_played","total_strokes"]].copy()

def join_frozen_predictions(snapshots,outcomes,min_coverage=.7):
    """Join on event and golfer; reject unverified or post-start snapshots.

    Expected input includes snapshot_captured_at, event_id, event_date, player,
    expected_round. Timezone-aware snapshot timestamps are required (UTC).
    """
    if snapshots is None or snapshots.empty: raise ValueError("No frozen pre-event snapshots")
    d=snapshots.copy()
    required=("event_id","event_date","player","expected_round","snapshot_captured_at")
    absent=[c for c in required if c not in d]
    if absent: raise ValueError("Missing snapshot columns: "+", ".join(absent))
    d["event_date"]=pd.to_datetime(d.event_date,utc=True,errors="coerce")
    d["snapshot_captured_at"]=pd.to_datetime(d.snapshot_captured_at,utc=True,errors="coerce")
    d["expected_round"]=pd.to_numeric(d.expected_round,errors="coerce")
    if d[list(required)].isna().any().any(): raise ValueError("Incomplete pre-event snapshot")
    if (d.snapshot_captured_at>=d.event_date).any():
        raise ValueError("Snapshot captured at or after tournament start date")
    if d.duplicated(["event_id","player"]).any(): raise ValueError("Duplicate frozen golfer/event")
    actual=outcomes.copy()
    actual["event_date"]=pd.to_datetime(actual.event_date,utc=True,errors="coerce")
    joined=d.merge(actual[["event_id","player","event_date","actual_round_strokes"]],
                   on=["event_id","player","event_date"],how="inner",validate="one_to_one")
    coverage=len(joined)/len(d)
    if coverage<min_coverage: raise ValueError(f"Outcome coverage {coverage:.1%} below {min_coverage:.0%}")
    if joined.empty: raise ValueError("No matching historical outcomes")
    joined["event_date"]=joined.event_date.dt.tz_localize(None)
    return joined,{"snapshot_rows":len(d),"matched":len(joined),"coverage":round(coverage,4),
                   "events":joined.event_id.nunique()}
