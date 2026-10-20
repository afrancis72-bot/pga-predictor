"""Leakage-aware historical evaluation for PGA weighting experiments.

Input: one row per golfer per event, with PRE-EVENT model features and
observed realized strokes per round. No historical observations are bundled.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from course_weights_v11_1 import apply_weighted_projection, DEFAULT_ALLOCATION, validate_allocation

REQUIRED=("event_id","event_date","player","expected_round","actual_round_strokes")

def evaluate_historical_weights(history, candidates=None, min_events=3, min_golfers=15):
    """Evaluate fixed, predeclared weight candidates against a baseline.

    Standardization is event-local. Each row must contain pre-event features;
    event_date is the tournament start. This function never fits weights to
    the held-out outcome. Candidate choice must use a separate training period.
    """
    if history is None or history.empty: raise ValueError("Historical pre-event data is required")
    d=history.copy()
    missing=[c for c in REQUIRED if c not in d]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    d["event_date"]=pd.to_datetime(d.event_date,errors="coerce")
    for c in ("expected_round","actual_round_strokes"):
        d[c]=pd.to_numeric(d[c],errors="coerce")
    if d[list(REQUIRED)].isna().any().any(): raise ValueError("Missing required values")
    if d.duplicated(["event_id","player"]).any(): raise ValueError("Duplicate golfer/event rows")
    if d.groupby("event_id").event_date.nunique().max()>1:
        raise ValueError("An event must have a single start date")
    sizes=d.groupby("event_id").size()
    eligible=sizes[sizes>=min_golfers].index
    d=d[d.event_id.isin(eligible)].copy()
    if d.event_id.nunique()<min_events:
        raise ValueError(f"Need at least {min_events} events with {min_golfers} golfers each")
    candidates={"default_55_25_10_5_5":dict(DEFAULT_ALLOCATION)} if candidates is None else candidates
    for a in candidates.values(): validate_allocation(a)
    rows=[]
    for event_id,g in d.groupby("event_id",sort=False):
        g=g.copy()
        observed=g.actual_round_strokes.to_numpy(dtype=float)
        scenarios={"baseline":g.expected_round.to_numpy(dtype=float)}
        for name,allocation in candidates.items():
            if name=="baseline": raise ValueError("baseline is a reserved scenario name")
            adjusted,audit=apply_weighted_projection(g,allocation)
            scenarios[name]=adjusted.expected_round.to_numpy(dtype=float)
        for name,estimate in scenarios.items():
            err=estimate-observed
            rows.append({"event_id":event_id,"event_date":g.event_date.iloc[0],
                         "scenario":name,"golfers":len(g),
                         "squared_error_sum":float(np.square(err).sum()),
                         "absolute_error_sum":float(np.abs(err).sum()),
                         "signed_error_sum":float(err.sum())})
    per_event=pd.DataFrame(rows).sort_values(["event_date","scenario"]).reset_index(drop=True)
    totals=per_event.groupby("scenario",as_index=False).agg(
        events=("event_id","nunique"),golfers=("golfers","sum"),
        squared_error_sum=("squared_error_sum","sum"),
        absolute_error_sum=("absolute_error_sum","sum"),
        signed_error_sum=("signed_error_sum","sum"))
    totals["rmse"]=np.sqrt(totals.squared_error_sum/totals.golfers)
    totals["mae"]=totals.absolute_error_sum/totals.golfers
    totals["bias"]=totals.signed_error_sum/totals.golfers
    base=float(totals.loc[totals.scenario=="baseline","rmse"].iloc[0])
    totals["rmse_improvement_vs_baseline"]=base-totals.rmse
    return totals.sort_values("rmse").reset_index(drop=True),per_event

def rolling_out_of_sample(history,candidates=None,min_train_events=3,min_golfers=15):
    """Choose among candidate allocations on strictly earlier events, score next event.

    Includes the baseline as a selectable option. Returns chronological
    predictions and comparison, never a same-event tuned result.
    """
    if candidates is None:
        candidates={"default_55_25_10_5_5":dict(DEFAULT_ALLOCATION)}
    d=history.copy()
    d["event_date"]=pd.to_datetime(d.event_date,errors="coerce")
    dates=d.groupby("event_id").event_date.min().sort_values()
    ordered=dates.index.tolist()
    folds=[]
    for idx in range(min_train_events,len(ordered)):
        test_id=ordered[idx]
        train_ids=[e for e in ordered[:idx] if dates[e]<dates[test_id]]
        if len(train_ids)<min_train_events: continue
        training=d[d.event_id.isin(train_ids)]
        try:
            leaderboard,_=evaluate_historical_weights(training,candidates,min_train_events,min_golfers)
            test_summary,_=evaluate_historical_weights(
                pd.concat([d[d.event_id==test_id]]),candidates,1,min_golfers)
        except ValueError: continue
        selected=str(leaderboard.iloc[0].scenario)
        chosen=test_summary[test_summary.scenario==selected].iloc[0]
        baseline=test_summary[test_summary.scenario=="baseline"].iloc[0]
        folds.append({"event_id":test_id,"event_date":dates[test_id],
                      "chosen_scenario":selected,"golfers":int(chosen.golfers),
                      "chosen_sse":float(chosen.squared_error_sum),
                      "baseline_sse":float(baseline.squared_error_sum)})
    if not folds: raise ValueError("Not enough chronological events for out-of-sample folds")
    out=pd.DataFrame(folds)
    comparison={"events":len(out),"golfers":int(out.golfers.sum()),
                "selected_rmse":float(np.sqrt(out.chosen_sse.sum()/out.golfers.sum())),
                "baseline_rmse":float(np.sqrt(out.baseline_sse.sum()/out.golfers.sum()))}
    comparison["rmse_improvement"]=comparison["baseline_rmse"]-comparison["selected_rmse"]
    return out,comparison
