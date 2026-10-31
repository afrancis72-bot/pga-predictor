"""Pre-event 2–3 tournament strokes-gained stability research.

A weather-disrupted or shortened event should be reported but not automatically
outweigh two normal, all-around tournaments. This module requires *event-level*
strokes gained for all four categories and pre-lock completion dates. A single
round or unknown source never passes evidence verification.

This is a research indicator, NOT a learned probability or a proven ROI edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("sg_ott","sg_approach","sg_arg","sg_putting")

def _truth(x):
    return str(x).strip().casefold() in ("true","1","yes","y","wind","extreme","abnormal")

def rolling_sg_evidence(events: pd.DataFrame, cutoff, *, last_events: int = 3):
    cols = ["key","recent_complete_sg_events","recent_disrupted_events",
            "recent_all_four_events","recent_rolling_positive_categories",
            "recent_all_four_positive","recent_min_component_sg",
            "recent_all_around_quality","recent_sg_events",
            *("recent_weighted_"+c for c in COMPONENTS)]
    if events is None or events.empty or not {"key","date"}.issubset(events):
        return pd.DataFrame(columns=cols)
    if last_events < 2 or last_events > 5:
        raise ValueError("Rolling event window must be two to five tournaments.")
    e=events.copy()
    e["date"]=pd.to_datetime(e["date"], errors="coerce",utc=True).dt.tz_localize(None)
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is not None: cutoff=cutoff.tz_localize(None)
    e=e[e["date"].notna() & e["date"].lt(cutoff.normalize())].copy()
    if e.empty:return pd.DataFrame(columns=cols)
    e["event_key"]=e["event_id"].fillna(e["date"].dt.strftime("%Y-%m-%d")).astype(str) if "event_id" in e else e["date"].dt.strftime("%Y-%m-%d")
    if "round" in e:
        e=e[pd.to_numeric(e["round"],errors="coerce").isna()].copy()
    e=e[~e.duplicated(["key","event_key"],keep=False)].copy()
    if e.empty:return pd.DataFrame(columns=cols)
    for c in COMPONENTS:
        if c not in e:e[c]=np.nan
        e[c]=pd.to_numeric(e[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    e["complete"]=e[list(COMPONENTS)].notna().all(axis=1)
    e["disrupted"]=e["abnormal_conditions"].apply(_truth) if "abnormal_conditions" in e else False
    if {"scheduled_rounds","completed_rounds"}.issubset(e):
        sched=pd.to_numeric(e["scheduled_rounds"],errors="coerce")
        done=pd.to_numeric(e["completed_rounds"],errors="coerce")
        e["disrupted"]=e["disrupted"] | (sched.gt(0)&done.lt(sched)&done.ge(1))
    records=[]
    for key,g in e.sort_values(["date","event_key"],ascending=[False,False]).groupby("key",sort=False):
        recent=g.head(last_events).copy()
        valid=recent.loc[recent.complete & ~recent.disrupted].copy()
        record={"key":key,"recent_complete_sg_events":len(valid),
                "recent_disrupted_events":int(recent.disrupted.sum()),
                "recent_all_four_events":int((recent.complete & (recent[list(COMPONENTS)]>0).all(axis=1)).sum()),
                "recent_rolling_positive_categories":np.nan,
                "recent_all_four_positive":pd.NA,
                "recent_min_component_sg":np.nan,
                "recent_all_around_quality":np.nan,
                "recent_sg_events":" | ".join(str(x) for x in recent["event_key"])}
        record.update({"recent_weighted_"+c:np.nan for c in COMPONENTS})
        # Require two different completed events after abnormal conditions are
        # removed. Do not fill missing SG with zero or calculate a single-event trend.
        if len(valid)>=2:
            weights=np.arange(len(valid),0,-1,dtype=float)
            weights/=weights.sum()
            means=np.average(valid[list(COMPONENTS)].to_numpy(float),axis=0,weights=weights)
            count=int(np.count_nonzero(means>0))
            record["recent_rolling_positive_categories"]=count
            record["recent_all_four_positive"]=bool(count==4)
            record["recent_min_component_sg"]=float(np.min(means))
            # A small value to rank evidence breadth; not a cut-probability.
            record["recent_all_around_quality"]=float(np.mean(np.clip(means,-2,2)) +
                                                        .20*np.clip(np.min(means),-2,2))
            record.update({"recent_weighted_"+c:float(v) for c,v in zip(COMPONENTS,means)})
        records.append(record)
    return pd.DataFrame(records,columns=cols)

def attach_rolling_sg(evidence, event_history, tournament_start):
    from reliability_features import _clean
    w=_clean(event_history)
    # Use the audited common identity normalization, but do not overwrite the
    # cut records or trust data with mismatched dates.
    metrics=rolling_sg_evidence(w,tournament_start)
    e=evidence.copy()
    if metrics.empty:
        for col in metrics.columns:
            if col!="key" and col not in e:e[col]=np.nan
        return e
    e["__key__"]=e["player"].map(lambda x:__import__("re").sub(r"[^a-z0-9]+"," ",str(x).casefold()).strip())
    result=e.merge(metrics,left_on="__key__",right_on="key",how="left",validate="one_to_one")
    return result.drop(columns=["__key__","key"])
