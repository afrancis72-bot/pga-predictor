"""Experimental auditable feature-family weighting for PGA V11.1.

No RickRunGood model scores, rankings, or weights are consumed.
Absent families are redistributed among available evidence; no fabricated news.
"""
import numpy as np
import pandas as pd

DEFAULT_ALLOCATION={"course_dna":.55,"recent_form":.25,"course_history":.10,"player_intelligence":.05,"context":.05}
DEFAULT_SKILLS={"sg_approach":.32,"sg_ott":.18,"sg_arg":.10,"sg_putting":.15,
                "rrg_sg_bermuda_l36":.10,"gir_pct":.08,"scrambling_pct":.07}
FAMILIES={
"course_dna":tuple(DEFAULT_SKILLS),
"recent_form":("form_sg_total","form_sg_approach","form_sg_ott","form_sg_putting","sg_total"),
"course_history":("history_fit","venue_fit","course_history_score"),
"player_intelligence":("verified_news_adjustment",),
"context":("verified_weather_fit","verified_travel_fit"),
}
def validate_allocation(allocation,course_cap=.60):
    if set(allocation)!=set(DEFAULT_ALLOCATION): raise ValueError("Missing or extra weighting families")
    vals=np.array(list(allocation.values()),dtype=float)
    if not np.isfinite(vals).all() or (vals<0).any(): raise ValueError("Weights must be finite and nonnegative")
    if abs(float(vals.sum())-1)>1e-6: raise ValueError("Weight families must total 100%")
    if allocation["course_dna"]>course_cap+1e-9: raise ValueError("Course DNA exceeds cap")
    return True

def _standardize(series):
    x=pd.to_numeric(series,errors="coerce")
    sd=x.std(ddof=0)
    return (x-x.mean())/sd if pd.notna(sd) and sd>1e-9 else pd.Series(0.,index=x.index)

def apply_weighted_projection(pred,allocation=None,skill_weights=None,course_cap=.60):
    """Blend standardized available feature families into expected-round adjustments.

    Baseline expected_round is retained; new signal shifts are bounded to 0.35
    strokes per round pending historical calibration. Missing families are not
    treated as negative golfer signals.
    """
    alloc=dict(DEFAULT_ALLOCATION if allocation is None else allocation)
    validate_allocation(alloc,course_cap)
    skills=dict(DEFAULT_SKILLS if skill_weights is None else skill_weights)
    if set(skills)!=set(DEFAULT_SKILLS): raise ValueError("Skill weights must cover the configured Course DNA metrics")
    if any(not np.isfinite(v) or v<0 for v in skills.values()) or abs(sum(skills.values())-1)>1e-6:
        raise ValueError("Course DNA skill weights must total 100%")
    d=pred.copy()
    if "expected_round" not in d: raise ValueError("Prediction missing expected_round")
    family_signals={}
    availability={}
    for family,cols in FAMILIES.items():
        present=[c for c in cols if c in d and pd.to_numeric(d[c],errors="coerce").notna().sum()>=3
                 and pd.to_numeric(d[c],errors="coerce").std(ddof=0)>1e-9]
        if not present: continue
        internal=skills if family=="course_dna" else {c:1. for c in cols}
        active_weights=np.array([internal[c] for c in present],dtype=float)
        active_weights/=active_weights.sum()
        signal=sum(_standardize(d[c]).fillna(0).to_numpy()*w for c,w in zip(present,active_weights))
        family_signals[family]=signal
        availability[family]=present
    active={k:v for k,v in alloc.items() if k in family_signals and v>0}
    if not active: return d,{"applied":False,"reason":"No usable weighted statistical families","available":availability}
    mass=sum(active.values())
    blended=sum(family_signals[k]*(v/mass) for k,v in active.items())
    adjustment=np.clip(-.35*blended,-.35,.35)
    d["v111_weighted_round_adjustment"]=adjustment
    d["expected_round"]=pd.to_numeric(d.expected_round,errors="coerce")+adjustment
    return d,{"applied":True,"requested_weights":alloc,
              "effective_weights":{k:round(v/mass,4) for k,v in active.items()},
              "available":availability,"round_adjustment_limit":.35}
