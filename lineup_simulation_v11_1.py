"""PGA V11.1 lineup-level portfolio scoring from correlated V11 simulation samples.

Scores are aligned across golfers by simulation row. This preserves the common
round shocks; never independently shuffle golfer outcomes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def evaluate_lineup_candidates(predictions: pd.DataFrame, samples: np.ndarray,
                               lineups: pd.DataFrame, top_fraction: float = 0.01) -> pd.DataFrame:
    """Evaluate lineup distributions; lineups must contain lineup,player.

    The top-fraction threshold is relative to *supplied candidate lineups*,
    not an estimated DraftKings opponent field. This is not contest ROI.
    """
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1")
    if not {"lineup","player"}.issubset(lineups.columns):
        raise ValueError("Lineups require lineup and player columns")
    if "player" not in predictions or predictions.player.duplicated().any():
        raise ValueError("Predictions require unique player names")
    scores=np.asarray(samples,dtype=float)
    if scores.ndim!=2 or scores.shape[1]!=len(predictions) or not np.isfinite(scores).all():
        raise ValueError("Samples must be finite (sims x prediction rows)")
    if lineups.empty: raise ValueError("No lineups provided")
    ix={p:i for i,p in enumerate(predictions.player)}
    groups=list(lineups.groupby("lineup",sort=False))
    matrix=np.zeros((len(predictions),len(groups)),dtype=np.float32)
    for k,(_,g) in enumerate(groups):
        if g.player.duplicated().any(): raise ValueError("Repeated golfer within lineup")
        for p in g.player:
            if p not in ix: raise ValueError(f"Unknown golfer: {p}")
            matrix[ix[p],k]=1
    totals=scores@matrix
    cutoff=np.quantile(totals,1-top_fraction,axis=1)
    result=[]
    for k,(name,g) in enumerate(groups):
        x=totals[:,k]
        result.append({"lineup":name,"mean_dk":float(x.mean()),
                       "median_dk":float(np.median(x)),
                       "p75_dk":float(np.quantile(x,.75)),
                       "p90_dk":float(np.quantile(x,.90)),
                       "p95_dk":float(np.quantile(x,.95)),
                       "sd_dk":float(x.std()),
                       "relative_top_fraction_pct":float(100*np.mean(x>=cutoff)),
                       "players":len(g)})
    return pd.DataFrame(result)

def rank_eight_lineups(evaluation: pd.DataFrame, balanced_count: int=6) -> pd.DataFrame:
    """Label exactly eight distinct candidate lineups: 6 balanced, 2 upside.

    This labels/ranks candidate scores, not a constrained portfolio optimizer.
    """
    if len(evaluation)<8 or evaluation.lineup.duplicated().any():
        raise ValueError("Need at least eight unique evaluated lineups")
    if balanced_count!=6:
        raise ValueError("Current eight-lineup configuration requires six balanced")
    e=evaluation.copy()
    # Balanced favors median/mean, upside favors top-decile and relative rank.
    e["balanced_score"]=.60*e.mean_dk+.40*e.median_dk
    e["upside_score"]=.60*e.p90_dk+.40*e.p95_dk
    balanced=e.nlargest(6,"balanced_score").copy()
    balanced["portfolio_role"]="balanced"
    remaining=e[~e.lineup.isin(balanced.lineup)]
    upside=remaining.nlargest(2,"upside_score").copy()
    upside["portfolio_role"]="upside"
    return pd.concat([balanced,upside],ignore_index=True)
