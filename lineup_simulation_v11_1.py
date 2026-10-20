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

def validate_eight_portfolio(lineups: pd.DataFrame, predictions: pd.DataFrame,
                             salary_cap: int=50000, salary_floor: int=46500,
                             max_exposure: float=.50, min_unique: int=3) -> dict:
    """Check hard constraints on the *actual* eight-lineup portfolio."""
    if not {"lineup","player"}.issubset(lineups):
        raise ValueError("Expected lineup and player columns")
    if not {"player","salary"}.issubset(predictions):
        raise ValueError("Predictions require player and salary")
    if lineups.lineup.nunique()!=8 or len(lineups)!=48:
        raise ValueError("Exactly eight six-golfer lineups required")
    if predictions.player.duplicated().any():
        raise ValueError("Duplicate golfers in predictions")
    salaries=predictions.set_index("player").salary
    groups=[]
    for name,g in lineups.groupby("lineup"):
        members=set(g.player)
        if len(members)!=6: raise ValueError(f"Lineup {name} has duplicate golfers")
        if not members.issubset(salaries.index):
            raise ValueError(f"Unknown golfer in lineup {name}")
        total=int(pd.to_numeric(salaries.loc[list(members)]).sum())
        if not salary_floor<=total<=salary_cap:
            raise ValueError(f"Lineup {name} salary {total} outside bounds")
        groups.append(members)
    for i in range(8):
        for j in range(i+1,8):
            if len(groups[i]&groups[j])>6-min_unique:
                raise ValueError(f"Lineups {i+1} and {j+1} violate uniqueness")
    counts=lineups.player.value_counts()
    cap=int(np.floor(8*max_exposure+1e-9))
    if (counts>cap).any():
        raise ValueError("Exposure cap exceeded")
    return {"valid":True,"lineups":8,"max_golfer_appearances":int(counts.max()),
            "exposure_cap_appearances":cap}

def select_eight_from_candidates(predictions: pd.DataFrame, samples: np.ndarray,
                                 candidates: pd.DataFrame, salary_floor: int=46500,
                                 max_exposure: float=.50, min_unique: int=3,
                                 balanced_count: int=6) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Greedy portfolio selection over a larger legal candidate bank.

    Uses correlated sample outcomes and a soft penalty for lineup-score
    covariance. Selection is heuristic, not proof of global optimality or ROI.
    """
    if balanced_count!=6: raise ValueError("Expected six balanced and two upside")
    ev=evaluate_lineup_candidates(predictions,samples,candidates)
    if len(ev)<8: raise ValueError("At least eight candidate lineups required")
    pool={key:g.copy() for key,g in candidates.groupby("lineup",sort=False)}
    salaries=predictions.set_index("player").salary
    eligible={}
    for _,row in ev.iterrows():
        key=row["lineup"]; g=pool[key]; members=frozenset(g.player)
        if len(members)!=6 or not members.issubset(salaries.index): continue
        salary=int(pd.to_numeric(salaries.loc[list(members)]).sum())
        if salary_floor<=salary<=50000: eligible[key]=members
    if len(eligible)<8: raise ValueError("Fewer than eight legal candidate lineups")
    ix={p:i for i,p in enumerate(predictions.player)}
    scores=np.asarray(samples,dtype=float)
    score_by_lineup={key:scores[:,[ix[p] for p in members]].sum(axis=1)
                     for key,members in eligible.items()}
    frame=ev[ev.lineup.isin(eligible)].set_index("lineup")
    # Comparable scale for each scoring criterion.
    balanced=.60*frame.mean_dk+.40*frame.median_dk
    upside=.60*frame.p90_dk+.40*frame.p95_dk
    norm=max(float(frame.mean_dk.std()),1.)
    selected=[]; counts={}; chosen_scores=[]
    cap=int(np.floor(8*max_exposure+1e-9))
    for slot in range(8):
        role="balanced" if slot<6 else "upside"
        base=balanced if role=="balanced" else upside
        best=None; best_value=-np.inf
        for key,members in eligible.items():
            if key in selected: continue
            if any(counts.get(p,0)>=cap for p in members): continue
            if any(len(members & eligible[old])>6-min_unique for old in selected): continue
            # Penalize highly correlated lineups; avoid degenerate zero-variance correlations.
            correlations=[]
            for prior in chosen_scores:
                a=score_by_lineup[key]
                if a.std()>1e-9 and prior.std()>1e-9:
                    correlations.append(float(np.corrcoef(a,prior)[0,1]))
            cov_penalty=max(0.,np.mean(correlations)) if correlations else 0.
            concentration=sum((counts.get(p,0)/max(cap,1))**2 for p in members)
            value=float(base.loc[key])-.25*norm*cov_penalty-.08*norm*concentration
            if value>best_value: best,best_value=key,value
        if best is None:
            raise ValueError(f"Candidate bank cannot satisfy constraints after {len(selected)} lineups")
        selected.append(best); chosen_scores.append(score_by_lineup[best])
        for p in eligible[best]: counts[p]=counts.get(p,0)+1
    roles={key:("balanced" if i<6 else "upside") for i,key in enumerate(selected)}
    chosen=candidates[candidates.lineup.isin(selected)].copy()
    chosen["portfolio_role"]=chosen.lineup.map(roles)
    report=ev[ev.lineup.isin(selected)].copy()
    report["portfolio_role"]=report.lineup.map(roles)
    report["selection_order"]=report.lineup.map({key:i+1 for i,key in enumerate(selected)})
    report=report.sort_values("selection_order").reset_index(drop=True)
    validate_eight_portfolio(chosen,predictions,salary_floor=salary_floor,
                             max_exposure=max_exposure,min_unique=min_unique)
    return chosen,report

def generate_candidate_bank(predictions: pd.DataFrame, locks=(), excludes=(),
                            salary_floor: int=46500, min_player_salary: int=6500,
                            target: int=400, draws: int=100000,
                            seed: int=42) -> pd.DataFrame:
    """Generate a diverse, salary-legal bank without portfolio exposure pruning.

    Hard portfolio constraints are applied later to the selected eight. This
    function does not use ownership or opponent-field assumptions.
    """
    if not {"player","salary","v11_mean_dk","v11_p90_dk"}.issubset(predictions):
        raise ValueError("Candidate generation needs player, salary, V11 mean and p90")
    d=predictions.copy().dropna(subset=["player","salary","v11_mean_dk","v11_p90_dk"])
    if d.player.duplicated().any(): raise ValueError("Duplicate golfers")
    locks=set(locks); excludes=set(excludes)
    if locks & excludes or len(locks)>6: raise ValueError("Invalid locks/exclusions")
    d=d[~d.player.isin(excludes)]
    d=d[pd.to_numeric(d.salary)>=min_player_salary].reset_index(drop=True)
    if not locks.issubset(set(d.player)): raise ValueError("Locked golfer unavailable")
    if len(d)<6: raise ValueError("Fewer than six eligible golfers")
    locked=d.index[d.player.isin(locks)].to_numpy(dtype=int)
    selectable=d.index[~d.player.isin(locks)].to_numpy(dtype=int)
    need=6-len(locked)
    if need>len(selectable): raise ValueError("Not enough golfers after locks")
    salary=pd.to_numeric(d.salary).to_numpy(dtype=int)
    mu=pd.to_numeric(d.v11_mean_dk).to_numpy(float)
    ceiling=pd.to_numeric(d.v11_p90_dk).to_numpy(float)
    rng=np.random.default_rng(seed)
    found={}
    for style in ("balanced","upside","uniform"):
        strength=(.8*mu+.2*ceiling) if style=="balanced" else (.35*mu+.65*ceiling)
        strength=(strength-strength.mean())/max(strength.std(),1e-8)
        for temp in (1.5,3.0,6.0):
            logw=np.zeros(len(selectable)) if style=="uniform" else strength[selectable]/temp
            for _ in range(max(1,draws//9//1000)):
                n=1000
                if need:
                    keys=rng.gumbel(size=(n,len(selectable)))+logw[None,:]
                    pick=selectable[np.argpartition(keys,-need,axis=1)[:,-need:]]
                else:
                    pick=np.empty((n,0),dtype=int)
                for row in pick:
                    idx=np.concatenate([locked,row])
                    total=int(salary[idx].sum())
                    if salary_floor<=total<=50000:
                        key=tuple(sorted(d.player.iloc[idx].astype(str)))
                        found[key]=total
                if len(found)>=target*3: break
    if len(found)<8:
        raise ValueError("Could not generate eight legal candidate lineups")
    # Seeded sample, not just top-scoring lineups, preserves diversity.
    keys=list(found)
    rng.shuffle(keys)
    keys=keys[:target]
    rows=[{"lineup":i+1,"player":p} for i,key in enumerate(keys) for p in key]
    return pd.DataFrame(rows)
