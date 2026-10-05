from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd

@dataclass
class ShowdownPortfolioSettings:
    lineup_count:int=20; salary_cap:int=50000; salary_floor:int=45000; roster_size:int=6
    max_exposure:float=.50; min_unique:int=2; seed:int=42; candidate_samples:int=90000

def optimize_showdown(d:pd.DataFrame, settings:ShowdownPortfolioSettings, locks=None, excludes=None):
    locks=set(locks or []); excludes=set(excludes or [])
    x=d[~d.player.isin(excludes)].copy(); x=x.dropna(subset=["player","salary","showdown_p90"])
    x["salary"]=pd.to_numeric(x.salary).astype(int)
    if not locks.issubset(set(x.player)): raise ValueError("A locked golfer is unavailable.")
    # Ceiling-first with mean/value support; no cut-equity term in one-round DFS.
    def z(c):
        a=pd.to_numeric(x[c],errors="coerce").fillna(0); sd=a.std(ddof=0); return (a-a.mean())/sd if sd else a*0
    x["obj"]=.55*z("showdown_p90")+.20*z("showdown_p95")+.15*z("showdown_mean")+.10*z("showdown_value")
    rng=np.random.default_rng(settings.seed); idx=np.arange(len(x)); vals=x.obj.to_numpy()
    lock_idx=set(x.index[x.player.isin(locks)])
    # use positional indices after reset
    x=x.reset_index(drop=True); lock_idx=set(x.index[x.player.isin(locks)]); avail=np.array([i for i in range(len(x)) if i not in lock_idx])
    need=settings.roster_size-len(lock_idx); cand={}
    for temp in (1.0,1.7,2.7):
        p=np.exp((x.obj.to_numpy()-x.obj.max())/temp); p=p[avail]/p[avail].sum()
        for _ in range(max(10000,settings.candidate_samples//3)):
            pick=list(lock_idx)+(list(rng.choice(avail,need,replace=False,p=p)) if need else [])
            lu=x.iloc[pick]; sal=int(lu.salary.sum())
            if settings.salary_floor<=sal<=settings.salary_cap:
                names=frozenset(lu.player.astype(str)); score=float(lu.obj.sum())+.000005*sal
                if names not in cand or score>cand[names][0]: cand[names]=(score,sal)
    ranked=sorted([(v[0],v[1],k) for k,v in cand.items()],reverse=True)
    if not ranked: raise RuntimeError("No feasible Showdown lineups. Relax salary floor or locks.")
    maxapp=max(1,int(np.floor(settings.max_exposure*settings.lineup_count+1e-9))); maxover=settings.roster_size-settings.min_unique
    selected=[]; expo={p:0 for p in x.player.astype(str)}
    for slot in range(settings.lineup_count):
        best=None; bs=-1e99
        for sc,sal,names in ranked:
            if names in selected or any(len(names&o)>maxover for o in selected): continue
            if any(expo[p]>=maxapp for p in names): continue
            penalty=sum((expo[p]/maxapp)**2 for p in names)
            adj=sc-(.22+.02*slot)*penalty
            if adj>bs: bs=adj; best=names
        if best is None: raise RuntimeError(f"Only built {len(selected)} lineups. Relax exposure/uniqueness/salary floor.")
        selected.append(best)
        for p in best: expo[p]+=1
    rows=[]; sums=[]
    for no,names in enumerate(selected,1):
        lu=x[x.player.isin(names)].sort_values("salary",ascending=False)
        sums.append({"lineup":no,"salary":int(lu.salary.sum()),"mean":lu.showdown_mean.sum(),"p90_sum":lu.showdown_p90.sum()})
        for _,r in lu.iterrows(): rows.append({"lineup":no,"player":r.player,"salary":int(r.salary),"showdown_mean":r.showdown_mean,"showdown_p90":r.showdown_p90,"showdown_p95":r.showdown_p95})
    port=pd.DataFrame(rows); summ=pd.DataFrame(sums); ex=port.groupby("player").size().sort_values(ascending=False).rename("lineups").reset_index(); ex["exposure_pct"]=100*ex.lineups/settings.lineup_count
    return port,summ,ex
