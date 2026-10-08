"""PGA single-round Showdown distribution model — V10.9.6.

Uses only signals already present in the weekly model: round rating, prior-round
adjustment, Course DNA, and available OTIS APP/OTT/ARG/PUTT percentiles.  It does
NOT claim to possess hole-level birdie/bogey data.  Output fantasy values are
transparent DK-style proxies for portfolio ranking, not official DK projections.
"""
import math, re
import numpy as np
import pandas as pd


def _key(x):
    return re.sub(r'[^a-z0-9]+',' ',str(x).casefold()).strip()


def _pct(s, default=50.0):
    return pd.to_numeric(s,errors='coerce').fillna(default).clip(1,99)


def build_showdown_pool(dk, ratings, round_no, otis=None, n_sims=25000, seed=4217):
    d=dk.copy(); r=ratings.copy()
    d['match_key']=d.player.map(_key); r['match_key']=r.player.map(_key)
    if 'Position' in d.columns:
        positions=d.Position.astype(str).str.upper()
        allowed=positions.isin(['G','GOLF','GOLFER','UTIL','FLEX'])
        if allowed.any(): d=d[allowed].copy()
    if 'Roster Position' in d.columns:
        d=d[~d['Roster Position'].astype(str).str.upper().str.contains('CPT')].copy()
    keep=['match_key','pre_round_rating','live_form_adj','live_rounds','round_rating','course_fit_signal']
    x=d.merge(r[keep],on='match_key',how='left',validate='many_to_one')
    x=x.dropna(subset=['round_rating']).copy()
    x['salary']=pd.to_numeric(x.salary,errors='coerce'); x=x[x.salary.between(3000,20000)].copy()
    if len(x)<6: raise ValueError('Fewer than six DraftKings golfers match OTIS ratings.')

    # Add only OTIS components that actually exist. Missing components are neutral (50), never zero.
    comps=['otis_fit_app','otis_fit_ott','otis_fit_arg','otis_fit_putt']
    if otis is not None and not otis.empty and 'player' in otis.columns:
        o=otis.copy(); o['match_key']=o.player.map(_key)
        avail=[c for c in comps if c in o.columns]
        if avail: x=x.merge(o[['match_key']+avail].drop_duplicates('match_key'),on='match_key',how='left')
    for c in comps:
        x[c]=_pct(x[c] if c in x.columns else pd.Series(50,index=x.index))

    # Transparent single-round traits from available percentiles.
    # APP/OTT/PUTT create scoring upside; APP/OTT/ARG create stability / bogey-resistance proxy.
    x['upside_signal']=(0.45*x.otis_fit_app+0.25*x.otis_fit_ott+0.20*x.otis_fit_putt+0.10*x.course_fit_signal).clip(1,99)
    x['stability_signal']=(0.45*x.otis_fit_app+0.25*x.otis_fit_ott+0.20*x.otis_fit_arg+0.10*x.course_fit_signal).clip(1,99)

    # Baseline is deliberately close to V10.9.5 so the audit isolates distribution/upside effects.
    base=55.0+(x.round_rating-50.0)*0.22
    # Small component tilt; this cannot overwhelm the validated round rating.
    mean=base+(x.upside_signal-50.0)*0.025+(x.stability_signal-50.0)*0.012
    # Volatility is player-specific: more upside -> wider right tail; more stability -> slightly less noise.
    sd=(11.5+(x.upside_signal-50.0)*0.035-(x.stability_signal-50.0)*0.018).clip(8.5,15.0)

    rng=np.random.default_rng(seed+int(round_no))
    z=rng.standard_normal((int(n_sims),len(x)))
    # Mild positive skew for high-upside golfers without inventing birdie counts.
    skew=np.maximum(z,0.0)**2-0.5
    sims=mean.to_numpy()[None,:]+sd.to_numpy()[None,:]*z+((x.upside_signal.to_numpy()-50)/50.0)[None,:]*1.8*skew
    sims=np.clip(sims,5,115)
    x['sd_proxy']=sims.std(axis=0)
    x['mean_proxy']=sims.mean(axis=0)
    x['p75_proxy']=np.percentile(sims,75,axis=0)
    x['p90_proxy']=np.percentile(sims,90,axis=0)
    x['p95_proxy']=np.percentile(sims,95,axis=0)
    # GPP ranking favors ceiling but still requires a strong mean.
    x['gpp_score']=0.55*x.mean_proxy+0.30*x.p90_proxy+0.15*x.p95_proxy
    x['value_gpp_per_1k']=x.gpp_score/(x.salary/1000.0)
    x['round']=int(round_no)
    cols=['player','salary','round','round_rating','pre_round_rating','live_form_adj','live_rounds','course_fit_signal',
          'otis_fit_app','otis_fit_ott','otis_fit_arg','otis_fit_putt','upside_signal','stability_signal',
          'mean_proxy','sd_proxy','p75_proxy','p90_proxy','p95_proxy','gpp_score','value_gpp_per_1k']
    return x[cols].sort_values('gpp_score',ascending=False).reset_index(drop=True)


def optimize_showdown(pool,n_lineups=4,max_exposure=.5,salary_floor=46500,seed=42):
    if len(pool)<6: raise ValueError('Need at least six golfers.')
    n_lineups=int(n_lineups); rng=np.random.default_rng(seed)
    salaries=pool.salary.to_numpy(dtype=int)
    score=pool.gpp_score.to_numpy(float)
    mean=pool.mean_proxy.to_numpy(float); p90=pool.p90_proxy.to_numpy(float)
    candidates={}
    # Candidate bank mixes pure strength and salary efficiency, then ranks by GPP distribution score.
    for i in range(50000):
        exponent=0.55+(i%11)*0.10
        weight=np.exp(np.clip((score-score.mean())/max(3.0,score.std())*exponent,-6,6))
        if i%4==0: weight*=np.clip(salaries.mean()/salaries,.55,1.9)
        ids=tuple(sorted(rng.choice(len(pool),size=6,replace=False,p=weight/weight.sum())))
        total=int(salaries[list(ids)].sum())
        if total<salary_floor or total>50000: continue
        gpp=float(score[list(ids)].sum()); mn=float(mean[list(ids)].sum()); ceil=float(p90[list(ids)].sum())
        if ids not in candidates or gpp>candidates[ids][0]: candidates[ids]=(gpp,total,mn,ceil)
    if not candidates: raise ValueError('No legal 6-golfer combinations found. Try lowering the salary floor.')
    ordered=sorted(candidates.items(),key=lambda item:(item[1][0],item[1][3],item[1][1]),reverse=True)
    cap=max(1,int(math.floor(n_lineups*max_exposure+1e-9)))
    if cap*len(pool)<n_lineups*6: raise ValueError('Exposure cap is mathematically infeasible for this field and lineup count.')
    counts=np.zeros(len(pool),dtype=int); chosen=[]
    for _ in range(n_lineups):
        best=None; best_val=-1e18
        for ids,(gpp,salary,mn,ceil) in ordered:
            if ids in chosen or any(counts[j]>=cap for j in ids): continue
            overlap=max((len(set(ids)&set(old)) for old in chosen),default=0)
            val=gpp-1.75*overlap
            if val>best_val: best=(ids,gpp,salary,mn,ceil); best_val=val
        if best is None: raise ValueError(f'Could construct only {len(chosen)} of {n_lineups} lineups under exposure cap; increase cap or reduce lineups.')
        ids,gpp,salary,mn,ceil=best; chosen.append(ids)
        for j in ids: counts[j]+=1
    rows=[]
    for i,ids in enumerate(chosen,1):
        rec={'Lineup':i,'Salary':int(salaries[list(ids)].sum()),'Mean proxy':round(float(mean[list(ids)].sum()),2),
             'P90 proxy':round(float(p90[list(ids)].sum()),2),'GPP distribution score':round(float(score[list(ids)].sum()),2)}
        for j,idx in enumerate(ids,1): rec[f'G{j}']=pool.iloc[idx].player
        rows.append(rec)
    exposure=pd.DataFrame({'Player':pool.player,'Lineups':counts,'Exposure %':np.round(100*counts/n_lineups,1)}).query('Lineups>0').sort_values(['Lineups','Player'],ascending=[False,True])
    return pd.DataFrame(rows),exposure.reset_index(drop=True)
