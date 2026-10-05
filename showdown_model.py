from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

APPROACH_BUCKETS = ["lt100", "100_125", "125_150", "150_175", "175_200", "200_plus"]

@dataclass
class ShowdownSettings:
    round_number: int = 1
    sims: int = 50000
    seed: int = 42
    round_affinity_weight: float = 0.16
    current_week_weight: float = 0.18
    course_approach_weight: float = 0.16
    scoring_style_weight: float = 0.14
    weather_weight: float = 0.08


def _num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)

def _z(s):
    x = pd.to_numeric(s, errors="coerce")
    x = x.fillna(x.median() if x.notna().any() else 0.0)
    sd = x.std(ddof=0)
    return (x-x.mean())/sd if sd and np.isfinite(sd) else pd.Series(0.0,index=x.index)

def _col(d, name, default=0.0):
    return _num(d[name], default) if name in d else pd.Series(default, index=d.index, dtype=float)

def build_showdown_projections(base: pd.DataFrame, round_history: pd.DataFrame|None=None,
                               current_week: pd.DataFrame|None=None,
                               course_mix: dict[str,float]|None=None,
                               settings: ShowdownSettings|None=None) -> pd.DataFrame:
    """Build round-specific DK Showdown distributions.

    Inputs are deliberately pre-result: baseline player skill + historical round splits +
    course approach mix + current-week underlying stats (for R2-R4). Missing features shrink
    to neutral rather than receiving a fabricated value.
    """
    s = settings or ShowdownSettings()
    d = base.copy()
    if "player" not in d or "salary" not in d:
        raise ValueError("Base file requires player and salary.")

    # Baseline skill. Prefer SG components; recent ball striking gets modest weight.
    sg_total = _col(d,"sg_total")
    sg_app = _col(d,"sg_approach")
    sg_ott = _col(d,"sg_ott")
    recent = _col(d,"recent_ball_striking") if "recent_ball_striking" in d else (_col(d,"recent5_sg_approach")+_col(d,"recent5_sg_ott"))/2
    d["baseline_skill_z"] = .45*_z(sg_total)+.25*_z(sg_app)+.15*_z(sg_ott)+.15*_z(recent)

    # Historical round affinity, sample-size shrunk toward player's overall baseline.
    d["round_affinity_z"] = 0.0
    d["round_sample"] = 0
    if round_history is not None and not round_history.empty:
        h = round_history.copy()
        rcol = f"r{s.round_number}_sg_total"
        ncol = f"r{s.round_number}_rounds"
        if "player" in h and rcol in h:
            cols=["player",rcol]+([ncol] if ncol in h else [])
            h=h[cols].drop_duplicates("player")
            d=d.merge(h,on="player",how="left")
            n=_col(d,ncol,0) if ncol in d else pd.Series(12.0,index=d.index)
            # 20-round prior prevents small-sample round narratives from dominating.
            shrink=n/(n+20.0)
            delta=_col(d,rcol,0)-sg_total
            d["round_affinity_z"]=_z(delta*shrink)
            d["round_sample"]=n.astype(int)

    # Course-weighted approach buckets. If absent, neutral; no proxy fabrication.
    mix=course_mix or {}
    weighted=pd.Series(0.0,index=d.index)
    coverage=pd.Series(0.0,index=d.index)
    totalw=sum(max(0,float(mix.get(b,0))) for b in APPROACH_BUCKETS)
    if totalw>0:
        for b in APPROACH_BUCKETS:
            w=max(0,float(mix.get(b,0)))/totalw
            c=f"sg_app_{b}"
            if c in d:
                ok=pd.to_numeric(d[c],errors="coerce").notna().astype(float)
                weighted += w*_col(d,c,0)
                coverage += w*ok
    d["course_approach_coverage"]=coverage
    d["course_approach_z"]=_z(weighted.where(coverage>0,0))

    # DFS scoring style: Birdies-or-Better gained and Bogey Avoidance are independent signals.
    bob = _col(d,"birdies_or_better_gained") if "birdies_or_better_gained" in d else _col(d,"birdies_gained")
    ba = _col(d,"bogey_avoidance_gained")
    d["scoring_style_z"]=.68*_z(bob)+.32*_z(ba)

    # Current-week underlying play (R2-R4): reward sustainable ball striking, regress putting.
    d["current_week_z"]=0.0
    if s.round_number>1 and current_week is not None and not current_week.empty and "player" in current_week:
        cw=current_week.drop_duplicates("player").copy()
        keep=[c for c in ["player","week_sg_app","week_sg_ott","week_sg_arg","week_sg_putt","tee_time_weather_edge","start_position"] if c in cw]
        d=d.merge(cw[keep],on="player",how="left")
        underlying=.52*_col(d,"week_sg_app")+.23*_col(d,"week_sg_ott")+.10*_col(d,"week_sg_arg")+.15*_col(d,"week_sg_putt")
        d["current_week_z"]=_z(underlying)
    d["weather_z"]=_z(_col(d,"tee_time_weather_edge")) if "tee_time_weather_edge" in d else 0.0

    # Mean round strength. Round affinity and approach fit enter upstream.
    d["round_strength"]=(
        .48*d.baseline_skill_z + s.round_affinity_weight*d.round_affinity_z +
        s.course_approach_weight*d.course_approach_z + s.scoring_style_weight*d.scoring_style_z +
        s.current_week_weight*d.current_week_z + s.weather_weight*d.weather_z
    )

    # Monte Carlo round scoring. Shared field shock preserves round-level correlation.
    rng=np.random.default_rng(s.seed); n=len(d)
    skill=d.round_strength.to_numpy(float)
    # Volatility is intentionally material in one-round golf; stronger birdie profiles get fatter upside.
    upside=np.clip(d.scoring_style_z.to_numpy(float),-2.5,2.5)
    field=rng.normal(0,.28,size=(s.sims,1))
    indiv=rng.normal(0,1.0,size=(s.sims,n))
    tail=rng.standard_t(df=5,size=(s.sims,n))*.16
    perf=skill[None,:]+field+indiv+tail*(1+.10*np.maximum(upside,0))[None,:]

    # Translate latent performance into birdies/bogeys and DK-like round points.
    birdie_rate=np.clip(3.55+.72*perf+.30*upside[None,:],.2,9.0)
    bogey_rate=np.clip(2.45-.40*perf-.24*np.clip(d.scoring_style_z.to_numpy(float),-3,3)[None,:],.15,7.0)
    birdies=rng.poisson(birdie_rate)
    bogeys=rng.poisson(bogey_rate)
    eagles=rng.binomial(2,np.clip(.035+.012*np.maximum(perf,0),.005,.14))
    # Pars fill remaining holes. DK event scoring approximation + finishing/bonus proxy via round rank.
    score_to_par=bogeys-birdies-2*eagles
    dk=birdies*3 + eagles*8 - bogeys*.5 + np.maximum(0,18-birdies-bogeys-eagles)*.5
    # Bonus proxy for 3-birdie streak / bogey-free round, simulated from event counts.
    dk += (birdies>=5)*3 + (bogeys==0)*3

    # R4: final-position points based on starting tournament position plus simulated final-round movement.
    if s.round_number==4 and current_week is not None and "start_position" in d:
        start=_col(d,"start_position",50).to_numpy(float)
        # Round performance moves leaderboard; this is intentionally bounded rather than deterministic.
        final_metric=start[None,:]-3.0*perf+rng.normal(0,2.0,size=(s.sims,n))
        ranks=np.argsort(np.argsort(final_metric,axis=1),axis=1)+1
        fp=np.select([ranks==1,ranks<=2,ranks<=3,ranks<=5,ranks<=10,ranks<=20,ranks<=30,ranks<=40],
                     [13,10,9,8,7,6,5,4],default=3)
        dk=dk+fp

    d["showdown_mean"]=dk.mean(axis=0)
    d["showdown_p75"]=np.percentile(dk,75,axis=0)
    d["showdown_p90"]=np.percentile(dk,90,axis=0)
    d["showdown_p95"]=np.percentile(dk,95,axis=0)
    d["round_birdies_mean"]=birdies.mean(axis=0)
    d["round_bogeys_mean"]=bogeys.mean(axis=0)
    d["round_score_mean"]=score_to_par.mean(axis=0)
    d["showdown_value"]=d.showdown_mean/(pd.to_numeric(d.salary,errors="coerce")/1000)
    return d
