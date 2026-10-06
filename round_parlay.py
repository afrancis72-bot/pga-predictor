import re, math
from io import StringIO
import numpy as np
import pandas as pd
import requests

UA={"User-Agent":"Mozilla/5.0 (compatible; PGA-Parlay-Model/1.0)"}

def _norm(s):
    return re.sub(r"[^a-z0-9]+"," ",str(s).casefold()).strip()

def _slug(s):
    return re.sub(r"[^a-z0-9]+","-",str(s).casefold()).strip("-")

def golfchannel_round_url(tournament, year, round_no):
    return f"https://www.golfchannel.com/pga-tour/news/{_slug(tournament)}-{int(year)}-round-{int(round_no)}-tee-times-groupings-how-to-watch"

def fetch_text(url):
    r=requests.get(url,headers=UA,timeout=15); r.raise_for_status(); return r.text

def parse_groupings_text(text, round_no=1):
    # Works on rendered/article text copied from Golf Channel/GolfDigest/PGA TOUR media.
    clean=re.sub(r"<[^>]+>","\n",text)
    clean=re.sub(r"&nbsp;"," ",clean)
    clean=re.sub(r"\s+"," ",clean)
    # tee-time pattern followed by three comma-separated names. Supports 7:45 p.m./8:45 a.m. -- A, B, C
    pat=re.compile(r"(?P<time>\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.)?(?:\s*/\s*\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.)?)?)\s*(?:--|—|–|-)?\s*(?P<a>[A-Z][A-Za-z .’'\-]+?),\s*(?P<b>[A-Z][A-Za-z .’'\-]+?),\s*(?P<c>[A-Z][A-Za-z .’'\-]+?)(?=\s+\d{1,2}:\d{2}|\s+(?:First|10th|Tenth) tee|$)")
    out=[]
    for i,m in enumerate(pat.finditer(clean),1):
        names=[m.group('a').strip(),m.group('b').strip(),m.group('c').strip()]
        if all(2 <= len(x) <= 45 for x in names):
            out.append({"group":i,"tee_time":m.group('time').strip(),"player1":names[0],"player2":names[1],"player3":names[2],"round":round_no})
    return pd.DataFrame(out)

def fetch_groupings(tournament, year, round_no, source_url=""):
    urls=[]
    if source_url.strip(): urls.append(source_url.strip())
    urls.append(golfchannel_round_url(tournament,year,round_no))
    errors=[]
    for u in dict.fromkeys(urls):
        try:
            html=fetch_text(u)
            # First try HTML tables.
            try:
                tabs=pd.read_html(StringIO(html))
                for t in tabs:
                    cols=[_norm(c) for c in t.columns]
                    if len(t.columns)>=4 and any('time' in c for c in cols):
                        # Find rows containing three plausible player fields after time.
                        rec=[]
                        for _,row in t.iterrows():
                            vals=[str(v).strip() for v in row.tolist() if str(v).strip() not in ('','nan')]
                            if len(vals)>=4:
                                names=vals[-3:]
                                rec.append({"group":len(rec)+1,"tee_time":vals[0],"player1":names[0],"player2":names[1],"player3":names[2],"round":round_no})
                        if rec: return pd.DataFrame(rec),u,"HTML table"
            except Exception: pass
            df=parse_groupings_text(html,round_no)
            if not df.empty: return df,u,"article text"
            errors.append(f"No groupings parsed from {u}")
        except Exception as e: errors.append(f"{u}: {e}")
    return pd.DataFrame(),"","; ".join(errors[-2:])

def normalize_groupings_upload(df, round_no):
    if df is None or df.empty: return pd.DataFrame()
    w=df.copy(); w.columns=[_norm(c).replace(' ','_') for c in w.columns]
    aliases={"time":"tee_time","tee":"tee_time","golfer1":"player1","golfer2":"player2","golfer3":"player3","player_1":"player1","player_2":"player2","player_3":"player3"}
    w=w.rename(columns={c:aliases.get(c,c) for c in w.columns})
    if not {'player1','player2','player3'}.issubset(w.columns): return pd.DataFrame()
    if 'group' not in w: w['group']=range(1,len(w)+1)
    if 'tee_time' not in w: w['tee_time']=''
    w['round']=round_no
    return w[['group','tee_time','player1','player2','player3','round']].copy()

def normalize_live_results(df):
    """Accept player plus round score / SG components. Lower score is better."""
    if df is None or df.empty: return pd.DataFrame()
    w=df.copy(); w.columns=[_norm(c).replace(' ','_') for c in w.columns]
    ren={"name":"player","golfer":"player","r1":"round_score","r2":"round_score","r3":"round_score","score":"round_score","strokes":"round_score",
         "sg_app":"sg_approach","approach":"sg_approach","sg_ott":"sg_off_tee","ott":"sg_off_tee","sg_arg":"sg_around_green","arg":"sg_around_green","sg_putt":"sg_putting","putting":"sg_putting"}
    w=w.rename(columns={c:ren.get(c,c) for c in w.columns})
    if 'player' not in w: return pd.DataFrame()
    keep=[c for c in ['player','round','round_score','sg_total','sg_approach','sg_off_tee','sg_around_green','sg_putting','gir','birdies','bogeys'] if c in w]
    w=w[keep].copy()
    for c in keep:
        if c not in ('player',): w[c]=pd.to_numeric(w[c],errors='coerce')
    return w

def build_round_ratings(otis, live_results, round_no):
    """Round-specific latent strength. OTIS Rank/Model intentionally ignored."""
    o=otis.copy()
    if o.empty: return pd.DataFrame()
    o['key']=o['player'].map(_norm)
    # OTIS percentiles: True Skill anchors; Course Fit/Form are bounded tilts.
    ts=pd.to_numeric(o.get('otis_true_skill',50),errors='coerce').fillna(50)
    cf=pd.to_numeric(o.get('otis_course_fit',50),errors='coerce').fillna(50)
    fm=pd.to_numeric(o.get('otis_form',50),errors='coerce').fillna(50)
    pre=0.62*ts + 0.23*cf + 0.15*fm
    o['pre_round_rating']=pre.clip(1,99)
    o['live_form_adj']=0.0
    o['live_rounds']=0
    if round_no>1 and live_results is not None and not live_results.empty:
        lr=live_results.copy(); lr['key']=lr['player'].map(_norm)
        # Aggregate only completed prior rounds.
        if 'round' in lr and lr['round'].notna().any(): lr=lr[lr['round'] < round_no]
        rows=[]
        for k,g in lr.groupby('key'):
            n=len(g)
            # Prefer sustainable SG components. Putting is deliberately discounted.
            sustainable=None
            comp=[]
            for c,wgt in [('sg_approach',0.45),('sg_off_tee',0.25),('sg_around_green',0.15),('sg_putting',0.15)]:
                if c in g and g[c].notna().any(): comp.append((wgt,float(g[c].mean())))
            if comp:
                sw=sum(w for w,_ in comp); sustainable=sum(w*x for w,x in comp)/sw
            elif 'sg_total' in g and g.sg_total.notna().any(): sustainable=float(g.sg_total.mean())*0.65
            elif 'round_score' in g and g.round_score.notna().any():
                # Field-relative scoring fallback, computed later.
                sustainable=np.nan
            rows.append((k,n,sustainable,float(g['round_score'].mean()) if 'round_score' in g and g.round_score.notna().any() else np.nan))
        live=pd.DataFrame(rows,columns=['key','live_rounds','sustainable','avg_score'])
        if not live.empty:
            if live.sustainable.isna().any() and live.avg_score.notna().any():
                med=live.avg_score.median(); live.loc[live.sustainable.isna(),'sustainable']=(med-live.loc[live.sustainable.isna(),'avg_score'])*0.55
            # Shrink early tournament evidence. ~1.8 rating points per SG, capped.
            live['live_form_adj']=(live.sustainable.fillna(0)*1.8*np.minimum(live.live_rounds/2.5,1.0)).clip(-7,7)
            o=o.drop(columns=['live_form_adj','live_rounds']).merge(live[['key','live_rounds','live_form_adj']],on='key',how='left')
            o['live_rounds']=o.live_rounds.fillna(0).astype(int); o['live_form_adj']=o.live_form_adj.fillna(0)
    o['round_rating']=(o.pre_round_rating+o.live_form_adj).clip(1,99)
    return o[['player','key','pre_round_rating','live_form_adj','live_rounds','round_rating']]

def simulate_groups(groupings, ratings, n_sims=100000, seed=42, tie_band=0.22):
    rng=np.random.default_rng(seed); rmap=ratings.set_index('key').to_dict('index') if not ratings.empty else {}
    rows=[]; sim_store={}
    for _,g in groupings.iterrows():
        names=[str(g.player1),str(g.player2),str(g.player3)]; keys=[_norm(x) for x in names]
        if any(k not in rmap for k in keys):
            missing=[n for n,k in zip(names,keys) if k not in rmap];
            for n in missing: rows.append({'group':g['group'],'tee_time':g.get('tee_time',''),'player':n,'status':'UNMATCHED'})
            continue
        rt=np.array([rmap[k]['round_rating'] for k in keys],float)
        # Convert rating edge to strokes; common group/weather shock cancels head-to-head but preserves coherent round worlds.
        mu=71.0-(rt-50.0)*0.055
        common=rng.normal(0,0.85,n_sims)
        indiv=rng.normal(0,2.55,(n_sims,3))
        scores=mu[None,:]+common[:,None]+indiv
        # Continuous tie band approximates integer-score ties without discretizing all latent performance.
        mins=scores.min(axis=1)
        for j,n in enumerate(names):
            diff=scores[:,j]-mins
            tied=(diff<=tie_band) & ((scores<=mins[:,None]+tie_band).sum(axis=1)>1)
            win=(scores[:,j] < np.min(np.delete(scores,j,axis=1),axis=1)-tie_band)
            loss=~win & ~tied
            rows.append({'group':int(g['group']),'tee_time':g.get('tee_time',''),'player':n,'status':'OK','round_rating':rt[j],
                         'win_pct':100*win.mean(),'push_pct':100*tied.mean(),'loss_pct':100*loss.mean(),'non_loss_pct':100*(win|tied).mean(),
                         'sim_mean_score':scores[:,j].mean(),'sim_sd':scores[:,j].std()})
        sim_store[int(g['group'])]=(names,scores)
    return pd.DataFrame(rows),sim_store

def build_six_leg_tickets(probs, ticket_count=5, max_player_overlap=4):
    p=probs[probs.status.eq('OK')].copy()
    if p.empty: return pd.DataFrame(),pd.DataFrame()
    # One pick per group: highest win probability. Candidate alternates remain visible in probabilities table.
    best=p.sort_values(['group','win_pct'],ascending=[True,False]).groupby('group',as_index=False).head(1)
    if len(best)<6: return pd.DataFrame(),pd.DataFrame()
    # Candidate pool emphasizes edge but creates diversified 6-leg combinations.
    best=best.sort_values('win_pct',ascending=False).reset_index(drop=True)
    candidates=[]
    top=best.head(min(14,len(best)))
    from itertools import combinations
    for idxs in combinations(range(len(top)),6):
        sub=top.iloc[list(idxs)]
        # Push-aware full-ticket probability: all legs avoid loss; strict 6/6 separately.
        strict=float(np.prod(sub.win_pct.values/100))
        survive=float(np.prod(sub.non_loss_pct.values/100))
        weakest=float(sub.win_pct.min())
        score=0.62*math.log(max(strict,1e-12))+0.28*math.log(max(survive,1e-12))+0.10*(weakest/100)
        candidates.append((score,strict,survive,weakest,tuple(sub.player),tuple(sub.group)))
    candidates.sort(reverse=True,key=lambda x:x[0])
    chosen=[]
    for c in candidates:
        s=set(c[4])
        if all(len(s & set(x[4]))<=max_player_overlap for x in chosen): chosen.append(c)
        if len(chosen)>=ticket_count: break
    summary=[]; legs=[]
    for t,c in enumerate(chosen,1):
        summary.append({'ticket':t,'strict_6_of_6_pct':100*c[1],'all_legs_non_loss_pct':100*c[2],'weakest_leg_win_pct':c[3],'legs':6})
        for leg,(player,grp) in enumerate(zip(c[4],c[5]),1):
            row=p[(p.player==player)&(p.group==grp)].iloc[0]
            legs.append({'ticket':t,'leg':leg,'group':grp,'player':player,'win_pct':row.win_pct,'push_pct':row.push_pct,'loss_pct':row.loss_pct,'tee_time':row.tee_time})
    return pd.DataFrame(summary),pd.DataFrame(legs)

def fetch_live_results(source_url, round_no):
    """Best-effort public leaderboard ingestion. Returns prior-round scores when exposed as HTML tables."""
    if not source_url or not source_url.strip(): return pd.DataFrame(),"No leaderboard URL supplied"
    try:
        html=fetch_text(source_url.strip())
        tabs=pd.read_html(StringIO(html))
        frames=[]
        for t in tabs:
            t.columns=[_norm(c).replace(' ','_') for c in t.columns]
            pcol=next((c for c in t.columns if c in ('player','name','golfer') or 'player' in c),None)
            if not pcol: continue
            for r in range(1,round_no):
                rcol=next((c for c in t.columns if c in (f'r{r}',f'round_{r}',f'round{r}')),None)
                if rcol:
                    x=t[[pcol,rcol]].rename(columns={pcol:'player',rcol:'round_score'}).copy(); x['round']=r; frames.append(x)
        if frames:
            out=pd.concat(frames,ignore_index=True); out['round_score']=pd.to_numeric(out.round_score,errors='coerce'); out=out.dropna(subset=['player','round_score'])
            return out,"public leaderboard HTML"
        return pd.DataFrame(),"Leaderboard page did not expose prior-round score tables; upload fallback available"
    except Exception as e:
        return pd.DataFrame(),f"Leaderboard fetch failed: {e}"
