import re, math, json, gzip, base64, os
from io import StringIO
import numpy as np
import pandas as pd
import requests

UA={"User-Agent":"Mozilla/5.0 (compatible; PGA-Parlay-Model/1.0)"}


PGA_GRAPHQL_URL="https://orchestrator.pgatour.com/graphql"
PGA_REST_URL="https://data-api.pgatour.com"
PGA_API_KEY_DEFAULT="da2-gsrx5bibzbb4njvhl7t37wqyl4"

def _pga_headers():
    return {
        "User-Agent": UA["User-Agent"], "Content-Type":"application/json",
        "Accept":"application/graphql-response+json, application/json",
        "x-api-key": os.environ.get("PGA_API_KEY") or PGA_API_KEY_DEFAULT,
        "x-pgat-platform":"web", "Origin":"https://www.pgatour.com",
        "Referer":"https://www.pgatour.com/"
    }

def _pga_graphql(query, variables, operation):
    body={"query":query,"variables":variables,"operationName":operation}
    last=None
    for _ in range(3):
        try:
            r=requests.post(PGA_GRAPHQL_URL,headers=_pga_headers(),json=body,timeout=30)
            r.raise_for_status(); obj=r.json()
            if obj.get("errors"): raise RuntimeError(str(obj["errors"][0].get("message",obj["errors"][0])))
            return obj.get("data") or {}
        except Exception as e: last=e
    raise last

def _decompress_payload(payload):
    raw=base64.b64decode(payload)
    try: raw=gzip.decompress(raw)
    except OSError: pass
    return json.loads(raw.decode("utf-8"))

def _find_tournament_id(tournament, year):
    # Official PGA TOUR schedule REST endpoint; recurse because the envelope has changed over time.
    r=requests.get(f"{PGA_REST_URL}/schedule/R/{int(year)}",headers={"User-Agent":UA["User-Agent"]},timeout=30)
    r.raise_for_status(); obj=r.json(); target=_norm(tournament)
    candidates=[]
    def walk(x):
        if isinstance(x,dict):
            vals=' '.join(str(v) for v in x.values() if isinstance(v,(str,int,float)))
            if target and target in _norm(vals):
                for k,v in x.items():
                    if isinstance(v,str) and re.fullmatch(r"R\d{7}",v): candidates.append(v)
                    if 'id' in str(k).casefold() and isinstance(v,str) and re.fullmatch(r"R\d{7}",v): candidates.append(v)
            for v in x.values(): walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(obj)
    return candidates[0] if candidates else ''

def _api_groupings(tournament, year, round_no, tournament_id=''):
    tid=str(tournament_id or '').strip()
    if not re.fullmatch(r"R\d{7}",tid):
        tid=_find_tournament_id(tournament,year)
    if not tid: return pd.DataFrame(),'', 'PGA TOUR API could not resolve tournament ID'
    q='query TeeTimesCompressedV2($teeTimesCompressedV2Id: ID!) { teeTimesCompressedV2(id: $teeTimesCompressedV2Id) { id payload } }'
    data=_pga_graphql(q,{"teeTimesCompressedV2Id":tid},"TeeTimesCompressedV2")
    node=data.get("teeTimesCompressedV2") or {}; payload=node.get("payload")
    if not payload: return pd.DataFrame(),tid,'PGA TOUR API returned no tee-time payload'
    parsed=_decompress_payload(payload); rec=[]
    for rnd in parsed.get("rounds") or []:
        if int(rnd.get("roundInt") or 0)!=int(round_no): continue
        for g in rnd.get("groups") or []:
            players=g.get("players") or []
            if len(players)!=3: continue
            names=[(x.get("displayName") or (str(x.get("firstName") or '')+' '+str(x.get("lastName") or '')).strip()).strip() for x in players]
            ts=g.get("teeTime")
            tee_time=pd.to_datetime(ts,unit='ms',utc=True).strftime('%Y-%m-%d %H:%M UTC') if ts else ''
            rec.append({"group":g.get("groupNumber"),"tee_time":tee_time,"tee":g.get("startTee",''),
                        "player1":names[0],"player2":names[1],"player3":names[2],"round":round_no})
    out=_validate_groups(pd.DataFrame(rec),round_no)
    return out,tid,(f'official PGA TOUR TeeTimes API ({len(out)} groups)' if not out.empty else f'PGA TOUR API has no valid R{round_no} groups yet')

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

def _validate_groups(df, round_no):
    """Accept only plausible 3-player groups; never silently guess pairings."""
    if df is None or df.empty:
        return pd.DataFrame()
    need=['player1','player2','player3']
    if not set(need).issubset(df.columns):
        return pd.DataFrame()
    w=df.copy()
    for c in need:
        w[c]=w[c].astype(str).str.strip()
    w=w[(w[need].apply(lambda r: all(2 <= len(x) <= 55 for x in r),axis=1)) &
        (w[need].apply(lambda r: len(set(x.casefold() for x in r))==3,axis=1))].copy()
    if w.empty: return w
    w=w.drop_duplicates(subset=need).reset_index(drop=True)
    w['group']=range(1,len(w)+1); w['round']=round_no
    if 'tee_time' not in w: w['tee_time']=''
    if 'tee' not in w: w['tee']=''
    return w[['group','tee_time','tee','player1','player2','player3','round']]

def _parse_pgatour_tables(html, round_no):
    """Parse PGA TOUR public tee-time HTML tables. Handles a Players cell containing 3 names."""
    try: tabs=pd.read_html(StringIO(html))
    except Exception: return pd.DataFrame()
    for t in tabs:
        cols={_norm(c):c for c in t.columns}
        time_col=next((v for k,v in cols.items() if k=='time' or 'time' in k),None)
        players_col=next((v for k,v in cols.items() if 'players' in k),None)
        tee_col=next((v for k,v in cols.items() if k=='tee' or 'tee' in k),None)
        if time_col is None or players_col is None: continue
        rec=[]
        for _,row in t.iterrows():
            raw=str(row[players_col]).strip()
            # PGA rendered tables often repeat country/name tokens; extract name-like chunks.
            raw=re.sub(r'Image:[^A-Z]*',' ',raw)
            raw=re.sub(r'\b(?:USA|JPN|CAN|ENG|AUS|KOR|RSA|SWE|DEN|NOR|IRL|SCO|ESP|FRA|GER|ARG|CHI|MEX|NZL|BEL|ITA|COL|TPE|CHN)\b',' ',raw)
            names=[]
            # Prefer separators if pandas preserved them.
            for part in re.split(r'\s{2,}|\||;|\n',raw):
                part=re.sub(r'\s+',' ',part).strip(' ,')
                if re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ .’'\-]+$",part) and ' ' in part and 3<=len(part)<=55:
                    if not names or _norm(part)!=_norm(names[-1]): names.append(part)
            if len(names)!=3:
                continue
            rec.append({'tee_time':str(row[time_col]).strip(),'tee':str(row[tee_col]).strip() if tee_col else '',
                        'player1':names[0],'player2':names[1],'player3':names[2]})
        out=_validate_groups(pd.DataFrame(rec),round_no)
        if not out.empty: return out
    return pd.DataFrame()

def fetch_groupings(tournament, year, round_no, source_url="", tournament_id=""):
    """Official PGA TOUR TeeTimes API first; rendered pages/articles are fallbacks."""
    errors=[]
    try:
        df,tid,note=_api_groupings(tournament,year,round_no,tournament_id)
        if not df.empty: return df,f"PGA TOUR API {tid}",note
        errors.append(note)
    except Exception as e:
        errors.append(f"PGA TOUR API: {e}")

    urls=[]
    if source_url.strip(): urls.append(source_url.strip())
    urls.append(golfchannel_round_url(tournament,year,round_no))
    for u in dict.fromkeys(urls):
        try:
            html=fetch_text(u)
            if 'pgatour.com' in u:
                df=_parse_pgatour_tables(html,round_no)
                if not df.empty: return df,u,f'PGA TOUR rendered table ({len(df)} groups)'
            df=_validate_groups(parse_groupings_text(html,round_no),round_no)
            if not df.empty: return df,u,f'validated article text ({len(df)} groups)'
            errors.append(f'No valid 3-player groups parsed from {u}')
        except Exception as e: errors.append(f'{u}: {e}')
    return pd.DataFrame(),'', '; '.join(errors[-4:])

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

def _baycurrent_independent_fit(o):
    """Independent Yokohama fit from available OTIS skill components.

    Only course-specific components we actually have are used.  The researched
    2026 DNA assigns 22 APP / 15 OTT / 5 ARG / 5 PUTT points to these inputs;
    they are renormalized across the available 47 points rather than inventing
    unavailable par-4, driving-distance, accuracy or bogey-avoidance data.
    """
    specs=[('otis_fit_app',22.0),('otis_fit_ott',15.0),('otis_fit_arg',5.0),('otis_fit_putt',5.0)]
    num=pd.Series(0.0,index=o.index); den=pd.Series(0.0,index=o.index)
    for col,w in specs:
        if col not in o.columns: continue
        x=pd.to_numeric(o[col],errors='coerce')
        ok=x.notna()
        num.loc[ok] += w*x.loc[ok]
        den.loc[ok] += w
    fallback=pd.to_numeric(o.get('otis_course_fit',50),errors='coerce').fillna(50)
    return (num/den.replace(0,np.nan)).fillna(fallback).clip(1,99)

def build_round_ratings(otis, live_results, round_no, tournament_name='', use_course_dna=True):
    """Round-specific latent strength. OTIS Rank/Model intentionally ignored."""
    o=otis.copy()
    if o.empty: return pd.DataFrame()
    o['key']=o['player'].map(_norm)
    # OTIS percentiles: True Skill anchors; course fit/form are bounded tilts.
    ts=pd.to_numeric(o.get('otis_true_skill',50),errors='coerce').fillna(50)
    generic_cf=pd.to_numeric(o.get('otis_course_fit',50),errors='coerce').fillna(50)
    is_baycurrent=('baycurrent' in str(tournament_name).casefold() or 'yokohama' in str(tournament_name).casefold())
    cf=_baycurrent_independent_fit(o) if (is_baycurrent and use_course_dna) else generic_cf
    fm=pd.to_numeric(o.get('otis_form',50),errors='coerce').fillna(50)
    pre=0.62*ts + 0.23*cf + 0.15*fm
    o['course_fit_signal']=cf
    o['course_fit_source']='Independent Yokohama DNA' if (is_baycurrent and use_course_dna) else 'OTIS Course Fit'
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
    return o[['player','key','pre_round_rating','course_fit_signal','course_fit_source','live_form_adj','live_rounds','round_rating']]

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
