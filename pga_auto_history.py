"""Auto-ingest pre-lock PGA cuts and event-level SG from public TOUR-facing data.

Provider: pga-tour-unofficial-api. No generated season totals, no live events,
no fabricated dates/cuts/strokes-gained. Unknown facts remain missing. A
COMPLETED schedule row plus parsed schedule end is a *date proxy*, not an
independently certified event completion timestamp. Caller can inspect audit.

Provider calls can be slow; Streamlit should cache a collection per event.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import re
import pandas as pd
import numpy as np

from dated_event_discovery import _display_end

SG_IDS = {2567: "sg_ott", 2568: "sg_approach",
          2569: "sg_arg", 2564: "sg_putting"}
REQUIRED = {"player_id", "display_name", "position"}
COMPLETED = {"COMPLETED", "FINAL", "OFFICIAL", "COMPLETE"}
FINISHED = re.compile(r"^(?:T)?[1-9][0-9]*$")
MISSED = {"MC", "CUT", "MISSED CUT", "MISSED_CUT"}
SKIP = {"WD", "W/D", "DQ", "DNS", "MDF", "WITHDRAWN", "DISQUALIFIED"}
EVENT_ID = re.compile(r"^R20[0-9]{5}$")


def _key(value):
    import unicodedata
    x = unicodedata.normalize("NFKD", str(value or ""))
    x = "".join(c for c in x if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", x.casefold()).strip()


def _numeric(value):
    try:
        # PGA API total column may be strings, or an explicit numeric total_num.
        clean = str(value).strip().replace(",", "").replace("−", "-")
        if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", clean): return np.nan
        value = float(clean)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def validate_completed_events(schedule, cutoff, *, lookback=38):
    """Schedule COMPLETED AND an unambiguous pre-cutoff end date are required."""
    needed = {"tournament_id", "display_date", "status"}
    if schedule is None or schedule.empty or not needed.issubset(schedule):
        return pd.DataFrame(), ["PGA schedule missing required IDs, dates, or completion status"]
    cutoff = pd.Timestamp(cutoff).normalize()
    rows, skipped = [], []
    for rec in schedule.to_dict("records"):
        event_id = str(rec.get("tournament_id") or "")
        status = str(rec.get("status") or "").strip().upper()
        year = int(rec.get("year") or event_id[1:5]) if EVENT_ID.fullmatch(event_id) else cutoff.year
        end = _display_end(rec.get("display_date"), year)
        reason = ""
        if not EVENT_ID.fullmatch(event_id): reason = "invalid event id"
        elif status not in COMPLETED: reason = "not completed"
        elif pd.isna(end): reason = "ambiguous schedule end"
        elif pd.Timestamp(end).tz_localize(None).normalize() >= cutoff:
            reason = "at/after prediction lock"
        if reason:
            skipped.append(f"{event_id}: {reason}")
            continue
        rows.append({"event_id": event_id, "name": str(rec.get("tournament_name") or ""),
                     "end_date": pd.Timestamp(end).tz_localize(None).normalize(),
                     "event_status": status})
    events = pd.DataFrame(rows)
    if events.empty:return events, skipped
    events = events.sort_values("end_date", ascending=False).drop_duplicates("event_id")
    return events.head(int(lookback)).reset_index(drop=True), skipped


def normalize_leaderboard(leaderboard, event, *, min_field=60):
    """Return audited cut labels, never mistake no-cut fields for all-made-cuts."""
    if leaderboard is None or leaderboard.empty or not REQUIRED.issubset(leaderboard):
        return pd.DataFrame(), "leaderboard absent or missing identity/finish fields"
    if len(leaderboard) < min_field:
        return pd.DataFrame(), "field too small; no-cut or incomplete"
    if leaderboard["player_id"].isna().any() or leaderboard.player_id.duplicated().any():
        return pd.DataFrame(), "ambiguous player identities"
    if leaderboard.attrs.get("tournament_status") and str(
        leaderboard.attrs["tournament_status"]).upper() not in COMPLETED:
        return pd.DataFrame(), "live leaderboard state"
    x = leaderboard[["player_id", "display_name", "position"]].copy()
    x["position"] = x["position"].astype(str).str.upper().str.strip()
    finished = x.position.map(lambda v: bool(FINISHED.fullmatch(v)))
    missed = x.position.isin(MISSED)
    if finished.sum() < 35 or missed.sum() < 4:
        return pd.DataFrame(), "explicit missed cuts absent or leaderboard incomplete"
    x = x[finished | missed].copy()
    x["made_cut"] = finished.loc[x.index].astype(bool)
    x["player"] = x["display_name"].astype(str).str.strip()
    x["player_id"] = x["player_id"].astype(str)
    x["key"] = x["player"].map(_key)
    x["date"] = event["end_date"].strftime("%Y-%m-%d")
    x["event_id"] = event["event_id"]
    x["event_name"] = event["name"]
    rounds = []
    for col in ("round_1","round_2","round_3","round_4"):
        if col in leaderboard:
            vals=pd.to_numeric(leaderboard[col],errors="coerce")
            rounds.append(vals.between(55,100))
    if rounds:
        counts=pd.concat(rounds,axis=1).sum(axis=1)
        event_rounds=int(counts.max())
        x["completed_rounds"] = counts.loc[x.index].astype(int)
    else:
        event_rounds=0
        x["completed_rounds"] = np.nan
    x["scheduled_rounds"]=4
    x["abnormal_conditions"] = bool(event_rounds in (2,3))
    x["cut_eligible"]=True
    x["date_status"]="COMPLETED_SCHEDULE_END_PROXY"
    return x, "accepted"


def normalize_event_sg(raw, *, completed_rounds):
    """Accept only total aggregate -1 rows and all four distinct SG components.

    For four-category comparability across 3/4-round tournaments, per-round
    averages are derived from tournament totals / known rounds played.
    """
    if raw is None or raw.empty:
        return None
    required={"stat_id", "round", "total"}
    if not required.issubset(raw):
        return None
    if not 2 <= int(completed_rounds) <= 4:
        return None
    x=raw.copy()
    x["id"]=pd.to_numeric(x["stat_id"],errors="coerce")
    x=x[pd.to_numeric(x["round"],errors="coerce").eq(-1)&x["id"].isin(SG_IDS)].copy()
    if "section" in x:
        x=x[x["section"].astype(str).str.casefold().eq("strokesgained")]
    if x.empty:return None
    results={}
    for sid, name in SG_IDS.items():
        group=x[x["id"].eq(sid)]
        if group.empty:return None
        values=[]
        for rec in group.to_dict("records"):
            value=_numeric(rec.get("total_num"))
            if pd.isna(value):value=_numeric(rec.get("total"))
            if not pd.isna(value): values.append(float(value))
        if not values or max(values)-min(values)>1e-6:return None
        val=values[0]
        if abs(val)>35:return None  # guardrail against wrong units and year-to-date
        results[name]=round(val/int(completed_rounds),6)
    return results


def collect_pre_event_history(players, tournament_start, *,
                               provider=None, lookback=38, max_workers=4,
                               max_sg_calls=360, min_field=60):
    """Return season cutoff cut history, 2–3 recent event SG, and provenance audit.

    Collect across last/current schedule years. Only valid completed leaderboard
    rows contribute made cuts. Scorecard stats are sampled only for roster
    golfers, and only for their most recent three qualifying prior events.
    Calls are strictly bounded to protect upstream service.
    """
    if provider is None:
        import pga_tour_api as provider
    cutoff=pd.Timestamp(tournament_start).normalize()
    names=pd.DataFrame({"player":list(map(str,players))}).drop_duplicates("player")
    names["key"]=names.player.map(_key)
    if names.key.duplicated().any():
        raise ValueError("Ambiguous current-field golfer names; automatic matching blocked")
    if not (1 <= len(names) <= 300):
        raise ValueError("Expected a field of 1–300 golfers")
    years=(cutoff.year-1,cutoff.year)
    schedules=[]; errors=[]
    for year in years:
        try:
            schedule=provider.pga_schedule(year,"R")
            if schedule is not None and not schedule.empty:
                schedules.append(schedule.copy())
            else:errors.append(f"{year}: empty season schedule")
        except Exception as exc:
            errors.append(f"{year}: schedule {type(exc).__name__}: {str(exc)[:100]}")
    if not schedules:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(
            [{"stage":"schedule","status":"unavailable","detail":"; ".join(errors)}])
    events, rejects=validate_completed_events(pd.concat(schedules,ignore_index=True),
                                                cutoff,lookback=lookback)
    logs=[{"stage":"schedule","status":"excluded","detail":s} for s in rejects[:20]]
    logs.extend({"stage":"schedule","status":"error","detail":s} for s in errors)
    if events.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(logs)
    event_rows=[]; allowed=set(names.key)
    for event in events.to_dict("records"):
        try:
            lb=provider.pga_leaderboard(event["event_id"])
            frame,status=normalize_leaderboard(lb,event,min_field=min_field)
            if frame.empty:
                logs.append({"stage":"leaderboard","status":"skipped",
                             "detail":f"{event['event_id']}: {status}"})
                continue
            matches=frame[frame.key.isin(allowed)].copy()
            event_rows.append(matches)
            logs.append({"stage":"leaderboard","status":"accepted",
                         "detail":f"{event['event_id']}: matched {len(matches)} of {len(frame)} cuts/finishes"})
        except Exception as exc:
            logs.append({"stage":"leaderboard","status":"error",
                         "detail":f"{event['event_id']}: {type(exc).__name__}: {str(exc)[:110]}"})
    if not event_rows:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(logs)
    hits=pd.concat(event_rows,ignore_index=True)
    # Reject colliding names/IDs rather than guess which golfer matches.
    ambiguities=hits.groupby("key").player_id.nunique()
    bad=set(ambiguities[ambiguities.gt(1)].index)
    if bad:
        hits=hits[~hits.key.isin(bad)]
        logs.append({"stage":"identity","status":"excluded",
                     "detail":f"{len(bad)} ambiguous golfer identity matches"})
    # Include only current-season starts in historical cut rate, matching
    # typical published season cuts-made/starts. Prior-year events only feed SG.
    season=hits[pd.to_datetime(hits.date).dt.year.eq(cutoff.year)].copy()
    snapshot=(season.groupby("key",as_index=False)
              .agg(events_played=("made_cut","size"),cuts_made=("made_cut","sum"),
                   as_of=("date","max")))
    snapshot=snapshot.merge(names,on="key",how="left")
    snapshot=snapshot[["player","events_played","cuts_made","as_of"]]
    # Build a list of latest three prior events per golfer. Scorecard API can
    # be slow; cap calls, record coverage, cache full result at caller.
    ordered=hits.sort_values(["date","event_id"],ascending=[False,False])
    tasks=[]
    for key,group in ordered.groupby("key"):
        for row in group.head(3).itertuples():
            if int(row.completed_rounds) >= 2:
                tasks.append((row.event_id,row.player_id,row.key,row.player,
                              row.date,int(row.completed_rounds),bool(row.abnormal_conditions)))
    tasks=tasks[:max(0,int(max_sg_calls))]
    def get_stat(task):
        event_id,pid,key,player,dt,rounds,short=task
        try:
            raw=provider.pga_scorecard_stats(event_id,pid,round="-1")
            vals=normalize_event_sg(raw,completed_rounds=rounds)
            if vals is None:return None, f"{event_id}/{pid}: SG unavailable/incomplete"
            return {"player":player,"date":dt,"event_id":event_id,
                    "made_cut":None,"completed_rounds":rounds,
                    "scheduled_rounds":4,"abnormal_conditions":short,
                    "sg_source":"PGA_TOUR_EVENT_SCORECARD_AGGREGATE_PER_ROUND",
                    **vals}, None
        except Exception as exc:
            return None,f"{event_id}/{pid}: {type(exc).__name__}: {str(exc)[:90]}"
    # Request no more than four concurrent external queries. Ensure errors
    # do not drop the valid cut records from other events.
    sg_rows=[]
    with ThreadPoolExecutor(max_workers=max(1,min(int(max_workers),4))) as pool:
        futures=[pool.submit(get_stat,task) for task in tasks]
        for future in as_completed(futures):
            record,error=future.result()
            if record:sg_rows.append(record)
            elif error:logs.append({"stage":"strokes_gained","status":"unavailable","detail":error})
    sg=pd.DataFrame(sg_rows)
    if not sg.empty:
        sg=sg.sort_values(["player","date"],ascending=[True,False]).reset_index(drop=True)
    logs.append({"stage":"summary","status":"completed",
                 "detail":f"Cut evidence: {len(snapshot)} golfers; "
                          f"SG complete events: {len(sg)}; "
                          f"SG requests: {len(tasks)}; pre-lock schedule end proxy"})
    return snapshot,sg,pd.DataFrame(logs)
