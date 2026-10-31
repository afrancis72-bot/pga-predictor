"""PGA pre-event reliability features, provenance and leakage protection.

Cut consistency and last-event four-category SG breadth are explicit research
features. Bounded adjustments are provisional; do NOT claim calibrated edge.
Only pre-event rows or timestamped season snapshots are eligible for forecasts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import re

SG = ("sg_ott", "sg_approach", "sg_arg", "sg_putting")
_ALIASES = {
    "sg_off_the_tee": "sg_ott", "sg_offtee": "sg_ott", "sg_off_tee": "sg_ott",
    "sg_around_the_green": "sg_arg", "sg_around_green": "sg_arg",
    "sg_approach_the_green": "sg_approach", "sg_app": "sg_approach",
    "sg_putt": "sg_putting", "sg_putts": "sg_putting",
    "event_date": "date", "start_date": "date", "tournament_date": "date",
    "player_name": "player", "name": "player", "golfer": "player",
    "cuts": "cuts_made", "starts": "events_played",
    "snapshot_date": "as_of", "stats_as_of": "as_of",
    "event_id": "event_id", "tournament_id": "event_id",
    "tournament": "event_id", "event": "event_id",
}

def _key(name):
    return re.sub(r"[^a-z0-9]+", " ", str(name).casefold()).strip()

def _clean(df):
    if df is None or df.empty:
        return pd.DataFrame()
    w = df.copy()
    ren = {}
    for c in w.columns:
        clean = re.sub(r"\s+", "_", str(c).strip().casefold())
        ren[c] = _ALIASES.get(clean, clean)
    w = w.rename(columns=ren)
    if w.columns.duplicated().any():
        raise ValueError("Ambiguous duplicate columns in historical PGA evidence.")
    if "player" in w:
        w["key"] = w["player"].map(_key)
        w = w[w["key"].ne("")].copy()
    return w

def _parse_cut(x):
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)) and x in (0, 1):
        return bool(x)
    t = str(x).strip().upper()
    if t in {"1", "YES", "Y", "TRUE", "MADE", "MADE CUT", "MADE_CUT"}: return True
    if t in {"0", "NO", "N", "FALSE", "MC", "CUT", "MISSED CUT", "MISSED_CUT", "MDF"}: return False
    return np.nan

def build_pre_event_evidence(players, season_stats=None, event_results=None,
                             tournament_start=None, prior_starts=8,
                             typical_cut_rate=.62):
    """Player keyed evidence; single-date cutoff blocks use of future outcomes.

    For season stats, an as_of snapshot timestamp is REQUIRED. Event history
    needs dated rows and a made_cut field. The latest complete event supplies
    SG breadth; missing stats never count as positive or negative.
    """
    if tournament_start is None:
        raise ValueError("Tournament start date required for historical evidence safety.")
    cutoff = pd.Timestamp(tournament_start)
    if cutoff.tzinfo is not None: cutoff = cutoff.tz_localize(None)
    cutoff = cutoff.normalize()
    if not 0 < typical_cut_rate < 1 or prior_starts < 0:
        raise ValueError("Invalid cut reliability prior.")
    if isinstance(players, pd.DataFrame): names=players["player"].astype(str).tolist()
    else: names=list(map(str,players))
    out = pd.DataFrame({"player":names})
    out["key"]=out["player"].map(_key)
    if out["key"].duplicated().any():
        raise ValueError("Duplicated player identities in the current field.")
    a = _clean(season_stats)
    e = _clean(event_results)
    cut_history = {}
    last_event = {}

    if not e.empty and {"key","date"}.issubset(e):
        e["date"] = pd.to_datetime(e["date"], errors="coerce", utc=True).dt.tz_localize(None)
        e = e[e["date"].notna() & (e["date"] < cutoff)].copy()
        e["event_key"] = e["event_id"].astype(str) if "event_id" in e else e["date"].dt.strftime("%Y-%m-%d")
        # A row denotes a player/event, not a round. If multiple records
        # represent the latest tournament, preserve its date but refuse to
        # mistake any one round for full-tournament SG.
        duplicated = e.duplicated(["key", "event_key"], keep=False)
        is_round = pd.to_numeric(e["round"], errors="coerce").notna() if "round" in e else pd.Series(False,index=e.index)
        unreliable_sg = duplicated | is_round
        for c in SG:
            if c in e: e.loc[unreliable_sg,c] = np.nan
        e = e.drop_duplicates(["key", "event_key"], keep="first")
        if "made_cut" in e:
            e["cut_bool"] = e["made_cut"].map(_parse_cut)
            valid = e[e["cut_bool"].notna()].copy()
            valid["year"] = valid["date"].dt.year
            valid = valid[valid["year"] == cutoff.year]
            for key, grp in valid.groupby("key"):
                cut_history[key] = (int(grp["cut_bool"].astype(int).sum()), len(grp))
        # Only a complete and dated last EVENT with four numeric SG components
        # can earn an all-four-positive flag. A single round doesn't qualify.
        for key, grp in e.sort_values("date", ascending=False).groupby("key",sort=False):
            row = grp.iloc[0]
            values = [pd.to_numeric(row.get(c),errors="coerce") for c in SG]
            complete = bool(np.all(np.isfinite(values))) if len(values)==4 else False
            last_event[key] = {
                "last_event_date": str(pd.Timestamp(row["date"]).date()),
                "last_event_name": str(row.get("event_id", "")).strip(),
                "last_event_sg_complete": complete,
                "last_event_positive_categories": int(sum(float(v)>0 for v in values)) if complete else np.nan,
                "last_event_all_four_positive": bool(all(float(v)>0 for v in values)) if complete else pd.NA,
                **{f"last_event_{c}": float(v) if np.isfinite(v) else np.nan for c,v in zip(SG,values)},
            }

    season_cuts = {}
    if not a.empty and {"key","cuts_made","events_played","as_of"}.issubset(a):
        a["as_of"] = pd.to_datetime(a["as_of"],errors="coerce",utc=True).dt.tz_localize(None)
        # Explicit as-of field; reject future snapshots or unknown year.
        a = a[a["as_of"].notna() & (a["as_of"] < cutoff)]
        if "season" in a: a = a[pd.to_numeric(a["season"],errors="coerce").eq(cutoff.year)]
        a = a.sort_values("as_of",ascending=False).drop_duplicates("key")
        for row in a.itertuples():
            made = pd.to_numeric(row.cuts_made,errors="coerce")
            starts = pd.to_numeric(row.events_played,errors="coerce")
            if np.isfinite(made) and np.isfinite(starts) and 0<=made<=starts and starts>=1 and float(starts).is_integer() and float(made).is_integer():
                season_cuts[row.key]=(int(made),int(starts),str(pd.Timestamp(row.as_of).date()))

    recs = []
    for key in out["key"]:
        row = {"key":key, "season_cuts_made": np.nan,
               "season_starts": np.nan,"historical_cut_pct": np.nan,
               "shrunk_cut_pct": np.nan,"cut_record_as_of":"","cut_evidence_source":"missing",
               "last_event_date":"","last_event_name":"","last_event_sg_complete":False,
               "last_event_positive_categories":np.nan,"last_event_all_four_positive":pd.NA}
        row.update({f"last_event_{c}": np.nan for c in SG})
        if key in season_cuts:
            made,starts,vintage = season_cuts[key]
            row["cut_record_as_of"] = vintage
            row["cut_evidence_source"] = "dated season snapshot"
        elif key in cut_history:
            made,starts = cut_history[key]
            row["cut_evidence_source"] = "dated event-level cut records"
            row["cut_record_as_of"] = "before tournament"
        else:
            made,starts = None,None
        if starts is not None:
            row["season_cuts_made"],row["season_starts"]=made,starts
            row["historical_cut_pct"] = 100. * made/starts
            row["shrunk_cut_pct"]=100. * (made+prior_starts*typical_cut_rate)/(starts+prior_starts)
        if key in last_event: row.update(last_event[key])
        recs.append(row)
    return out.merge(pd.DataFrame(recs),on="key",how="left",validate="one_to_one").drop(columns="key")


def snapshot_last_events(snapshot):
    """Convert ONE optional weekly snapshot into complete previous-event SG rows."""
    w=_clean(snapshot)
    required={"player","last_event_date","last_event_sg_ott","last_event_sg_approach",
              "last_event_sg_arg","last_event_sg_putting"}
    if not required.issubset(w):
        return pd.DataFrame()
    ren={"last_event_date":"date", "last_event_name":"event_id",
         **{f"last_event_{c}":c for c in SG}}
    cols=list(required)+(["last_event_name"] if "last_event_name" in w else [])
    result=w[cols].rename(columns=ren).copy()
    if "event_id" not in result: result["event_id"]=result["date"]
    return result

def apply_reliability_research(projections, evidence, *, no_cut=False):
    """Bounded, explicit exploratory adjustments to expected strokes.

    Center cut history on the comparable evidence-covered field; do not
    inject fixed strength from average TOUR cut rates into a different field.
    No evidence -> unchanged forecast. No-cut -> cut consistency not applied.
    The coefficients require future walk-forward calibration.
    """
    if "expected_round" not in projections:
        raise ValueError("Expected round input missing.")
    d=projections.merge(evidence,on="player",how="left",validate="one_to_one")
    before=pd.to_numeric(d["expected_round"],errors="coerce")
    if before.isna().any():
        raise ValueError("Expected round input includes invalid values.")
    rate=pd.to_numeric(d.get("shrunk_cut_pct"),errors="coerce")
    starts=pd.to_numeric(d.get("season_starts"),errors="coerce")
    qualified=rate[starts.ge(8) & rate.notna()]
    cut_strokes=pd.Series(0.,index=d.index)
    if not no_cut and len(qualified)>=10:
        reference=float(qualified.median())
        trust=(starts/24.).clip(0.,1.).fillna(0.)
        cut_strokes=(-(rate-reference)*0.008*trust).clip(-.18,.18).fillna(0.)
    # A strong single week is NOT enough to change a forecast. Two completed,
    # comparable events with verified OTT/APP/ARG/PUTT observations are required
    # before the 2–3 event stability feature affects expected strokes.
    count=pd.to_numeric(d.get("recent_rolling_positive_categories",pd.Series(np.nan,index=d.index)),errors="coerce")
    n_events=pd.to_numeric(d.get("recent_complete_sg_events",pd.Series(np.nan,index=d.index)),errors="coerce")
    confirmed=n_events.ge(2) & count.notna()
    # Conservative bounded correction (less than 0.10 strokes / round).
    # Heuristic coefficients MUST be replaced by walk-forward fitted evidence.
    breadth_strokes=count.map({0:.06,1:.03,2:0.,3:-.025,4:-.075}).where(confirmed,0.).fillna(0.)
    d["expected_round_pre_reliability"]=before
    d["cut_history_strokes_adjustment"]=cut_strokes
    d["four_category_strokes_adjustment"]=breadth_strokes
    d["multi_event_form_source"]="two-plus complete prior events"
    d["expected_round"]=(before+cut_strokes+breadth_strokes).clip(-6.,6.)
    d["reliability_strokes_adjustment"]=d["expected_round"]-before
    d["cut_reference_covered_field_pct"]=float(qualified.median()) if len(qualified)>=10 else np.nan
    d["reliability_input_coverage"]=(
        (starts.ge(8) & rate.notna()).astype(int) +
        confirmed.astype(int)
    )
    return d
