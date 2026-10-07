from pathlib import Path
from datetime import date
import pandas as pd
import re
import streamlit as st

from international_ingestion import search_courses, geocode_course, location_detail, fetch_course_detail, fetch_course_holes, fetch_weather, IngestionError
from portfolio_optimizer import PortfolioSettings, optimize_portfolio
from pga_predictor_pro import Config, predict_from_dataframes
from round_parlay import fetch_groupings, normalize_groupings_upload, fetch_live_results, normalize_live_results, build_round_ratings, simulate_groups, build_six_leg_tickets, golfchannel_round_url

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="PGA Predictor Pro", page_icon="⛳", layout="wide")
st.title("⛳ PGA Predictor Pro — Tournament Model V10.7.0")
st.caption("$0 course/weather ingestion + DraftKings field + OTIS Advanced Course-Fit player layer")

@st.cache_data
def load_repo_csv(name):
    path = ROOT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load_weekly(upload, name):
    return pd.read_csv(upload) if upload is not None else load_repo_csv(name)

def validate_advanced_input(df, kind):
    """Validate optional advanced inputs without changing model logic.

    Repository fallbacks that do not satisfy the current contract are ignored
    instead of crashing a weekly build. Explicit uploads remain visible errors.
    """
    if df is None or df.empty:
        return pd.DataFrame(), "not supplied"

    work = df.copy()
    work.columns = [str(c).replace("\ufeff", "").strip() for c in work.columns]
    lower = {c.casefold(): c for c in work.columns}

    required = {
        "player_stats": {"player"},
        "results": {"player", "date"},
        "course_history": {"player"},
    }[kind]

    missing = [c for c in required if c not in lower]
    if missing:
        return pd.DataFrame(), "incompatible: missing " + ", ".join(missing)

    # Normalize required identity/date column casing when necessary.
    ren = {}
    for req in required:
        actual = lower[req]
        if actual != req:
            ren[actual] = req
    if ren:
        work = work.rename(columns=ren)

    if "player" in work.columns:
        work["player"] = work["player"].astype(str).str.strip()
        work = work[~work["player"].str.casefold().isin({"", "nan", "none", "player", "name"})].copy()

    if kind == "results":
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work[work["date"].notna()].copy()
        if work.empty:
            return pd.DataFrame(), "incompatible: no valid dated result rows"

    return work.reset_index(drop=True), "valid"


def _clean_cols(df):
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def normalize_otis_course_fit(df):
    """Normalize OTIS Advanced Course-Fit export for the current weekly field.

    Rank and Model are retained for audit only and are never used as predictive inputs.
    OTIS component scores are within-field percentiles, so they are used only within
    the current tournament field. Missing values remain missing.
    """
    if df is None or df.empty:
        return pd.DataFrame(), "empty"
    w = _clean_cols(df)
    lookup = {c.casefold(): c for c in w.columns}
    required = ["player", "true skill", "course fit", "form"]
    missing = [c for c in required if c not in lookup]
    if missing:
        return pd.DataFrame(), "unrecognized OTIS Course-Fit CSV (missing " + ", ".join(missing) + ")"
    ren = {
        lookup["player"]: "player",
        lookup["true skill"]: "otis_true_skill",
        lookup["course fit"]: "otis_course_fit",
        lookup["form"]: "otis_form",
    }
    optional = {
        "rank": "otis_rank", "model": "otis_model_audit",
        "fit: app": "otis_fit_app", "fit: ott": "otis_fit_ott",
        "fit: arg": "otis_fit_arg", "fit: putt": "otis_fit_putt",
        "fit: history": "otis_fit_history", "form rds": "otis_form_rds",
        "venue rds": "otis_venue_rds",
    }
    for src, dst in optional.items():
        if src in lookup: ren[lookup[src]] = dst
    w = w.rename(columns=ren)
    keep = [c for c in ["player", "otis_rank", "otis_model_audit", "otis_true_skill", "otis_course_fit", "otis_form", "otis_fit_app", "otis_fit_ott", "otis_fit_arg", "otis_fit_putt", "otis_fit_history", "otis_form_rds", "otis_venue_rds"] if c in w.columns]
    w = w[keep].copy()
    w["player"] = w["player"].astype(str).str.strip()
    for c in keep:
        if c != "player": w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[~w["player"].str.casefold().isin({"", "nan", "none", "player", "name"})].copy()
    w = w.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True)
    advanced = all(c in w.columns for c in ["otis_fit_app","otis_fit_ott","otis_fit_arg","otis_fit_putt","otis_fit_history","otis_form_rds","otis_venue_rds"])
    return w, ("OTIS Advanced Course-Fit" if advanced else "OTIS Course-Fit")

def normalize_otis_season_stats(df, target_season=2026):
    """Normalize an OTIS Season stats CSV into player_stats.csv-compatible rows."""
    if df is None or df.empty:
        return pd.DataFrame(), "empty"
    w = _clean_cols(df)
    lookup = {c.casefold(): c for c in w.columns}
    # OTIS documents player + season and the stat slugs below. Be tolerant of name/player_name.
    pcol = next((lookup[x] for x in ("player","player_name","name") if x in lookup), None)
    scol = next((lookup[x] for x in ("season","year") if x in lookup), None)
    if pcol is None:
        return pd.DataFrame(), "unrecognized OTIS season stats (missing player/name)"
    if scol is not None:
        season_text = w[scol].astype(str).str.strip()
        mask = season_text.eq(str(target_season))
        if mask.any():
            w = w[mask].copy()
    if pcol != "player": w = w.rename(columns={pcol:"player"})
    w["player"] = w["player"].astype(str).str.strip()
    # Keep OTIS native stat names; build_features already consumes published stat columns when present.
    numeric = [
        "sg_total","sg_off_the_tee","sg_approach","sg_around_green","sg_putting","sg_tee_to_green",
        "measured_rounds","driving_distance","driving_accuracy_pct","gir_pct","scrambling_pct",
        "putts_per_round","scoring_avg","birdie_avg","par3_scoring_avg","par4_scoring_avg",
        "par5_scoring_avg","events_played","cuts_made","top10s","wins","earnings_usd"
    ]
    for c in numeric:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[~w["player"].str.casefold().isin({"","nan","none","player","name"})].copy()
    return w.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True), "OTIS season stats"

def normalize_otis_event_results(df):
    """Normalize an OTIS Event results CSV into results.csv-compatible rows."""
    if df is None or df.empty:
        return pd.DataFrame(), "empty"
    w = _clean_cols(df)
    lookup = {c.casefold(): c for c in w.columns}
    pcol = next((lookup[x] for x in ("player","player_name","name") if x in lookup), None)
    dcol = next((lookup[x] for x in ("date","event_date","start_date","tournament_date") if x in lookup), None)
    if pcol is None:
        return pd.DataFrame(), "unrecognized OTIS event results (missing player/name)"
    if dcol is None:
        return pd.DataFrame(), "unrecognized OTIS event results (missing date)"
    ren={}
    if pcol!="player": ren[pcol]="player"
    if dcol!="date": ren[dcol]="date"
    if ren: w=w.rename(columns=ren)
    w["player"]=w["player"].astype(str).str.strip()
    w["date"]=pd.to_datetime(w["date"], errors="coerce")
    w=w[w["date"].notna() & ~w["player"].str.casefold().isin({"","nan","none","player","name"})].copy()
    return w.reset_index(drop=True), "OTIS event results"

def build_course_history_from_otis(results_df, course_name):
    """Derive current-venue history from OTIS event results; never invent missing history."""
    if results_df is None or results_df.empty or not str(course_name).strip():
        return pd.DataFrame(), "not available"
    w=results_df.copy()
    lookup={str(c).casefold():c for c in w.columns}
    ccol=next((lookup[x] for x in ("course","course_name","venue") if x in lookup), None)
    if ccol is None:
        return pd.DataFrame(), "OTIS results have no course column"
    def norm(x):
        return "".join(ch for ch in str(x).casefold() if ch.isalnum())
    target=norm(course_name)
    vals=w[ccol].astype(str)
    exact=vals.map(norm).eq(target)
    if not exact.any():
        # Conservative token containment fallback for branding variants.
        tokens=[t for t in re.split(r"\W+", str(course_name).casefold()) if len(t)>=4 and t not in {"golf","course","club"}]
        if tokens:
            exact=vals.str.casefold().map(lambda x: all(t in x for t in tokens))
    h=w[exact].copy()
    if h.empty:
        return pd.DataFrame(), "no OTIS starts matched current course"
    # build_features accepts a player-keyed history table; preserve published OTIS fields.
    return h.reset_index(drop=True), f"{len(h)} OTIS course-history starts"

def normalize_dk_players(df):
    """Accept native DraftKings PGA salary exports or the app's normalized players.csv."""
    if df is None or df.empty:
        return pd.DataFrame(), "empty"

    work = df.copy()
    # Strip BOM/whitespace that can appear in downloaded CSV headers.
    work.columns = [str(c).replace("\ufeff", "").strip() for c in work.columns]
    lookup = {str(c).strip().casefold(): c for c in work.columns}

    # Already-normalized weekly input.
    if "player" in lookup and "salary" in lookup:
        player_col, salary_col = lookup["player"], lookup["salary"]
        source = "normalized"
    else:
        # Native DraftKings PGA exports normally use Name + Salary.
        player_col = None
        for candidate in ("name", "player name", "golfer", "player"):
            if candidate in lookup:
                player_col = lookup[candidate]
                break
        salary_col = None
        for candidate in ("salary", "dk salary"):
            if candidate in lookup:
                salary_col = lookup[candidate]
                break
        if player_col is None or salary_col is None:
            return work, "unrecognized"
        source = "DraftKings native"

    work["player"] = work[player_col].astype(str).str.strip()
    work["salary"] = pd.to_numeric(
        work[salary_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )

    # Remove blank/header-like/bad rows while preserving all useful DK columns.
    bad_names = {"", "nan", "none", "name", "player"}
    work = work[
        work["player"].str.casefold().notna()
        & ~work["player"].str.casefold().isin(bad_names)
        & work["salary"].notna()
        & (work["salary"] > 0)
    ].copy()
    work["salary"] = work["salary"].astype(int)

    # DraftKings native salary exports can flag withdrawn/inactive golfers in a Status column.
    # Treat OUT as a hard exclusion before simulation so an inactive golfer can never reach
    # projections or the optimizer. Preserve every other status unchanged.
    status_col = next((lookup[k] for k in ("status", "player status") if k in lookup), None)
    if status_col is not None:
        status = work[status_col].fillna("").astype(str).str.strip().str.casefold()
        work = work[~status.isin({"out"})].copy()

    # A native salary file should contain one row per golfer; protect the model from duplicates.
    work = work.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True)
    return work, source

@st.cache_data(ttl=3600)
def cached_course_search(query, api_key, country_hint):
    return search_courses(query, api_key, country_hint)

@st.cache_data(ttl=86400)
def cached_osm_search(query, country_hint, resolver_version="10.6.4a"):
    # resolver_version deliberately invalidates cached misses after resolver updates.
    return geocode_course(query, country_hint)

@st.cache_data(ttl=3600)
def cached_course_detail(course_id, api_key):
    return fetch_course_detail(course_id, api_key)

@st.cache_data(ttl=3600)
def cached_course_holes(course_id, tournament, api_key):
    return fetch_course_holes(course_id, tournament, api_key)

@st.cache_data(ttl=1800)
def cached_weather_gca(course_id, tournament, start_date, api_key):
    detail = cached_course_detail(course_id, api_key)
    return fetch_weather(detail, tournament, start_date=start_date, days=7)

@st.cache_data(ttl=1800)
def cached_weather_location(lat, lon, tournament, start_date):
    return fetch_weather({"latitude": lat, "longitude": lon}, tournament, start_date=start_date, days=7)

st.sidebar.header("Tournament Setup")
tournament_name = st.sidebar.text_input("Tournament", placeholder="Current tournament")
course_query = st.sidebar.text_input("Course", placeholder="Current course")
tournament_start = st.sidebar.date_input("Tournament-week weather start", value=date.today())

st.sidebar.subheader("DraftKings")
players_up = st.sidebar.file_uploader("DraftKings field / players.csv (required)", type="csv", key="players")

st.sidebar.subheader("$0 internet ingestion")
st.sidebar.caption("Course/scorecard: Golf Courses API when available. Global location fallback: OpenStreetMap/Nominatim. Weather: Open-Meteo. No PGA TOUR scraping is used.")
gca_key = st.sidebar.text_input("Golf Courses API key (optional)", type="password", help="Optional richer scorecard source. If blank or no match is found, the app uses the global OpenStreetMap location fallback.")
country_hint = st.sidebar.text_input("Country / region hint (recommended)", placeholder="Japan, Bermuda, Mexico...")
if "course_matches" not in st.session_state: st.session_state.course_matches = []
if "selected_course_id" not in st.session_state: st.session_state.selected_course_id = None
if "course_lookup_note" not in st.session_state: st.session_state.course_lookup_note = ""
if st.sidebar.button("Find course online"):
    st.session_state.selected_course_id = None
    st.session_state.course_matches = []
    st.session_state.course_lookup_note = ""
    try:
        gca_matches = cached_course_search(course_query, gca_key, country_hint) if gca_key.strip() else []
        for r in gca_matches: r["source"] = "Golf Courses API"
        if gca_matches:
            st.session_state.course_matches = gca_matches
            st.session_state.course_lookup_note = "Rich course database match found."
        else:
            osm_matches = cached_osm_search(course_query, country_hint)
            st.session_state.course_matches = osm_matches
            st.session_state.course_lookup_note = (
                "Golf Courses API had no match; global OpenStreetMap location fallback used." if gca_key.strip()
                else "Global OpenStreetMap location lookup used."
            )
            if not osm_matches:
                st.session_state.course_lookup_note = "No match after course database + automatic global alias/location search."
    except Exception as exc:
        st.session_state.course_lookup_note = f"Lookup failed: {exc}"

if st.session_state.course_lookup_note:
    (st.sidebar.warning if "no match" in st.session_state.course_lookup_note.casefold() or "failed" in st.session_state.course_lookup_note.casefold() else st.sidebar.info)(st.session_state.course_lookup_note)

selected_course = None
if st.session_state.course_matches:
    labels = [f"[{r.get('source','Course source')}] {r['name']} — {r['city']}, {r['state']}, {r['country']}".strip(" —,") for r in st.session_state.course_matches]
    choice = st.sidebar.selectbox("Matched course/location", range(len(labels)), format_func=lambda i: labels[i])
    selected_course = st.session_state.course_matches[choice]
    st.session_state.selected_course_id = selected_course["id"]
    st.sidebar.success(f"Course identity ready via {selected_course.get('source','online source')}")

st.sidebar.subheader("OTIS Golf data")
st.sidebar.caption("Upload the weekly OTIS Course-Fit CSV exported from Model → Course fit → Advanced. The app uses True Skill, Course Fit and Form; OTIS Rank/Model are audit-only and never drive projections.")
otis_fit_up = st.sidebar.file_uploader("OTIS — Advanced Course-Fit CSV", type="csv", key="otis_fit")

st.sidebar.subheader("Advanced model inputs")
st.sidebar.caption("Optional direct overrides. Use these only if you already have model-contract CSVs.")
stats_up = st.sidebar.file_uploader("player_stats.csv override", type="csv", key="stats")
results_up = st.sidebar.file_uploader("results.csv override", type="csv", key="results")
history_up = st.sidebar.file_uploader("course_history.csv override", type="csv", key="history")
holes_up = st.sidebar.file_uploader("course_holes.csv (overrides online course)", type="csv", key="holes")
weather_up = st.sidebar.file_uploader("weather.csv (overrides Open-Meteo)", type="csv", key="weather")

if not tournament_name.strip() or not course_query.strip():
    st.info("Enter the current tournament and course in the sidebar.")
    st.stop()

players_raw = load_weekly(players_up, "players.csv")
players, dk_format = normalize_dk_players(players_raw)
# Advanced-data precedence:
# explicit model-contract override > OTIS weekly Course-Fit export > repository fallback.
otis_fit_raw = pd.read_csv(otis_fit_up) if otis_fit_up is not None else pd.DataFrame()
otis_fit, otis_fit_status = normalize_otis_course_fit(otis_fit_raw)

if stats_up is not None:
    player_stats_raw = pd.read_csv(stats_up)
    player_stats, stats_status = validate_advanced_input(player_stats_raw, "player_stats")
elif not otis_fit.empty:
    player_stats, stats_status = otis_fit, otis_fit_status
else:
    player_stats_raw = load_repo_csv("player_stats.csv")
    player_stats, stats_status = validate_advanced_input(player_stats_raw, "player_stats")

if results_up is not None:
    results_raw = pd.read_csv(results_up)
    results, results_status = validate_advanced_input(results_raw, "results")
else:
    results_raw = load_repo_csv("results.csv")
    results, results_status = validate_advanced_input(results_raw, "results")

if history_up is not None:
    history_raw = pd.read_csv(history_up)
    history, history_status = validate_advanced_input(history_raw, "course_history")
else:
    history_raw = load_repo_csv("course_history.csv")
    history, history_status = validate_advanced_input(history_raw, "course_history")

# Invalid repository fallbacks are deliberately treated as absent. An explicit
# upload with a bad schema is still surfaced to the user below.

# Course holes: explicit upload > selected open source > repository fallback.
if holes_up is not None:
    holes = pd.read_csv(holes_up); holes_source = "uploaded"
elif selected_course is not None:
    if selected_course.get("source") == "Golf Courses API":
        try:
            holes, selected_detail = cached_course_holes(st.session_state.selected_course_id, tournament_name, gca_key)
            holes_source = "Golf Courses API" if not holes.empty else "course identified; scorecard/holes unavailable"
        except Exception as exc:
            holes = pd.DataFrame(); holes_source = f"course identified; scorecard failed: {exc}"
    else:
        holes = pd.DataFrame(); holes_source = "location verified; scorecard/holes unavailable"
else:
    holes = load_repo_csv("course_holes.csv"); holes_source = "repository fallback" if not holes.empty else "not supplied"

# Weather: explicit upload > selected open source > repository fallback.
if weather_up is not None:
    weather = pd.read_csv(weather_up); weather_source = "uploaded"
elif selected_course is not None:
    try:
        if selected_course.get("source") == "Golf Courses API":
            weather = cached_weather_gca(st.session_state.selected_course_id, tournament_name, tournament_start, gca_key)
        else:
            weather = cached_weather_location(float(selected_course["latitude"]), float(selected_course["longitude"]), tournament_name, tournament_start)
        weather_source = "Open-Meteo"
    except Exception as exc:
        weather = pd.DataFrame(); weather_source = f"failed: {exc}"
else:
    weather = load_repo_csv("weather.csv"); weather_source = "repository fallback" if not weather.empty else "not supplied"

dk_ready = (players_up is not None and not players.empty and {"player", "salary"}.issubset(players.columns) and dk_format != "unrecognized")
if players_up is None:
    st.sidebar.info("DK field: not uploaded yet — course/weather lookup is still available")
elif players_raw.empty:
    st.sidebar.error("DK field: uploaded file is empty")
elif dk_format == "unrecognized":
    st.sidebar.error("DK field: unrecognized format. Upload the untouched DraftKings PGA salary CSV or a players.csv with player/salary columns.")
elif players.empty:
    st.sidebar.error("DK field: no valid golfer rows found after normalization")

st.sidebar.subheader("Data integrity")
if dk_ready:
    st.sidebar.success(f"DK field: {len(players)} golfers ✓ ({dk_format})")
advanced_rows = [
    ("player layer", stats_up, otis_fit_up, player_stats, stats_status),
    ("results", results_up, None, results, results_status),
    ("course_history", history_up, None, history, history_status),
]
for name, override_up, otis_up, df, status in advanced_rows:
    if override_up is not None and not df.empty:
        st.sidebar.success(f"{name}: direct override ✓")
    elif otis_up is not None and not df.empty:
        st.sidebar.success(f"{name}: {status} ✓ ({len(df)} golfers)")
    elif override_up is not None or otis_up is not None:
        st.sidebar.error(f"{name}: {status}")
    elif status == "valid" and not df.empty:
        st.sidebar.warning(f"{name}: repository fallback (unverified)")
    elif str(status).startswith("incompatible"):
        st.sidebar.warning(f"{name}: ignored repository fallback — {status}")
    else:
        st.sidebar.info(f"{name}: not supplied")
(st.sidebar.success if holes_source in ("uploaded","Golf Courses API") else st.sidebar.warning)(f"course_holes: {holes_source}")
(st.sidebar.success if weather_source in ("uploaded","Open-Meteo") else st.sidebar.warning)(f"weather: {weather_source}")

# Exact-name field coverage check after both weekly files are normalized.
otis_match_count = 0
otis_unmatched = []
if dk_ready and not otis_fit.empty:
    dk_names = set(players["player"].astype(str).str.strip().str.casefold())
    otis_names = set(otis_fit["player"].astype(str).str.strip().str.casefold())
    otis_match_count = len(dk_names & otis_names)
    otis_unmatched = sorted(dk_names - otis_names)
    if otis_match_count == len(dk_names):
        st.sidebar.success(f"DK ↔ OTIS match: {otis_match_count}/{len(dk_names)} ✓")
    else:
        st.sidebar.warning(f"DK ↔ OTIS coverage: {otis_match_count}/{len(dk_names)}; {len(otis_unmatched)} using neutral fallback")

# Prevent a stale keyed file from silently masquerading as this week's tournament.
def tournament_coverage(df):
    if df.empty or "tournament" not in df.columns: return None
    vals=df["tournament"].dropna().astype(str).str.strip().str.casefold()
    return bool((vals == tournament_name.strip().casefold()).any())
coverage_errors=[]
for label, df in [("course_history.csv",history),("course_holes.csv",holes),("weather.csv",weather)]:
    match=tournament_coverage(df)
    if match is False:
        coverage_errors.append(f"{label} has a tournament column but no rows for '{tournament_name}'.")

st.sidebar.success(f"{tournament_name} — {course_query}")
sims=st.sidebar.selectbox("Monte Carlo simulations",[25000,50000,100000],index=1)

# Make the limitations visible rather than silently implying full automation.
with st.expander("V10.6.6 source coverage", expanded=True):
    c1,c2,c3,c4=st.columns(4)
    c1.metric("DK field", f"{len(players)} golfers ✓" if dk_ready else "Awaiting upload")
    c2.metric("Course/holes", "Scorecard auto ✓" if holes_source=="Golf Courses API" else holes_source)
    c3.metric("Weather", "Auto ✓" if weather_source=="Open-Meteo" else weather_source)
    c4.metric("OTIS player layer", f"{otis_match_count}/{len(players)} matched" if dk_ready and not otis_fit.empty else ("Uploaded" if not otis_fit.empty else "Awaiting upload"))
    if otis_fit_up is None:
        st.warning("Upload the weekly OTIS Advanced Course-Fit CSV before treating projections as production-ready.")
    elif dk_ready and otis_match_count < len(players):
        st.warning(f"OTIS coverage is {otis_match_count}/{len(players)} ({otis_match_count/len(players):.1%}). The {len(players)-otis_match_count} unmatched golfers remain in the field and use the model neutral baseline for OTIS Skill/Form/Fit rather than being assigned zero strength.")
    elif dk_ready:
        st.success("Weekly player layer complete: full DK ↔ OTIS coverage.")

if "prediction" not in st.session_state: st.session_state.prediction=None; st.session_state.prediction_key=None
run_key=(tournament_name,course_query,str(tournament_start),sims,getattr(players_up,"name",None),getattr(otis_fit_up,"name",None),getattr(stats_up,"name",None),getattr(results_up,"name",None),getattr(history_up,"name",None),holes_source,weather_source,st.session_state.selected_course_id, selected_course.get("source") if selected_course else None)
if st.button("Build current-week projections", type="primary"):
    if players_up is None:
        st.error("Upload the current DraftKings field before building projections.")
    elif players_raw.empty:
        st.error("The uploaded DraftKings file is empty.")
    elif dk_format == "unrecognized":
        st.error("Unrecognized DraftKings format. Upload the untouched PGA DKSalaries CSV or a normalized players.csv.")
    elif players.empty:
        st.error("No valid golfer rows were found after DraftKings normalization.")
    elif otis_fit_up is None:
        st.error("Upload the OTIS Advanced Course-Fit CSV before building production projections.")
    elif otis_fit.empty:
        st.error(f"OTIS Course-Fit CSV could not be used: {otis_fit_status}.")
    elif otis_match_count == 0:
        st.error("None of the DraftKings golfers matched the OTIS file. Check that the correct weekly OTIS export was uploaded.")
    elif otis_match_count / len(players) < 0.70:
        st.error(f"OTIS coverage is only {otis_match_count}/{len(players)} ({otis_match_count/len(players):.1%}). This is below the 70% safety floor; verify the weekly files before building projections.")
    elif stats_up is not None and stats_status != "valid":
        st.error(f"Uploaded player_stats.csv is {stats_status}.")
    elif results_up is not None and results_status != "valid":
        st.error(f"Uploaded results.csv is {results_status}.")
    elif history_up is not None and history_status != "valid":
        st.error(f"Uploaded course_history.csv is {history_status}.")
    elif coverage_errors:
        for msg in coverage_errors:
            st.error(msg + " Replace/refresh that input before running.")
    else:
      try:
        with st.spinner(f"Running {sims:,} tournament simulations..."):
            pred=predict_from_dataframes(Config(tournament=tournament_name,sims=int(sims),seed=42),players,player_stats,results,history,holes,weather)
            pred=pred.rename(columns={"dk_proxy":"dk_points_proxy"})
            st.session_state.prediction=pred; st.session_state.prediction_key=run_key
        st.success(f"Built projections for {len(pred)} golfers using {sims:,} simulations.")
      except Exception as exc: st.error(str(exc))

mc=st.session_state.prediction
if mc is None:
    st.info("Upload the current DraftKings field, fetch/select the course, verify the source panel, then build projections.")
    st.stop()
if st.session_state.prediction_key != run_key:
    st.warning("Tournament settings or weekly inputs changed. Rebuild projections before optimizing."); st.stop()

page=st.sidebar.radio("View",["Model Dashboard","Portfolio Optimizer","Round 3-Ball / 6-Leg Builder"])
if page=="Model Dashboard":
    c1,c2,c3,c4=st.columns(4); c1.metric("Field",len(mc)); c2.metric("Simulations",f"{sims:,}"); c3.metric("Salary floor","$6,500"); c4.metric("DK salary cap","$50,000")
    st.subheader(f"{tournament_name} Simulation — {course_query}")
    display_cols=[c for c in ["player","salary","win_pct","top5_pct","top10_pct","top20_pct","make_cut_pct","expected_finish","dk_points_proxy","points_per_1k","course_fit_ceiling"] if c in mc.columns]
    sort_options=[c for c in ["win_pct","top10_pct","make_cut_pct","dk_points_proxy","course_fit_ceiling","salary"] if c in mc.columns]
    sort_col=st.selectbox("Sort by",sort_options)
    st.dataframe(mc.sort_values(sort_col,ascending=(sort_col=="salary"))[display_cols],width="stretch",hide_index=True)
    st.download_button("Download current-week projections",mc.to_csv(index=False),f"{tournament_name.replace(' ','_')}_projections.csv","text/csv")
elif page=="Portfolio Optimizer":
    st.subheader("DraftKings Portfolio Optimizer")
    a,b,c,d=st.columns(4); lineup_count=a.number_input("Lineups",1,20,10); max_exposure=b.slider("Max exposure",0.10,1.00,0.50,0.05); min_unique=c.number_input("Minimum unique golfers",1,5,3); salary_floor=d.number_input("Minimum lineup salary",40000,50000,46500,100)
    min_player_salary=st.number_input("Minimum golfer salary",6000,10000,6500,100); strategy=st.selectbox("Strategy",["GPP Ceiling","Balanced / Single Entry","Cut Equity"])
    names=sorted(mc.player.dropna().astype(str).unique()); locks=st.multiselect("Lock golfers",names); excludes=st.multiselect("Exclude golfers",[n for n in names if n not in locks])
    st.info("Frozen optimizer defaults: 50% max exposure for 20-max, 3 minimum unique, $6,500 golfer floor, $46,500 lineup floor.")
    if st.button("Generate portfolio",type="primary"):
        settings=PortfolioSettings(lineup_count=int(lineup_count),salary_cap=50000,salary_floor=int(salary_floor),min_player_salary=int(min_player_salary),roster_size=6,max_exposure=float(max_exposure),min_unique=int(min_unique),strategy=strategy,seed=42)
        try:
            with st.spinner("Optimizing portfolio..."): portfolio,summary,exposure=optimize_portfolio(mc,settings,locks,excludes)
            st.success(f"Generated {len(summary)} lineups"); st.subheader("Lineup Summary"); st.dataframe(summary,width="stretch",hide_index=True); st.subheader("Lineups"); st.dataframe(portfolio,width="stretch",hide_index=True); st.subheader("Exposure"); st.dataframe(exposure,width="stretch",hide_index=True)
            st.download_button("Download lineups CSV",portfolio.to_csv(index=False),"pga_lineups.csv","text/csv"); st.download_button("Download exposure CSV",exposure.to_csv(index=False),"pga_exposure.csv","text/csv")
        except Exception as exc: st.error(str(exc))

else:
    st.subheader("Round 3-Ball / 6-Leg Parlay Builder")
    st.caption("Round-specific 3-ball probabilities using True Skill + Course Fit + pre-event Form, with shrunk in-tournament form updates for R2-R4. OTIS Rank/Model are never predictive inputs.")
    a,b,c,d=st.columns(4)
    round_no=int(a.selectbox("Round",[1,2,3,4],index=0))
    round_sims=int(b.selectbox("Round simulations",[25000,50000,100000],index=2))
    ticket_count=int(c.number_input("6-leg tickets",1,10,5))
    max_overlap=int(d.slider("Max shared picks between tickets",0,5,4))
    year=int(tournament_start.year)
    default_group_url="https://www.pgatour.com/tournaments/pga/playerschamp/index/tee-times"
    grouping_url=st.text_input("Public grouping URL (fallback/override)",value=default_group_url,help="The app tries PGA TOUR first. Paste a tournament-specific PGA TOUR tee-times URL here if needed; Golf Channel round articles remain a fallback.")
    grouping_up=st.file_uploader("Grouping CSV fallback (group, tee_time, player1, player2, player3)",type="csv",key=f"groupings_r{round_no}")
    leaderboard_url=st.text_input("Public leaderboard URL for prior-round form (R2-R4)",value="https://www.pgatour.com/tournaments/pga/playerschamp/index" if round_no>1 else "",disabled=(round_no==1))
    live_up=st.file_uploader("Prior-round results CSV fallback (player, round, round_score and/or SG components)",type="csv",key=f"live_r{round_no}",disabled=(round_no==1))

    if st.button("Pull groupings + build round model",type="primary"):
        if otis_fit.empty:
            st.error("Upload the OTIS Advanced Course-Fit CSV first; the round model requires the Course DNA player layer.")
        else:
            if grouping_up is not None:
                groups=normalize_groupings_upload(pd.read_csv(grouping_up),round_no); gsrc="uploaded grouping CSV"; gnote="manual fallback"
            else:
                with st.spinner("Pulling public groupings..."):
                    groups,gsrc,gnote=fetch_groupings(tournament_name,year,round_no,grouping_url)
            live=pd.DataFrame(); lnote="R1: no in-tournament adjustment"
            if round_no>1:
                if live_up is not None:
                    live=normalize_live_results(pd.read_csv(live_up)); lnote="uploaded prior-round results"
                else:
                    with st.spinner("Pulling prior-round leaderboard data..."):
                        live,lnote=fetch_live_results(leaderboard_url,round_no)
            if groups.empty:
                st.error(f"Could not parse public groupings ({gnote}). Use the grouping CSV fallback so the model never guesses pairings.")
            else:
                ratings=build_round_ratings(otis_fit,live,round_no)
                probs,_=simulate_groups(groups,ratings,n_sims=round_sims,seed=42)
                summary,legs=build_six_leg_tickets(probs,ticket_count=ticket_count,max_player_overlap=max_overlap)
                st.session_state.round_parlay={"groups":groups,"ratings":ratings,"probs":probs,"summary":summary,"legs":legs,"gsrc":gsrc,"gnote":gnote,"lnote":lnote,"round":round_no}
    rp=st.session_state.get("round_parlay")
    if rp and rp.get("round")==round_no:
        st.success(f"Round {round_no} model built from {len(rp['groups'])} groups. Grouping source: {rp['gsrc']} ({rp['gnote']}).")
        if round_no>1:
            (st.success if not rp['ratings'].empty and rp['ratings'].live_rounds.max()>0 else st.warning)(f"In-tournament form source: {rp['lnote']}")
        unmatched=rp['probs'][rp['probs'].status.eq('UNMATCHED')] if not rp['probs'].empty else pd.DataFrame()
        if not unmatched.empty: st.warning(f"{len(unmatched)} grouping names did not match the OTIS player layer and were excluded rather than guessed.")
        st.subheader("3-Ball probabilities")
        ok=rp['probs'][rp['probs'].status.eq('OK')].copy()
        if not ok.empty:
            st.dataframe(ok.sort_values(['group','win_pct'],ascending=[True,False]),width="stretch",hide_index=True)
            st.download_button("Download round probabilities",ok.to_csv(index=False),f"round_{round_no}_3ball_probabilities.csv","text/csv")
        st.subheader("6-Leg ticket portfolio")
        if rp['summary'].empty:
            st.warning("Fewer than six fully matched groups are available; no six-leg ticket was created.")
        else:
            st.dataframe(rp['summary'],width="stretch",hide_index=True)
            st.dataframe(rp['legs'],width="stretch",hide_index=True)
            st.caption("Strict 6/6 = all six selected golfers win outright. All-legs non-loss includes simulated tie/push outcomes. Sportsbook odds are intentionally not assumed; add offered odds later for EV analysis.")
            st.download_button("Download 6-leg tickets",rp['legs'].to_csv(index=False),f"round_{round_no}_six_leg_tickets.csv","text/csv")

st.divider()
st.caption("V10.7.1 hardens grouping ingestion with PGA TOUR-first validated 3-player groups. V10.7.0 adds the Round 3-Ball / 6-Leg Builder. Existing V10.6.6b tournament projection and DFS optimizer logic is preserved. Round model uses OTIS True Skill/Course Fit/Form plus shrunk prior-round evidence for R2-R4; public grouping/leaderboard ingestion has CSV fallbacks and never guesses missing groupings.")
