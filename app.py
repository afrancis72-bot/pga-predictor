from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st

from international_ingestion import search_courses, geocode_course, location_detail, fetch_course_detail, fetch_course_holes, fetch_weather, IngestionError
from portfolio_optimizer import PortfolioSettings, optimize_portfolio
from pga_predictor_pro import Config, predict_from_dataframes

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="PGA Predictor Pro", page_icon="⛳", layout="wide")
st.title("⛳ PGA Predictor Pro — Tournament Model V10.6.4")
st.caption("$0 multi-source international course/weather ingestion + manual DraftKings field + frozen tournament model/optimizer")

@st.cache_data
def load_repo_csv(name):
    path = ROOT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load_weekly(upload, name):
    return pd.read_csv(upload) if upload is not None else load_repo_csv(name)

@st.cache_data(ttl=3600)
def cached_course_search(query, api_key, country_hint):
    return search_courses(query, api_key, country_hint)

@st.cache_data(ttl=86400)
def cached_osm_search(query, country_hint):
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

st.sidebar.subheader("Advanced model inputs")
st.sidebar.caption("Optional weekly uploads. Repository fallbacks remain available until a lawful free current-stat API is identified.")
stats_up = st.sidebar.file_uploader("player_stats.csv", type="csv", key="stats")
results_up = st.sidebar.file_uploader("results.csv", type="csv", key="results")
history_up = st.sidebar.file_uploader("course_history.csv", type="csv", key="history")
holes_up = st.sidebar.file_uploader("course_holes.csv (overrides online course)", type="csv", key="holes")
weather_up = st.sidebar.file_uploader("weather.csv (overrides Open-Meteo)", type="csv", key="weather")

if not tournament_name.strip() or not course_query.strip():
    st.info("Enter the current tournament and course in the sidebar.")
    st.stop()

players = load_weekly(players_up, "players.csv")
player_stats = load_weekly(stats_up, "player_stats.csv")
results = load_weekly(results_up, "results.csv")
history = load_weekly(history_up, "course_history.csv")

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

dk_ready = (players_up is not None and not players.empty and {"player", "salary"}.issubset(players.columns))
if players_up is None:
    st.sidebar.info("DK field: not uploaded yet — course/weather lookup is still available")
elif players.empty:
    st.sidebar.error("DK field: uploaded file is empty")
elif not {"player", "salary"}.issubset(players.columns):
    st.sidebar.error("DK field: needs 'player' and 'salary' columns")

st.sidebar.subheader("Data integrity")
if dk_ready:
    st.sidebar.success(f"DK field: uploaded ({len(players)} golfers)")
for name, up, df in [("player_stats.csv",stats_up,player_stats),("results.csv",results_up,results),("course_history.csv",history_up,history)]:
    if up is not None: st.sidebar.success(f"{name}: uploaded")
    elif not df.empty: st.sidebar.warning(f"{name}: repository fallback")
    else: st.sidebar.info(f"{name}: not supplied")
(st.sidebar.success if holes_source in ("uploaded","Golf Courses API") else st.sidebar.warning)(f"course_holes: {holes_source}")
(st.sidebar.success if weather_source in ("uploaded","Open-Meteo") else st.sidebar.warning)(f"weather: {weather_source}")

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
with st.expander("V10.6.4 source coverage", expanded=True):
    c1,c2,c3,c4=st.columns(4)
    c1.metric("DK field", "Uploaded ✓" if dk_ready else "Awaiting upload")
    c2.metric("Course/holes", "Scorecard auto ✓" if holes_source=="Golf Courses API" else holes_source)
    c3.metric("Weather", "Auto ✓" if weather_source=="Open-Meteo" else weather_source)
    advanced_fresh=sum(x is not None for x in (stats_up,results_up,history_up))
    c4.metric("Advanced stats/form/history", f"{advanced_fresh}/3 fresh uploads")
    if advanced_fresh < 3:
        st.warning("Advanced player stats/results/course history are not fully automated at $0 yet. Repository fallbacks can be used for testing, but should not be treated as current-week data unless you have verified them.")

if "prediction" not in st.session_state: st.session_state.prediction=None; st.session_state.prediction_key=None
run_key=(tournament_name,course_query,str(tournament_start),sims,getattr(players_up,"name",None),getattr(stats_up,"name",None),getattr(results_up,"name",None),getattr(history_up,"name",None),holes_source,weather_source,st.session_state.selected_course_id, selected_course.get("source") if selected_course else None)
if st.button("Build current-week projections", type="primary"):
    if players_up is None:
        st.error("Upload the current DraftKings field before building projections.")
    elif players.empty:
        st.error("The uploaded DraftKings file is empty.")
    elif not {"player", "salary"}.issubset(players.columns):
        st.error("DraftKings players.csv must contain 'player' and 'salary' columns.")
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

page=st.sidebar.radio("View",["Model Dashboard","Portfolio Optimizer"])
if page=="Model Dashboard":
    c1,c2,c3,c4=st.columns(4); c1.metric("Field",len(mc)); c2.metric("Simulations",f"{sims:,}"); c3.metric("Salary floor","$6,500"); c4.metric("DK salary cap","$50,000")
    st.subheader(f"{tournament_name} Simulation — {course_query}")
    display_cols=[c for c in ["player","salary","win_pct","top5_pct","top10_pct","top20_pct","make_cut_pct","expected_finish","dk_points_proxy","points_per_1k","course_fit_ceiling"] if c in mc.columns]
    sort_options=[c for c in ["win_pct","top10_pct","make_cut_pct","dk_points_proxy","course_fit_ceiling","salary"] if c in mc.columns]
    sort_col=st.selectbox("Sort by",sort_options)
    st.dataframe(mc.sort_values(sort_col,ascending=(sort_col=="salary"))[display_cols],width="stretch",hide_index=True)
    st.download_button("Download current-week projections",mc.to_csv(index=False),f"{tournament_name.replace(' ','_')}_projections.csv","text/csv")
else:
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

st.divider()
st.caption("V10.6.4 ingestion release. Course data: Golf Courses API when available; global location fallback: OpenStreetMap/Nominatim; weather: Open-Meteo. Predictive model/optimizer unchanged. OpenStreetMap data © OpenStreetMap contributors, ODbL.")
