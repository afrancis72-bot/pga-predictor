from pathlib import Path
import pandas as pd
import streamlit as st

from portfolio_optimizer import PortfolioSettings, optimize_portfolio
from pga_predictor_pro import Config, predict_from_dataframes

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="PGA Predictor Pro", page_icon="⛳", layout="wide")
st.title("⛳ PGA Predictor Pro — Tournament Model V10.5")
st.caption("Tournament-agnostic weekly ingestion + frozen V10.3 predictive/optimizer logic")

@st.cache_data
def load_repo_csv(name):
    path = ROOT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load_weekly(upload, name):
    return pd.read_csv(upload) if upload is not None else load_repo_csv(name)

st.sidebar.header("Tournament Setup")
tournament_name = st.sidebar.text_input("Tournament", placeholder="Current tournament")
course_name = st.sidebar.text_input("Course", placeholder="Current course")
st.sidebar.subheader("Current-week data")
st.sidebar.caption("Upload fresh weekly files. If omitted, the generic repository file is used and clearly flagged below.")
players_up = st.sidebar.file_uploader("DK field / players.csv (required)", type="csv", key="players")
stats_up = st.sidebar.file_uploader("player_stats.csv", type="csv", key="stats")
results_up = st.sidebar.file_uploader("results.csv", type="csv", key="results")
history_up = st.sidebar.file_uploader("course_history.csv", type="csv", key="history")
holes_up = st.sidebar.file_uploader("course_holes.csv", type="csv", key="holes")
weather_up = st.sidebar.file_uploader("weather.csv", type="csv", key="weather")

if not tournament_name.strip() or not course_name.strip():
    st.info("Enter the current tournament and course in the sidebar.")
    st.stop()

players = load_weekly(players_up, "players.csv")
player_stats = load_weekly(stats_up, "player_stats.csv")
results = load_weekly(results_up, "results.csv")
history = load_weekly(history_up, "course_history.csv")
holes = load_weekly(holes_up, "course_holes.csv")
weather = load_weekly(weather_up, "weather.csv")

if players.empty:
    st.error("A current-week players/DraftKings field file is required.")
    st.stop()
if "player" not in players.columns:
    st.error("players.csv must contain a 'player' column.")
    st.stop()
if "salary" not in players.columns:
    st.error("players.csv must contain a DraftKings 'salary' column for lineup optimization.")
    st.stop()

sources = {
    "players.csv": players_up,
    "player_stats.csv": stats_up,
    "results.csv": results_up,
    "course_history.csv": history_up,
    "course_holes.csv": holes_up,
    "weather.csv": weather_up,
}
st.sidebar.subheader("Weekly data integrity")
for name, up in sources.items():
    if up is not None:
        st.sidebar.success(f"{name}: uploaded")
    elif (ROOT / name).exists():
        st.sidebar.warning(f"{name}: repository fallback")
    else:
        st.sidebar.info(f"{name}: not supplied")

# Tournament-key checks where the data contract supports them.
def tournament_coverage(df):
    if df.empty or "tournament" not in df.columns:
        return None
    vals = df["tournament"].dropna().astype(str).str.strip().str.casefold()
    return bool((vals == tournament_name.strip().casefold()).any())

for label, df in [("course_history.csv", history), ("course_holes.csv", holes), ("weather.csv", weather)]:
    match = tournament_coverage(df)
    if match is False:
        st.error(f"{label} contains a tournament column but has no rows for '{tournament_name}'. Load the current week's data before running.")
        st.stop()

if players_up is None:
    st.warning("The DK field is currently coming from repository players.csv. For a real weekly run, upload the current DraftKings field so stale players/salaries cannot be used silently.")

st.sidebar.success(f"{tournament_name} — {course_name}")
sims = st.sidebar.selectbox("Monte Carlo simulations", [25000, 50000, 100000], index=1)

if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.prediction_key = None

run_key = (tournament_name, course_name, sims, getattr(players_up, "name", None), getattr(stats_up, "name", None), getattr(results_up, "name", None), getattr(history_up, "name", None), getattr(holes_up, "name", None), getattr(weather_up, "name", None))

if st.button("Build current-week projections", type="primary"):
    try:
        with st.spinner(f"Running {sims:,} tournament simulations..."):
            pred = predict_from_dataframes(
                Config(tournament=tournament_name, sims=int(sims), seed=42),
                players, player_stats, results, history, holes, weather
            )
            pred = pred.rename(columns={"dk_proxy": "dk_points_proxy"})
            st.session_state.prediction = pred
            st.session_state.prediction_key = run_key
        st.success(f"Built projections for {len(pred)} golfers using {sims:,} simulations.")
    except Exception as exc:
        st.error(str(exc))

mc = st.session_state.prediction
if mc is None:
    st.info("Load/verify the current-week data, then click **Build current-week projections**. V10.5 generates the simulation internally; no prebuilt Monte Carlo or model-input CSV is required.")
    st.stop()
if st.session_state.prediction_key != run_key:
    st.warning("Tournament settings or weekly files changed. Rebuild projections before optimizing.")
    st.stop()

page = st.sidebar.radio("View", ["Model Dashboard", "Portfolio Optimizer"])
if page == "Model Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Field", len(mc)); c2.metric("Simulations", f"{sims:,}"); c3.metric("Salary floor", "$6,500"); c4.metric("DK salary cap", "$50,000")
    st.subheader(f"{tournament_name} Simulation — {course_name}")
    display_cols = [c for c in ["player","salary","win_pct","top5_pct","top10_pct","top20_pct","make_cut_pct","expected_finish","dk_points_proxy","points_per_1k","course_fit_ceiling"] if c in mc.columns]
    sort_options = [c for c in ["win_pct","top10_pct","make_cut_pct","dk_points_proxy","course_fit_ceiling","salary"] if c in mc.columns]
    sort_col = st.selectbox("Sort by", sort_options)
    st.dataframe(mc.sort_values(sort_col, ascending=(sort_col=="salary"))[display_cols], width="stretch", hide_index=True)
    st.download_button("Download current-week projections", mc.to_csv(index=False), f"{tournament_name.replace(' ','_')}_projections.csv", "text/csv")
else:
    st.subheader("DraftKings Portfolio Optimizer")
    a,b,c,d=st.columns(4)
    lineup_count=a.number_input("Lineups",1,20,10); max_exposure=b.slider("Max exposure",0.10,1.00,0.50,0.05); min_unique=c.number_input("Minimum unique golfers",1,5,3); salary_floor=d.number_input("Minimum lineup salary",40000,50000,46500,100)
    min_player_salary=st.number_input("Minimum golfer salary",6000,10000,6500,100)
    strategy=st.selectbox("Strategy",["GPP Ceiling","Balanced / Single Entry","Cut Equity"])
    names=sorted(mc.player.dropna().astype(str).unique()); locks=st.multiselect("Lock golfers",names); excludes=st.multiselect("Exclude golfers",[n for n in names if n not in locks])
    st.info("Frozen V10.3 optimizer defaults preserved: 50% max exposure for 20-max, 3 minimum unique, $6,500 golfer floor, $46,500 lineup floor.")
    if st.button("Generate portfolio", type="primary"):
        settings=PortfolioSettings(lineup_count=int(lineup_count),salary_cap=50000,salary_floor=int(salary_floor),min_player_salary=int(min_player_salary),roster_size=6,max_exposure=float(max_exposure),min_unique=int(min_unique),strategy=strategy,seed=42)
        try:
            with st.spinner("Optimizing portfolio..."):
                portfolio,summary,exposure=optimize_portfolio(mc,settings,locks,excludes)
            st.success(f"Generated {len(summary)} lineups")
            st.subheader("Lineup Summary"); st.dataframe(summary,width="stretch",hide_index=True)
            st.subheader("Lineups"); st.dataframe(portfolio,width="stretch",hide_index=True)
            st.subheader("Exposure"); st.dataframe(exposure,width="stretch",hide_index=True)
            st.download_button("Download lineups CSV",portfolio.to_csv(index=False),"pga_lineups.csv","text/csv")
            st.download_button("Download exposure CSV",exposure.to_csv(index=False),"pga_exposure.csv","text/csv")
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption("V10.5 ingestion release: current-week inputs → frozen tournament model → Monte Carlo → optimizer. Refresh field, course data, weather, tee times and withdrawals before lineup lock.")
