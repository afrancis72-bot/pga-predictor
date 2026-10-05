from pathlib import Path
import pandas as pd
import streamlit as st

from portfolio_optimizer import PortfolioSettings, optimize_portfolio
from course_fit_ceiling import COURSE_PROFILES, add_course_fit_ceiling
from course_fit_simulation import resimulate_with_course_fit_calibrated

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="PGA Predictor Pro", page_icon="⛳", layout="wide")
st.title("⛳ PGA Predictor Pro — Tournament Model V10.4")
st.caption("Tournament-agnostic V10.3 predictive logic + weekly data contract")

st.sidebar.header("Tournament Setup")
tournament_name = st.sidebar.text_input("Tournament", placeholder="Current tournament")
course_name = st.sidebar.text_input("Course", placeholder="Current course")
mc_upload = st.sidebar.file_uploader("Upload current-week simulation CSV", type="csv", key="mc")
model_upload = st.sidebar.file_uploader("Upload current-week model-input CSV", type="csv", key="model")

if not tournament_name.strip() or not course_name.strip():
    st.info("Enter the current tournament and course in the sidebar.")
    st.stop()

if mc_upload is None or model_upload is None:
    st.warning("V10.4 has no tournament-specific fallback data. Upload BOTH current-week files before running the model: simulation CSV and model-input CSV.")
    st.stop()

mc = pd.read_csv(mc_upload)
model = pd.read_csv(model_upload)

required_mc = {"player", "salary", "win_pct", "top10_pct", "make_cut_pct", "dk_points_proxy"}
missing_mc = sorted(required_mc - set(mc.columns))
if missing_mc:
    st.error("Simulation CSV is missing required columns: " + ", ".join(missing_mc))
    st.stop()

if "player" not in model.columns:
    st.error("Model-input CSV must contain a 'player' column.")
    st.stop()

# Weekly-data integrity gate: prevent labels from silently describing a different field.
mc_players = set(mc["player"].dropna().astype(str).str.strip())
model_players = set(model["player"].dropna().astype(str).str.strip())
overlap = mc_players & model_players
coverage = len(overlap) / max(len(mc_players), 1)
if coverage < 0.90:
    st.error(f"Weekly input mismatch: only {coverage:.0%} of simulation players are present in the model-input file. Load matching files for {tournament_name}.")
    st.stop()
elif coverage < 1.0:
    st.sidebar.warning(f"Field match: {coverage:.0%}. Review missing model-input players before lineup lock.")
else:
    st.sidebar.success("Weekly field integrity check: 100% match")

# Merge the best available confidence field into the simulation table.
if "model_data_confidence" not in mc.columns:
    for conf_col in ["model_data_confidence_v10", "model_data_confidence_v9", "model_data_confidence_v8", "model_data_confidence"]:
        if conf_col in model.columns:
            mc = mc.merge(model[["player", conf_col]].rename(columns={conf_col: "model_data_confidence"}), on="player", how="left")
            break

st.sidebar.subheader("Course-Fit Monte Carlo (V10.2c)")
use_course_fit = st.sidebar.toggle(
    "Enable calibrated Course-Fit simulation",
    value=False,
    help="Reprices the loaded weekly simulation using bounded course fit, then calibrates outputs back to the loaded baseline scale."
)
profile = st.sidebar.selectbox("Course profile", list(COURSE_PROFILES), index=0, disabled=not use_course_fit)

if use_course_fit:
    scored_model = add_course_fit_ceiling(model, profile)
    mc = resimulate_with_course_fit_calibrated(mc, scored_model, sims=50000, seed=42)
    st.sidebar.caption("Course fit is applied upstream in 50,000 Monte Carlo simulations. Outputs are calibrated to the loaded weekly baseline; the optimizer receives no separate course-fit bonus.")

st.sidebar.success(f"{tournament_name} — {course_name}")
st.sidebar.caption("Data: current-week uploaded files")
page = st.sidebar.radio("View", ["Model Dashboard", "Portfolio Optimizer"])

if page == "Model Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Field", len(mc))
    c2.metric("Simulations", "50,000" if use_course_fit else "Loaded baseline")
    c3.metric("Salary floor", "$6,500")
    c4.metric("DK salary cap", "$50,000")

    st.subheader(f"{tournament_name} Simulation — {course_name}")
    display_cols = [c for c in [
        "player", "salary", "win_pct", "top5_pct", "top10_pct", "top20_pct",
        "make_cut_pct", "expected_finish", "dk_points_proxy", "dk_value_per_1000",
        "model_data_confidence", "course_fit_ceiling", "course_fit_coverage", "course_fit_profile"
    ] if c in mc.columns]
    sort_options = [c for c in ["win_pct", "top10_pct", "make_cut_pct", "dk_points_proxy", "course_fit_ceiling", "salary"] if c in mc.columns]
    sort_col = st.selectbox("Sort by", sort_options)
    ascending = sort_col == "salary"
    st.dataframe(mc.sort_values(sort_col, ascending=ascending)[display_cols], width="stretch", hide_index=True)
else:
    st.subheader("DraftKings Portfolio Optimizer")
    st.caption("Portfolio-wide exposure and uniqueness constraints are enforced across all six-golfer lineups.")

    a, b, c, d = st.columns(4)
    lineup_count = a.number_input("Lineups", 1, 20, 10)
    max_exposure = b.slider("Max exposure", 0.10, 1.00, 0.50, 0.05)
    min_unique = c.number_input("Minimum unique golfers", 1, 5, 3)
    salary_floor = d.number_input("Minimum lineup salary", 40000, 50000, 46500, 100)

    min_player_salary = st.number_input("Minimum golfer salary", 6000, 10000, 6500, 100, help="Hard rule: golfers below this salary are excluded from every lineup.")
    strategy = st.selectbox("Strategy", ["GPP Ceiling", "Balanced / Single Entry", "Cut Equity"])
    names = sorted(mc.player.dropna().astype(str).unique())
    locks = st.multiselect("Lock golfers", names)
    excludes = st.multiselect("Exclude golfers", [n for n in names if n not in locks])

    st.info("V10.3 predictive/optimizer defaults preserved: 50% max exposure for 20-max PGA portfolios, 3 minimum unique golfers, $6,500 player floor, and $46,500 lineup floor.")

    if st.button("Generate portfolio", type="primary"):
        settings = PortfolioSettings(
            lineup_count=int(lineup_count), salary_cap=50000,
            salary_floor=int(salary_floor), min_player_salary=int(min_player_salary), roster_size=6,
            max_exposure=float(max_exposure), min_unique=int(min_unique),
            strategy=strategy, seed=42,
        )
        try:
            with st.spinner("Optimizing portfolio..."):
                portfolio, summary, exposure = optimize_portfolio(mc, settings, locks, excludes)
            st.success(f"Generated {len(summary)} lineups")
            st.subheader("Lineup Summary")
            st.dataframe(summary, width="stretch", hide_index=True)
            st.subheader("Lineups")
            for number in summary.lineup:
                lu = portfolio[portfolio.lineup == number]
                salary = int(lu.salary.sum())
                with st.expander(f"Lineup {number} — ${salary:,}", expanded=(number == 1)):
                    cols = [c for c in ["player", "salary", "win_pct", "top10_pct", "make_cut_pct", "dk_points_proxy", "course_fit_ceiling"] if c in lu.columns]
                    st.dataframe(lu[cols], width="stretch", hide_index=True)
            st.subheader("Exposure")
            st.dataframe(exposure, width="stretch", hide_index=True)
            st.download_button("Download lineups CSV", portfolio.to_csv(index=False), "pga_lineups.csv", "text/csv")
            st.download_button("Download exposure CSV", exposure.to_csv(index=False), "pga_exposure.csv", "text/csv")
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption("V10.4 architecture release: no tournament-specific fallback data. Refresh field, course inputs, weather, tee times and withdrawals before lineup lock.")
