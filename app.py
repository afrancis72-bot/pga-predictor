from pathlib import Path
import pandas as pd
import streamlit as st

from portfolio_optimizer import PortfolioSettings, optimize_portfolio

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="PGA Predictor Pro", page_icon="⛳", layout="wide")
st.title("⛳ PGA Predictor Pro")
st.caption("Tournament probabilities + DraftKings portfolio optimizer")

# Production fallback for the current tournament. Future weeks can be loaded
# directly from the sidebar without changing app.py.
DEFAULT_MC = DATA / "bank_of_utah_monte_carlo_50000_v10_QC.csv"
DEFAULT_MODEL = DATA / "bank_of_utah_model_input_v10.csv"

st.sidebar.header("Tournament Setup")
tournament_name = st.sidebar.text_input("Tournament", "Bank of Utah Championship")
course_name = st.sidebar.text_input("Course", "Black Desert Resort")
mc_upload = st.sidebar.file_uploader("Upload simulation CSV (optional)", type="csv", key="mc")
model_upload = st.sidebar.file_uploader("Upload model-input CSV (optional)", type="csv", key="model")

@st.cache_data
def read_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if mc_upload is not None:
    mc = pd.read_csv(mc_upload)
    mc_source = "Uploaded simulation"
elif DEFAULT_MC.exists():
    mc = read_path(str(DEFAULT_MC))
    mc_source = "V10 QC — repository"
else:
    st.error("No simulation file is available. Upload a tournament simulation CSV.")
    st.stop()

if model_upload is not None:
    model = pd.read_csv(model_upload)
elif DEFAULT_MODEL.exists():
    model = read_path(str(DEFAULT_MODEL))
else:
    model = pd.DataFrame()

required = {"player", "salary", "win_pct", "top10_pct", "make_cut_pct", "dk_points_proxy"}
missing = sorted(required - set(mc.columns))
if missing:
    st.error("Simulation CSV is missing required columns: " + ", ".join(missing))
    st.stop()

# Merge the best available confidence field into the simulation table.
if "model_data_confidence" not in mc.columns and not model.empty and "player" in model.columns:
    for conf_col in ["model_data_confidence_v10", "model_data_confidence_v9", "model_data_confidence_v8", "model_data_confidence"]:
        if conf_col in model.columns:
            mc = mc.merge(model[["player", conf_col]].rename(columns={conf_col: "model_data_confidence"}), on="player", how="left")
            break

st.sidebar.success(f"{tournament_name} — {course_name}")
st.sidebar.caption(f"Data: {mc_source}")
page = st.sidebar.radio("View", ["Model Dashboard", "Portfolio Optimizer"])

if page == "Model Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Field", len(mc))
    c2.metric("Simulations", "50,000" if "50000" in DEFAULT_MC.name and mc_upload is None else "Loaded")
    c3.metric("Salary floor", "$6,500")
    c4.metric("DK salary cap", "$50,000")

    st.subheader(f"{tournament_name} Simulation")
    display_cols = [c for c in [
        "player", "salary", "win_pct", "top5_pct", "top10_pct", "top20_pct",
        "make_cut_pct", "expected_finish", "dk_points_proxy", "dk_value_per_1000",
        "model_data_confidence"
    ] if c in mc.columns]
    sort_options = [c for c in ["win_pct", "top10_pct", "make_cut_pct", "dk_points_proxy", "salary"] if c in mc.columns]
    sort_col = st.selectbox("Sort by", sort_options)
    ascending = sort_col == "salary"
    st.dataframe(mc.sort_values(sort_col, ascending=ascending)[display_cols], use_container_width=True, hide_index=True)
else:
    st.subheader("DraftKings Portfolio Optimizer")
    st.caption("Portfolio-wide exposure and uniqueness constraints are enforced across all six-golfer lineups.")

    a, b, c, d = st.columns(4)
    lineup_count = a.number_input("Lineups", 1, 20, 10)
    max_exposure = b.slider("Max exposure", 0.10, 1.00, 0.60, 0.05)
    min_unique = c.number_input("Minimum unique golfers", 1, 5, 3)
    salary_floor = d.number_input("Minimum lineup salary", 40000, 50000, 46500, 100)

    min_player_salary = st.number_input(
        "Minimum golfer salary", 6000, 10000, 6500, 100,
        help="Hard rule: golfers below this salary are excluded from every lineup."
    )
    strategy = st.selectbox("Strategy", ["GPP Ceiling", "Balanced / Single Entry", "Cut Equity"])
    names = sorted(mc.player.dropna().astype(str).unique())
    locks = st.multiselect("Lock golfers", names)
    excludes = st.multiselect("Exclude golfers", [n for n in names if n not in locks])

    st.info("Production defaults: 60% max exposure, 3 minimum unique golfers, $6,500 player floor, $46,500 lineup floor.")

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
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.subheader("Lineups")
            for number in summary.lineup:
                lu = portfolio[portfolio.lineup == number]
                salary = int(lu.salary.sum())
                with st.expander(f"Lineup {number} — ${salary:,}", expanded=(number == 1)):
                    cols = [c for c in ["player", "salary", "win_pct", "top10_pct", "make_cut_pct", "dk_points_proxy"] if c in lu.columns]
                    st.dataframe(lu[cols], use_container_width=True, hide_index=True)
            st.subheader("Exposure")
            st.dataframe(exposure, use_container_width=True, hide_index=True)
            st.download_button("Download lineups CSV", portfolio.to_csv(index=False), "pga_lineups.csv", "text/csv")
            st.download_button("Download exposure CSV", exposure.to_csv(index=False), "pga_exposure.csv", "text/csv")
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption("Model outputs are projections, not guarantees. Refresh field, weather, tee times and withdrawals before lineup lock.")
