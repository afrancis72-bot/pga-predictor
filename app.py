from pathlib import Path
import pandas as pd
import streamlit as st

from portfolio_optimizer import PortfolioSettings, optimize_portfolio

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="PGA Predictor Pro", page_icon="⛳", layout="wide")
st.title("⛳ PGA Predictor Pro")
st.caption("Tournament probabilities + DraftKings portfolio optimizer")

mc_path = DATA / "bank_of_utah_monte_carlo_50000_v8.csv"
model_path = DATA / "bank_of_utah_model_input_v8.csv"

if not mc_path.exists():
    st.error("Missing data/bank_of_utah_monte_carlo_50000_v8.csv")
    st.stop()

mc = pd.read_csv(mc_path)
model = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()

# Merge confidence into simulation output if necessary.
if "model_data_confidence" not in mc.columns and not model.empty:
    conf_col = "model_data_confidence_v8" if "model_data_confidence_v8" in model.columns else "model_data_confidence"
    if conf_col in model.columns:
        mc = mc.merge(model[["player", conf_col]].rename(columns={conf_col:"model_data_confidence"}), on="player", how="left")

st.sidebar.header("Tournament")
st.sidebar.success("Bank of Utah Championship — Black Desert Resort")

page = st.sidebar.radio("View", ["Model Dashboard", "10-Lineup Optimizer"])

if page == "Model Dashboard":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Field", len(mc))
    c2.metric("Simulations", "50,000")
    c3.metric("Recent-form coverage", "88 / 115")
    c4.metric("Salary cap", "$50,000")

    st.subheader("Tournament Simulation")
    display_cols = [c for c in [
        "player","salary","win_pct","top5_pct","top10_pct","top20_pct",
        "make_cut_pct","expected_finish","dk_points_proxy","dk_value_per_1000",
        "model_data_confidence"
    ] if c in mc.columns]
    sort_col = st.selectbox("Sort by", [c for c in ["win_pct","top10_pct","make_cut_pct","dk_value_per_1000","salary"] if c in mc.columns])
    ascending = sort_col == "salary"
    st.dataframe(mc.sort_values(sort_col, ascending=ascending)[display_cols], use_container_width=True, hide_index=True)

else:
    st.subheader("DraftKings Portfolio Optimizer")
    st.caption("Builds the requested number of six-golfer lineups simultaneously with exposure and uniqueness constraints.")

    a,b,c,d = st.columns(4)
    lineup_count = a.number_input("Lineups", 1, 20, 10)
    max_exposure = b.slider("Max exposure", 0.10, 1.00, 0.70, 0.05)
    min_unique = c.number_input("Minimum unique golfers", 1, 5, 2)
    salary_floor = d.number_input("Minimum salary used", 40000, 50000, 46500, 100)

    min_player_salary = st.number_input("Minimum golfer salary", 6000, 10000, 6500, 100, help="Hard rule: golfers below this salary are excluded from every lineup.")
    strategy = st.selectbox("Strategy", ["GPP Ceiling", "Balanced / Single Entry", "Cut Equity"])
    names = sorted(mc.player.dropna().astype(str).unique())
    locks = st.multiselect("Lock golfers", names)
    excludes = st.multiselect("Exclude golfers", [n for n in names if n not in locks])

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
                    st.dataframe(lu[["player","salary","win_pct","top10_pct","make_cut_pct","dk_points_proxy"]], use_container_width=True, hide_index=True)
            st.subheader("Exposure")
            st.dataframe(exposure, use_container_width=True, hide_index=True)
            st.download_button("Download lineups CSV", portfolio.to_csv(index=False), "pga_lineups.csv", "text/csv")
            st.download_button("Download exposure CSV", exposure.to_csv(index=False), "pga_exposure.csv", "text/csv")
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption("Model outputs are projections, not guarantees. Refresh field, weather, tee times and withdrawals before lineup lock.")
