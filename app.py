import streamlit as st
import pandas as pd
from pathlib import Path

from pga_predictor_pro import Config, predict, optimize_lineups

st.set_page_config(
    page_title="PGA Predictor Pro",
    page_icon="⛳",
    layout="wide"
)

st.title("⛳ PGA Predictor Pro")
st.caption(
    "Tournament predictions • Course fit • Weather • "
    "Monte Carlo simulation • DFS analytics"
)

st.sidebar.header("Tournament Settings")

tournament = st.sidebar.text_input(
    "Tournament",
    value="PGA Tournament"
)

simulations = st.sidebar.select_slider(
    "Monte Carlo Simulations",
    options=[1000, 5000, 10000, 25000, 50000, 100000],
    value=25000
)

seed = st.sidebar.number_input(
    "Simulation Seed",
    min_value=1,
    max_value=999999,
    value=42
)

st.sidebar.divider()

st.sidebar.header("DFS Settings")

salary_cap = st.sidebar.number_input(
    "DraftKings Salary Cap",
    min_value=40000,
    max_value=60000,
    value=50000,
    step=100
)

number_lineups = st.sidebar.slider(
    "Number of Lineups",
    min_value=1,
    max_value=100,
    value=20
)

st.sidebar.divider()

st.sidebar.info(
    "PGA Predictor Pro combines recent form, strokes gained, "
    "course history, course architecture, approach-distance fit, "
    "weather and Monte Carlo simulation."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏆 Tournament Predictor",
        "📊 Player Explorer",
        "💰 DFS",
        "🗂 Data"
    ]
)

with tab1:

    st.header("Tournament Prediction")

    st.write(
        "Run the model to estimate each golfer's probability "
        "of winning, finishing Top 5/10/20, making the cut, "
        "and expected tournament finish."
    )

    if st.button(
        "▶ Run Tournament Simulation",
        type="primary",
        use_container_width=True
    ):

        try:

            config = Config(
                tournament=tournament,
                sims=int(simulations),
                seed=int(seed),
                salary_cap=int(salary_cap)
            )

            with st.spinner(
                f"Running {simulations:,} tournament simulations..."
            ):

                predictions = predict(config)

            st.session_state["predictions"] = predictions

            st.success(
                f"Completed {simulations:,} simulations."
            )

        except Exception as e:

            st.error(
                "The model could not run yet."
            )

            st.exception(e)

    if "predictions" in st.session_state:

        predictions = st.session_state["predictions"].copy()

        st.subheader("Model Leaders")

        c1, c2, c3, c4 = st.columns(4)

        if len(predictions):

            leader = predictions.iloc[0]

            c1.metric(
                "Model #1",
                leader["player"]
            )

            if "win_pct" in leader:

                c2.metric(
                    "Win Probability",
                    f"{leader['win_pct']:.1%}"
                )

            if "top10_pct" in leader:

                c3.metric(
                    "Top 10 Probability",
                    f"{leader['top10_pct']:.1%}"
                )

            if "expected_finish" in leader:

                c4.metric(
                    "Expected Finish",
                    f"{leader['expected_finish']:.1f}"
                )

        display_columns = [
            "player",
            "model_rank",
            "win_pct",
            "top5_pct",
            "top10_pct",
            "top20_pct",
            "make_cut_pct",
            "expected_finish",
            "median_finish",
            "model_strength",
            "course_history_score",
            "approach_fit",
            "weather_adj",
            "tee_wave_adj",
            "dk_proxy",
            "salary",
            "points_per_1k",
            "ownership",
            "leverage"
        ]

        display_columns = [
            c for c in display_columns
            if c in predictions.columns
        ]

        table = predictions[
            display_columns
        ].copy()

        percentage_columns = [
            "win_pct",
            "top5_pct",
            "top10_pct",
            "top20_pct",
            "make_cut_pct"
        ]

        column_config = {}

        for column in percentage_columns:

            if column in table.columns:

                column_config[column] = (
                    st.column_config.ProgressColumn(
                        column.replace("_", " ").title(),
                        format="%.1%%",
                        min_value=0,
                        max_value=1
                    )
                )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )

        st.download_button(
            "Download Predictions CSV",
            data=predictions.to_csv(
                index=False
            ),
            file_name=(
                tournament
                .replace(" ", "_")
                +
                "_predictions.csv"
            ),
            mime="text/csv"
        )


with tab2:

    st.header("Player Explorer")

    if "predictions" not in st.session_state:

        st.info(
            "Run the Tournament Predictor first."
        )

    else:

        predictions = st.session_state["predictions"]

        selected_player = st.selectbox(
            "Select Golfer",
            predictions["player"].tolist()
        )

        golfer = predictions[
            predictions["player"]
            ==
            selected_player
        ].iloc[0]

        st.subheader(selected_player)

        c1, c2, c3, c4, c5 = st.columns(5)

        if "win_pct" in golfer:
            c1.metric(
                "Win",
                f"{golfer['win_pct']:.1%}"
            )

        if "top5_pct" in golfer:
            c2.metric(
                "Top 5",
                f"{golfer['top5_pct']:.1%}"
            )

        if "top10_pct" in golfer:
            c3.metric(
                "Top 10",
                f"{golfer['top10_pct']:.1%}"
            )

        if "top20_pct" in golfer:
            c4.metric(
                "Top 20",
                f"{golfer['top20_pct']:.1%}"
            )

        if "make_cut_pct" in golfer:
            c5.metric(
                "Make Cut",
                f"{golfer['make_cut_pct']:.1%}"
            )

        st.divider()

        st.subheader("Model Factors")

        factors = {}

        factor_columns = {
            "model_strength": "Overall Model Strength",
            "course_history_score": "Course History",
            "approach_fit": "Approach Distance Fit",
            "weather_adj": "Weather Adjustment",
            "tee_wave_adj": "Tee Wave Adjustment",
            "ceiling_score": "Ceiling",
            "form_sg_total": "Recent SG Total",
            "form_sg_approach": "Recent Approach",
            "form_sg_ott": "Recent Off The Tee",
            "form_sg_arg": "Recent Around Green",
            "form_sg_putting": "Recent Putting"
        }

        for column, label in factor_columns.items():

            if (
                column in golfer.index
                and
                pd.notna(golfer[column])
            ):

                factors[label] = float(
                    golfer[column]
                )

        if factors:

            chart_data = pd.DataFrame(
                {
                    "Factor": factors.keys(),
                    "Score": factors.values()
                }
            ).set_index("Factor")

            st.bar_chart(
                chart_data
            )

            st.dataframe(
                chart_data,
                use_container_width=True
            )

        st.subheader("DFS")

        dfs1, dfs2, dfs3, dfs4 = st.columns(4)

        if "salary" in golfer:
            dfs1.metric(
                "Salary",
                (
                    f"${golfer['salary']:,.0f}"
                    if pd.notna(golfer["salary"])
                    else "N/A"
                )
            )

        if "dk_proxy" in golfer:
            dfs2.metric(
                "DK Projection",
                f"{golfer['dk_proxy']:.2f}"
            )

        if "points_per_1k" in golfer:
            dfs3.metric(
                "Value",
                f"{golfer['points_per_1k']:.2f}"
            )

        if "leverage" in golfer:
            dfs4.metric(
                "Leverage",
                f"{golfer['leverage']:.2f}"
            )


with tab3:

    st.header("DraftKings Lineup Builder")

    if "predictions" not in st.session_state:

        st.info(
            "Run the Tournament Predictor before generating lineups."
        )

    else:

        predictions = st.session_state["predictions"]

        if st.button(
            "Generate DFS Lineups",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Optimizing lineups..."
                ):

                    lineups = optimize_lineups(
                        predictions,
                        salary_cap=int(
                            salary_cap
                        ),
                        roster_size=6,
                        count=int(
                            number_lineups
                        ),
                        seed=int(seed)
                    )

                st.session_state[
                    "lineups"
                ] = lineups

            except Exception as e:

                st.error(
                    "Unable to generate lineups."
                )

                st.exception(e)

        if "lineups" in st.session_state:

            lineups = st.session_state[
                "lineups"
            ]

            st.dataframe(
                lineups,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download DFS Lineups",
                data=lineups.to_csv(
                    index=False
                ),
                file_name=(
                    tournament
                    .replace(" ", "_")
                    +
                    "_lineups.csv"
                ),
                mime="text/csv"
            )


with tab4:

    st.header("Model Data")

    st.write(
        "The prediction engine reads five data files."
    )

    required_files = {
        "players.csv":
            "Current player statistics, salary and ownership.",

        "results.csv":
            "Historical tournament and round results.",

        "course_history.csv":
            "Player history at the tournament/course.",

        "course_holes.csv":
            "Hole yardage, contours, hazards and course architecture.",

        "weather.csv":
            "Forecast conditions and tee-time weather."
    }

    for filename, description in required_files.items():

        path = Path(
            "data"
        ) / filename

        if path.exists():

            st.success(
                f"✓ {filename}"
            )

            try:

                df = pd.read_csv(
                    path
                )

                st.caption(
                    f"{description} — {len(df):,} rows"
                )

                with st.expander(
                    f"View {filename}"
                ):

                    st.dataframe(
                        df.head(100),
                        use_container_width=True
                    )

            except Exception:

                st.warning(
                    f"{filename} exists but could not be read."
                )

        else:

            st.error(
                f"✗ {filename} missing"
            )

            st.caption(
                description
            )

    st.divider()

    st.subheader(
        "Upload / Preview Data"
    )

    uploaded = st.file_uploader(
        "Upload a CSV to inspect it",
        type=["csv"]
    )

    if uploaded is not None:

        try:

            preview = pd.read_csv(
                uploaded
            )

            st.write(
                f"{len(preview):,} rows × "
                f"{len(preview.columns)} columns"
            )

            st.dataframe(
                preview,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "Could not read this CSV."
            )

            st.exception(e)


st.divider()

st.caption(
    "PGA Predictor Pro • Statistical forecasts are estimates, "
    "not guarantees of tournament outcomes."
)
