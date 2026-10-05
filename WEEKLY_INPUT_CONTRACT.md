# PGA Predictor V10.4 — Weekly Input Contract

V10.4 is tournament-agnostic. It contains no embedded tournament field, course, simulation, or model-input fallback.

## Required each week in the Streamlit app
1. Tournament name
2. Course name
3. Current-week simulation CSV
4. Matching current-week model-input CSV

### Simulation CSV — required columns
- `player`
- `salary`
- `win_pct`
- `top10_pct`
- `make_cut_pct`
- `dk_points_proxy`

Optional outputs used when present include `top5_pct`, `top20_pct`, `expected_finish`, `expected_4r_to_par_if_made_cut`, and `dk_value_per_1000`.

### Model-input CSV
Must contain `player`. Course-Fit uses whichever supported columns are available. Missing statistics are not treated as zero; coverage is tracked.

Generic Course-Fit columns include:
- `sg_approach`
- `recent5_sg_approach` or `form_sg_approach`
- `recent_ball_striking`
- `sg_ott`
- `driving_accuracy`
- `driving_distance`
- `driving_fit` (generic course-specific driving suitability input)
- `sg_putting`
- `sg_arg`
- `course_history_z`, `course_history_score`, or legacy `course_history_z_v8`
- `model_data_confidence` or versioned equivalents

## Integrity gate
The app compares the player field in both weekly files. Below 90% overlap stops the app; partial overlap above that threshold produces a warning.

## Predictive logic
V10.4 is an architecture/data-contract release. The V10.3 optimizer defaults and calibrated Course-Fit Monte Carlo methodology are preserved. No Bank of Utah or Black Desert data is bundled or used as fallback.

## Upstream generic engine
`pga_predictor_pro.py` remains CSV-driven and supports the generic source files `players.csv`, `player_stats.csv`, `results.csv`, `course_history.csv`, `course_holes.csv`, and `weather.csv` when running the upstream prediction pipeline.
