# PGA V11.1 Historical Calibration — Data Contract

The Results / Calibration page accepts a CSV with **one golfer per tournament**, with:
- `event_id`: stable tournament identifier (e.g., `2025-bermuda`)
- `event_date`: **tournament start date** (`YYYY-MM-DD`)
- `player`: golfer name
- `expected_round`: frozen **pre-event** projected strokes **relative to par per round** (e.g., -1.1)
- `actual_round_strokes`: realized average strokes **relative to par per completed round**, not gross strokes and not DK points (e.g., -0.5)
- Optional **pre-event** columns: `sg_approach`, `sg_ott`, `sg_arg`, `sg_putting`, `rrg_sg_bermuda_l36`, `gir_pct`, `scrambling_pct`, `form_sg_total`, `history_fit`, `verified_news_adjustment`, `verified_weather_fit`, `verified_travel_fit`.

**Never** substitute final event strokes-gained statistics for pre-event stats. Record the snapshot's capture timestamp and data source separately. No retrospective snapshot reconstruction from final stats.

Calibration requires at least three events with at least 15 golfers each. Chronological holdouts require four or more eligible events. Missing optional families are reported and their weights are redistributed; this does **not** prove the absent family is irrelevant.

Reported RMSE/MAE/bias concern strokes relative to par, **not** DraftKings points, make-cut probabilities, contest cash rates, or ROI. Those require separately captured outcomes and payout structures.

The app currently compares a fixed default allocation against the baseline, then chooses from candidates based on earlier events only. It does **not** automatically optimize weights, and no improved performance has been demonstrated without verified historical inputs.

## Data availability checkpoint (2026-10-19)
The repository's `results.csv` and `course_history.csv` are headers only. The repository's `player_stats.csv` has golfer rows but no event-dated frozen snapshots. Neither is sufficient for a valid historical backtest. RickRunGood current-week exports are not historical pre-event snapshots.
