# PGA Predictor V10.6 — Weekly Input Contract

V10.6 keeps the DraftKings field manual and automates only sources with published API/open-data access suitable for automated use.

## Required for a production weekly run
1. Tournament name
2. Course name
3. Current DraftKings field CSV with `player` and `salary`
4. Select a matched course from OpenGolfAPI, or upload `course_holes.csv`

## Automated at $0
- Course search and hole/scorecard data: OpenGolfAPI (ODbL)
- Course weather forecast: Open-Meteo (CC BY 4.0)

Automated hole data supplies par/yardage from the open source. Course-DNA fields not present in the source are left at the model's neutral defaults rather than invented.

## Advanced inputs
`player_stats.csv`, `results.csv`, and `course_history.csv` remain optional uploads/repository fallbacks. V10.6 does not scrape PGA TOUR. Repository fallback files are visibly flagged and must not be assumed current.

## Integrity rules
- A current DraftKings upload is required before simulation.
- Tournament-keyed files with no matching current tournament are blocked.
- Internet-source failures are surfaced; no silent stale substitution is performed.
- Explicit `course_holes.csv` / `weather.csv` uploads override automated sources.

## Predictive logic
V10.6 is an ingestion/UI release. It does not intentionally alter the predictive weights, Monte Carlo simulation, or optimizer logic in `pga_predictor_pro.py` / `portfolio_optimizer.py`.
