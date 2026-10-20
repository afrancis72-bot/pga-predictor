# PGA V11.1 development branch — course history and wind layer

This branch is **experimental**. The production app on main is unchanged.

## Added
- `venue_history_v11_1.py` computes distinct venue, comparable-course, and wind-conditioned historical signals.
- Prior seasons only; exponential recency weighting; field-strength adjustment if supplied.
- Sample-size shrinkage to a neutral prior; missing weather or missing history does not count as a bad performance.
- Output includes `venue_fit`, `comp_fit`, `wind_fit`, `history_fit`, and effective-round coverage.

## Expected input
`player,course,year,strokes_gained_per_round,rounds`; optional `wind_mph,field_strength_factor`.
Course names must be normalized before matching.

## Next steps before release
1. Collect verified Port Royal hole pars and historic scoring, and actual historical player/venue results. Repository placeholder CSVs do not provide these.
2. Validate comparable-course weights out-of-sample and avoid double-counting existing approach/accuracy inputs.
3. Connect this feature layer to the V11 golf-first engine, with weather scenarios shared by tee-time wave.
4. Backtest against prior years, including DK scoring and cut accuracy; benchmark against unchanged V11.
5. Upgrade the portfolio optimizer to use full simulated lineup distributions, contest payouts, and exactly eight entries (six balanced/two upside).
6. Test integration and deploy only after review.

**Do not treat this branch as production-ready or its scores as calibrated forecasts.**
