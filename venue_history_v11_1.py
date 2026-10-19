"""PGA V11.1: transparent, sample-aware venue and comparable-course fit.

Input history: player, course, year, strokes_gained_per_round, rounds.
Optional: wind_mph, comparable_weight, field_strength_factor.
All scores are relative, not guaranteed DK projections.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = {"player", "course", "year", "strokes_gained_per_round", "rounds"}

def venue_history_features(players: pd.DataFrame, history: pd.DataFrame,
                           venue: str, comp_weights: dict[str, float],
                           current_year: int, wind_target_mph: float = 20.0,
                           prior_rounds: float = 12.0) -> pd.DataFrame:
    """Return separate, shrunk venue/comp/wind features with coverage.

    Historical inputs must be pre-event; future years are excluded.
    A missing category remains neutral (zero), never a fabricated negative.
    """
    if "player" not in players:
        raise ValueError("players requires a player column")
    missing = REQUIRED - set(history.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")
    h = history.copy()
    h["year"] = pd.to_numeric(h.year, errors="coerce")
    h["rounds"] = pd.to_numeric(h.rounds, errors="coerce")
    h["strokes_gained_per_round"] = pd.to_numeric(h.strokes_gained_per_round, errors="coerce")
    h = h[(h.year < current_year) & (h.rounds > 0) & h.strokes_gained_per_round.notna()].copy()
    h["recency"] = np.power(0.8, (current_year - h.year - 1).clip(lower=0))
    h["field_factor"] = pd.to_numeric(h.get("field_strength_factor", 1.0), errors="coerce") if "field_strength_factor" in h else 1.0
    h["field_factor"] = np.clip(h["field_factor"].fillna(1.0), 0.5, 1.5) if isinstance(h["field_factor"], pd.Series) else 1.0
    h["weight"] = h.rounds * h.recency * h.field_factor
    h["category"] = np.where(h.course.eq(venue), "venue", np.where(h.course.isin(comp_weights), "comp", "other"))
    h["weight"] *= np.where(h.category.eq("comp"), h.course.map(comp_weights).fillna(0), 1.0)
    wind = pd.to_numeric(h["wind_mph"], errors="coerce") if "wind_mph" in h else pd.Series(np.nan, index=h.index)
    h["wind_weight"] = h.weight * np.exp(-0.5 * ((wind - wind_target_mph) / 8.0) ** 2)
    h.loc[wind.isna(), "wind_weight"] = 0.0
    output = players[["player"]].drop_duplicates().copy()
    for label, subset, weight_col in [
        ("venue", h[h.category.eq("venue")], "weight"),
        ("comp", h[h.category.eq("comp")], "weight"),
        ("wind", h[h.category.isin(["venue", "comp"])], "wind_weight"),
    ]:
        sub = subset.assign(wx=subset.strokes_gained_per_round * subset[weight_col])
        g = sub.groupby("player", as_index=False).agg(weighted_sum=("wx", "sum"), effective_rounds=(weight_col, "sum"))
        g[f"{label}_fit"] = g.weighted_sum / (g.effective_rounds + prior_rounds)
        g = g.rename(columns={"effective_rounds": f"{label}_effective_rounds"})
        output = output.merge(g[["player", f"{label}_fit", f"{label}_effective_rounds"]], on="player", how="left")
        output[f"{label}_fit"] = output[f"{label}_fit"].fillna(0.0)
        output[f"{label}_effective_rounds"] = output[f"{label}_effective_rounds"].fillna(0.0)
    output["history_fit"] = (0.40 * output.venue_fit + 0.40 * output.comp_fit + 0.20 * output.wind_fit)
    output["history_coverage"] = np.minimum(1.0, (output.venue_effective_rounds + output.comp_effective_rounds) / 24.0)
    return output
