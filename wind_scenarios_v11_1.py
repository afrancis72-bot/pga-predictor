"""Shared round and tee-wave weather scenarios for PGA V11.1.

This module supplies *correlated* weather effects for downstream golf simulation.
It does not infer forecasts or claim empirical calibration.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def shared_weather_shocks(sims: int, golfer_waves: list[str], round_wind_mph: list[float],
                          seed: int = 42, stroke_penalty_per_10mph: float = 0.20,
                          wave_sd_mph: float = 3.0, day_sd_mph: float = 2.0) -> np.ndarray:
    """Return [simulation, golfer, round] shifts in strokes relative to baseline.

    Same-wave golfers share a draw; both waves share each round's day shock.
    Shifts are positive for harder conditions. Wind baseline is 10 mph.
    This is an explicitly provisional weather-response assumption.
    """
    if sims < 1 or len(round_wind_mph) != 4 or not golfer_waves:
        raise ValueError("Require positive simulations, golfer waves, and four round winds")
    wind = np.asarray(round_wind_mph, dtype=float)
    if not np.isfinite(wind).all() or (wind < 0).any():
        raise ValueError("Round wind must be finite and nonnegative")
    if stroke_penalty_per_10mph < 0 or min(wave_sd_mph, day_sd_mph) < 0:
        raise ValueError("Weather response and uncertainty cannot be negative")
    rng = np.random.default_rng(seed)
    waves = pd.Index(golfer_waves).astype(str)
    unique = list(dict.fromkeys(waves))
    wave_index = np.array([unique.index(w) for w in waves], dtype=int)
    day = rng.normal(0, day_sd_mph, (sims, 1, 4))
    wave = rng.normal(0, wave_sd_mph, (sims, len(unique), 4))
    actual = wind[None, None, :] + day + wave[:, wave_index, :]
    # The wind penalty is nonlinear: severe winds increase mistakes disproportionately.
    excess = np.maximum(actual - 10.0, 0.0) / 10.0
    return (stroke_penalty_per_10mph * excess ** 1.4).astype(np.float32)

def weather_diagnostics(shocks: np.ndarray) -> dict:
    if shocks.ndim != 3:
        raise ValueError("Expected sims x golfers x rounds")
    return {"mean_round_stroke_shift": float(shocks.mean()),
            "p90_round_stroke_shift": float(np.quantile(shocks, .90)),
            "golfer_count": int(shocks.shape[1]),
            "rounds": int(shocks.shape[2])}
