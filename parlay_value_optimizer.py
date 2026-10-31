"""Price-aware PGA 3-ball research optimization.

NO model-generated recommendation is authorized until out-of-sample win/push/loss
calibration has been independently established. The current round simulator
contains provisional assumptions. This module never fabricates sportsbook odds.
Four-from-six is 15 separate four-leg wagers; a pushed leg pays decimal 1.0.
Cross-group independence is an explicit approximation, not a validated fact.
"""
from __future__ import annotations

from itertools import combinations
import math
import numpy as np
import pandas as pd

def american_decimal(odds):
    """Convert standard American odds into total-return decimal odds."""
    v = float(odds)
    if not math.isfinite(v) or (-100 < v < 100) :
        raise ValueError("Odds must be standard American prices, e.g. +150 or -110.")
    return (1.0 + v / 100.0) if v > 0 else (1.0 + 100.0 / abs(v))

def evaluate_prices(probs: pd.DataFrame, prices: pd.DataFrame,
                    *, round_no: int | None = None, tournament: str | None = None,
                    win_haircut_pp: float = 4.0) -> pd.DataFrame:
    """Match verified 3-ball lines to simulated players and estimate priced value.

    Each price must reference an exact (group, player) from this round's simulation.
    For 3-balls, bookmaker ties push, not win. Two-ball prices require a distinct
    two-player simulation and are deliberately rejected.
    """
    required = {"group", "player", "odds"}
    if prices is None or not required.issubset(prices.columns):
        raise ValueError("Odds input needs group, player, odds columns.")
    if not 0 <= win_haircut_pp <= 25:
        raise ValueError("Probability uncertainty haircut must be 0–25 percentage points.")
    x = prices.copy()
    x = x[x["odds"].notna() & x["odds"].astype(str).str.strip().ne("")].copy()
    if x.empty:
        return pd.DataFrame(columns=["group", "player", "odds", "raw_roi", "stressed_roi"])
    x["group"] = pd.to_numeric(x["group"], errors="raise").astype(int)
    x["player"] = x["player"].astype(str).str.strip()
    if x.duplicated(["group", "player"]).any():
        raise ValueError("Duplicate sportsbook price for the same group and player.")
    if "market_type" in x and not x["market_type"].fillna("3ball").astype(str).str.lower().isin(["3ball", "3-ball", "three-ball"]).all():
        raise ValueError("Only 3-ball prices are supported. Two-ball ties and probabilities differ.")
    if "tie_rule" in x and not x["tie_rule"].fillna("push").astype(str).str.lower().eq("push").all():
        raise ValueError("3-ball tie settlement must be push; verify the sportsbook market.")
    if round_no is not None and "round" in x:
        values = pd.to_numeric(x["round"], errors="raise")
        if values.isna().any() or not values.eq(int(round_no)).all():
            raise ValueError("Bookmaker CSV is for a different round.")
    if tournament and "tournament" in x:
        values = x["tournament"].dropna().astype(str).str.strip()
        if not values.str.casefold().eq(str(tournament).strip().casefold()).all():
            raise ValueError("Bookmaker CSV tournament does not match setup.")
    x["odds"] = pd.to_numeric(x["odds"], errors="raise")
    x["decimal_odds"] = x["odds"].apply(american_decimal)

    p = probs[probs["status"].eq("OK")].copy()
    if p.duplicated(["group", "player"]).any():
        raise ValueError("Simulation output contains ambiguous group/player identities.")
    p = p[["group", "player", "win_pct", "push_pct", "loss_pct"]]
    out = x.merge(p, on=["group", "player"], how="left", validate="one_to_one")
    if out["win_pct"].isna().any():
        bad = out.loc[out["win_pct"].isna(), ["group", "player"]].to_dict("records")
        raise ValueError(f"Sportsbook prices unmatched to this round's official 3-balls: {bad[:4]}")
    for c in ["win_pct", "push_pct", "loss_pct"]:
        out[c] = pd.to_numeric(out[c], errors="raise")
    if (out[["win_pct", "push_pct", "loss_pct"]] < 0).any().any() or not np.allclose(
        out[["win_pct", "push_pct", "loss_pct"]].sum(axis=1), 100., atol=.25
    ):
        raise ValueError("Invalid win/push/loss probabilities from simulation.")
    out["p_win"] = out["win_pct"] / 100.
    out["p_push"] = out["push_pct"] / 100.
    out["p_loss"] = out["loss_pct"] / 100.
    # Stress test: transfer percentage points from WIN into LOSS; never erase pushes.
    out["stress_win"] = (out["p_win"] - win_haircut_pp / 100.).clip(lower=0.)
    out["stress_loss"] = 1. - out["stress_win"] - out["p_push"]
    out["break_even_nonpush"] = 1. / out["decimal_odds"]
    out["win_cond_nonpush"] = out["p_win"] / (out["p_win"] + out["p_loss"])
    out["raw_roi"] = out["p_win"] * (out["decimal_odds"] - 1) - out["p_loss"]
    out["stressed_roi"] = out["stress_win"] * (out["decimal_odds"] - 1) - out["stress_loss"]
    out["conditional_edge_pp"] = 100 * (out["win_cond_nonpush"] - out["break_even_nonpush"])
    out["stressed_edge_pp"] = 100 * (out["stress_win"] / (1. - out["p_push"]) - out["break_even_nonpush"])
    return out.sort_values(["stressed_roi", "raw_roi"], ascending=False).reset_index(drop=True)

def _at_least_four_win_probability(win_probs):
    # Pushes are non-wins for "4/6 winning" statistics.
    dp = np.zeros(7, dtype=float)
    dp[0] = 1.
    for p in win_probs:
        dp[1:] = dp[1:] * (1.-p) + dp[:-1] * p
        dp[0] *= 1.-p
    return float(dp[4:].sum())

def _ticket_metrics(six: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Exact independent-group expected returns for six-leg and 15 four-folds."""
    if len(six) != 6 or six["group"].nunique() != 6:
        raise ValueError("Six distinct three-ball groups are required.")
    x = six.reset_index(drop=True)
    d = x["decimal_odds"].to_numpy(float)
    p = x["p_win"].to_numpy(float)
    push = x["p_push"].to_numpy(float)
    stress = x["stress_win"].to_numpy(float)
    factors = p * d + push
    stressed_factors = stress * d + push
    fours = list(combinations(range(6), 4))
    rr = np.array([np.prod(factors[list(ix)]) for ix in fours])
    rr_stress = np.array([np.prod(stressed_factors[list(ix)]) for ix in fours])
    rr_rows = [{
        "four_leg": i+1,
        "players": " | ".join(x.iloc[list(ix)]["player"]),
        "decimal_odds_without_push": float(np.prod(d[list(ix)])),
        "model_roi": float(rr[i]-1),
        "stressed_roi": float(rr_stress[i]-1),
    } for i,ix in enumerate(fours)]
    m = {
        "players": " | ".join(x["player"]),
        "groups": " | ".join(map(str, x["group"])),
        "all_six_win_pct": float(100 * np.prod(p)),
        "at_least_four_win_pct": float(100 * _at_least_four_win_probability(p)),
        "six_leg_roi": float(np.prod(factors) - 1),
        "six_leg_stressed_roi": float(np.prod(stressed_factors) - 1),
        "round_robin_4of6_roi": float(rr.mean() - 1),
        "round_robin_stressed_roi": float(rr_stress.mean() - 1),
        "round_robin_bets": 15,
        "total_bets_if_both": 16,
    }
    return m, pd.DataFrame(rr_rows)

def optimize_round_prices(priced: pd.DataFrame, *, ticket_count: int = 3,
                          max_shared_players: int = 4, min_stressed_leg_roi: float = .05,
                          min_stressed_ticket_roi: float = .05,
                          historically_calibrated: bool = False):
    """Select risk-adjusted six-group research tickets; never force an actual bet.

    Price comparison is meaningful only if inputs are correct. A ticket becomes
    'BET CANDIDATE' only if independent historical calibration is certified by
    caller AND every individual leg plus both ticket structures clears the
    conservative EV threshold. Production UI always passes False until audit.
    """
    if ticket_count < 1 or ticket_count > 10 or not 0 <= max_shared_players <= 5:
        raise ValueError("Invalid ticket count or maximum overlap.")
    if not 0 <= min_stressed_leg_roi <= 1 or not 0 <= min_stressed_ticket_roi <= 1:
        raise ValueError("EV thresholds must be fractions between zero and one.")
    if priced is None or priced.empty:
        return {"decision": "NO BET — no verified sportsbook prices", "tickets": pd.DataFrame(),
                "four_leg_bets": pd.DataFrame(), "qualified_legs": pd.DataFrame()}
    qualified = priced.loc[(priced["stressed_roi"] >= min_stressed_leg_roi) &
                           (priced["stressed_edge_pp"] > 0)].copy()
    # A single golfer per 3-ball, selected by stressed value, not raw likelihood.
    qualified = (qualified.sort_values("stressed_roi", ascending=False)
                 .drop_duplicates("group").head(12).reset_index(drop=True))
    if qualified["group"].nunique() < 6:
        return {"decision": f"NO BET — {len(qualified)} of 6 minimum distinct groups clear the stress test",
                "tickets": pd.DataFrame(), "four_leg_bets": pd.DataFrame(), "qualified_legs": qualified}
    candidate = []
    for ix in combinations(range(len(qualified)), 6):
        six = qualified.iloc[list(ix)]
        metrics, legs = _ticket_metrics(six)
        if (metrics["six_leg_stressed_roi"] < min_stressed_ticket_roi or
                metrics["round_robin_stressed_roi"] < min_stressed_ticket_roi):
            continue
        # Better-priced upside under a modest risk penalty for thin marginal edge.
        rank = .5 * metrics["six_leg_stressed_roi"] + .5 * metrics["round_robin_stressed_roi"]
        candidate.append((rank, six, metrics, legs))
    candidate.sort(key=lambda v: v[0], reverse=True)
    chosen, rows, four = [], [], []
    for _, six, metrics, fours in candidate:
        players = set(six["player"])
        if any(len(players & prev) > max_shared_players for prev in chosen):
            continue
        chosen.append(players)
        ticket_no = len(chosen)
        metrics["ticket"] = ticket_no
        metrics["decision"] = "BET CANDIDATE" if historically_calibrated else "RESEARCH ONLY — UNCALIBRATED"
        rows.append(metrics)
        fours.insert(0, "ticket", ticket_no)
        four.append(fours)
        if len(chosen) == ticket_count:
            break
    if not rows:
        decision = "NO BET — no six-leg plus 4-of-6 portfolio clears the stress test"
    elif historically_calibrated:
        decision = "BET CANDIDATES — verify live odds and betting terms before placing"
    else:
        decision = "NO BET RECOMMENDED — model not historically calibrated; research candidates only"
    return {"decision": decision, "tickets": pd.DataFrame(rows),
            "four_leg_bets": pd.concat(four, ignore_index=True) if four else pd.DataFrame(),
            "qualified_legs": qualified}
