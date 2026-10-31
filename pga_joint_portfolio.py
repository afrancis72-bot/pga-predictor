"""Joint PGA six-golfer tournament evaluation.

All golfers are evaluated in the SAME underlying simulated tournament, not
by adding individual 90th-percentile outcomes.  Input columns must be
identified explicitly to prevent the NASCAR-style data-alignment defect.
"""
from __future__ import annotations
import numpy as np


def joint_candidate_metrics(candidates, scenario_players, scenario_scores, scenario_cuts,
                            strategy="GPP Ceiling", progress_callback=None):
    """Return per-candidate same-scenario score, P90/P95/P99 and 6-for-6 chance.

    candidates: iterable of frozenset[str] golfer names, six each.
    scenario_players: exact display names in the original simulation order.
    scenario_scores / scenario_cuts: 2-D arrays with exactly matching columns.
    """
    names = [str(x) for x in scenario_players]
    if len(names) != len(set(names)):
        raise ValueError("PGA joint simulations contain duplicate golfer names")
    score = np.asarray(scenario_scores, dtype=float)
    cut = np.asarray(scenario_cuts, dtype=bool)
    if score.ndim != 2 or cut.shape != score.shape or score.shape[1] != len(names):
        raise ValueError("PGA joint score/cut matrices do not match golfer column identities")
    if score.shape[0] < 100 or not np.isfinite(score).all():
        raise ValueError("PGA joint score matrix has too few scenarios or non-finite scores")
    row = {name: j for j, name in enumerate(names)}
    options = list(candidates)
    if not options:
        return {}
    if any(len(c)!=6 or any(n not in row for n in c) for c in options):
        raise ValueError("PGA joint candidate includes missing or invalid golfer identities")
    results = {}
    mode = str(strategy).lower()
    # Small batches bound memory usage even for 5,000 simulated tournaments
    # and several thousand candidate six-golfer lineups.
    for base in range(0, len(options), 96):
        group = options[base:base+96]
        indices = np.asarray([[row[name] for name in sorted(candidate)] for candidate in group], dtype=int)
        totals = score[:, indices].sum(axis=2)
        made_all = cut[:, indices].all(axis=2)
        means = totals.mean(axis=0)
        p90 = np.percentile(totals, 90, axis=0)
        p95 = np.percentile(totals, 95, axis=0)
        p99 = np.percentile(totals, 99, axis=0)
        six = made_all.mean(axis=0)
        for i, candidate in enumerate(group):
            mu, q90, q95, q99, p6 = map(float, (means[i], p90[i], p95[i], p99[i], six[i]))
            if "cut" in mode:
                # Prioritizes lineups that can get all six through the cut.
                objective = 0.50*mu + 0.20*q90 + 0.30*q95 + 130.0*p6
            elif "balanced" in mode or "single" in mode:
                objective = 0.55*mu + 0.25*q90 + 0.20*q95 + 65.0*p6
            else:
                objective = 0.35*mu + 0.30*q90 + 0.35*q95 + 25.0*p6
            results[candidate] = {
                "objective": float(objective),
                "mean": mu, "p90": q90, "p95": q95, "p99": q99,
                "six_of_six_probability": p6,
            }
        if progress_callback:
            progress_callback(min(0.94, 0.88 + 0.06*(base+len(group))/len(options)),
                              f"Evaluating simulated tournament outcomes: {min(base+len(group),len(options)):,}/{len(options):,}")
    return results
