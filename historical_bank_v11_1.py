"""Historical PGA tournament-log bank: normalize, merge and query verified records.

Missing ShotLink components remain NaN. Tournament-level SG totals are not
per-round statistics and must not be treated as strokes relative to par.
"""
from __future__ import annotations

import pandas as pd

KEY = ("date", "tour", "golfer", "tournament", "course")
STATS = ("ott", "app", "bs", "arg", "putt", "sg", "t2g", "tot")
FIELDS = KEY + ("position",) + STATS
CORE = ("ott", "app", "arg", "putt")


def normalize_logs(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical CSV records; preserve absent ShotLink stats as nulls."""
    if raw is None or raw.empty:
        raise ValueError("Tournament log input is empty")
    d = raw.copy()
    required = KEY + ("position",)
    missing = [c for c in required if c not in d]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    for col in KEY + ("position",):
        d[col] = d[col].astype("string").str.strip()
    for col in KEY[1:] + ("position",):
        if d[col].isna().any() or d[col].eq("").any():
            raise ValueError("Missing " + col)
    dates = pd.to_datetime(d["date"], errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError("Invalid tournament date")
    d["date"] = dates.dt.strftime("%Y-%m-%d")
    d["tour"] = d["tour"].str.upper()
    for col in STATS:
        if col not in d:
            d[col] = float("nan")
        d[col] = pd.to_numeric(d[col].replace({"—": None, "–": None, "-": None, "": None}), errors="coerce")
    d["shotlink_complete"] = d[list(CORE)].notna().all(axis=1)
    d["year"] = dates.dt.year.astype(int)
    d["finish_status"] = d["position"].str.upper().map(
        lambda p: "CUT" if p == "CUT" else "WD" if p == "WD" else "DQ" if p == "DQ" else "FINISHED"
    )
    if d.duplicated(list(KEY)).any():
        raise ValueError("Duplicate golfer/event records in input")
    return d[list(FIELDS) + ["year", "finish_status", "shotlink_complete"]]


def merge_logs(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Idempotent merge, preferring incoming nonmissing values; flag conflicts."""
    fresh = normalize_logs(incoming)
    if existing is None or existing.empty:
        return fresh, {"added": len(fresh), "updated": 0, "unchanged": 0, "conflicts": 0}
    old = normalize_logs(existing)
    base = old.set_index(list(KEY))
    added = updated = unchanged = conflicts = 0
    for _, row in fresh.iterrows():
        key = tuple(row[k] for k in KEY)
        if key not in base.index:
            base.loc[key, ["position", *STATS]] = [row["position"], *(row[s] for s in STATS)]
            added += 1
            continue
        changed = False
        for col in ("position", *STATS):
            prior, new = base.loc[key, col], row[col]
            if pd.isna(new):
                continue
            if pd.notna(prior) and str(prior) != str(new):
                conflicts += 1
                continue  # Preserve original; manual review required.
            if pd.isna(prior):
                base.loc[key, col] = new
                changed = True
        if changed:
            updated += 1
        else:
            unchanged += 1
    combined = base.reset_index()[list(FIELDS)]
    return normalize_logs(combined), {"added": added, "updated": updated, "unchanged": unchanged, "conflicts": conflicts}


def query_logs(bank: pd.DataFrame, *, years=None, tournaments=None, golfers=None,
               courses=None, categories=None, start_date=None, end_date=None,
               shotlink_only=False) -> pd.DataFrame:
    """Filter stored records by year/event/golfer/course/date/stat categories."""
    d = normalize_logs(bank)
    for column, values in (("year", years), ("tournament", tournaments),
                           ("golfer", golfers), ("course", courses)):
        if values is not None:
            if isinstance(values, (str, int)):
                values = [values]
            d = d[d[column].isin(values)]
    if start_date is not None:
        d = d[d["date"] >= pd.Timestamp(start_date).strftime("%Y-%m-%d")]
    if end_date is not None:
        d = d[d["date"] <= pd.Timestamp(end_date).strftime("%Y-%m-%d")]
    if shotlink_only:
        d = d[d["shotlink_complete"]]
    if categories is not None:
        if isinstance(categories, str):
            categories = [categories]
        invalid = set(categories) - set(STATS)
        if invalid:
            raise ValueError("Unknown categories: " + ", ".join(sorted(invalid)))
        columns = list(KEY) + ["position", "year", "finish_status", "shotlink_complete"] + list(categories)
        return d[columns].sort_values(["date", "tournament", "golfer"]).reset_index(drop=True)
    return d.sort_values(["date", "tournament", "golfer"]).reset_index(drop=True)
