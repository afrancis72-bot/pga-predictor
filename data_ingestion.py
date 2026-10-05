"""V10.6 internet ingestion adapters.

Only sources whose published access terms support automated API use are included.
Predictive logic lives elsewhere; this module only fetches/normalizes inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
import requests
import pandas as pd

OPEN_GOLF = "https://api.opengolfapi.org"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
UA = "PGA-Predictor-V10.6/1.0"

class IngestionError(RuntimeError):
    pass

def _get_json(url: str, *, params: dict | None = None, timeout: int = 20) -> Any:
    try:
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        raise IngestionError(f"Internet ingestion failed: {exc}") from exc

def _records(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "courses", "items"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
        # Some OpenGolf endpoints return a single object.
        if payload:
            return [payload]
    return []

def search_courses(query: str, limit: int = 10) -> list[dict]:
    if not query.strip():
        return []
    payload = _get_json(f"{OPEN_GOLF}/v1/courses/search", params={"q": query.strip(), "limit": limit})
    rows = _records(payload)
    out = []
    for row in rows:
        cid = row.get("id") or row.get("course_id") or row.get("slug")
        name = row.get("name") or row.get("course_name") or row.get("display_name")
        city = row.get("city") or (row.get("location") or {}).get("city") if isinstance(row.get("location"), dict) else row.get("city")
        state = row.get("state") or (row.get("location") or {}).get("state") if isinstance(row.get("location"), dict) else row.get("state")
        if cid and name:
            out.append({"id": str(cid), "name": str(name), "city": city or "", "state": state or "", "raw": row})
    return out

def fetch_course_detail(course_id: str) -> dict:
    payload = _get_json(f"{OPEN_GOLF}/v1/courses/{course_id}")
    rows = _records(payload)
    return rows[0] if rows else {}

def fetch_course_holes(course_id: str, tournament: str) -> pd.DataFrame:
    payload = _get_json(f"{OPEN_GOLF}/v1/courses/{course_id}/holes")
    rows = _records(payload)
    # Endpoint shapes can nest holes under a course object.
    if len(rows) == 1 and isinstance(rows[0].get("holes"), list):
        rows = [x for x in rows[0]["holes"] if isinstance(x, dict)]
    normalized = []
    for i, row in enumerate(rows, start=1):
        hole = row.get("hole") or row.get("number") or row.get("hole_number") or i
        par = row.get("par")
        yardage = row.get("yardage") or row.get("yards")
        normalized.append({
            "tournament": tournament,
            "hole": hole,
            "par": par,
            "yardage": yardage,
            # Unknown DNA fields remain neutral, matching model defaults.
            "fairway_width": 30.0,
            "rough_severity": 0.5,
            "water": 0.5,
            "bunker_density": 0.5,
            "green_size": 6000.0,
            "wind_exposure": 0.5,
            "elevation_change": 25.0,
        })
    df = pd.DataFrame(normalized)
    if not df.empty:
        df["par"] = pd.to_numeric(df["par"], errors="coerce")
        df["yardage"] = pd.to_numeric(df["yardage"], errors="coerce")
    return df

def _coords(detail: dict) -> tuple[float, float] | None:
    candidates = [detail]
    if isinstance(detail.get("location"), dict): candidates.append(detail["location"])
    if isinstance(detail.get("coordinates"), dict): candidates.append(detail["coordinates"])
    for obj in candidates:
        lat = obj.get("latitude", obj.get("lat"))
        lon = obj.get("longitude", obj.get("lon", obj.get("lng")))
        try:
            if lat is not None and lon is not None:
                return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
    return None

def fetch_weather(detail: dict, tournament: str, start_date: date | None = None, days: int = 7) -> pd.DataFrame:
    coords = _coords(detail)
    if not coords:
        raise IngestionError("Selected course did not provide latitude/longitude; upload weather.csv instead.")
    lat, lon = coords
    start = start_date or date.today()
    end = start + timedelta(days=max(1, min(days, 16)) - 1)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    payload = _get_json(OPEN_METEO, params=params)
    hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
    times = hourly.get("time", [])
    if not times:
        raise IngestionError("Open-Meteo returned no hourly forecast for the selected course/date window.")
    df = pd.DataFrame({
        "tournament": tournament,
        "time": times,
        "temperature_f": hourly.get("temperature_2m", []),
        "rain_prob": hourly.get("precipitation_probability", []),
        "wind_mph": hourly.get("wind_speed_10m", []),
        "wind_gust_mph": hourly.get("wind_gusts_10m", []),
    })
    return df
