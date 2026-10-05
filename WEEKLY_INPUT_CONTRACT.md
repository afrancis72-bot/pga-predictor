# PGA Predictor V10.6.2 — Weekly Input Contract

## Production weekly workflow
1. Enter tournament, course, tournament-week weather start date.
2. Paste your free Golf Courses API key (password-masked in Streamlit).
3. Add a country/region hint for international events (Japan, Bermuda, Mexico, etc.).
4. Click **Find course online**, review the ranked match, and select the correct course.
5. Upload the current DraftKings field CSV. It must contain `player` and `salary`.
6. Verify course/holes and weather show as current/automatic before simulation.
7. Upload current player_stats/results/course_history when available. Repository fallbacks are testing fallbacks, not proof of current-week freshness.
8. Build projections, then optimize.

## International course source
Golf Courses API (`golfcoursesapi.com`) is used for course search, coordinates and scorecard data. A free API key is required. Course matching uses the entered course name plus an optional country/region hint and displays ranked candidates instead of silently accepting the first response.

## Weather
Open-Meteo is queried from the selected course coordinates for the selected tournament-week start date.

## Guardrails
- No PGA TOUR scraping.
- No silent course-search failures: HTTP/auth/no-match failures are shown in the sidebar.
- DraftKings upload is required only when building projections, not when testing course/weather ingestion.
- Manual `course_holes.csv` and `weather.csv` override internet sources.
- Unknown course-DNA fields remain neutral rather than being fabricated.
- Predictive model and portfolio optimizer are not intentionally changed by V10.6.2.
