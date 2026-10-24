# AGENTS.md — Istanbul Rooftop Solar Suitability Tool

## Project
A web app that tells an Istanbul resident whether their roof is suitable for solar panels.
The user clicks a building on a map and draws roof obstacles on the roof map. The app returns:
- suitability score (%)
- installable capacity (kWp)
- annual production (kWh)
- savings range (TL)
- CO2 reduction (t/yr)
- shading heatmap
- confidence level
- an AI-written explanation in Turkish

Context: GreenTech / COP31 program, Task 3 "Build and Show". Deadline: 2026-10-21 23:59 (Europe/Istanbul).
Solo developer. Principle: **narrow but working beats feature-rich but broken.**

`SPEC.md` is the functional source of truth (what to build). This file defines how to work.

## Working style (mandatory)
1. **Plan first.** Before writing code, show a short plan (files to create or change, functions, tests) and WAIT for approval.
2. **One task at a time.** Only touch files the current task needs.
3. **Tests for every module.** Use pytest. Tests must run WITHOUT network: mock all HTTP (PVGIS, YOLO service) and all Gemini calls.
4. **Finish every task with a walkthrough:**
   - what changed
   - how to run it
   - test results
   - what is real vs placeholder
5. **No extra features.** Build nothing beyond the current stage in SPEC.md. If something is unclear, ask.
6. **Never guess API parameters.** Check the official docs (PVGIS, pyarrow, shapely, google-genai, Ultralytics, Render) and cite the doc page in the plan.

## Tech stack
- Python 3.11
- UI: streamlit, streamlit-folium, folium; charts with Altair (ships with Streamlit)
- Network and geometry: requests, shapely, pyproj
- Science: pvlib, pandas, numpy, pillow
- AI: google-genai (Gemini API)
- Tests: pytest
- Pin exact versions (`==`) in requirements.txt.
- **Do NOT use** osmnx, geopandas, torch or any local ML model in the main app. They are too heavy for free hosting (~512 MB RAM).
  - Buildings come from the local Overture extract `data/buildings/*.parquet`, read with pyarrow (already a Streamlit dependency). The app only reads the precomputed `height_ml_m` column; do NOT add lightgbm, overturemaps, duckdb or geopandas.
  - Overpass is only a possible future refresh source; the app does not call it.
  - Use shapely + pyproj for geometry.
- YOLO runs ONLY in the separate `yolo_service/` (FastAPI + onnxruntime, no torch).
- Model preparation and validation notebooks run on **Kaggle** (torch and ultralytics are allowed there).

## Repository layout
```
app.py                    Streamlit UI (Turkish)
core/config.py            constants, model names, get_gemini_api_key(), get_yolo_url()
core/buildings.py         local Overture buildings (GeoParquet), click lookup, neighbours, height rule
core/horizon.py           72-value horizon from neighbours (+ terrain), main shading direction
core/solar.py             PVGIS PVcalc + printhorizon, caching, retries
core/vision.py            Gemini roof perception (surface, obstacles, roof type, image view)
core/yolo_client.py       HTTP client for yolo_service (optional, fail-safe)
core/fusion.py            merge Gemini + YOLO detections
core/calc.py              usable area, tariff and bill functions, confidence
core/building_mode.py     apartment-building mode: flat count, roof segments, equal sharing, payback
core/layout.py            hip/gable roof faces, panel layout (vectorized grid search), capacity priority rule
core/drawings.py          roof-view drawings: Draw GeoJSON parsing, labels, generations
core/report.py            "Hesap dökümü", PVGIS verification links, downloadable report (no LLM)
core/cost.py              installation cost bands (numbers from config only)
core/metrics.py           shading heatmap (month x hour), monthly chart data, Istanbul comparison
core/explain.py           grounded Turkish explanation + number check + template fallback
core/agent.py             Roof Agent (Gemini function calling over core tools)
yolo_service/             FastAPI app: POST /detect, GET /health, model/*.onnx
scripts/                  one-off scripts (e.g. build_istanbul_reference.py)
notebooks/                Kaggle notebooks (ONNX export, validation, solar atlas crop)
data/buildings/             Overture extract with ML height + height_model_report.json
data/                     demo_buildings.json, istanbul_reference.json, istanbul_gti layer
tests/                    pytest, one test file per module
```

## Hard rules
- **Coordinates:** input is WGS84 (EPSG:4326). Metric geometry uses UTM 35N (EPSG:32635).
- **Horizon format:** a list of exactly 72 elevation angles in degrees, 5° steps. Index 0 = North, then clockwise (index 18 = East, 36 = South, 54 = West). This matches PVGIS `userhorizon`.
- **PVGIS aspect convention:** 0 = South, 90 = West, -90 = East. PVGIS outputs (e.g. `printhorizon`) also use South-based azimuths; always convert them to the North-based horizon format above.
- **PVGIS calls:**
  - server-side only
  - cached with `st.cache_data` (or an in-module cache for non-UI code)
  - retry on HTTP 429 and 5xx (500, 502, 503, 504, 529) up to 3 times with 1/2/4 s backoff
  - max 30 requests/second
  - de-duplicate before submitting: build the set of unique requests first, then run them in parallel (max 4 workers); the module cache is guarded per key so the same request never fires twice
- **Building data:**
  - the file is loaded lazily on first use (never at import time); a missing or unreadable file returns None and the UI falls back to the manual area
  - all footprints are projected once to EPSG:32635 at load time and indexed with a shapely STRtree; per click only the clicked point is projected
  - the prototype covers only Erenköy / Suadiye / Göztepe (`BUILDINGS_BBOX`); outside it the status is `kapsam_disi`, never confused with `bina_bulunamadi`
- **Building height rule, always in this order:**
  1. `"user"`: floors typed by the user × 3 m (target building only)
  2. `"height_tag"`: `height`
  3. `"levels_tag"`: `num_floors` × 3 m
  4. `"ml_estimate"`: `height_ml_m` (LightGBM estimate, precomputed on Kaggle)
  5. `"neighbour_median"`: median of known neighbour heights within 150 m (known = height or num_floors only, never estimates)
  6. `"default"`: median of all known heights in the loaded file (`FALLBACK_HEIGHT_M` = 15 m only when the file has none)

  Always record the source of each height as one of these six labels.
- **Secrets:**
  - The API key is read only through `core/config.get_gemini_api_key()`: env var `GEMINI_API_KEY` first, then `st.secrets["GEMINI_API_KEY"]`.
  - Wrap the `st.secrets` access in try/except: Streamlit raises an error when no secrets file exists (e.g. on Render, where only env vars are used).
  - If the key is missing, show a friendly Turkish error.
  - Never log, print or display the key.
  - `.streamlit/secrets.toml` must be in .gitignore.
- **Privacy (KVKK):** Store no personal data.
- **Attribution:** always show "© OpenStreetMap katkıcıları · Bina verisi: Overture Maps Foundation (ODbL), Microsoft ML Buildings · Güneş verisi: PVGIS (JRC)".
- **No Google Maps / Google Earth imagery** anywhere in the AI pipeline (Terms of Service). Esri imagery is never sent to any server, to Gemini or to any remote model, and never stored. The only processing is an optional in-browser super-resolution model (Real-ESRGAN, static/models) whose output stays in the browser. Keep the Esri attribution visible.
- **Numbers never come from an LLM.** Tariff, CO2 factor, installation cost bands and all results come from config or tools. Savings use the EPDK two-tier tariff in `core/calc.py`, valued PER FLAT after the production is split equally among the flats (bill-based when the user gives an average monthly flat bill, a wide fallback range otherwise).
- **Language:** UI text in Turkish. Code, comments, docstrings and commit messages in English.
- **Licensing:** Ultralytics (YOLOE) is AGPL-3.0, so the GitHub repo stays public. Keep a LICENSE file and credit all data sources in README.
- **Robustness:**
  - The app must never crash because an optional part failed (YOLO service, Gemini, building data).
  - Degrade gracefully, lower the confidence and explain why in Turkish.

## AI architecture (core of the solution)
The solution has two AI components:
1. **Building height estimate:** LightGBM, precomputed on Kaggle into the `height_ml_m` column (the app only reads it).
2. **Explanation (Gemini):** a grounded Turkish explanation where every reason is tied to a metric.

Rules:
- The agent and the explainer NEVER invent numbers. `explain.py` checks every number in the text against the tool outputs; on mismatch it retries once, then falls back to a template.
- Log every tool call. Show the log in a collapsible "YZ'nin adımları" section.
- Model names live in config only:
  - `AGENT_MODEL`: Flash class
  - `EXPLAIN_MODEL`: Flash-Lite class

  Verify current model IDs in the Gemini API docs.

## Deployment
- Primary: Render, two free web services from the same repo. Backup: Streamlit Community Cloud.
- **Service 1 (app):**
  - Build: `pip install -r requirements.txt`
  - Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
  - Env vars: `GEMINI_API_KEY`, `YOLO_URL`
- **Service 2 (YOLO):**
  - Root directory: `yolo_service`
  - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Python version on Render:**
  - Pin it with a `.python-version` file containing `3.11`, in the repo root AND in `yolo_service/`.
  - Render's default for new services is a much newer Python, which may break pinned packages.
  - If the `PYTHON_VERSION` env var is used instead, it must be a fully qualified version (e.g. `3.11.9`).
- Keep the app's memory under ~400 MB: no heavy imports at module top level when avoidable.
- Free services sleep after 15 min of inactivity. The app must handle a slow (~60 s) YOLO wake-up with a timeout and a Gemini-only fallback.
