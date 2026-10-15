# AGENTS.md — Istanbul Rooftop Solar Suitability Tool

## Project
A web app that tells an Istanbul resident whether their roof is suitable for solar panels.
The user clicks a building on a map and can optionally upload a roof photo. The app returns:
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
3. **Tests for every module.** Use pytest. Tests must run WITHOUT network: mock all HTTP (PVGIS, Overpass, YOLO service) and all Gemini calls.
4. **Finish every task with a walkthrough:**
   - what changed
   - how to run it
   - test results
   - what is real vs placeholder
5. **No extra features.** Build nothing beyond the current stage in SPEC.md. If something is unclear, ask.
6. **Never guess API parameters.** Check the official docs (PVGIS, Overpass, google-genai, Ultralytics, Render) and cite the doc page in the plan.

## Tech stack
- Python 3.11
- UI: streamlit, streamlit-folium, folium; charts with Altair (ships with Streamlit)
- Network and geometry: requests, shapely, pyproj
- Science: pvlib, pandas, numpy, pillow
- AI: google-genai (Gemini API)
- Tests: pytest
- Pin exact versions (`==`) in requirements.txt.
- **Do NOT use** osmnx, geopandas, torch or any local ML model in the main app. They are too heavy for free hosting (~512 MB RAM).
  - Query OpenStreetMap directly through the Overpass API with requests.
  - Use shapely + pyproj for geometry.
- YOLO runs ONLY in the separate `yolo_service/` (FastAPI + onnxruntime, no torch).
- Model preparation and validation notebooks run on **Kaggle** (torch and ultralytics are allowed there).

## Repository layout
```
app.py                    Streamlit UI (Turkish)
core/config.py            constants, model names, get_gemini_api_key(), get_yolo_url()
core/buildings.py         Overpass query, target building + neighbours, height rule
core/horizon.py           72-value horizon from neighbours (+ terrain), main shading direction
core/solar.py             PVGIS PVcalc + printhorizon, caching, retries
core/vision.py            Gemini roof perception (surface, obstacles, roof type, image view)
core/yolo_client.py       HTTP client for yolo_service (optional, fail-safe)
core/fusion.py            merge Gemini + YOLO detections
core/calc.py              area, kWp, score, kWh, CO2, TL, confidence
core/metrics.py           shading heatmap (month x hour), monthly chart data, Istanbul comparison
core/explain.py           grounded Turkish explanation + number check + template fallback
core/agent.py             Roof Agent (Gemini function calling over core tools)
yolo_service/             FastAPI app: POST /detect, GET /health, model/*.onnx
scripts/                  one-off scripts (e.g. build_istanbul_reference.py)
notebooks/                Kaggle notebooks (ONNX export, validation, solar atlas crop)
data/demo_images/         real roof photos + SOURCES.md (source, author, license, link)
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
  - retry on HTTP 429/529 up to 3 times with 1/2/4 s backoff
  - max 30 requests/second
- **Overpass calls:**
  - send a descriptive User-Agent
  - use a cache and a timeout
  - use a fallback endpoint if the main one fails
- **Building height rule, always in this order:**
  1. `height` tag
  2. `building:levels` × 3 m
  3. median of known neighbour heights
  4. 15 m default

  Always record the source of each height as one of: `"height_tag"`, `"levels_tag"`, `"neighbour_median"`, `"default"`.
- **Secrets:**
  - The API key is read only through `core/config.get_gemini_api_key()`: env var `GEMINI_API_KEY` first, then `st.secrets["GEMINI_API_KEY"]`.
  - Wrap the `st.secrets` access in try/except: Streamlit raises an error when no secrets file exists (e.g. on Render, where only env vars are used).
  - If the key is missing, show a friendly Turkish error.
  - Never log, print or display the key.
  - `.streamlit/secrets.toml` must be in .gitignore.
- **Privacy (KVKK):** uploaded images are processed in memory only and never written to disk. Store no personal data.
- **Attribution:** always show "© OpenStreetMap katkıcıları · Güneş verisi: PVGIS (JRC)". Demo images show their source and license.
- **No Google Maps / Google Earth imagery** anywhere in the AI pipeline (Terms of Service). Satellite tiles on the map are for display only.
- **Numbers never come from an LLM.** Tariff, CO2 factor and all results come from config or tools.
- **Language:** UI text in Turkish. Code, comments, docstrings and commit messages in English.
- **Licensing:** Ultralytics (YOLOE) is AGPL-3.0, so the GitHub repo stays public. Keep a LICENSE file and credit all data sources in README.
- **Robustness:**
  - The app must never crash because an optional part failed (YOLO service, Gemini, Overpass).
  - Degrade gracefully, lower the confidence and explain why in Turkish.

## AI architecture (core of the solution)
The solution is AI-native. Gemini has four roles:
1. **Perception:** detect the roof surface and obstacles in a roof photo; return JSON boxes (and masks when available).
2. **Classification:** roof type (flat or pitched) and image view (top-down or oblique). Requested in the same call as perception.
3. **Roof Agent:** orchestrates the core tools via function calling, handles missing data by stating assumptions, and adjusts confidence.
4. **Explanation:** a grounded Turkish explanation where every reason is tied to a metric.

Rules:
- YOLOE (via yolo_service) is a second, independent perception model. `core/fusion.py` marks detections confirmed by both models.
- The agent and the explainer NEVER invent numbers. `explain.py` checks every number in the text against the tool outputs; on mismatch it retries once, then falls back to a template.
- Log every tool call. Show the log in a collapsible "YZ'nin adımları" section.
- Model names live in config only:
  - `VISION_MODEL` and `AGENT_MODEL`: Flash class
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
