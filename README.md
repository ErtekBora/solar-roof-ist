# İstanbul Çatı GES Uygunluk Aracı

Istanbul Rooftop Solar Suitability Tool (GreenTech / COP31, Task 3 "Build and Show").
A resident clicks a building on a map and optionally uploads a roof photo; the app estimates
suitability, installable capacity (kWp), annual production, savings, CO2 reduction, shading and
confidence, with an AI-written Turkish explanation. Built only from open data and AI.

> Status: **Day 0 skeleton.** The map runs; the core modules are stubs. See `SPEC.md` for the plan.

## Run locally (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then edit the key (or set $env:GEMINI_API_KEY)
python -m pytest
streamlit run app.py
```

## Data sources and licenses

- Building data: © OpenStreetMap contributors and Microsoft ML Buildings, ODbL, via Overture Maps Foundation; building heights without a tag are an ML estimate (LightGBM trained on this data, no GlobalBuildingAtlas data)
- Solar data: PVGIS v5.3, European Commission JRC
- Satellite map tiles (display only): Esri World Imagery
- Roof photo demo set: Roboflow rooftop1 (CC BY 4.0), RID / TUM (research use), Wikimedia Commons (CC licenses); each file is listed in `data/demo_images/SOURCES.md`
- Irradiation layer: Global Solar Atlas (CC BY 4.0)
- CO2 factor: ETKB TUESEF 2021; usable-area ratio: Ember; tariff: EPDK (AG tek terim mesken, 2026-04-04)

## License

AGPL-3.0 (see `LICENSE`). The optional YOLO service uses Ultralytics YOLOE, which is AGPL-3.0,
so this repository stays public under the same license.
