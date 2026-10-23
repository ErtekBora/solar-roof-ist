# İstanbul Çatı GES Uygunluk Aracı

Istanbul Rooftop Solar Suitability Tool (GreenTech / COP31, Task 3 "Build and Show").
A resident clicks a building on a map and draws roof obstacles on the roof map; the app estimates
suitability, installable capacity (kWp), annual production, savings, CO2 reduction, shading and
confidence, with an AI-written Turkish explanation. Built only from open data and AI.

> Status: **Day 0 skeleton.** The map runs; the core modules are stubs. See `SPEC.md` for the plan.

## Screenshots

<!-- TODO: add docs/roof_view.png — "Çatı görünümü": the selected building on Esri World Imagery (display only)
     with the computed panel layout, a dropped north face and a user-drawn obstacle. -->
_Screenshot placeholder: Çatı görünümü (panel yerleşimi ve çizilen engeller)._

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
- Satellite map tiles (display only, never sent to a model, stored or analysed): Esri World Imagery — "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
- Irradiation layer: Global Solar Atlas (CC BY 4.0)
- CO2 factor: ETKB TUESEF 2021; usable-area ratio: Ember; tariff: EPDK (AG tek terim mesken, 2026-04-04)

## License

AGPL-3.0 (see `LICENSE`). The optional YOLO service uses Ultralytics YOLOE, which is AGPL-3.0,
so this repository stays public under the same license.
