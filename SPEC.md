# SPEC.md — Istanbul Rooftop Solar Suitability Tool

## 1. Problem and objective
Istanbul has a large rooftop solar potential that is almost unused. The main barrier is an awareness gap: residents do not know if their roof is suitable.

Objective: given a building in Istanbul (and optionally a roof photo), produce a trustworthy, explainable suitability assessment in seconds, using only open data and AI. No physical data collection.

**Product goal (Bina modu, the unit is the APARTMENT BUILDING):** flat owners in an apartment building decide together to install ONE rooftop PV system, pay once, and share the production EQUALLY among all flats to cut their electricity bills. The building manager (or any resident) clicks the building on the map; the app uses all available data and asks the minimum. There is no single-flat framing anywhere in the UI. Public and institutional buildings are out of scope. See section 6b.

## 2. User flow
1. User opens the app and sees a map of Istanbul (centre 40.9720, 29.0760, zoom 15, with the covered Erenköy / Suadiye / Göztepe area outlined).
2. User clicks a building (Prototype+) or a location plus a manual roof area (Demo).
3. Optional: the user uploads a roof photo or picks a demo image from `data/demo_images/`.
4. Steps in the main area after a building is found (see section 6b): (a) building summary (footprint, setback area, floors/height and its source); (b) "Daire sayısı" pre-filled with the estimate, "Tahmini — farklıysa düzeltin"; (c) a prominent "Çatı fotoğrafı yükleyin (önerilir)" step with guidance, a file uploader and "Fotoğrafsız devam et" (bundled demo photos only under "Örnek fotoğrafla dene", never applied to the real building); (d) optional: Çatı yapısı (+ "Düz teras payı" slider, pre-filled from the photo) and the average monthly flat bill; (e) "Hesapla" and the results. The sidebar only holds "Sıfırla", help and the attribution. Only the bill survives a new map click; "Sıfırla" clears it too.
5. The app shows:
   - building cards (kWp, kWh, CO2, total installation cost range) and per-flat cards (one-time cost, annual savings, payback, bill coverage), plus score and confidence
   - detection overlay on the photo
   - charts: shading heatmap, monthly production, horizon
   - the AI explanation
   - a collapsible "YZ'nin adımları" panel with the tool-call log (MVP)

## 3. Data sources (open data only)
| Data | Source | Notes |
|---|---|---|
| Building footprints, `height`, `num_floors`, `height_ml_m` | Local Overture Maps extract `data/buildings/erenkoy_suadiye_goztepe_ml.parquet` (OpenStreetMap + Microsoft ML Buildings), ML height added on Kaggle | ODbL, attribution required; Prototype covers lon 29.050–29.100, lat 40.950–40.990 only |
| Production, irradiation, losses | PVGIS v5.3 `PVcalc` | JRC, free, server-side |
| Terrain horizon | PVGIS `printhorizon` | resampled to 72 values |
| Sun position, clear-sky irradiance | pvlib (local) | Haurwitz clear-sky model (zenith only) |
| Roof photos | user upload; demo set from Roboflow rooftop1 (CC BY 4.0), RID / TUM (research use), Wikimedia Commons (CC licenses) | NO synthetic images. Every demo file is listed in `data/demo_images/SOURCES.md` |
| City irradiation layer | Global Solar Atlas GTI/GHI GeoTIFF (CC BY 4.0), cropped to Istanbul on Kaggle and exported as PNG + bounds JSON | display layer only |
| Istanbul reference yield | `scripts/build_istanbul_reference.py`: PVGIS optimal at Fatih, Kadıköy, Beşiktaş, Bakırköy, Üsküdar → mean saved to `data/istanbul_reference.json` | never hard-code an invented value |
| CO2 factor | ETKB TUESEF 2021, 0.6345 tCO2/MWh | config constant |
| Tariff | EPDK AG tek terim mesken, yürürlük 2026-04-04 (`TARIFF_SOURCE`) | config constants, two tiers; the user gives a monthly bill instead of a tariff |
| Usable area ratio without photo | Ember: 0.276 | config constant |

## 4. External APIs
### PVGIS PVcalc
- Endpoint: `https://re.jrc.ec.europa.eu/api/v5_3/PVcalc`
- Common params: `lat`, `lon`, `peakpower=1`, `loss=14`, `outputformat=json`
- **Ideal run:** `optimalangles=1`, `usehorizon=0`
- **Terrain run (Demo):** `optimalangles=0`, `angle` and `aspect` = the optimal values returned by the ideal run, `usehorizon=1` (PVGIS terrain horizon)
- **Shaded run (Prototype+):** `optimalangles=0`, same `angle`/`aspect`, `usehorizon=1`, `userhorizon=<72 comma-separated values>` where values = max(building horizon, terrain horizon)
- Note: when `optimalangles=1`, PVGIS ignores `angle` and `aspect`. So the ideal run must come first, and its optimal angle/aspect (in the response's `inputs` section — verify the path) are reused.
- Assumption: the flat segment is installed at the optimal tilt and orientation. Pitched faces use a fixed slope and the face azimuth (**Face run:** `optimalangles=0`, `angle=DEFAULT_ROOF_PITCH_DEG` (30), `aspect=<face>`, `usehorizon=1`; one cached call per face, `core/solar.get_face_yield`).
- Read from the response (verify field names in the docs):
  - `E_y` (kWh/kWp/yr)
  - `E_m` (12 monthly values)
  - `H(i)_y` (kWh/m²/yr)
  - `l_tg` (temperature loss %)
  - `l_total`
  - the optimal angle and aspect

### PVGIS printhorizon
- Endpoint: `https://re.jrc.ec.europa.eu/api/v5_3/printhorizon?lat=..&lon=..&outputformat=json`
- Resample to 72 values (North = 0, clockwise, 5° steps).
- **Azimuth conversion is mandatory:** PVGIS outputs use a South-based azimuth (0 = South, -90 = East, +90 = West, ±180 = North).
  - Convert with `north_based = (pvgis_az + 180) % 360`.
  - Confirm the convention against one real `printhorizon` response before coding (e.g. a hill known to be on one side).
  - Add a unit test: PVGIS azimuth 0 → index 36 (South), -90 → index 18 (East), 90 → index 54 (West).

### Buildings (local Overture extract; Overpass is only a possible future refresh source)
- Public Overpass servers repeatedly answered "502 at capacity", so the app does not depend on them.
- Loading: `pyarrow.parquet.read_table` on first use (columns id, geometry, height, num_floors, height_ml_m, is_underground, class, sources); a missing or unreadable file returns None and the UI uses the manual area.
- Cleaning at load: WKB decoded, underground and empty geometries skipped, invalid ones repaired with `buffer(0)`, a MultiPolygon reduced to its largest part, everything projected once to EPSG:32635, parts under 1 m² dropped. Built into a shapely STRtree.
- Lookup `find_building(index, lat, lon)`: outside the bbox → `kapsam_disi`; inside a footprint → that building; else the nearest footprint within 15 m; else `bina_bulunamadi`. Overlapping footprints: prefer `OpenStreetMap`, then the largest area (same tie-break for equal distances).
- Neighbours: all footprints within 150 m (metric) of the target.
- Height rule and source labels: see AGENTS.md. `roof_shape`, `roof_direction` and `roof_material` are ignored (almost empty; in Overture `roof_direction` is the ridge bearing, not a facing direction).
- Model card: parquet schema metadata key `height_ml` (bytes), else `data/buildings/height_model_report.json`, else None (the UI then omits the MAE sentence).

### Gemini API (google-genai)
- `client = genai.Client(api_key=get_gemini_api_key())`
- Perception: `generate_content` with image + prompt and `response_mime_type="application/json"`.
- `box_2d` = [ymin, xmin, ymax, xmax], normalized 0–1000.
- Agent: function calling with the tools in section 5. Use automatic function calling or a manual loop, whichever lets us log each call.
- Cache perception results by image hash.

### YOLO service (ours)
- `POST {YOLO_URL}/detect` (multipart image) → `[{"label", "confidence", "box": [x1, y1, x2, y2]}]` in pixels
- `GET {YOLO_URL}/health`
- Timeout 60 s. On any failure return `None` and continue with Gemini only.

## 5. Tools (pure functions in core/, each unit-tested)
| Tool | Signature (indicative) | Returns |
|---|---|---|
| get_building | `find_building(index, lat, lon)` | `{status, message, building: {id, polygon, area_m2, setback_polygons, setback_area_m2, height_m, height_source, data_source}, neighbours[] with distance_m}` |
| compute_horizon | `compute_horizon(target, neighbours, terrain72=None)` | `{horizon72, main_shading_direction, max_angle_deg}` |
| get_solar_yield | `get_solar_yield(lat, lon, horizon72=None)` | `{ideal: {...}, shaded: {...}, angle, aspect}` |
| detect_roof_objects | `detect_roof_objects(image_bytes)` | `{roof_type, image_view, roof_area_px, obstacles[], obstacle_ratio, sources: [gemini, yolo]}` |
| analyze_building | `analyze_building(roof_area_m2, usable_base_area_m2, obstacle_ratio, image_view, structure, flat_share_pct, n_flats, n_flats_estimated, bill_tl, flat_solar, faces)` (`core/building_mode.py`; replaces the removed single-roof `compute_capacity`) | all result numbers (sections 6 and 6b) |
| run_scenario | `run_scenario(base_result, change)`, e.g. "move water tanks" → obstacle_ratio reduced | new numbers + delta |
| optimize_layout | `optimize_layout(usable_polygon_m, scenario, exclusion_zones_m=None)` | `{panel_count, kwp, panels: [rectangles], tilt, azimuth, row_pitch_m, coverage_ratio}` |

**Panel layout optimization (Prototype; tool `optimize_layout`, pure geometry, no LLM):**
- The LLM never places panels or produces panel counts. It chooses scenarios and explains results; the tool computes.
- Panel size and power are config assumptions shown in the "Varsayımlar" box: `PANEL_LENGTH_M = 1.72`, `PANEL_WIDTH_M = 1.13`, `PANEL_KWP = 0.40`.
- **Input:** the usable polygon (footprint shrunk by `SETBACK_M`, in metres, EPSG:32635) minus optional exclusion zones.
- **Search:** grid search over layout rotation (aligned to the polygon's main axis ± 0–90° in 5° steps) and grid offset (a few sub-panel shifts in x and y). Keep the layout with the most panels fully inside the polygon; break ties by the azimuth closest to south.
- **Flat roofs:**
  - panels are tilted, so rows need spacing to avoid self-shading
  - row pitch = L·cos(tilt) + L·sin(tilt) / tan(h_min), where h_min = solar elevation at solar noon on 21 December (≈ 90° − lat − 23.44°)
  - run for both tilt scenarios (optimal and `FLAT_LOW_TILT_DEG`)
- **Pitched roofs:**
  - panels lie flat on each used roof face (no row spacing)
  - each face is a half of the usable polygon split along the ridge line
- When a layout exists, kWp = panel_count × PANEL_KWP; this replaces the m²/kWp ratio. The ratio remains the fallback (manual area, no polygon).
- **Exclusion zones from the photo:**
  - the demo photo is not geo-referenced to the building, so photo obstacles are NOT placed on the map polygon
  - on the map, obstacles reduce the panel count proportionally: panel_count × (1 − obstacle_ratio), rounded down
  - for **top-down** photos only, draw a second layout on the photo itself in pixel space: roof polygon minus obstacle boxes, with metres-per-pixel = sqrt(roof_area_m2 ÷ roof_area_px). This shows panels avoiding the detected objects. Skip this for oblique photos.
- **Map display:** draw the panel rectangles on the folium map over the building (converted back to WGS84).
- **Unit tests:**
  - 10 × 10 m square, no setback, panels flat (pitched case) → at least 5 × 8 = 40 panels (floor(10 / 1.72) × floor(10 / 1.13)); assert ≥ 40 and no overlaps
  - all panels lie inside the polygon
  - an exclusion zone in the centre reduces the count
  - row pitch for tilt 30° at lat 41° matches the formula

**Horizon ray-casting:**
- For each neighbour, compute the azimuth from the target centroid to each neighbour vertex (and densified edge points).
- angle = atan((h_neighbour − h_target) / distance), clamped at ≥ 0.
- Exclude the target building itself; ignore neighbours that are not taller than the target; use a minimum distance of 1 m to avoid division blow-ups.
- Each of the 72 bins keeps the maximum angle.
- Final horizon = element-wise max(building horizon, terrain horizon).
- Unit test: a neighbour 30 m due west and 15 m taller → 26.57° in the West bin.

**Roof geometry and panel orientation (Prototype):**
- **Roof area** comes from the building footprint (Overture/OSM, in EPSG:32635), replacing the manual input. Manual input remains as a fallback when no building is found.
- **Geometric edge setback:**
  - instead of the constant EDGE_FACTOR, shrink the footprint inward by `SETBACK_M` (config, default 1.0 m, an explicit assumption): `usable_polygon = footprint.buffer(-SETBACK_M)`
  - if the result is empty (very small roof), usable area = 0, with a Turkish reason
  - EDGE_FACTOR (0.85) is used only when there is no polygon (manual area)
- **Flat roofs:**
  - panels are rack-mounted, so orientation is a design choice
  - run PVGIS for two scenarios: (a) optimal tilt/azimuth, (b) low tilt `FLAT_LOW_TILT_DEG` (config, default 10°, explicit assumption) facing south (aspect 0)
  - show both; the headline uses (a)
  - explain in Turkish that (b) fits more panels with less row shading and lower wind load
- **Pitched roofs: orientation source, in this priority order:**
  1. **User override:** UI select "Çatı yönü" (K, KD, D, GD, G, GB, B, KB) and a tilt slider "Çatı eğimi (°)".
  2. **OSM tags** `roof:direction` (degrees or compass letters) and `roof:angle`.
  3. **Ridge estimate from the polygon:**
     - take `footprint.minimum_rotated_rectangle`; its long side is the ridge direction
     - the two roof faces face perpendicular to the ridge (two opposite azimuths)
     - tilt = `DEFAULT_ROOF_PITCH_DEG` (config, default 30°, explicit assumption shown in the UI)
  4. **Fallback:** optimal orientation (today's behaviour), with a Turkish reason.
- **Two-face calculation (building mode uses the ridge estimate, source 3):**
  - run PVGIS once per face (convert compass azimuth to the PVGIS aspect convention: 0 = South, −90 = East, 90 = West, 180 = North)
  - each face gets half of the pitched PLAN area; its SURFACE area is plan ÷ cos(30°) and kWp = surface ÷ 5.5 (section 6b)
  - include a face only if its specific yield ≥ 80% of the flat segment's SHADED specific yield (same terrain-horizon basis, `FACE_MIN_YIELD_RATIO`); a dropped face contributes 0 kWp, 0 kWh and 0 cost
  - report both faces and which ones were used, dropped or unavailable
- Record `yon_kaynagi` as one of `"user"`, `"osm_tag"`, `"ridge_estimate"`, `"optimal_assumed"`.
- Unit tests:
  - ridge azimuth of a rectangle rotated 30°
  - compass ↔ PVGIS aspect conversion: S → 0, E → −90, W → 90, N → 180
  - setback on a 10 × 10 m square with 1 m → 64 m²
  - empty result on a 1.5 × 1.5 m square

## 6. Calculations
| Quantity | Formula |
|---|---|
| Usable area (building polygon + photo, Prototype+) | area(footprint.buffer(−SETBACK_M)) × (1 − obstacle_ratio) |
| Usable area (building polygon, no photo) | area(footprint.buffer(−SETBACK_M)) × (0.276 ÷ 0.85), labelled "ortalama varsayım" |
| Usable area (manual area + photo) | roof_area × 0.85 × (1 − obstacle_ratio) |
| Usable area (no photo) | roof_area × 0.276, labelled "ortalama varsayım" in the UI |
| obstacle_ratio | obstacle area ÷ roof area in pixels (scale-free; union of obstacle boxes/masks clipped to the roof) |
| kWp | building mode: flat segment = flat PLAN area ÷ 6.5; pitched face = face SURFACE area (plan ÷ cos 30°) ÷ 5.5; see section 6b |
| Suitability (%) | E_shaded ÷ E_ideal × 100 |
| Shading loss (%) | 100 − suitability |
| Annual kWh | kWp × E_shaded |
| CO2 (t/yr) | kWh × 0.6345 ÷ 1000 |
| Savings range (TL/yr, PER FLAT) | with an average flat bill: C = kwh_from_bill(bill); for each of 12 monthly values P = (segment-summed, shaded, roof-scaled production) ÷ N: saving = bill_tl(C) − bill_tl(max(C − P × share, 0)), share 1.0 (monthly netting); summed over months. Without a bill: (annual kWh ÷ N) × [LOW, HIGH], labelled "geniş tahmin: kademe bilinmiyor" |

**Tariff and bill-based savings (Prototype P1):**
- `LOW = 2.918965 × 1.26`, `HIGH = 4.320708 × 1.26` TL/kWh (EPDK, excluding tax × `TAX_FACTOR`, applied once); only consumption above `LIMIT` = 8 kWh/day × 30 = 240 kWh/month is billed at HIGH; `BILL_AT_LIMIT` = LIMIT × LOW ≈ 882.7 TL.
- `bill_tl(kwh)`, `kwh_from_bill(bill)` (None, NaN or ≤ 0 → 0 kWh) and `monthly_savings(C, production, share)` are closed-form; PV offsets the HIGH-tier kWh first; surplus beyond consumption is valued at 0.
- The production passed to the savings is ROOF-scaled: kWp × E_m of the SHADED PVGIS run (never raw E_m); the 12 values sum to the displayed annual kWh.
- Coverage = annual savings ÷ (12 × bill), capped at 100 %, now computed on the PER-FLAT share of the production. The building mode no longer shows a "one flat cannot use the production" warning; the production is already split equally (section 6b).

**Confidence:**
- **Yüksek:** top-down photo AND ≥ 70% of neighbour heights from tags, i.e. `height_source in ("height_tag", "levels_tag")`. Both count, because most Istanbul buildings in OSM only have `building:levels`.
- **Orta:** one of the two conditions is missing
- **Düşük:** no photo AND most heights are defaults

In the Demo stage (no building data yet), confidence depends on the photo only: top-down = Orta, oblique or no photo = Düşük.

**Building mode confidence** (section 6b): same base as P1 below (polygon or top-down photo → Orta, otherwise Düşük); roof structure "Bilmiyorum" drops one level; a user-set Karma counts as known; a pitched face whose PVGIS call failed (and that the chosen structure uses) drops one level; never below Düşük. Hints may include "Daire sayısını kontrol edin", "Ortalama daire faturasını girin", "Çatı yapısını seçin", "Çatı fotoğrafı yükleyin". Flat count and bill do not change the level.

**Prototype P1 confidence** (only inputs that change a shown result count; replaces the height-based rule above until P2):
- building polygon (with or without a usable top-down photo) → Orta; manual area without photo → Düşük; manual area with a top-down photo → Orta; Yüksek stays unreachable until P2 neighbour shading.
- roof type "Bilmiyorum" drops one level (not below Düşük).
- building height and the bill do NOT affect confidence in P1 (height is shown as information; the empty-bill case is labelled on the TL card only). TODO P2: height joins the confidence.
- the UI lists only actions that would improve the result (top-down photo, roof type, a bill for the TL card).
- a bundled demo photo is never applied to a real clicked building (its vision result is only displayed); a user-uploaded photo is assumed to show the clicked building.

Orientation also adds a reason (Prototype+):
- `yon_kaynagi` = `"user"` or `"osm_tag"` → positive reason ("çatı yönü biliniyor")
- `"ridge_estimate"` → "çatı yönü bina şeklinden tahmin edildi"
- `"optimal_assumed"` → "çatı yönü bilinmiyor; en uygun yön varsayıldı"

For pitched roofs, `"optimal_assumed"` prevents "Yüksek" confidence.

Always list the reasons for the confidence level.

## 6b. Building mode (Bina modu)
Added after the Prototype P1 work; the unit is the apartment building. Pure, separately tested functions: `estimate_flat_count`, `compute_segments`, `estimate_install_cost`, `allocate_equal_share`, `payback_range` (`core/building_mode.py`, `core/cost.py`). The Roof Agent will call them later.

**Inputs (sidebar, stable widget keys, always rendered).**
- *Daire sayısı* (N, min 1): pre-filled `floors = num_floors if known else max(1, round(height_m / 3))`; `N = max(1, round(floors × footprint_area_m2 / FLAT_GROSS_AREA_M2))`, `FLAT_GROSS_AREA_M2 = 120` (assumption: gross area per flat including walls and stairs). Shown as "Tahmini (kat sayısı × taban alanı ÷ 120 m²) — lütfen kontrol edin" until edited. N is clamped to ≥ 1 in every function that divides by it.
- *Ortalama aylık daire faturası (TL)*: optional, no default. No installation quote and no common-area bill are asked.
- *Çatı yapısı*: Tamamen düz / Tamamen eğimli / Karma (teras + eğimli) / Bilmiyorum (default). Karma shows the slider "Düz teras payı (%)" (0–100, step 5, default 50). A roof photo may pre-select flat / pitched / Karma at 50 % but never "Bilmiyorum"; a bundled demo photo is never applied to a real building.
- Flow: steps (d) and (e) appear once a photo is uploaded, a demo is tried or "Fotoğrafsız devam et" is pressed; the photo step restarts for every new building. Widgets of hidden steps lose their state, so the bill is mirrored in a plain session key and restored before its widget is created.
- Persistence: only the bill persists across map clicks. A different building id re-estimates N, resets the structure to "Bilmiyorum" and the slider to 50, set in `st.session_state` before the widgets are created. No per-building edits are stored. "Sıfırla" clears all of it including the bill. Structure, slider, N and bill changes recompute from the session state with no PVGIS call.

**Roof segments and capacity.** Usable PLAN area (setback + usable-share/photo logic as before) is split into a flat segment (plan × flat share) and a pitched segment (plan × (1 − flat share)).
- Flat segment: `kWp = flat plan area ÷ 6.5` (6.5 m²/kWp is per PLAN area, row spacing included); PVGIS optimal tilt/aspect (the existing `get_solar_yield` run).
- Pitched segment: ridge along the long axis of the footprint's `minimum_rotated_rectangle` (EPSG:32635); two faces with azimuths perpendicular to the ridge; slope `DEFAULT_ROOF_PITCH_DEG` (30). Each face gets half the pitched PLAN area; `face surface = face plan ÷ cos 30°`; `face kWp = surface ÷ 5.5` (5.5 m²/kWp is per SURFACE area, panels flush with the roof). North-based azimuth → PVGIS aspect (0 = S, 90 = W, −90 = E). Without a footprint (manual area) an N–S ridge is assumed (`DEFAULT_RIDGE_AZIMUTH_DEG = 0`, East/West faces).
- A face below 80 % of the flat segment's shaded specific yield is dropped: "Kuzeye bakan çatı yüzü verimsiz olduğu için hesaba katılmadı". Dropped: 0 kWp, 0 kWh, 0 cost, its area simply unused.
- "Bilmiyorum": both extremes (100 % flat, 100 % pitched) are computed; kWp, kWh, cost, savings and payback are ranges `[min, max]` over the scenarios (never assuming which is larger); confidence drops one level.
- On every "Hesapla": the flat segment (2 cached PVGIS calls: optimal + terrain) and both faces (1 cached call each) run in parallel. If a face call fails after the retries, the other segments are used and "Bir çatı yüzü için güneş verisi alınamadı; sonuç eksik olabilir" is shown (face marked unavailable). If the flat call fails, the existing error is shown.
- Totals and the 12 monthly kWh values are sums over the used segments; the monthly chart stacks the segments (grouped per scenario for "Bilmiyorum").

**Installation cost tool** (`core/cost.py`, numbers ONLY from config). `COST_SOURCE` = "Satıcı fiyat sayfaları, 2026 (KDV ve kurulum dahil; panel, inverter, konstrüksiyon, kablolama, proje ve EDAŞ başvurusu). Gerçek fiyat teklifle netleşir." `COST_BANDS_TL_PER_KWP`, derived from published 2026 totals (5 kWp 120,000–180,000 TL; 10 kWp 220,000–320,000 TL; 15–20 kWp 350,000–550,000 TL): kWp < 7.5 → 24,000–36,000 TL/kWp; 7.5 ≤ kWp < 12.5 → 22,000–32,000; kWp ≥ 12.5 → 17,500–36,700. `COST_BAND_MAX_SOURCED_KWP = 20`: above it the last band is kept and flagged "kaynak aralığı dışında, tahmin daha belirsiz".
- `estimate_install_cost(kwp) → (cost_low, cost_high, band, flags)` for ONE kWp value; kWp ≤ 0 → `(0, 0, None, ["kurulabilir panel yok"])`.
- Several scenarios: call it once per scenario and combine: total range = `[min of all cost_low, max of all cost_high]`. Per-flat range = total ÷ N.
- Always listed exclusions: çatı güçlendirme (eski binalarda), ~10–12 yılda bir inverter değişimi, bakım, panel verim kaybı, kat malikleri karar süreci.

**Equal sharing, savings, payback** (mesken tariff, existing `bill_tl`, `kwh_from_bill`, `monthly_savings`). Each month the production (shaded PVGIS run, roof-scaled, used segments only) is split equally among N flats (equal kWh per flat), and the money is shared as an EQUAL TL DISCOUNT per flat (everyone pays the installation equally).
- With a bill, **monthly netting (aylık mahsuplaşma), share = 1.0 only**: `C = kwh_from_bill(bill)`; building monthly savings = `N × monthly_savings(C, production_m ÷ N, 1.0)` (the month's production offsets that month's consumption up to the consumption; surplus 0 TL); summed to a year; the fixed discount per flat = yearly total ÷ N ("Daire başına sabit indirim", shown as TL/yıl and as the yearly average TL/ay). There is no 0.5 lower bound; the displayed ranges come only from the roof-structure scenarios and the cost bands. Shown: "Aylık mahsuplaşma varsayımıyla hesaplanmıştır." and "Daire başına faturanın yaklaşık %X–%Y kadarı karşılanır" (cap 100 %) with the remaining average monthly bill range.
- Notes under the card: "Yıllık ortalamadır; kışın daha az, yazın daha fazla indirim olur ve indirim faturayı aşamaz." and "Fiziksel paylaşımda (her daireye eşit kWh) tüketimi yüksek daireler kademe farkı nedeniyle biraz daha fazla kazanır. Herkes kurulumu eşit ödediği için tasarrufun yönetim tarafından aidattan eşit düşülmesi önerilir; bu, sistemin ortak sayaca bağlandığı yaygın uygulamayla da uyumludur." Individual flat bills are never asked for.
- Assumption (shown under the savings card and passed to the explainer): "Girilen fatura yılın her ayı için ortalama tüketim kabul edilmiştir; mevsimsel farklar (klima, ısıtıcı) dikkate alınmamıştır. Yaz aylarındaki fazla üretim 0 TL sayıldığı için tasarruf ihtiyatlı hesaplanmıştır."
- Without a bill: per-flat savings = (annual kWh ÷ N) × [LOW, HIGH] (tax included once; monthly netting removes the self-use uncertainty, only the tariff tier is unknown), labelled "geniş tahmin: kademe bilinmiyor".
- `payback_range(cost_low, cost_high, sav_low, sav_high)`: `low = cost_low ÷ sav_high`, `high = cost_high ÷ sav_low` (per flat). A savings bound below 1 TL/yr gives None for that bound ("hesaplanamaz (tasarruf çok düşük)"); a payback above `PAYBACK_MAX_YEARS` (50) is also None, so the UI never shows absurd years; both None hides the card and says why. Never raises. The payback ignores degradation, maintenance and tariff changes.
- Building cards: total kWp, annual kWh, CO₂, total cost range. Per-flat cards: one-time cost, annual savings, payback, coverage (only with a bill).
- Legal note, always visible: "Üretimin dairelere eşit paylaştırılması mevzuata, kat malikleri kararına ve dağıtım şirketinin onayına tabidir. En yaygın uygulama sistemin ortak alan sayacına bağlanmasıdır; bu durumda ortak alan tüketimini aşan üretim değerlendirilemeyebilir ve geri ödeme uzar."

**Explainer.** The payload carries: roof structure and shares, segments used/dropped/unavailable, N and whether it is estimated, building kWp/kWh/CO₂, total and per-flat cost range, per-flat savings range, payback range or "hesaplanamaz", coverage range and the constant-consumption assumption (with a bill), the cost band and its flags. The LLM computes nothing; the number check applies. Forbidden words: "maliyet" and "yatırım" are allowed only when a cost range is in the payload; "amorti" and "geri ödeme" only when a numeric payback range is in the payload; the rest of `FINANCIAL_TERMS` stays forbidden.

## 7. AI components
- **Perception prompt:**
  - find "roof surface" plus obstacles: water tank, air conditioner unit, chimney, solar panel, satellite dish, skylight, other
  - return JSON with label, `box_2d`, optional mask, `roof_type` (flat or pitched), `image_view` (top-down or oblique)
- **Single class list:** define `OBSTACLE_CLASSES` once in `core/config.py`: `["water tank", "air conditioner unit", "chimney", "solar panel", "satellite dish", "skylight"]`.
  - The Gemini prompt, the YOLOE `set_classes` call on Kaggle and `fusion.py` all use this exact list.
  - Gemini may additionally return `"other"`. YOLO never returns it, so `"other"` detections can never be "doğrulandı"; they are always "kontrol et".
  - `"roof surface"` is not an obstacle class; it is only used for the roof area.
- **Coordinate conversion:** `vision.py` must return the image width and height with its result. Gemini `box_2d` [ymin, xmin, ymax, xmax] (0–1000) → pixels: x = xmin / 1000 × width, y = ymin / 1000 × height (same for max). YOLO boxes are already in pixels [x1, y1, x2, y2].
- **Fusion:** match Gemini and YOLO boxes by IoU ≥ 0.3 (after converting both to pixels).
  - Only boxes with the same label are matched, after normalizing to `OBSTACLE_CLASSES`.
  - matched → "doğrulandı"
  - single-model → "kontrol et"
  - The UI shows an editable confirmation table.
- **Roof Agent:**
  - system prompt: Turkish-speaking solar advisor; must call tools for every number; must state assumptions; never ask for personal data
  - typical order: get_building → compute_horizon → get_solar_yield → detect_roof_objects (if photo) → optimize_layout (per scenario) → analyze_building
  - **scenario role (MVP):**
    - the agent decides which layout scenarios to run: flat → optimal tilt vs low tilt; pitched → which faces to use
    - it calls optimize_layout once per scenario and compares panel count, kWp and annual kWh
    - it recommends one scenario with a short Turkish reason (e.g. more panels vs more yield per panel)
    - all numbers come from the tool outputs
- **Explanation:**
  - input = the result JSON only
  - output: 3 reasons, each tied to a metric (irradiation vs Istanbul average, shading loss and when it happens, obstacle area); 2 recommendations; a structural-check warning
  - Number check: every number in the text must appear in the JSON (allow rounding). Retry once, then use the Turkish template.

## 8. Metrics and visuals
- **Metric cards:**
  - suitability, kWp, kWh, TL range, CO2, confidence
  - annual irradiation (kWh/m²) with % difference vs the Istanbul reference
  - specific yield (kWh/kWp), shading loss + main direction, temperature loss, obstacle ratio, usable area
- **Shading heatmap (month × hour):**
  - for the 15th of each month, hourly sun position (pvlib)
  - a cell is shaded if sun elevation < horizon at that azimuth
  - colour = lost clear-sky irradiance (Haurwitz); Altair heatmap
- **Monthly production:** ideal vs shaded (Altair grouped bars).
- **Horizon chart:** 72-value profile with sun paths for 21 Jun and 21 Dec.
- **Photo overlay:** roof (green), obstacles (red), confirmed vs check labels.
- **"Varsayımlar ve kaynaklar" box (Demo-2 onwards):** a compact table under the results. Every number shown is tagged with one of:
  - "Ölçüm/veri" (PVGIS, building footprint, photo detection)
  - "Kullanıcı girdisi"
  - "Varsayım" (0.276, 0.85, SETBACK_M, tilt defaults, self-consumption range, tariff tax factor, CO2 factor)
  - "YZ tahmini" (ML building height)
  - "Sonraki aşamada" (features not yet active)

  Each row shows its source. Rows added in Prototype P1 (numbers come from `height_model_report.json`, never typed into UI text):
  - **Bina yüksekliği:** value and source label; for `ml_estimate` it is a YZ tahmini (LightGBM) with the cross-validated MAE (6.78 m, 41 % within ±3 m; 3,376 buildings with known height, 5-fold CV, value hidden). Baselines: constant 15 m 13.05 m, neighbour median 7.86 m, GlobalBuildingAtlas raw 11.02 m. The GBA-assisted variant (5.88 m) was REJECTED: GBA heights are CC BY-NC 4.0, incompatible with ODbL data and the AGPL repo, so the shipped column contains no GBA-derived values.
  - **Kenar boşluğu:** `SETBACK_M` (Varsayım).
  - **Elektrik tarifesi:** LOW / HIGH TL/kWh incl. tax, `TARIFF_SOURCE`, tax factor approximate (Varsayım).
  - **Aylık fatura / Tasarruf aralığı:** bill-based (Kullanıcı girdisi) or wide estimate when consumption is unknown.
  - **Daire sayısı, Çatı yapısı, Paylaşım, Kurulum maliyeti, Geri ödeme, Tüketim varsayımı:** see section 6b; each row carries its status tag and source. Surplus production beyond a flat's consumption counts as 0 TL.
- **Map layers:**
  - OSM (default)
  - satellite tiles (display only)
  - Istanbul irradiation layer (Prototype+)
  - demo buildings coloured by score

## 9. Output JSON (keys in Turkish, used by UI, agent and explainer)
Building mode adds: `cati_yapisi, duz_pay_yuzde, daire_sayisi, daire_sayisi_tahmini, senaryolar[] (segmentler[], kwp, kwh, aylik_kwh[12], maliyet_tl, paylasim), kurulabilir_guc_kwp_aralik, yillik_uretim_kwh_aralik, co2_ton_yil_aralik, toplam_maliyet_tl_aralik, daire_basi_maliyet_tl_aralik, daire_basi_tasarruf_tl_aralik, geri_odeme_yil_aralik, fatura_karsilama_yuzde_aralik, kalan_fatura_tl_ay_aralik, tuketim_varsayimi, maliyet_bayraklari, uyarilar`.

`bina_id, lat, lon, cati_tipi, goruntu_turu, toplam_cati_alani_m2, engel_orani, engel_alani_m2, kullanilabilir_alan_m2, kurulabilir_guc_kwp, ideal_uretim_kwh_kwp, golgeli_uretim_kwh_kwp, yillik_isinim_kwh_m2, istanbul_ort_kiyas_yuzde, sicaklik_kaybi_yuzde, golge_kaybi_yuzde, ana_golge_yonu, uygunluk_puani_yuzde, yillik_uretim_kwh, aylik_uretim_kwh[12], yillik_tasarruf_tl_aralik[2], co2_ton_yil, ufuk_cizgisi_72[72], yukseklik_kaynaklari{}, panel_egimi_derece, panel_yonu_derece, yon_kaynagi, cati_yuzleri[], egim_senaryolari[], panel_sayisi, panel_yerlesimi[], onerilen_senaryo, varsayimlar[], guven_seviyesi, guven_nedenleri[], tespitler[], yz_aciklama_raporu, yz_adimlari[]`

## 10. Stages and acceptance criteria
| Stage | Target date | Scope | Done when |
|---|---|---|---|
| Day 0 | Fri 16 Oct | skeleton and map (step 1, local); GitHub + first Render deploy (step 2) | step 1: the map runs locally and the smoke test passes; step 2: the public Render URL shows the map |
| Demo | Sat 17 Oct | manual roof area + roof type; photo (upload or demo) → Gemini perception + overlay; PVGIS ideal + terrain run; calc; metric cards; monthly chart; "Varsayımlar ve kaynaklar" box; explanation | on the live URL, a demo image + area gives cards + explanation without errors |
| Prototype | Mon 19 Oct | Local Overture building + neighbours (P1: click lookup, ML height, bill-based savings); roof area from OSM + geometric setback; roof orientation (user override → OSM tags → ridge estimate → optimal) with two-face PVGIS for pitched roofs and two tilt scenarios for flat roofs; panel layout optimization (optimize_layout) drawn on the map; horizon; shaded PVGIS run; shading heatmap; horizon chart; confidence; Istanbul reference; irradiation layer; 5 demo buildings | all 5 demo buildings give results; horizon, ridge and setback unit tests pass |
| MVP | Tue 20 Oct | Roof Agent + "YZ'nin adımları" + scenario selection over optimize_layout; yolo_service on Render + fusion; number check; run_scenario | agent logs ≥ 5 tool calls; app still works with the YOLO service off |
| Freeze | Wed 21 Oct 18:00 | bug fixes only | screen recording done, texts written |

## 11. Validation (no field data)
- **A. Synthetic geometry tests** (pytest): horizon angle, area, kWp, score.
- **B. Detection quality** (Kaggle): recall/precision of Gemini and YOLO on labelled Roboflow rooftop1 (flat roofs) and RID (pitched roofs) samples.
- **C. Demo-image consistency:** repeated runs on the same image give stable obstacle ratios.
- **D. Physics sanity:**
  - PVGIS vs pvlib order of magnitude
  - specific yield within ~1,100–1,450 kWh/kWp for Istanbul
  - ±1 floor height sensitivity of the score

## 12. Out of scope (for this deadline)
User accounts, storing results, permits/regulation RAG, LightGBM height model, 3D/DSM per-pixel irradiance, payment or installer marketplace.
