# SPEC.md — Istanbul Rooftop Solar Suitability Tool

## 1. Problem and objective
Istanbul has a large rooftop solar potential that is almost unused. The main barrier is an awareness gap: residents do not know if their roof is suitable.

Objective: given a building in Istanbul (and optionally a roof photo), produce a trustworthy, explainable suitability assessment in seconds, using only open data and AI. No physical data collection.

## 2. User flow
1. User opens the app and sees a map of Istanbul (center 41.015, 28.979, zoom 12).
2. User clicks a building (Prototype+) or a location plus a manual roof area (Demo).
3. Optional: the user uploads a roof photo or picks a demo image from `data/demo_images/`.
4. Optional: the user adjusts the electricity tariff (TL/kWh) and roof type override.
5. The app shows:
   - result cards: score, kWp, kWh, TL range, CO2, confidence
   - detection overlay on the photo
   - charts: shading heatmap, monthly production, horizon
   - the AI explanation
   - a collapsible "YZ'nin adımları" panel with the tool-call log (MVP)

## 3. Data sources (open data only)
| Data | Source | Notes |
|---|---|---|
| Building footprints, `height`, `building:levels` | OpenStreetMap via Overpass API | ODbL, attribution required |
| Production, irradiation, losses | PVGIS v5.3 `PVcalc` | JRC, free, server-side |
| Terrain horizon | PVGIS `printhorizon` | resampled to 72 values |
| Sun position, clear-sky irradiance | pvlib (local) | Haurwitz clear-sky model (zenith only) |
| Roof photos | user upload; demo set from Roboflow rooftop1 (CC BY 4.0), RID / TUM (research use), Wikimedia Commons (CC licenses) | NO synthetic images. Every demo file is listed in `data/demo_images/SOURCES.md` |
| City irradiation layer | Global Solar Atlas GTI/GHI GeoTIFF (CC BY 4.0), cropped to Istanbul on Kaggle and exported as PNG + bounds JSON | display layer only |
| Istanbul reference yield | `scripts/build_istanbul_reference.py`: PVGIS optimal at Fatih, Kadıköy, Beşiktaş, Bakırköy, Üsküdar → mean saved to `data/istanbul_reference.json` | never hard-code an invented value |
| CO2 factor | ETKB TUESEF 2021, 0.6345 tCO2/MWh | config constant |
| Tariff | EPDK residential tariff | config default set by the developer, editable in the UI |
| Usable area ratio without photo | Ember: 0.276 | config constant |

## 4. External APIs
### PVGIS PVcalc
- Endpoint: `https://re.jrc.ec.europa.eu/api/v5_3/PVcalc`
- Common params: `lat`, `lon`, `peakpower=1`, `loss=14`, `outputformat=json`
- **Ideal run:** `optimalangles=1`, `usehorizon=0`
- **Terrain run (Demo):** `optimalangles=0`, `angle` and `aspect` = the optimal values returned by the ideal run, `usehorizon=1` (PVGIS terrain horizon)
- **Shaded run (Prototype+):** `optimalangles=0`, same `angle`/`aspect`, `usehorizon=1`, `userhorizon=<72 comma-separated values>` where values = max(building horizon, terrain horizon)
- Note: when `optimalangles=1`, PVGIS ignores `angle` and `aspect`. So the ideal run must come first, and its optimal angle/aspect (in the response's `inputs` section — verify the path) are reused.
- Assumption: panels are installed at the optimal tilt and orientation. For pitched roofs the real orientation is unknown, so add this to the confidence reasons.
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

### Overpass
- Endpoint: `https://overpass-api.de/api/interpreter` (POST)
- Fallback: `https://overpass.kumi.systems/api/interpreter`
- Query: `[out:json][timeout:25]; ( way["building"](around:150,{lat},{lon}); relation["building"](around:150,{lat},{lon}); ); out geom tags;`
- Ways become polygons directly. For multipolygon relations, build the polygon from the `outer` members. Skip anything that is not a valid polygon.
- Target building = the polygon containing the click point; if none, the nearest polygon within 20 m; otherwise "bina bulunamadı" (ask for manual area).

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
| get_building | `get_building(lat, lon, radius_m=150)` | target `{id, polygon, area_m2, height_m, height_source}` + `neighbours[]` with the same fields and `distance_m` |
| compute_horizon | `compute_horizon(target, neighbours, terrain72=None)` | `{horizon72, main_shading_direction, max_angle_deg}` |
| get_solar_yield | `get_solar_yield(lat, lon, horizon72=None)` | `{ideal: {...}, shaded: {...}, angle, aspect}` |
| detect_roof_objects | `detect_roof_objects(image_bytes)` | `{roof_type, image_view, roof_area_px, obstacles[], obstacle_ratio, sources: [gemini, yolo]}` |
| compute_capacity | `compute_capacity(roof_area_m2, obstacle_ratio, roof_type, yield_ideal, yield_shaded, tariff)` | all result numbers (section 6) |
| run_scenario | `run_scenario(base_result, change)`, e.g. "move water tanks" → obstacle_ratio reduced | new numbers + delta |

**Horizon ray-casting:**
- For each neighbour, compute the azimuth from the target centroid to each neighbour vertex (and densified edge points).
- angle = atan((h_neighbour − h_target) / distance), clamped at ≥ 0.
- Exclude the target building itself; ignore neighbours that are not taller than the target; use a minimum distance of 1 m to avoid division blow-ups.
- Each of the 72 bins keeps the maximum angle.
- Final horizon = element-wise max(building horizon, terrain horizon).
- Unit test: a neighbour 30 m due west and 15 m taller → 26.57° in the West bin.

**Roof geometry and panel orientation (Prototype):**
- **Roof area** comes from the OSM footprint (in EPSG:32635), replacing the manual input. Manual input remains as a fallback when no building is found.
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
- **Two-face calculation (pitched, sources 2–3 without a single direction):**
  - run PVGIS once per face (convert compass azimuth to the PVGIS aspect convention: 0 = South, −90 = East, 90 = West)
  - each face gets half of the usable area
  - include a face only if its specific yield ≥ 80% of the ideal (optimal) yield
  - report both faces and which ones were used
- Record `yon_kaynagi` as one of `"user"`, `"osm_tag"`, `"ridge_estimate"`, `"optimal_assumed"`.
- Unit tests:
  - ridge azimuth of a rectangle rotated 30°
  - compass ↔ PVGIS aspect conversion: S → 0, E → −90, W → 90, N → 180
  - setback on a 10 × 10 m square with 1 m → 64 m²
  - empty result on a 1.5 × 1.5 m square

## 6. Calculations
| Quantity | Formula |
|---|---|
| Usable area (OSM polygon + photo, Prototype+) | area(footprint.buffer(−SETBACK_M)) × (1 − obstacle_ratio) |
| Usable area (OSM polygon, no photo) | area(footprint.buffer(−SETBACK_M)) × (0.276 ÷ 0.85), labelled "ortalama varsayım" |
| Usable area (manual area + photo) | roof_area × 0.85 × (1 − obstacle_ratio) |
| Usable area (no photo) | roof_area × 0.276, labelled "ortalama varsayım" in the UI |
| obstacle_ratio | obstacle area ÷ roof area in pixels (scale-free; union of obstacle boxes/masks clipped to the roof) |
| kWp | usable_area ÷ 6.5 (flat) or ÷ 5.5 (pitched) |
| Suitability (%) | E_shaded ÷ E_ideal × 100 |
| Shading loss (%) | 100 − suitability |
| Annual kWh | kWp × E_shaded |
| CO2 (t/yr) | kWh × 0.6345 ÷ 1000 |
| Savings range (TL/yr) | low = kWh × tariff × 0.5, high = kWh × tariff × 1.0 (self-consumption assumption shown in the UI) |

**Confidence:**
- **Yüksek:** top-down photo AND ≥ 70% of neighbour heights from tags, i.e. `height_source in ("height_tag", "levels_tag")`. Both count, because most Istanbul buildings in OSM only have `building:levels`.
- **Orta:** one of the two conditions is missing
- **Düşük:** no photo AND most heights are defaults

In the Demo stage (no building data yet), confidence depends on the photo only: top-down = Orta, oblique or no photo = Düşük.

Orientation also adds a reason (Prototype+):
- `yon_kaynagi` = `"user"` or `"osm_tag"` → positive reason ("çatı yönü biliniyor")
- `"ridge_estimate"` → "çatı yönü bina şeklinden tahmin edildi"
- `"optimal_assumed"` → "çatı yönü bilinmiyor; en uygun yön varsayıldı"

For pitched roofs, `"optimal_assumed"` prevents "Yüksek" confidence.

Always list the reasons for the confidence level.

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
  - typical order: get_building → compute_horizon → get_solar_yield → detect_roof_objects (if photo) → compute_capacity
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
  - "Ölçüm/veri" (PVGIS, OSM, photo detection)
  - "Kullanıcı girdisi"
  - "Varsayım" (0.276, 0.85, SETBACK_M, tilt defaults, self-consumption range, CO2 factor)
  - "Sonraki aşamada" (features not yet active)

  Each row shows its source.
- **Map layers:**
  - OSM (default)
  - satellite tiles (display only)
  - Istanbul irradiation layer (Prototype+)
  - demo buildings coloured by score

## 9. Output JSON (keys in Turkish, used by UI, agent and explainer)
`bina_id, lat, lon, cati_tipi, goruntu_turu, toplam_cati_alani_m2, engel_orani, engel_alani_m2, kullanilabilir_alan_m2, kurulabilir_guc_kwp, ideal_uretim_kwh_kwp, golgeli_uretim_kwh_kwp, yillik_isinim_kwh_m2, istanbul_ort_kiyas_yuzde, sicaklik_kaybi_yuzde, golge_kaybi_yuzde, ana_golge_yonu, uygunluk_puani_yuzde, yillik_uretim_kwh, aylik_uretim_kwh[12], yillik_tasarruf_tl_aralik[2], co2_ton_yil, ufuk_cizgisi_72[72], yukseklik_kaynaklari{}, panel_egimi_derece, panel_yonu_derece, yon_kaynagi, cati_yuzleri[], egim_senaryolari[], varsayimlar[], guven_seviyesi, guven_nedenleri[], tespitler[], yz_aciklama_raporu, yz_adimlari[]`

## 10. Stages and acceptance criteria
| Stage | Target date | Scope | Done when |
|---|---|---|---|
| Day 0 | Fri 16 Oct | skeleton and map (step 1, local); GitHub + first Render deploy (step 2) | step 1: the map runs locally and the smoke test passes; step 2: the public Render URL shows the map |
| Demo | Sat 17 Oct | manual roof area + roof type; photo (upload or demo) → Gemini perception + overlay; PVGIS ideal + terrain run; calc; metric cards; monthly chart; "Varsayımlar ve kaynaklar" box; explanation | on the live URL, a demo image + area gives cards + explanation without errors |
| Prototype | Mon 19 Oct | Overpass building + neighbours; roof area from OSM + geometric setback; roof orientation (user override → OSM tags → ridge estimate → optimal) with two-face PVGIS for pitched roofs and two tilt scenarios for flat roofs; horizon; shaded PVGIS run; shading heatmap; horizon chart; confidence; Istanbul reference; irradiation layer; 5 demo buildings | all 5 demo buildings give results; horizon, ridge and setback unit tests pass |
| MVP | Tue 20 Oct | Roof Agent + "YZ'nin adımları"; yolo_service on Render + fusion; number check; run_scenario | agent logs ≥ 5 tool calls; app still works with the YOLO service off |
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
