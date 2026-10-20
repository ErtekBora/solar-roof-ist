"""Streamlit UI (Turkish): Istanbul rooftop solar suitability tool."""
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import shapely
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from core import buildings, building_mode, cost, drawings, explain, layout, report
from core.config import (
    BUILDING_ZOOM,
    BUILDINGS_BBOX,
    CO2_FACTOR_T_PER_MWH,
    COST_SOURCE,
    DEFAULT_RIDGE_AZIMUTH_DEG,
    DEFAULT_ROOF_PITCH_DEG,
    EDGE_FACTOR,
    ESRI_ATTRIBUTION,
    ESRI_TILES_URL,
    EXPLAIN_MODEL,
    FLAT_GROSS_AREA_M2,
    FLAT_SHARE_DEFAULT_PCT,
    LEVEL_HEIGHT_M,
    M2_PER_KWP_FLAT,
    M2_PER_KWP_PITCHED,
    MAP_CENTER,
    MAP_ZOOM,
    MAX_UPLOAD_MB,
    PANEL_KWP,
    PANEL_LENGTH_M,
    PANEL_WIDTH_M,
    ROOF_VIEW_ZOOM,
    SETBACK_M,
    TARIFF_SOURCE,
    USABLE_RATIO_NO_PHOTO,
    MissingApiKeyError,
    get_gemini_api_key,
)
from core.formatting import percent_from_ratio, round_half_up, tr_num
from core.solar import SolarDataError, fetch_roof_yields
from core.vision import (
    OBSTACLE_LABELS_TR,
    VisionError,
    compute_obstacle_ratio,
    detect_roof_objects,
    draw_overlay,
)

log = logging.getLogger(__name__)
MONTHS_TR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
ROOF_CHOICES = list(building_mode.STRUCTURE_LABELS_TR.values())
STRUCTURE_FROM_CHOICE = {label: key for key, label in building_mode.STRUCTURE_LABELS_TR.items()}
UNKNOWN_CHOICE = building_mode.STRUCTURE_LABELS_TR["unknown"]
MIXED_CHOICE = building_mode.STRUCTURE_LABELS_TR["mixed"]
# session keys: stable widget keys, plus bookkeeping for the per-building reset
KEY_BILL, KEY_FLATS, KEY_STRUCTURE, KEY_SHARE = "bill_input", "n_flats", "roof_structure", "flat_share"
KEY_BUILDING, KEY_FLATS_ESTIMATE, KEY_PENDING_ROOF = "_input_building", "_flats_estimate", "_pending_roof"
KEY_BILL_KEEP, KEY_PHOTO_STEP, KEY_MANUAL_AREA = "_bill_keep", "_photo_step", "manual_area"
KEY_BILL_SHOWN = "_bill_shown"
KEY_PITCHED, KEY_PITCHED_KEEP, KEY_DRAWINGS, KEY_DRAW_MODE = "pitched_type", "_pitched_keep", "_roof_drawings", "draw_mode"
PITCHED_CHOICES = list(building_mode.PITCHED_TYPE_LABELS_TR.values())
PITCHED_FROM_CHOICE = {label: key for key, label in building_mode.PITCHED_TYPE_LABELS_TR.items()}
HIP_CHOICE = building_mode.PITCHED_TYPE_LABELS_TR["hip"]
DRAW_OBSTACLE, DRAW_TERRACE = "Engel çiz", "Teras çiz"
ROOF_VIEW_CAPTION = (
    "Uydu görüntüsü yalnızca gösterim içindir; tarihi farklı olabilir ve bina sınırıyla birkaç metre kayabilir."
)
NOTE_AVERAGE = "Ortalama engel payı düşüldü — engelleri çizerek netleştirin"
NOTE_PHOTO = "Fotoğraftaki engel payı düşüldü — engelleri çizerek netleştirin"
NOTE_FLAT_ASSUMED = "düz çatı varsayımıyla"
NOTE_FLAT_ASSUMED_GEOMETRY = "düz çatı varsayımıyla; çatı yapısını seçerseniz sonuç tek değere iner"
DROPPED_FACE_TOOLTIP = "Verimsiz yüz — panel konmadı"
NUMERIC_LABEL_SOURCE = {
    "ortalama": "ortalama orana göre",
    "fotograf": "fotoğraftaki engel oranına göre",
    "engel_yok": "engel yok beyanına göre",
}
TRASH_NOTE = (
    "Haritadaki çöp kutusu yalnızca son yeniden çizimden sonraki şekilleri siler; eskiler için 'Son çizimi geri al' "
    "veya 'Engelleri temizle'."
)
ZERO_PITCHED_WARNING = "Eğimli yüzler için sonuç üretilemedi; Hesapla'yı tekrar deneyin veya çatı yapısını değiştirin."
ZERO_GENERIC_WARNING = "Kurulabilir panel alanı bulunamadı; çatı yapısını veya çatı alanını kontrol edin."
FACE_STATUS_TR = {
    "kullanildi": "kullanıldı",
    "elendi": "elendi (düz bölüm veriminin %80'inin altında)",
    "alinamadi": "güneş verisi alınamadı",
    "kullanilmadi": "alan yok",
}
HELP_STEPS = (
    "1. Haritada bir apartmana tıklayın.\n"
    "2. Daire sayısını kontrol edin.\n"
    "3. Çatı fotoğrafı yükleyin (önerilir).\n"
    "4. İsteğe bağlı bilgileri girin ve Hesapla'ya basın.\n"
    "5. Çatı görünümünde engelleri çizerek sonucu netleştirin."
)
DEMO_NONE = "Seçilmedi"
PHOTO_STEP_TITLE = "Çatı fotoğrafı yükleyin (önerilir)"
PHOTO_GUIDANCE = (
    "Çatıya ya da terasa çıkıp köşelerden çekin; fotoğraf yalnızca bellekte işlenir, kaydedilmez."
)
PHOTO_BENEFIT = "Fotoğraf, ortalama kullanılabilir alan varsayımının yerine geçer ve sonucun güvenini artırır."
DEMO_DIR = Path(__file__).parent / "data" / "demo_images"
DEMO_PHOTO_NOTE = (
    "Bu, uygulamayla birlikte gelen bir örnek fotoğraftır (Pexels / Roboflow, lisans bilgisi altta); seçtiğiniz "
    "binanın fotoğrafı değildir. YZ'nin çatıda neleri tespit ettiğini göstermek için kullanılır ve hesaba katılmaz. "
    "Kendi çatınızın fotoğrafını yukarıdaki 'Çatı fotoğrafı yükleyin' adımından ekleyebilirsiniz."
)
DEMO_EXPANDER = "Örnek fotoğrafla dene (stok fotoğraf, hesaba katılmaz)"
ACCENT = "#e8710a"
ATTRIBUTION = (
    "© OpenStreetMap katkıcıları · Bina verisi: Overture Maps Foundation (ODbL), Microsoft ML Buildings · "
    "Güneş verisi: PVGIS (JRC)"
)
HEIGHT_SOURCE_TAG = {
    "user": "Kullanıcı girdisi",
    "height_tag": "Ölçüm/veri",
    "levels_tag": "Ölçüm/veri",
    "ml_estimate": "YZ tahmini",
    "neighbour_median": "Varsayım",
    "default": "Varsayım",
}


@st.cache_resource(show_spinner="Bina verisi yükleniyor…")
def get_building_index():
    """Load the local buildings file once per server process; None when it is missing or unreadable."""
    return buildings.load_buildings()


# ------------------------------------------------------------------------ map

def build_map():
    """Base map: tile layers and the layer control ONLY. Every dynamic element lives in build_overlay()."""
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Uydu (yalnızca görüntüleme)",
        control=True,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def _latlon(ring):
    """[[lon, lat], ...] -> [[lat, lon], ...] for folium."""
    return [[lat, lon] for lon, lat in ring]


def build_overlay(lookup, point):
    """One FeatureGroup with every dynamic element: coverage box, neighbours, target, setback, click marker."""
    fg = folium.FeatureGroup(name="Seçim ve bina verisi")
    min_lon, min_lat, max_lon, max_lat = BUILDINGS_BBOX
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]], color="#2563eb", weight=2, dash_array="6", fill=False,
        tooltip="Prototip kapsama alanı: Erenköy / Suadiye / Göztepe",
    ).add_to(fg)
    if lookup and lookup["status"] == "ok":
        for n in lookup["neighbours"]:
            folium.Polygon(_latlon(n["polygon"]), color="#888888", weight=1, fill=True, fill_opacity=0.05).add_to(fg)
        b = lookup["building"]
        folium.Polygon(
            _latlon(b["polygon"]), color=ACCENT, weight=3, fill=True, fill_color=ACCENT, fill_opacity=0.25,
            tooltip="Seçilen bina",
        ).add_to(fg)
        for ring in b["setback_polygons"]:
            folium.Polygon(_latlon(ring), color=ACCENT, weight=2, dash_array="5", fill=False).add_to(fg)
    if point:
        folium.CircleMarker(point, radius=5, color="#dc2626", fill=True, fill_opacity=0.9, tooltip="Tıklanan nokta").add_to(fg)
    return fg


def updated_point(stored, clicked):
    """Return (point, changed): the stored point is replaced only by a NEW, different click."""
    if not clicked:
        return stored, False
    new = (clicked["lat"], clicked["lng"])
    return (new, True) if new != stored else (stored, False)


def take_recenter(state):
    """One-shot centre/zoom request: returned once, then removed from the state."""
    return state.pop("recenter", None)


def ensure_lookup(state, index, point):
    """Building lookup for the selected point (cached in the state); a found building sets the one-shot recenter."""
    if point is None:
        return None
    cached = state.get("lookup")
    if cached and cached["point"] == point:
        return cached["result"]
    result = buildings.find_building(index, point[0], point[1])
    state["lookup"] = {"point": point, "result": result}
    if result["status"] == "ok":
        state["recenter"] = {"center": result["building"]["centroid"], "zoom": BUILDING_ZOOM}
    return result


def clear_results(state):
    """A new map click drops the stored results of the previous building."""
    for key in ("analysis", "result", "explanation", "lookup"):
        state.pop(key, None)


# ------------------------------------------------------------- calculation

def build_result(bundle, inputs, obstacle_ratio=None, image_view=None, geometry=None, capacity_source=None, drawn=None):
    """Compute the full building result from already fetched PVGIS data (no network).

    bundle: {"solar": get_solar_yield result, "faces": [{name, azimuth_deg, aspect, weight, solar|None}]}.
    geometry: per-scenario panel layout counts (user-drawn obstacles), see building_mode.analyze_building.
    """
    result = building_mode.analyze_building(
        pitched_type=inputs.get("pitched_type", "gable"),
        geometry=geometry,
        capacity_source=capacity_source,
        drawn=drawn,
        roof_area_m2=inputs["area"],
        usable_base_area_m2=inputs.get("usable_base_area_m2"),
        obstacle_ratio=obstacle_ratio,
        image_view=image_view,
        structure=inputs["structure"],
        flat_share_pct=inputs.get("flat_share_pct", FLAT_SHARE_DEFAULT_PCT),
        n_flats=inputs["n_flats"],
        n_flats_estimated=inputs.get("n_flats_estimated", False),
        bill_tl=inputs.get("bill"),
        flat_solar=bundle["solar"],
        faces=bundle["faces"],
    )
    result.update(lat=inputs["lat"], lon=inputs["lon"])
    result["pvgis_istekleri"] = report.pvgis_requests(inputs["lat"], inputs["lon"], bundle["solar"], bundle["faces"], result)
    return result


def calculate(
    lat, lon, area, structure="flat", bill=None, n_flats=1, n_flats_estimated=False,
    ridge_azimuth_deg=DEFAULT_RIDGE_AZIMUTH_DEG, pitched_type="hip",
):
    """Fetch PVGIS yields (flat + every face) and compute the full result dict (no photo, terrain shading only)."""
    bundle = fetch_solar_bundle(lat, lon, ridge_azimuth_deg, pitched_type)
    inputs = {
        "lat": lat, "lon": lon, "area": area, "structure": structure, "bill": bill, "n_flats": n_flats,
        "n_flats_estimated": n_flats_estimated, "pitched_type": pitched_type,
    }
    return build_result(bundle, inputs)


def fetch_solar_bundle(lat, lon, ridge_azimuth_deg, pitched_type="hip"):
    """PVGIS for the flat segment and the faces (de-duplicated, in parallel). A failed face becomes solar=None."""
    specs = building_mode.face_specs(ridge_azimuth_deg, pitched_type)
    fetched = fetch_roof_yields(lat, lon, [s["aspect"] for s in specs])
    return {"solar": fetched["solar"], "faces": [{**s, "solar": fetched["face_yields"].get(s["aspect"])} for s in specs]}


def footprint_of(inputs):
    """Metric footprint polygon of the selected building, or None (manual area)."""
    wkb = inputs.get("footprint_wkb_m")
    return shapely.from_wkb(wkb) if wkb else None


def pvgis_aspects(inputs):
    """Every face aspect PVGIS needs on 'Hesapla' (hip and gable together): 4 unique values for a building."""
    footprint = footprint_of(inputs)
    if footprint is None:
        specs = building_mode.face_specs(DEFAULT_RIDGE_AZIMUTH_DEG, "hip") + building_mode.face_specs(DEFAULT_RIDGE_AZIMUTH_DEG, "gable")
        return sorted({s["aspect"] for s in specs})
    return sorted({f["aspect"] for kind in layout.PITCHED_TYPES for f in layout.roof_faces(footprint, kind)})


def face_bundle(analysis, inputs, terrace=None):
    """{"solar", "faces"} for the selected pitched type, from the PVGIS data fetched on 'Hesapla' (no network).

    With a footprint the faces and their plan-area weights come from the roof geometry (minus the terrace for a
    drawn Karma terrace); without one from the ridge assumption.
    """
    pitched_type = inputs.get("pitched_type", "hip")
    footprint = footprint_of(inputs)
    if footprint is None:
        specs = building_mode.face_specs(DEFAULT_RIDGE_AZIMUTH_DEG, pitched_type)
        model = None
    else:
        model = layout.roof_model(footprint, pitched_type, terrace, "mixed" if terrace is not None else "pitched")
        weights = layout.face_weights(model["faces"])
        specs = [
            {"name": f["name"], "azimuth_deg": f["azimuth_deg"], "aspect": f["aspect"], "weight": w}
            for f, w in zip(model["faces"], weights)
        ]
    yields = analysis.get("face_yields") or {}
    for s in specs:
        if s["aspect"] not in yields:
            log.warning("No PVGIS request was made for face %s (aspect %r); fetched aspects: %s", s["name"], s["aspect"], sorted(yields))
    return {"solar": analysis["solar"], "faces": [{**s, "solar": yields.get(s["aspect"])} for s in specs]}, model


@st.cache_data(show_spinner=False, max_entries=64)
def cached_layout(footprint_wkb, pitched_type, terrace_wkb, kind, obstacle_wkbs, flat_tilt_deg, lat, skip_faces):
    """Roof model + panel layout of one scenario (pure geometry, cached by its inputs)."""
    footprint = shapely.from_wkb(footprint_wkb)
    terrace = shapely.from_wkb(terrace_wkb) if terrace_wkb else None
    model = layout.roof_model(footprint, pitched_type, terrace, kind)
    obstacles = [shapely.from_wkb(w) for w in obstacle_wkbs]
    layouts = layout.layout_model(model, flat_tilt_deg=flat_tilt_deg, lat=lat, obstacles=obstacles, skip_faces=skip_faces)
    return {"model": model, "layouts": layouts}


def _scenario_kind(flat_share):
    return "flat" if flat_share >= 1 else "pitched" if flat_share <= 0 else "mixed"


def _geometry_counts(roof):
    """{segment: {plan_m2, panels}} of one scenario's layout (geometry mode input to the calculation)."""
    model, layouts = roof["model"], roof["layouts"]
    out = {}
    if model["flat"] is not None:
        out[layout.FLAT_SEGMENT] = {"plan_m2": layout.area(model["flat"]), "panels": layouts.get(layout.FLAT_SEGMENT, {"count": 0})["count"]}
    for f in model["faces"]:
        name = layout.face_segment_name(f["name"])
        out[name] = {"plan_m2": f["plan_m2"], "panels": layouts[name]["count"] if name in layouts else 0}
    return out


def evaluate(analysis, inputs, detection, included_ids, draw=None):
    """Result + roof view for the current inputs and drawings (no network). Returns (result, ratio, roof or None).

    Priority for the usable capacity: user-drawn obstacles (panel count x 0.40 kWp) > photo obstacle ratio >
    average usable share. Without a drawing the numeric kWp stays and only that many panels are drawn.
    """
    ratio, view = photo_ratio(inputs, detection, included_ids)
    footprint = footprint_of(inputs)
    structure = inputs["structure"]
    has_draw = footprint is not None and draw is not None
    terrace = drawings.terrace_geom(draw) if has_draw and structure == "mixed" else None
    summary = drawings.summary(draw) if has_draw else {"count": 0, "area_m2": 0.0, "terrace": False, "marked": False}
    summary["terrace"] = terrace is not None
    mode, hint = ("numeric", None)
    if has_draw:
        mode, hint = layout.decide_capacity_mode(structure, summary["count"], draw["empty_confirmed"], terrace is not None)
    summary["marked"] = mode != "numeric"

    bundle, pitched_model = face_bundle(analysis, inputs, terrace)
    run_inputs = dict(inputs)
    if terrace is not None and pitched_model is not None:
        flat_area = layout.area(pitched_model["flat"])
        total = flat_area + sum(f["plan_m2"] for f in pitched_model["faces"])
        run_inputs["flat_share_pct"] = 100 * flat_area / total if total > 0 else 0.0
    source = None
    if mode == "numeric_clean":
        ratio, source = 0.0, "engel_yok"
    result = build_result(bundle, run_inputs, ratio, view, capacity_source=source, drawn=summary)
    if footprint is None:
        return result, ratio, None

    skip = tuple(sorted(building_mode.unusable_faces(bundle["solar"], bundle["faces"])))
    obstacle_wkbs = tuple(g.wkb for g in drawings.obstacle_geoms(draw)) if has_draw else ()
    roofs = []
    for _, share in building_mode.roof_scenarios(structure, run_inputs.get("flat_share_pct", FLAT_SHARE_DEFAULT_PCT)):
        kind = _scenario_kind(share)
        roofs.append(cached_layout(
            inputs["footprint_wkb_m"], inputs.get("pitched_type", "hip"), terrace.wkb if terrace is not None and kind == "mixed" else None,
            kind, obstacle_wkbs, float(bundle["solar"]["angle"]), float(inputs["lat"]), skip,
        ))
    if mode == "geometry":
        result = build_result(
            bundle, run_inputs, ratio, view, geometry=[_geometry_counts(r) for r in roofs], capacity_source="geometri",
            drawn=summary,
        )
    shown = result["senaryolar"][0]  # 'Bilmiyorum': the flat extreme is drawn
    limits = None if mode == "geometry" else {s["ad"]: s["panel_sayisi"] for s in shown["segmentler"] if s["durum"] == "kullanildi"}
    notes = []
    if structure == "unknown":
        notes.append(NOTE_FLAT_ASSUMED_GEOMETRY if mode == "geometry" else NOTE_FLAT_ASSUMED)
    if structure == "mixed" and terrace is None:
        notes.append(layout.MSG_NO_TERRACE_DRAWN)
    if hint:
        notes.append(hint)
    if mode == "numeric":
        notes.append(NOTE_PHOTO if ratio is not None else NOTE_AVERAGE)
    roof = {**roofs[0], "limits": limits, "notes": notes, "scenario": shown, "mode": mode, "skip": skip, "footprint": footprint}
    return result, ratio, roof


def clean_bill(value):
    """Monthly bill in TL, or None when empty, zero, negative or not a finite number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < float("inf") else None


# ------------------------------------------------------------ input state

def building_key(building):
    return building["id"] if building else "manual"


def estimate_flats(building, area):
    """Pre-filled flat count: floors from data (or height / 3 m) x footprint / FLAT_GROSS_AREA_M2, always >= 1."""
    if building is None:
        return building_mode.estimate_flat_count(None, None, area)
    return building_mode.estimate_flat_count(building.get("num_floors"), building.get("height_m"), building["area_m2"])


def sync_building_inputs(state, building, area):
    """Set the per-building inputs BEFORE their widgets exist; only the bill survives a new building.

    A new building id re-estimates Daire sayısı, resets the roof structure to 'Bilmiyorum' and the terrace slider to
    its default. Returns True when it reset.
    """
    key = building_key(building)
    if state.get(KEY_BUILDING) == key:
        return False
    estimate = estimate_flats(building, area)
    state[KEY_FLATS] = estimate
    state[KEY_FLATS_ESTIMATE] = estimate
    state[KEY_STRUCTURE] = UNKNOWN_CHOICE
    state[KEY_SHARE] = FLAT_SHARE_DEFAULT_PCT
    state[KEY_PITCHED] = state[KEY_PITCHED_KEEP] = HIP_CHOICE
    state[KEY_DRAWINGS] = drawings.new_state(key)  # drawings belong to one building
    state[KEY_BUILDING] = key
    state.pop(KEY_PHOTO_STEP, None)  # the photo step starts again for a new building
    return True


def ensure_defaults(state, bill_was_shown=True):
    """Widgets that are only shown in later steps lose their state while hidden: restore their values first.

    The bill is mirrored in a plain key (KEY_BILL_KEEP) so it survives hidden runs and new clicks.
    """
    state.setdefault(KEY_FLATS, state.get(KEY_FLATS_ESTIMATE, 1))
    state.setdefault(KEY_STRUCTURE, UNKNOWN_CHOICE)
    state.setdefault(KEY_SHARE, FLAT_SHARE_DEFAULT_PCT)
    state.setdefault(KEY_PITCHED, state.get(KEY_PITCHED_KEEP, HIP_CHOICE))  # hidden for flat / 'Bilmiyorum'
    if not bill_was_shown:  # a hidden widget loses its state: restore it from the mirror before it is created
        state[KEY_BILL] = state.get(KEY_BILL_KEEP)


def remember_bill():
    """on_change of the bill widget."""
    st.session_state[KEY_BILL_KEEP] = st.session_state.get(KEY_BILL)



def drawing_state(state, building):
    """The roof drawings of the selected building (a fresh state when missing or for another building)."""
    draw = state.get(KEY_DRAWINGS)
    if draw is None or draw["building"] != building_key(building):
        draw = state[KEY_DRAWINGS] = drawings.new_state(building_key(building))
    return draw


def apply_pending_roof(state):
    """A photo's roof-structure pre-selection is applied on the next run, before the widgets are created."""
    pending = state.pop(KEY_PENDING_ROOF, None)
    if pending:
        state[KEY_STRUCTURE], state[KEY_SHARE] = pending


def reset_inputs(state):
    """'Sıfırla': clears the bill too and forces the per-building reset on the next run."""
    state[KEY_BILL] = None
    state[KEY_BILL_KEEP] = None
    state.pop(KEY_BUILDING, None)


def photo_roof_choice(detection):
    """((roof structure label, flat share %), note) pre-selected from the photo; never 'Bilmiyorum'."""
    detected = detection["roof_type"]
    if detected == "mixed":
        return (MIXED_CHOICE, FLAT_SHARE_DEFAULT_PCT), "Çatı karma görünüyor; düz teras payı %50 olarak ön seçildi, kaydırıcıdan değiştirebilirsiniz."
    key = "flat" if detected == "flat" else "pitched"
    return (building_mode.STRUCTURE_LABELS_TR[key], FLAT_SHARE_DEFAULT_PCT), None


def collect_inputs(state, analysis, override=None):
    """Stored Hesapla inputs + the sidebar widget values -> the inputs the calculation uses (no network).

    override = (structure label, flat share %) replaces the widget values (used right after a photo pre-selection).
    """
    label, share = override or (state.get(KEY_STRUCTURE, UNKNOWN_CHOICE), state.get(KEY_SHARE, FLAT_SHARE_DEFAULT_PCT))
    n = cost.clamp_flats(state.get(KEY_FLATS))
    preselect = analysis.get("roof_preselect")
    if label == UNKNOWN_CHOICE:
        tag = "Varsayım"
    else:
        tag = "Ölçüm/veri" if preselect and preselect[0] == label else "Kullanıcı girdisi"
    pitched = state.get(KEY_PITCHED_KEEP) or state.get(KEY_PITCHED) or HIP_CHOICE
    return {
        **analysis["inputs"], "structure": STRUCTURE_FROM_CHOICE[label], "flat_share_pct": share, "roof_tag": tag,
        "pitched_type": PITCHED_FROM_CHOICE.get(pitched, "hip"),
        "n_flats": n, "n_flats_estimated": n == state.get(KEY_FLATS_ESTIMATE),
        "bill": clean_bill(state.get(KEY_BILL, state.get(KEY_BILL_KEEP))), "building": analysis.get("building"),
    }


def photo_applies(building, is_demo):
    """A bundled demo photo never describes a real clicked building; an uploaded photo is assumed to be of it."""
    return not (building is not None and is_demo)


def selected_ids(editor_df):
    """Obstacle ids ticked in the confirmation table."""
    return {int(i) for i in editor_df.index[editor_df["Dahil"]]}


def recompute(bundle, inputs, detection, included_ids):
    """Recompute ratio and result from cached PVGIS + detection only (no Gemini, no PVGIS call).

    With a photo that applies to the building the ratio is a float (0.0 = clean roof); None means no usable photo.
    A bundled demo photo never applies to a real building (inputs["photo_applies"] is False).
    """
    ratio, view = photo_ratio(inputs, detection, included_ids)
    result = build_result(bundle, inputs, obstacle_ratio=ratio, image_view=view)
    return result, ratio


def photo_ratio(inputs, detection, included_ids):
    """(obstacle ratio, image view) of a photo that applies to the building; (None, None) otherwise."""
    if detection is None or not inputs.get("photo_applies", True):
        return None, None
    boxes = [o["box_px"] for o in detection["obstacles"] if o["id"] in included_ids]
    ratio = compute_obstacle_ratio(detection["roof_polygon_px"], boxes, detection["width"], detection["height"])
    return ratio, detection["image_view"]


def format_savings_short(lo, hi):
    """Card-sized TL range: thousands with one decimal ("27,5 – 55,0 bin TL")."""
    if hi < 1000:
        a, b, unit, decimals = round_half_up(lo, 0), round_half_up(hi, 0), "TL", 0
    else:
        a, b, unit, decimals = round_half_up(lo / 1000, 1), round_half_up(hi / 1000, 1), "bin TL", 1
    if a == b:
        return f"{tr_num(a, decimals)} {unit}"
    return f"{tr_num(a, decimals)} – {tr_num(b, decimals)} {unit}"


def format_range(values, decimals=0, unit=""):
    """'4,2 – 6,0 kWp'; one value when both ends are equal after rounding."""
    lo, hi = (round_half_up(v, decimals) for v in values)
    text = tr_num(lo, decimals) if lo == hi else f"{tr_num(lo, decimals)} – {tr_num(hi, decimals)}"
    return f"{text} {unit}".strip()


def payback_text(payback):
    """Turkish payback range; an uncomputable bound reads 'hesaplanamaz (tasarruf çok düşük)'. None if both are None."""
    lo, hi = payback
    if lo is None and hi is None:
        return None
    if lo is None or hi is None:
        parts = [f"{tr_num(v, 1)} yıl" if v is not None else "hesaplanamaz (tasarruf çok düşük)" for v in (lo, hi)]
        return " – ".join(parts)
    return f"{format_range([lo, hi], 1)} yıl"


# ------------------------------------------------------------- explanation

def resolve_api_key():
    """Return (key, None) or (None, Turkish message). Main thread only: it may read st.secrets."""
    try:
        return get_gemini_api_key(), None
    except MissingApiKeyError as exc:
        return None, str(exc)


def explained_detection(detection, included_ids, inputs=None):
    """Detection restricted to the ticked obstacles (what the explanation may mention).

    None when there is no photo or the photo does not apply to the building (inputs["photo_applies"] False).
    """
    if detection is None or (inputs is not None and not inputs.get("photo_applies", True)):
        return None
    return {**detection, "obstacles": [o for o in detection["obstacles"] if o["id"] in included_ids]}


def generate_explanation(result, detection, demo_photo, api_key):
    """Build the explanation state {text, source, key}. Without a key the template is used (no Gemini call)."""
    key = explain.payload_key(explain.build_payload(result, detection, demo_photo))
    if api_key is None:
        text = explain.build_fallback_explanation(result, detection, demo_photo)
        return {"text": text, "source": "template", "key": key}
    out = explain.explain_result(result, detection, api_key=api_key, demo_photo=demo_photo)
    return {"text": out["text"], "source": out["source"], "key": key}


def explanation_is_stale(explanation, result, detection, demo_photo):
    """True when the stored explanation was generated for different numbers than the current result."""
    return explanation["key"] != explain.payload_key(explain.build_payload(result, detection, demo_photo))


def regenerate_explanation(result, detection, demo_photo):
    """Re-explain the current result (button handler, main thread) and return the new explanation state."""
    api_key, _ = resolve_api_key()
    return generate_explanation(result, detection, demo_photo, api_key)


# ------------------------------------------------------------------- photos

def parse_sources(path=DEMO_DIR / "SOURCES.md"):
    """Parse SOURCES.md into {filename: {source, author, license, link}}."""
    rows = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].endswith((".jpg", ".png")):
            rows[cells[0]] = dict(zip(("source", "author", "license", "link"), cells[1:]))
    return rows


def source_caption(filename):
    info = parse_sources().get(filename)
    if not info:
        return "Kaynak bilgisi bulunamadı."
    link = info["link"] if info["link"].startswith("http") else "kaynak bağlantısı eklenecek"
    return f"Kaynak: {info['source']} · Yazar: {info['author']} · Lisans: {info['license']} · {link}"


# ------------------------------------------------------------ building text

def height_text(building, report):
    """Turkish one-liner for the building height and where it comes from (information only in P1)."""
    h = tr_num(building["height_m"], 1)
    source = building["height_source"]
    if source == "user":
        return f"Bina yüksekliği: {h} m (girdiğiniz kat sayısı × {LEVEL_HEIGHT_M} m)."
    if source == "height_tag":
        return f"Bina yüksekliği: {h} m (kaynak: yükseklik verisi)."
    if source == "levels_tag":
        return f"Bina yüksekliği: {h} m (kaynak: kat sayısı verisi × {LEVEL_HEIGHT_M} m)."
    if source == "ml_estimate":
        mae = (report or {}).get("cv_mae_m")
        tail = f"; doğrulamada ortalama hata ±{tr_num(mae, 2)} m" if isinstance(mae, (int, float)) else ""
        return f"Bina yüksekliği: {h} m (YZ tahmini, LightGBM{tail})."
    if source == "neighbour_median":
        return f"Bina yüksekliği: {h} m (varsayım: komşu binaların ortanca yüksekliği)."
    return f"Bina yüksekliği: {h} m (varsayım: veri setindeki bilinen yüksekliklerin ortancası)."


def input_classes(inputs, has_photo_applied, building, result=None):
    """Which inputs were measured, estimated and assumed (Turkish bullet texts).

    A user drawing or the "Engel yok, çatı boş" statement (result "guc_kaynagi" geometri / engel_yok) replaces the
    photo/average obstacle line.
    """
    result = result or {}
    marked = result.get("guc_kaynagi") in ("geometri", "engel_yok")
    measured, estimated, assumed = ["Güneş verisi (ışınım, verim): PVGIS"], [], []
    if building is not None:
        measured.append("Çatı alanı: bina izi")
        assumed.append(f"Kenar boşluğu: {tr_num(SETBACK_M, 1)} m")
        src = building["height_source"]
        if src == "ml_estimate":
            estimated.append("Bina yüksekliği: YZ tahmini")
        elif src in ("neighbour_median", "default"):
            assumed.append("Bina yüksekliği: veri tabanlı varsayım")
        else:
            measured.append("Bina yüksekliği: veri")
    else:
        assumed.append("Çatı alanı: elle girilen değer")
    if marked and result.get("cizilen_engel_sayisi"):
        measured.append("Engeller: kullanıcı çizimi")
    elif marked:
        measured.append("Engel yok (kullanıcı beyanı)")
    elif has_photo_applied:
        measured.append("Engel oranı: fotoğraf analizi")
    else:
        assumed.append("Engeller: fotoğraf yok, ortalama kullanılabilir alan oranı")
    if inputs["structure"] == "unknown":
        assumed.append("Çatı yapısı: bilinmiyor, tamamen düz ve tamamen eğimli çatı için aralık hesaplandı")
    if inputs.get("n_flats_estimated"):
        estimated.append(f"Daire sayısı: kat sayısı × taban alanı ÷ {FLAT_GROSS_AREA_M2} m² tahmini")
    if inputs.get("bill") is None:
        assumed.append("Tüketim: fatura girilmedi")
    return measured, estimated, assumed


def show_lookup(lookup, index):
    """Message under the map for the selected point."""
    if index is None:
        st.warning(buildings.MSG_NO_DATA)
        return
    if lookup is None:
        st.info("Haritada kapsama alanı içindeki (mavi çerçeve) bir binaya tıklayın.")
    elif lookup["status"] == "ok":
        b = lookup["building"]
        st.success(
            f"Bina bulundu: taban alanı {tr_num(b['area_m2'], 1)} m²; {tr_num(SETBACK_M, 1)} m kenar boşluğu sonrası "
            f"{tr_num(b['setback_area_m2'], 1)} m² (veri: {b['data_source']})."
        )
        floors = b.get("num_floors")
        floors_text = f"Kat sayısı verisi: {tr_num(floors, 0)}. " if floors else "Kat sayısı verisi yok. "
        st.caption(floors_text + height_text(b, index.report))
        if b["match_distance_m"] > 0:
            st.caption(f"Tıkladığınız noktaya en yakın bina seçildi (~{tr_num(b['match_distance_m'], 1)} m).")
    else:
        st.info(lookup["message"])


def structure_text(r):
    label = building_mode.STRUCTURE_LABELS_TR[r["cati_yapisi"]]
    return f"{label} (düz teras payı %{r['duz_pay_yuzde']})" if r["duz_pay_yuzde"] is not None else label


def assumption_rows(r, inputs, detection, explanation=None, report=None):
    """'Varsayımlar ve kaynaklar' table: one row per shown number (SPEC section 8)."""
    building = inputs.get("building")
    applied = detection is not None and inputs.get("photo_applies", True)
    if building is not None:
        rows = [
            ("Çatı alanı", f"{tr_num(building['area_m2'], 1)} m²", "Ölçüm/veri", f"Bina izi ({building['data_source']}, Overture Maps)"),
            ("Kenar boşluğu", f"{tr_num(SETBACK_M, 1)} m", "Varsayım", "Bina izi içeri doğru küçültülür; panel konmayan şerit"),
            ("Bina yüksekliği", f"{tr_num(building['height_m'], 1)} m", HEIGHT_SOURCE_TAG[building["height_source"]], height_text(building, report)),
        ]
    else:
        rows = [("Çatı alanı", f"{tr_num(inputs['area'], 1)} m²", "Kullanıcı girdisi", "Kenar çubuğunda elle girilen değer")]
    flats_note = (
        f"Tahmin: kat sayısı × taban alanı ÷ {FLAT_GROSS_AREA_M2} m² (daire başına brüt alan varsayımı; duvar ve merdiven dahil)"
        if r["daire_sayisi_tahmini"] else "Girdiğiniz değer"
    )
    rows += [
        ("Daire sayısı", f"{r['daire_sayisi']}", "Varsayım" if r["daire_sayisi_tahmini"] else "Kullanıcı girdisi", flats_note),
        ("Çatı yapısı", structure_text(r), inputs["roof_tag"], "Fotoğraf analizi, kullanıcı seçimi veya 'Bilmiyorum' (iki uç durum hesaplanır)"),
    ]
    if r["cati_yapisi"] != "flat":
        rows.append((
            "Eğimli çatı tipi", building_mode.PITCHED_TYPE_LABELS_TR[r["egimli_tip"]], "Kullanıcı girdisi / varsayılan",
            "Kırma (4 yüz) varsayılandır: pilot bölgede en yaygın eğimli çatı; yüzler bina izinin en küçük dikdörtgeni üzerinde kurulur",
        ))
    rows.append((
        "Panel", f"{tr_num(PANEL_LENGTH_M, 2)} × {tr_num(PANEL_WIDTH_M, 2)} m, {tr_num(PANEL_KWP, 2)} kWp", "Varsayım",
        f"Panel sayısı: {format_range(r['panel_sayisi_aralik'])}; yerleşim çatı görünümünde",
    ))
    if r["guc_kaynagi"] == "geometri":
        rows.append((
            "Engeller", f"{r['cizilen_engel_sayisi']} çizim, {tr_num(r['cizilen_engel_alani_m2'], 1)} m²", "Kullanıcı girdisi",
            "Çatı görünümünde çizildi; kurulabilir güç = panel sayısı × 0,40 kWp, kullanılabilir alan oranı kullanılmadı",
        ))
    elif r["guc_kaynagi"] == "engel_yok":
        rows.append(("Engeller", "yok", "Kullanıcı girdisi", "Kullanıcı çatının boş olduğunu onayladı (engel oranı 0)"))
    if applied:
        view = "Kuşbakışı" if detection["image_view"] == "top-down" else "Açılı"
        rows += [
            ("Engel oranı", f"%{tr_num(percent_from_ratio(r['engel_orani']), 1)}", "Ölçüm/veri", "Gemini fotoğraf analizi; tabloda işaretlediğiniz engeller"),
            ("Görüntü türü", view, "Ölçüm/veri", "Gemini fotoğraf analizi"),
        ]
        if building is None:
            rows.append(("Kenar payı çarpanı", f"{EDGE_FACTOR}", "Varsayım", "Çatı kenarlarında panel konmayan şerit"))
    elif building is None:
        rows.append(("Kullanılabilir alan oranı", f"{USABLE_RATIO_NO_PHOTO}", "Varsayım", "Ember ortalaması (fotoğraf yok)"))
    else:
        rows.append(("Kullanılabilir alan oranı", f"{tr_num(USABLE_RATIO_NO_PHOTO / EDGE_FACTOR, 3)}", "Varsayım", building_mode.USABLE_RATIO_NOTE))
    ridge = building["ridge_azimuth_deg"] if building is not None else DEFAULT_RIDGE_AZIMUTH_DEG
    rows += [
        ("Kullanılabilir alan (plan)", f"{tr_num(r['kullanilabilir_alan_m2'], 1)} m²", "Varsayım", "Alan × oran; düz ve eğimli bölümlere bölünür"),
        ("Düz bölüm gücü", f"{tr_num(M2_PER_KWP_FLAT, 1)} m²/kWp", "Varsayım", "Plan alanı ÷ 6,5 (sıra aralığı dahil); PVGIS'in en uygun eğim ve yönü"),
        ("Eğimli yüzler", f"{tr_num(M2_PER_KWP_PITCHED, 1)} m²/kWp", "Varsayım", f"Yüzey alanı (plan ÷ cos {DEFAULT_ROOF_PITCH_DEG}°) ÷ 5,5; eğim {DEFAULT_ROOF_PITCH_DEG}°, paneller çatıya paralel; eğimli alan yüzlere plan alanları oranında dağıtılır"),
        (
            "Mahya yönü", f"{tr_num(ridge, 0)}°", "Ölçüm/veri" if building is not None else "Varsayım",
            "Bina izinin uzun ekseni; yüzler mahyaya dik" if building is not None else "Bina izi yok: kuzey-güney mahya varsayıldı",
        ),
        ("Kurulabilir güç (bina)", format_range(r["kurulabilir_guc_kwp_aralik"], 1, "kWp"), "Varsayım", "Kullanılan çatı bölümlerinin toplamı"),
        ("Düz bölüm eğimi / yönü", f"{r['panel_egim_derece']}° / {r['panel_yon_derece']}°", "Varsayım", "PVGIS'in en uygun eğim ve yönü"),
        ("Yıllık ışınım", f"{tr_num(r['yillik_isinim_kwh_m2'])} kWh/m²", "Ölçüm/veri", "PVGIS (JRC)"),
        ("Özgül verim (düz bölüm)", f"{tr_num(r['golgeli_uretim_kwh_kwp'])} kWh/kWp", "Ölçüm/veri", "PVGIS (JRC), arazi gölgesi dahil"),
        ("Sıcaklık kaybı", f"%{tr_num(r['sicaklik_kaybi_yuzde'], 1)}", "Ölçüm/veri", "PVGIS (JRC)"),
        ("Uygunluk (arazi gölgesi)", f"%{tr_num(r['uygunluk_puani_yuzde'], 1)}", "Ölçüm/veri", "PVGIS gölgeli ÷ gölgesiz verim"),
        ("Yıllık üretim (bina)", format_range(r["yillik_uretim_kwh_aralik"], 0, "kWh"), "Ölçüm/veri", "Bölüm gücü × PVGIS özgül verimi"),
        ("Kurulum maliyeti", format_savings_short(*r["toplam_maliyet_tl_aralik"]), "Varsayım", COST_SOURCE),
        ("Elektrik tarifesi", f"{tr_num(r['tarife_dusuk_tl_kwh'], 2)} / {tr_num(r['tarife_yuksek_tl_kwh'], 2)} TL/kWh", "Varsayım", f"{TARIFF_SOURCE}; vergiler dahil yaklaşık (×1,26)"),
    ]
    if r["tasarruf_yontemi"] == "fatura":
        rows += [
            ("Ortalama aylık daire faturası", f"{tr_num(r['fatura_tl_ay'])} TL", "Kullanıcı girdisi", f"Tüketim yaklaşık {tr_num(r['tuketim_kwh_ay'])} kWh/ay (iki kademeli tarifeden)"),
            ("Tasarruf", "fatura temelli", "Varsayım", "Aylık mahsuplaşma: ayın üretimi o ayın tüketimini tüketim kadar düşürür; fazla üretim 0 TL. Aralık yalnızca çatı senaryolarından gelir"),
            ("Tüketim varsayımı", "her ay aynı", "Varsayım", r["tuketim_varsayimi"]),
        ]
    else:
        rows.append(("Tasarruf aralığı", "geniş tahmin: kademe bilinmiyor", "Varsayım", "Kademe bilinmiyor: daire payı × [düşük kademe, yüksek kademe] (aylık mahsuplaşma)"))
    rows += [
        ("Paylaşım", "eşit TL indirim", "Varsayım", "Bina tasarrufu = N × daire tasarrufu (ortalama fatura); daire başına indirim = toplam ÷ N; kWh de eşit bölünür"),
        ("Geri ödeme", "basit", "Varsayım", "Daire başına maliyet ÷ yıllık tasarruf; bozulma, bakım ve tarife değişimi dikkate alınmadı"),
        ("CO₂ faktörü", f"{CO2_FACTOR_T_PER_MWH} tCO₂/MWh", "Varsayım", "ETKB TUESEF 2021"),
        ("Güven seviyesi", r["guven_seviyesi"], "Varsayım", "Alan kaynağı, fotoğraf, çatı yapısı ve çatı yüzü verisine bağlı kural"),
        ("Komşu bina gölgesi", "—", "Sonraki aşamada", "Bina yüksekliği ve komşu binalarla"),
    ]
    if explanation is not None:
        if explanation["source"] == "gemini":
            note = f"{EXPLAIN_MODEL}; metindeki her sayı sonuçlarla karşılaştırıldı"
            rows.append(("YZ açıklaması", "Gemini", "YZ yorumu (sayılar doğrulandı)", note))
        else:
            note = "YZ yanıtı kullanılamadı; sonuçlardan üretilen sabit metin"
            rows.append(("YZ açıklaması", "Standart şablon", "Şablon", note))
    if inputs.get("demo_photo"):
        rows.append(("Demo fotoğrafı", "—", "Varsayım", "Demo fotoğrafı örnektir; seçilen binanın kendi çatısı değildir."))
    return pd.DataFrame(rows, columns=["Değer", "Sonuç", "Durum", "Kaynak / not"])


# --------------------------------------------------------------------- views

def monthly_chart(result):
    rows = building_mode.chart_rows(result, MONTHS_TR)
    if not rows:
        st.info("Üretim yapan çatı bölümü yok; aylık üretim grafiği gösterilmiyor.")
        return
    groups = {row["Grup"] for row in rows}
    enc = dict(
        x=alt.X("Ay:N", sort=MONTHS_TR, title=None),
        y=alt.Y("kWh:Q", title="Aylık üretim (kWh)"),
        color=alt.Color("Seri:N", title=None),
        tooltip=["Ay", "Grup", "Seri", alt.Tooltip("kWh:Q", format=",.1f")],
    )
    if len(groups) > 1:
        enc["xOffset"] = "Grup:N"
    chart = alt.Chart(pd.DataFrame(rows)).mark_bar().encode(**enc).properties(height=320)
    st.altair_chart(chart, use_container_width=True)


def segment_table(result):
    """One row per roof segment and scenario (area rules visible to the user)."""
    status = {
        "kullanildi": "kullanıldı", "elendi": "verimsiz, hesaba katılmadı", "alinamadi": "güneş verisi alınamadı",
        "kullanilmadi": "alan ayrılmadı",
    }
    rows = []
    for sc in result["senaryolar"]:
        for seg in sc["segmentler"]:
            if seg["durum"] == "kullanilmadi":
                continue
            rows.append({
                "Senaryo": sc["etiket"], "Bölüm": seg["ad"], "Durum": status[seg["durum"]],
                "Plan alanı (m²)": round(seg["plan_m2"], 1),
                "Yüzey alanı (m²)": round(seg["yuzey_m2"], 1) if "yuzey_m2" in seg else round(seg["plan_m2"], 1),
                "kWp": round(seg["kwp"], 1), "kWh/yıl": round(sum(seg["aylik_kwh"])),
            })
    return pd.DataFrame(rows)


def coverage_text(r):
    """'Daire başına faturanın yaklaşık %X–%Y kadarı karşılanır' (grammatical for any number), or None without a bill."""
    if r["tasarruf_yontemi"] != "fatura":
        return None
    lo, hi = (int(round_half_up(v, 0)) for v in r["fatura_karsilama_yuzde_aralik"])
    span = f"%{lo}" if lo == hi else f"%{lo}–%{hi}"
    return f"Daire başına faturanın yaklaşık {span} kadarı karşılanır."


def remaining_bill_text(r):
    """Remaining average monthly flat bill range, or None without a bill."""
    if r["tasarruf_yontemi"] != "fatura":
        return None
    return f"Kalan ortalama aylık daire faturası: yaklaşık {format_range(r['kalan_fatura_tl_ay_aralik'], 0, 'TL')}."


def has_capacity(r):
    """False when no scenario has any installable kWp (e.g. every pitched face dropped or without PVGIS data)."""
    return r["kurulabilir_guc_kwp_aralik"][1] > 0


def zero_capacity_warning(r):
    pitched = any(seg["tur"] == "face" and seg["plan_m2"] > 0 for sc in r["senaryolar"] for seg in sc["segmentler"])
    return ZERO_PITCHED_WARNING if pitched else ZERO_GENERIC_WARNING


def face_status_lines(r):
    """One line per roof segment and scenario with its status; nothing disappears silently."""
    multi = len(r["senaryolar"]) > 1
    lines = []
    for sc in r["senaryolar"]:
        for seg in sc["segmentler"]:
            if seg["tur"] == "flat" and seg["durum"] == "kullanilmadi":
                continue  # no terrace share in this scenario: not a face
            prefix = f"{sc['etiket']} · " if multi else ""
            detail = f" ({tr_num(seg['kwp'], 1)} kWp)" if seg["durum"] == "kullanildi" else ""
            if seg["durum"] == "elendi" and seg.get("verim_orani") is not None:
                detail = f" (verim oranı %{tr_num(seg['verim_orani'] * 100)})"
            lines.append(f"- {prefix}{seg['ad']}: {FACE_STATUS_TR[seg['durum']]}{detail}")
    return lines


def show_face_status(r):
    st.markdown("**Çatı yüzleri**")
    st.markdown("\n".join(face_status_lines(r)))


def _capacity_cards(r):
    """Building and per-flat cards (kWp, kWh, CO2, cost, discount, payback, coverage)."""
    st.markdown("**Bina geneli**")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Kurulabilir güç", format_range(r["kurulabilir_guc_kwp_aralik"], 1, "kWp"))
    b2.metric("Yıllık üretim", format_range(r["yillik_uretim_kwh_aralik"], 0, "kWh"))
    b3.metric("CO₂ azaltımı", format_range(r["co2_ton_yil_aralik"], 1, "ton/yıl"))
    cost_lo, cost_hi = r["toplam_maliyet_tl_aralik"]
    b4.metric(
        "Toplam kurulum maliyeti", format_savings_short(cost_lo, cost_hi),
        help=f"Kesin aralık: {tr_num(cost_lo)} – {tr_num(cost_hi)} TL",
    )
    b4.caption("Tahmini aralık; gerçek fiyat teklifle netleşir.")

    st.markdown(f"**Daire başına ({r['daire_sayisi']} daire, üretim eşit paylaşılır)**")
    d1, d2, d3, d4 = st.columns(4)
    pf_lo, pf_hi = r["daire_basi_maliyet_tl_aralik"]
    d1.metric("Tek seferlik maliyet", format_savings_short(pf_lo, pf_hi), help=f"Kesin aralık: {tr_num(pf_lo)} – {tr_num(pf_hi)} TL")
    lo, hi = r["daire_basi_tasarruf_tl_aralik"]
    d2.metric("Daire başına sabit indirim (yıllık)", format_savings_short(lo, hi), help=f"Kesin aralık: {tr_num(lo)} – {tr_num(hi)} TL")
    d2.caption(f"Aylık ortalama: {format_range(r['daire_basi_indirim_tl_ay_aralik'], 0, 'TL/ay')}.")
    if r["tasarruf_yontemi"] == "fatura":
        d2.caption("Ortalama daire faturasına göre hesaplandı; vergiler dahil yaklaşık.")
    else:
        d2.caption("Geniş tahmin: kademe bilinmiyor; vergiler dahil yaklaşık. Ortalama daire faturasını girin.")
    high_share = building_mode.high_share_note(r)
    if high_share:
        d2.caption(high_share)
    payback = payback_text(r["geri_odeme_yil_aralik"])
    if payback is not None:
        d3.metric("Geri ödeme süresi (aralık)", payback)
        d3.caption("Bozulma, bakım ve tarife değişimi dikkate alınmadı.")
    else:
        d3.caption("Geri ödeme süresi hesaplanamadı (tasarruf çok düşük ya da kurulabilir panel yok).")
    if r["tasarruf_yontemi"] == "fatura":
        cov_lo, cov_hi = (int(round_half_up(v, 0)) for v in r["fatura_karsilama_yuzde_aralik"])
        d4.metric("Faturanın karşılanan kısmı", f"%{cov_lo}" if cov_lo == cov_hi else f"%{cov_lo} – %{cov_hi}")


def show_cards(r, inputs=None, applied=False, building=None):
    if has_capacity(r):
        _capacity_cards(r)
    else:
        st.warning(zero_capacity_warning(r))  # no 0 TL cost/discount cards; the rest of the page stays
    show_face_status(r)

    c1, c2, c3 = st.columns(3)
    c1.metric("Uygunluk (arazi gölgesi)", f"%{tr_num(r['uygunluk_puani_yuzde'], 1)}")
    c1.caption("Şu an yalnızca arazi (tepe) gölgesi hesaba katılıyor. Komşu bina gölgesi bir sonraki aşamada eklenecek.")
    c2.metric("Güven seviyesi", r["guven_seviyesi"])
    if r.get("guven_iyilestirme"):
        c2.caption("Sonucu iyileştirmek için: " + " ".join(r["guven_iyilestirme"]))
    c3.metric("Daire sayısı", f"{r['daire_sayisi']}")
    if r["daire_sayisi_tahmini"]:
        c3.caption(f"Tahmini (kat sayısı × taban alanı ÷ {FLAT_GROSS_AREA_M2} m²) — lütfen kontrol edin.")

    e1, e2, e3 = st.columns(3)
    e1.metric("Yıllık ışınım", f"{tr_num(r['yillik_isinim_kwh_m2'])} kWh/m²")
    e2.metric("Özgül verim", f"{tr_num(r['golgeli_uretim_kwh_kwp'])} kWh/kWp")
    e3.metric("Sıcaklık kaybı", f"%{tr_num(r['sicaklik_kaybi_yuzde'], 1)}")
    if r["engel_orani"] is not None:
        e4, _, _ = st.columns(3)
        e4.metric("Engel oranı", f"%{tr_num(percent_from_ratio(r['engel_orani']), 1)}")

    for warning in r["uyarilar"]:
        st.warning(warning + ".")
    for flag in r["maliyet_bayraklari"] if has_capacity(r) else []:
        st.warning(cost.flag_message(flag))
    for text in (coverage_text(r), remaining_bill_text(r)) if has_capacity(r) else ():
        if text:
            st.info(text)
    if r["tuketim_varsayimi"]:
        st.caption(r["tuketim_varsayimi"])
    if r["mahsuplasma_notu"]:
        st.caption(r["mahsuplasma_notu"])
    st.caption(
        f"Tarife: {TARIFF_SOURCE}; vergiler dahil yaklaşık, kesin tutar değildir. Aylık üretim dairelere eşit bölünür; "
        "dairenin tüketimini aşan fazla üretim 0 TL sayılır."
    )
    st.caption(r["indirim_notu"])
    st.caption(r["paylasim_notu"])
    st.caption(f"Maliyet kaynağı: {COST_SOURCE}")
    st.caption("Maliyete dahil olmayanlar: " + "; ".join(cost.EXCLUSIONS_TR) + ".")
    st.warning(building_mode.LEGAL_NOTE)

    with st.expander("Güven seviyesinin nedenleri", expanded=True):
        for reason in r["guven_nedenleri"]:
            st.write(f"- {reason}")
        if inputs is not None:
            measured, estimated, assumed = input_classes(inputs, applied, building, r)
            for title, items in (("Ölçülen", measured), ("Tahmin edilen", estimated), ("Varsayılan", assumed)):
                if items:
                    st.write(f"**{title}:** " + "; ".join(items))


def show_monthly(r):
    st.subheader("Aylık üretim")
    monthly_chart(r)
    st.dataframe(segment_table(r), hide_index=True, width="stretch")
    st.caption(
        f"Düz bölüm en uygun eğim ({r['panel_egim_derece']}°) ve yönde; eğimli yüzler {DEFAULT_ROOF_PITCH_DEG}° eğimle "
        f"mahyaya dik yönlerde varsayıldı. Kullanılabilir plan alanı: {tr_num(r['kullanilabilir_alan_m2'], 1)} m²."
    )


EXPLANATION_CAPTIONS = {
    "gemini": "Kaynak: Gemini (tüm sayılar doğrulandı)",
    "template": "Kaynak: Standart şablon (YZ yanıtı kullanılamadı)",
}


def render_explanation_text(text):
    """Verdict (first line) in bold, bullets as a list, the warning line as a warning box."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return
    st.markdown(f"**{lines[0]}**")
    warn = "\u26a0"
    body = [ln for ln in lines[1:] if not ln.startswith(("-", warn))]
    if body:
        st.write(" ".join(body))
    bullets = [ln for ln in lines[1:] if ln.startswith("-")]
    if bullets:
        st.markdown("\n".join(bullets))
    for ln in lines[1:]:
        if ln.startswith(warn):
            st.warning(ln.replace("\ufe0f", "").replace(warn, "").strip())


def explanation_section(analysis, result, detection):
    """'Yapay Zekâ Değerlendirmesi': last explanation, a stale notice and a manual refresh (no auto Gemini call)."""
    explanation = st.session_state.get("explanation")
    if not explanation:
        return
    demo = analysis["inputs"].get("demo_photo", False) and detection is not None
    st.subheader("Yapay Zekâ Değerlendirmesi")
    if analysis.get("key_error") and not analysis.get("vision_error"):
        st.warning(analysis["key_error"])
    if explanation_is_stale(explanation, result, detection, demo):
        st.info("Değerler değişti; açıklama güncel değil.")
        if st.button("Açıklamayı YZ ile güncelle"):
            with st.spinner("Açıklama güncelleniyor…"):
                st.session_state["explanation"] = regenerate_explanation(result, detection, demo)
            st.rerun()
    render_explanation_text(explanation["text"])
    st.caption(EXPLANATION_CAPTIONS[explanation["source"]])


def _skip_photo():
    st.session_state[KEY_PHOTO_STEP] = "skip"


def photo_step(state):
    """Main-area photo step. Returns (image_bytes or None, filename or None, is_demo, ready).

    ready is True once a photo is uploaded, a demo is tried or the user chose to continue without a photo.
    A bundled demo is only offered under 'Örnek fotoğrafla dene' and is never applied to a real building.
    """
    st.subheader(PHOTO_STEP_TITLE)
    st.write(PHOTO_GUIDANCE)
    uploaded = st.file_uploader(PHOTO_STEP_TITLE, type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.caption(PHOTO_BENEFIT)
    image_bytes, name, is_demo = None, None, False
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Dosya çok büyük; en fazla {MAX_UPLOAD_MB} MB yükleyebilirsiniz.")
        else:
            image_bytes, name = uploaded.getvalue(), uploaded.name
    with st.expander(DEMO_EXPANDER):
        files = sorted(p.name for p in DEMO_DIR.glob("roof_*.jpg"))
        if files:
            chosen = st.selectbox("Örnek fotoğraf", [DEMO_NONE, *files])
            if chosen != DEMO_NONE:
                st.caption(source_caption(chosen))
                if image_bytes is None:
                    image_bytes, name, is_demo = (DEMO_DIR / chosen).read_bytes(), chosen, True
        else:
            st.info("Örnek fotoğraf bulunamadı.")
    skipped = state.get(KEY_PHOTO_STEP) == "skip"
    if image_bytes is None and not skipped:
        st.button("Fotoğrafsız devam et", on_click=_skip_photo)
    elif image_bytes is None:
        st.caption("Fotoğrafsız devam ediliyor: kullanılabilir alan ortalama orana göre tahmin edilecek.")
    return image_bytes, name, is_demo, image_bytes is not None or skipped


def photo_section(state, applies):
    """Show detection messages, the confirmation table and the overlay ids; return the ticked obstacle ids or None."""
    detection = state.get("detection")
    if state.get("vision_error"):
        st.warning(f"{state['vision_error']} Fotoğrafsız hesap gösteriliyor; güven seviyesi düşük.")
    if detection is None:
        return None

    st.subheader("Fotoğraf analizi")
    if not applies:
        st.info(DEMO_PHOTO_NOTE)
    if detection["multiple_buildings"]:
        st.warning("Fotoğrafta birden fazla bina var; sonuç ana çatı için hesaplandı. Mümkünse tek çatıyı kırpın.")
    if detection["image_view"] == "oblique":
        st.info("Açılı fotoğraf: alan ölçümü yaklaşık; güven düşürüldü.")
    if detection["roof_source"] == "box":
        st.info("Çatı sınırı kutu olarak alındı; sonuç yaklaşık.")
    if state.get("roof_note"):
        st.info(state["roof_note"])

    obstacles = detection["obstacles"]
    if obstacles and applies:
        df = pd.DataFrame(
            {
                "No": [o["id"] for o in obstacles],
                "Dahil": [True] * len(obstacles),
                "Engel": [OBSTACLE_LABELS_TR[o["label"]] for o in obstacles],
                "Güven": [round(o["confidence"] * 100) for o in obstacles],
            },
            index=[o["id"] for o in obstacles],
        )
        st.caption("Çatıda gerçekten engel olmayanların işaretini kaldırın; sonuç anında güncellenir.")
        edited = st.data_editor(
            df,
            key=f"obstacles_{detection['image_hash'][:8]}",
            disabled=["No", "Engel", "Güven"],
            hide_index=True,
            column_config={
                "No": st.column_config.NumberColumn("No", format="%d", width="small"),
                "Dahil": st.column_config.CheckboxColumn("Dahil"),
                "Güven": st.column_config.NumberColumn("Güven", format="%d%%"),
            },
            width="stretch",
        )
        return selected_ids(edited)
    if not obstacles:
        st.write("Çatıda engel tespit edilmedi.")
    return {o["id"] for o in obstacles} if not applies else set()


def flats_step(state):
    """Step b: Daire sayısı, pre-filled with the estimate."""
    st.subheader("Daire sayısı")
    st.number_input(
        "Daire sayısı", min_value=1, step=1, key=KEY_FLATS,
        help=f"Tahmin: kat sayısı × taban alanı ÷ {FLAT_GROSS_AREA_M2} m² (daire başına brüt alan varsayımı). "
        "Sistemin maliyeti ve üretimi bu sayıdaki daire arasında eşit paylaşılır.",
    )
    if cost.clamp_flats(state.get(KEY_FLATS)) == state.get(KEY_FLATS_ESTIMATE):
        st.caption("Tahmini — farklıysa düzeltin")


def optional_inputs_step(state):
    """Step d: roof structure (+ terrace slider) and the average flat bill, both optional."""
    st.subheader("İsteğe bağlı bilgiler")
    state[KEY_BILL_SHOWN] = True
    choice = st.radio("Çatı yapısı", ROOF_CHOICES, key=KEY_STRUCTURE, horizontal=True)
    if choice in (building_mode.STRUCTURE_LABELS_TR["pitched"], MIXED_CHOICE):
        # mirrored in a plain key: the choice is still used ('Bilmiyorum' extremes) while the widget is hidden
        state[KEY_PITCHED_KEEP] = st.radio(
            "Eğimli çatı tipi", PITCHED_CHOICES, key=KEY_PITCHED, horizontal=True,
            help="Kırma çatı (4 yüz) pilot bölgede en yaygın eğimli çatıdır; beşik çatıda 2 yüz vardır.",
        )
    if choice == MIXED_CHOICE:
        st.slider("Düz teras payı (%)", min_value=0, max_value=100, step=5, key=KEY_SHARE)
        st.caption("Çatı görünümünde teras alanını çizerseniz pay çizimden hesaplanır.")
    st.number_input(
        "Ortalama aylık daire faturası (TL)", value=None, min_value=0.0, step=50.0, format="%.0f", key=KEY_BILL,
        on_change=remember_bill,
        help="İsteğe bağlı. Boş bırakırsanız tasarruf geniş bir aralık olarak gösterilir.",
    )


def _reset_clicked():
    reset_inputs(st.session_state)


# ---------------------------------------------------------------- roof view

def build_roof_map(center):
    """Static part of the roof map: Esri World Imagery (DISPLAY ONLY, attribution kept) and the Draw control
    (rectangle and polygon only; editing off, deleting on)."""
    # no scroll-wheel zoom: page scrolling must not be hijacked (the +/- buttons stay)
    m = folium.Map(location=center, zoom_start=ROOF_VIEW_ZOOM, max_zoom=21, tiles=None, scroll_wheel_zoom=False)
    folium.TileLayer(
        tiles=ESRI_TILES_URL, attr=ESRI_ATTRIBUTION, name="Uydu (yalnızca görüntüleme)", max_native_zoom=19, max_zoom=21,
    ).add_to(m)
    Draw(
        export=False, position="topleft", show_geometry_on_click=False,
        draw_options={
            "polyline": False, "circle": False, "marker": False, "circlemarker": False,
            "rectangle": {"shapeOptions": {"color": "#f59e0b"}}, "polygon": {"shapeOptions": {"color": "#f59e0b"}},
        },
        edit_options={"edit": False, "remove": True},
    ).add_to(m)
    return m


def _metric_ring_latlon(coords):
    xs, ys = zip(*coords)
    lons, lats = drawings.to_wgs84(list(xs), list(ys))
    return [[float(la), float(lo)] for lo, la in zip(lons, lats)]


def roof_label(roof, result):
    """Label of the drawn scenario. Geometry mode: 'N panel · N×0,40 kWp · …'. Numeric mode: the kWp is not a panel
    count, so it reads '≈X kWp (<source>; paneller temsilidir) · …' without the count."""
    sc = roof["scenario"]
    discount = format_range(sc["paylasim"]["indirim_ay_ortalama_tl"], 0)
    tail = f"{tr_num(sc['kwh'])} kWh/yıl · daire başına ~{discount} TL/ay"
    if roof["mode"] == "geometry":
        return f"{drawn_panel_count(roof)} panel · {tr_num(sc['kwp'], 1)} kWp · {tail}"
    source = NUMERIC_LABEL_SOURCE.get(result.get("guc_kaynagi"), NUMERIC_LABEL_SOURCE["ortalama"])
    return f"≈{tr_num(sc['kwp'], 1)} kWp ({source}; paneller temsilidir) · {tail}"


def drawn_panels(roof):
    """{segment: (n, 4, 2) metric corners} actually drawn: all panels in geometry mode, else the numeric count."""
    out = {}
    for name, lay in roof["layouts"].items():
        limit = None if roof["limits"] is None else roof["limits"].get(name, 0)
        out[name] = lay["panels"] if limit is None else lay["panels"][:limit]
    return out


def drawn_panel_count(roof):
    return sum(len(p) for p in drawn_panels(roof).values())


def build_roof_overlay(roof, draw, result):
    """Dynamic layers of the roof map: footprint, setback, panels, dropped faces, user shapes and the label."""
    fg = folium.FeatureGroup(name="Çatı")
    for ring in drawings.polygon_to_latlon(roof["footprint"]):
        folium.Polygon(ring, color="#ffffff", weight=2, fill=False, interactive=False).add_to(fg)
    if roof["model"]["setback"] is not None:
        for ring in drawings.polygon_to_latlon(roof["model"]["setback"]):
            folium.Polygon(ring, color="#ffffff", weight=1, dash_array="5", fill=False, interactive=False).add_to(fg)
    for face in roof["model"]["faces"]:
        if layout.face_segment_name(face["name"]) in roof["skip"] and face["polygon"] is not None:
            for ring in drawings.polygon_to_latlon(face["polygon"]):
                folium.Polygon(
                    ring, color="#6b7280", weight=1, fill=True, fill_color="#6b7280", fill_opacity=0.35,
                    tooltip=DROPPED_FACE_TOOLTIP,
                ).add_to(fg)
            for line in layout.hatch_lines(face["polygon"]):
                folium.PolyLine(_metric_ring_latlon(line.coords), color="#374151", weight=1, interactive=False).add_to(fg)
    for panels in drawn_panels(roof).values():
        for corners in panels:
            folium.Polygon(
                _metric_ring_latlon(corners), color="#1d4ed8", weight=1, fill=True, fill_color="#3b82f6",
                fill_opacity=0.75, interactive=False,
            ).add_to(fg)
    for shape in drawings.stored_shapes(draw):
        color = "#16a34a" if shape["kind"] == drawings.TERRACE else "#dc2626"
        for ring in drawings.polygon_to_latlon(shape["geom"]):
            folium.Polygon(ring, color=color, weight=2, fill=True, fill_color=color, fill_opacity=0.3, interactive=False).add_to(fg)
    bounds = roof["footprint"].bounds
    lon, lat = drawings.to_wgs84((bounds[0] + bounds[2]) / 2, bounds[3])
    folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(
            icon_size=(320, 24), icon_anchor=(160, 34),
            html=(
                "<div style='background:rgba(17,24,39,.85);color:#fff;padding:3px 8px;border-radius:4px;"
                f"font-size:12px;white-space:nowrap;text-align:center'>{roof_label(roof, result)}</div>"
            ),
        ),
    ).add_to(fg)
    return fg


def _clear_drawings():
    drawings.clear_all(st.session_state[KEY_DRAWINGS])


def _confirm_empty():
    drawings.confirm_empty(st.session_state[KEY_DRAWINGS])


def _undo_drawing():
    drawings.undo_last(st.session_state[KEY_DRAWINGS])


def roof_view_section(state, roof, result, building):
    """'Çatı görünümü': the second map (isolated from the building selection) and the drawing tools.

    Only `all_drawings` is returned from this map; its clicks never change the selected building. A change to the
    stored drawings reruns once so the cards above use it; a real deletion/replacement also bumps the map key.
    """
    st.subheader("Çatı görünümü")
    draw = drawing_state(state, building)
    modes = [DRAW_OBSTACLE, DRAW_TERRACE] if result["cati_yapisi"] == "mixed" else [DRAW_OBSTACLE]
    if state.get(KEY_DRAW_MODE) not in modes:
        state.pop(KEY_DRAW_MODE, None)  # widget not created yet in this run: safe to reset
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    mode_label = c1.radio("Çizim modu", modes, key=KEY_DRAW_MODE, horizontal=True)
    c2.button("Engelleri temizle", on_click=_clear_drawings, help="Tüm çizimleri (teras dahil) siler.")
    c3.button("Son çizimi geri al", on_click=_undo_drawing)
    c4.button("Engel yok, çatı boş", on_click=_confirm_empty)
    st.caption(
        "Haritanın solundaki araçlarla dikdörtgen veya çokgen çizin. Engeller kırmızı, teras yeşil gösterilir. "
        "Çizimleri silmek için 'Engelleri temizle' veya 'Son çizimi geri al' düğmelerini kullanın."
    )
    for note in roof["notes"]:
        st.info(note)
    data = st_folium(
        build_roof_map(list(building["centroid"])), key=drawings.map_key(draw), returned_objects=["all_drawings"],
        feature_group_to_add=build_roof_overlay(roof, draw, result), height=520, use_container_width=True,
    )
    mode = drawings.TERRACE if mode_label == DRAW_TERRACE else drawings.OBSTACLE
    change = drawings.sync(draw, (data or {}).get("all_drawings"), mode, roof["footprint"])
    if change["changed"]:
        st.rerun()  # once: the cards above and the map key follow the new drawings (a notice waits in the state)
    notice, draw["notice"] = draw.get("notice"), None
    if notice:
        st.warning(notice)
    st.caption(TRASH_NOTE)
    st.caption(ROOF_VIEW_CAPTION + " " + ESRI_ATTRIBUTION)
    st.caption(report.IMAGERY_NOTE)


def breakdown_section(result):
    """'Hesap dökümü' and the PVGIS verification links (all from the result dict, no LLM)."""
    with st.expander("Hesap dökümü"):
        st.dataframe(pd.DataFrame(report.breakdown_rows(result)), hide_index=True, width="stretch")
        st.markdown("**PVGIS'te kendiniz doğrulayın**")
        st.caption(report.PVGIS_CAPTION)
        for label, url in report.pvgis_links(result):
            st.link_button(f"PVGIS'te kendiniz doğrulayın — {label}", url)


def download_section(result, assumptions):
    """'Raporu indir': one Markdown file built in memory (no disk writes)."""
    st.download_button(
        "Raporu indir", data=report.report_markdown(result, assumptions).encode("utf-8"),
        file_name="cati_ges_raporu.md", mime="text/markdown",
    )


# ---------------------------------------------------------------------- main

def main():
    st.set_page_config(page_title="İstanbul Çatı GES Uygunluk Aracı", layout="wide")

    st.title("İstanbul Çatı GES Uygunluk Aracı")
    st.caption(
        "Haritadaki mavi çerçeve içinde (Erenköy / Suadiye / Göztepe) bir apartmana tıklayın; daireler birlikte tek bir "
        "çatı güneş sistemi kurup üretimi eşit paylaşırsa ne kazanacağını öğrenin."
    )

    state = st.session_state
    index = get_building_index()
    point = state.get("selected_point")
    lookup = ensure_lookup(state, index, point)
    recenter = take_recenter(state)
    view = {"center": recenter["center"], "zoom": recenter["zoom"]} if recenter else {}
    map_data = st_folium(
        build_map(), height=500, use_container_width=True, returned_objects=["last_clicked"], key="map",
        feature_group_to_add=build_overlay(lookup, point), **view,
    )
    point, changed = updated_point(point, (map_data or {}).get("last_clicked"))
    if changed:
        clear_results(state)
        state["selected_point"] = point
        st.rerun()
    show_lookup(lookup, index)
    building = lookup["building"] if lookup and lookup["status"] == "ok" else None

    with st.sidebar:
        st.header("Yardım")
        st.markdown(HELP_STEPS)
        st.button("Sıfırla", on_click=_reset_clicked, help="Fatura, daire sayısı ve çatı yapısı girdilerini temizler.")
        st.caption(ATTRIBUTION)

    image_bytes, photo_name, is_demo, pressed = None, None, False, False
    area = 150.0
    if point:
        st.markdown("---")
        st.caption(f"Seçilen konum: {point[0]:.5f}, {point[1]:.5f}")
        if building is None:
            area = st.number_input(
                "Bina bulunamazsa kullanılacak çatı alanı (m²)", min_value=1.0, value=150.0, step=10.0, key=KEY_MANUAL_AREA
            )
        sync_building_inputs(state, building, area)
        apply_pending_roof(state)
        ensure_defaults(state, bill_was_shown=state.pop(KEY_BILL_SHOWN, False))
        flats_step(state)
        image_bytes, photo_name, is_demo, ready = photo_step(state)
        if ready:
            optional_inputs_step(state)
            pressed = st.button("Hesapla", type="primary")

    preselect_rerun = False
    if pressed:
        if not point:
            st.error("Önce haritadan bir konum seçmelisiniz.")
        else:
            api_key, key_error = resolve_api_key()  # main thread: workers never touch st.*
            footprint_inputs = {"footprint_wkb_m": building["footprint_wkb_m"] if building else None}
            try:
                detection, vision_error = None, None
                # Gemini runs on its own thread so it never blocks one of the PVGIS workers
                with ThreadPoolExecutor(max_workers=1) as vision_pool:
                    vision_future = None
                    if image_bytes is not None:
                        if api_key:
                            vision_future = vision_pool.submit(detect_roof_objects, image_bytes, api_key)
                        else:
                            vision_error = key_error
                    with st.spinner("PVGIS verileri ve fotoğraf analizi bekleniyor, lütfen bekleyin…"):
                        # flat + every unique face aspect (hip and gable), so later sub-type changes need no call
                        bundle = fetch_roof_yields(point[0], point[1], pvgis_aspects(footprint_inputs))
                        if vision_future is not None:
                            try:
                                detection = vision_future.result()
                            except VisionError as exc:
                                vision_error = str(exc)
                # A bundled demo photo never describes a real clicked building: show it, do not apply it.
                applies = photo_applies(building, is_demo)
                roof_preselect, roof_note = None, None
                if detection is not None and applies:
                    roof_preselect, roof_note = photo_roof_choice(detection)
                analysis = {
                    **bundle,
                    "detection": detection,
                    "vision_error": vision_error,
                    "key_error": key_error,
                    "roof_note": roof_note,
                    "roof_preselect": roof_preselect,
                    "photo_bytes": image_bytes if detection else None,
                    "building": building,
                    "inputs": {
                        "lat": point[0], "lon": point[1],
                        "area": building["area_m2"] if building else area,
                        "usable_base_area_m2": building["setback_area_m2"] if building else None,
                        "photo_applies": applies,
                        "demo_photo": bool(detection) and is_demo,
                        **footprint_inputs,
                    },
                }
                state["analysis"] = analysis
                inputs = collect_inputs(state, analysis, override=roof_preselect)
                all_ids = {o["id"] for o in detection["obstacles"]} if detection else set()
                first, _, _ = evaluate(analysis, inputs, detection, all_ids, drawing_state(state, building))
                eff = explained_detection(detection, all_ids, inputs)
                with st.spinner("YZ açıklaması hazırlanıyor…"):
                    state["explanation"] = generate_explanation(first, eff, bool(inputs["demo_photo"] and eff), api_key)
                if roof_preselect is not None and roof_preselect != (state.get(KEY_STRUCTURE), state.get(KEY_SHARE)):
                    state[KEY_PENDING_ROOF] = roof_preselect  # widgets exist already: apply on the next run
                    preselect_rerun = True
            except SolarDataError as exc:
                for k in ("analysis", "result", "explanation"):
                    state.pop(k, None)
                st.error(str(exc))
            except ValueError:
                for k in ("analysis", "result", "explanation"):
                    state.pop(k, None)
                st.error("Girdiler geçersiz görünüyor. Lütfen çatı alanını kontrol edin.")
    if preselect_rerun:
        st.rerun()

    analysis = state.get("analysis")
    if analysis:
        detection = analysis["detection"]
        applies = analysis["inputs"]["photo_applies"]
        st.markdown("---")
        inputs = collect_inputs(state, analysis)
        if analysis.get("roof_preselect"):
            st.caption("Çatı yapısı fotoğraf analizine göre seçildi; kenar çubuğundan değiştirebilirsiniz.")
        ids = photo_section(analysis, applies)
        draw = drawing_state(state, inputs["building"])
        result, _ratio, roof = evaluate(analysis, inputs, detection, ids or set(), draw)
        state["result"] = result
        if detection is not None:
            st.image(
                draw_overlay(analysis["photo_bytes"], detection, ids),
                caption="Yeşil: çatı sınırı · Kırmızı: dahil edilen engeller · Gri: hariç tutulan engeller",
            )
        st.subheader(f"Sonuçlar ({result['lat']:.4f}, {result['lon']:.4f})")
        if inputs["building"] is not None:
            # TODO(P2): building height joins the confidence and drives neighbour shading.
            st.caption(height_text(inputs["building"], index.report if index else None))
        eff_detection = explained_detection(detection, ids or set(), inputs)
        show_cards(result, inputs, eff_detection is not None, inputs["building"])
        if roof is not None:
            roof_view_section(state, roof, result, inputs["building"])
        else:
            st.caption("Çatı görünümü yalnızca haritadan seçilen binalar için gösterilir (bina izi gerekir).")
        explanation_section(analysis, result, eff_detection)
        show_monthly(result)
        breakdown_section(result)
        st.subheader("Varsayımlar ve kaynaklar")
        assumptions = assumption_rows(result, inputs, detection, state.get("explanation"), index.report if index else None)
        st.dataframe(assumptions, hide_index=True, width="stretch")
        download_section(result, assumptions.itertuples(index=False, name=None))



if __name__ == "__main__":
    main()
