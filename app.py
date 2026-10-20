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
from core.report import ratio_panel_text
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
KEY_BILL_KEEP, KEY_MANUAL_AREA = "_bill_keep", "manual_area"
# obstacle source: ONE active source; the other sources' data stays in the state and is ignored meanwhile
KEY_SOURCE, KEY_DEMO, KEY_DEMO_CACHE = "obstacle_source", "demo_choice", "_demo_cache"
KEY_PHOTO, KEY_PHOTO_LIVE, KEY_PHOTO_RESULT = "_photo", "_photo_uploader_live", "_photo_result"
SRC_DRAW = "Haritada çizerek (önerilen)"
SRC_PHOTO = "Kendi çatı fotoğrafımla (YZ analizi)"
SRC_DEMO = "Örnek fotoğrafla YZ'yi dene (hesaba katılmaz)"
SOURCE_CHOICES = [SRC_DRAW, SRC_PHOTO, SRC_DEMO]
SOURCE_LABEL = "Çatı engelleri nereden gelsin?"
PHOTO_FIELDS = ("detection", "vision_error", "roof_note", "roof_preselect", "photo_bytes")
KEY_BILL_SHOWN = "_bill_shown"
KEY_PITCHED, KEY_PITCHED_KEEP, KEY_DRAWINGS, KEY_DRAW_MODE = "pitched_type", "_pitched_keep", "_roof_drawings", "draw_mode"
PITCHED_CHOICES = list(building_mode.PITCHED_TYPE_LABELS_TR.values())
PITCHED_FROM_CHOICE = {label: key for key, label in building_mode.PITCHED_TYPE_LABELS_TR.items()}
HIP_CHOICE = building_mode.PITCHED_TYPE_LABELS_TR["hip"]
DRAW_OBSTACLE, DRAW_TERRACE = "Engel çiz", "Teras çiz"
ALIGN_CLICK_LABEL, ALIGN_CANCEL_LABEL = "Çatıya tıklayarak hizala", "İptal"
ALIGN_CLICK_CAPTION = "Uydu görüntüsünde çatının ortasına bir kez tıklayın."
ROOF_VIEW_PRE_CAPTION = (
    "Çatı fotoğrafınız yoksa uydu görüntüsüne bakarak çerçeveyi hizalayın ve engelleri şimdi çizebilirsiniz."
)
ALIGN_APPROX_CAPTION = (
    "Çerçeve çatıyla birebir örtüşmeyebilir (bina izi zemindeki sınırdır); çerçeveyi çatının ortasına getirmeniz yeterli."
)
ALIGN_CAPTION = (
    "Yüksek binalarda uydu görüntüsü çatıyı yana kaymış gösterir. Beyaz çerçeveyi görüntüdeki çatının üstüne getirin, "
    "engelleri sonra çizin. Hesap yine bina izi (OpenStreetMap) üzerinden yapılır."
)
# Vega cannot take a Turkish locale through st.altair_chart: swap the separators of the tick labels (2,200 -> 2.200)
TR_AXIS_LABEL_EXPR = "replace(format(datum.value, ',.0f'), ',', '.')"
ROOF_VIEW_CAPTION = (
    "Uydu görüntüsü (Esri World Imagery) yalnızca gösterim içindir; tarihi farklı olabilir ve yüksek binalarda çatı "
    "birkaç metre kaymış görünebilir. Görüntü hiçbir modele gönderilmez, saklanmaz veya analiz edilmez. "
    + ESRI_ATTRIBUTION
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
    "1. Haritada binanıza tıklayın.\n"
    "2. Daire sayısını kontrol edin.\n"
    "3. Engel kaynağını seçin: haritada çizin (önerilen) veya çatı fotoğrafı yükleyin.\n"
    "4. Gerekirse çerçeveyi çatıya hizalayın.\n"
    "5. İsteğe bağlı bilgileri girip Hesapla'ya basın."
)
NOT_USED_LAYOUT = "kullanılmadı (panel yerleşimi)"
NO_FLAT_SECTION = "kullanılmadı (düz bölüm yok)"
PHOTO_STEP_TITLE = "Çatı fotoğrafı yükleyin"
PHOTO_GUIDANCE = (
    "Çatıya ya da terasa çıkıp köşelerden çekin; fotoğraf yalnızca bellekte işlenir, kaydedilmez."
)
PHOTO_BENEFIT = "Fotoğraf, ortalama kullanılabilir alan varsayımının yerine geçer ve sonucun güvenini artırır."
PHOTO_MODE_CAPTION = "Fotoğraf modunda engeller fotoğraftan alınır; çizmek için 'Haritada çizerek' seçin."
DEMO_MODE_CAPTION = "Örnek modunda hesap ortalama orana göre yapılır; engel çizmek için 'Haritada çizerek' seçin."
NO_BUILDING_DRAW_CAPTION = "Çizim yalnızca haritadan seçilen binalar için yapılabilir; ortalama oran kullanılacak."
RATIO_NO_PANELS_CAPTION = (
    "Panel yerleşimini görmek için 'Haritada çizerek' seçip engelleri çizin veya 'Engel yok, çatı boş' deyin."
)
DEMO_DIR = Path(__file__).parent / "data" / "demo_images"
DEMO_SAMPLES = ["roof_03.jpg", "roof_05.jpg", "roof_08.jpg", "roof_10.jpg", "roof_14.jpg"]
DEMO_LABELS = [f"Örnek {i}" for i in range(1, len(DEMO_SAMPLES) + 1)]
DEMO_NOT_YOURS = "Bu fotoğraf seçtiğiniz binaya ait değildir; YZ'nin çatıda neleri bulduğunu göstermek içindir."
DEMO_ANALYZE_LABEL = "YZ ile analiz et"
DEMO_SPINNER = "Fotoğraf analiz ediliyor (1–2 dakika sürebilir)…"
DEMO_ROOF_TYPE_TR = {"flat": "düz", "pitched": "eğimli", "mixed": "karma (düz teras + eğimli)"}
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
    state[KEY_SOURCE] = SRC_DRAW  # the obstacle source starts again for a new building
    for stale in (KEY_DEMO, KEY_PHOTO, KEY_PHOTO_LIVE, KEY_PHOTO_RESULT):
        state.pop(stale, None)
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


def building_yield_text(r):
    """'1.402 – 1.460 kWh/kWp' (building yield = annual kWh / installed kWp), or None when no scenario has kWp."""
    values = [v for v in r.get("ozgul_verim_bina_aralik") or [] if v is not None]
    return format_range([min(values), max(values)], 0, "kWh/kWp") if values else None


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
    setback_area = r["kenar_sonrasi_alan_m2"] if r.get("kenar_sonrasi_alan_m2") is not None else r["toplam_cati_alani_m2"]
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
    panel_dims = f"{tr_num(PANEL_LENGTH_M, 2)} × {tr_num(PANEL_WIDTH_M, 2)} m, {tr_num(PANEL_KWP, 2)} kWp"
    if r["guc_kaynagi"] == "geometri":
        rows.append(("Panel", panel_dims, "Varsayım", f"Panel sayısı: {format_range(r['panel_sayisi_aralik'])}; yerleşim çatı görünümünde"))
    else:  # a ratio gives a kWp, not a layout
        rows.append(("Panel", ratio_panel_text(r), "Varsayım", f"Panel boyutu: {panel_dims}"))
    layout_mode = r["guc_kaynagi"] == "geometri"  # kWp comes from the panel layout (drawn obstacles or "Engel yok")
    if layout_mode:
        rows.append((
            "Engeller", f"{r['cizilen_engel_sayisi']} çizim, {tr_num(r['cizilen_engel_alani_m2'], 1)} m²", "Kullanıcı girdisi",
            "Çatı görünümünde çizildi; kurulabilir güç = panel sayısı × 0,40 kWp, kullanılabilir alan oranı kullanılmadı",
        ))
    elif r["guc_kaynagi"] == "engel_yok":
        rows.append(("Engeller", "yok", "Kullanıcı girdisi", "Kullanıcı çatının boş olduğunu onayladı (engel oranı 0)"))
    elif r["guc_kaynagi"] == "fotograf":
        rows.append((
            "Engeller", f"fotoğraf analizi, %{tr_num(percent_from_ratio(r['engel_orani']), 1)}", "Ölçüm/veri",
            "Kendi çatı fotoğrafınızdan YZ ile; kullanılabilir alan = alan × (1 − engel oranı)",
        ))
    else:
        rows.append(("Engeller", "ortalama oran", "Varsayım", "Çizim veya fotoğraf yok: ortalama kullanılabilir alan oranı"))
    if layout_mode:
        rows.append(("Kullanılabilir alan oranı", NOT_USED_LAYOUT, "Kullanıcı girdisi", "Kurulabilir güç panel yerleşiminden gelir"))
    if applied:
        view = "Kuşbakışı" if detection["image_view"] == "top-down" else "Açılı"
        rows += [
            ("Engel oranı", f"%{tr_num(percent_from_ratio(r['engel_orani']), 1)}", "Ölçüm/veri", "Gemini fotoğraf analizi; tabloda işaretlediğiniz engeller"),
            ("Görüntü türü", view, "Ölçüm/veri", "Gemini fotoğraf analizi"),
        ]
        if building is None:
            rows.append(("Kenar payı çarpanı", f"{EDGE_FACTOR}", "Varsayım", "Çatı kenarlarında panel konmayan şerit"))
    elif layout_mode:
        pass  # the unused-ratio row is added above
    elif building is None:
        rows.append(("Kullanılabilir alan oranı", f"{USABLE_RATIO_NO_PHOTO}", "Varsayım", "Ember ortalaması (fotoğraf yok)"))
    else:
        rows.append(("Kullanılabilir alan oranı", f"{tr_num(USABLE_RATIO_NO_PHOTO / EDGE_FACTOR, 3)}", "Varsayım", building_mode.USABLE_RATIO_NOTE))
    ridge = building["ridge_azimuth_deg"] if building is not None else DEFAULT_RIDGE_AZIMUTH_DEG
    rows += [
        *(
            [("Kenar boşluğu sonrası alan", f"{tr_num(setback_area, 1)} m²", "Ölçüm/veri", "Panel yerleşiminin yapıldığı alan")]
            if layout_mode else []
        ),
        (
            ("Kullanılabilir alan (plan)", NOT_USED_LAYOUT, "Varsayım", "Panel sayısı yerleşimden gelir")
            if layout_mode else
            ("Kullanılabilir alan (plan)", f"{tr_num(r['kullanilabilir_alan_m2'], 1)} m²", "Varsayım", "Alan × oran; düz ve eğimli bölümlere bölünür")
        ),
        (
            ("Düz bölüm gücü", NOT_USED_LAYOUT, "Varsayım", "Panel sayısı yerleşimden gelir")
            if layout_mode else
            ("Düz bölüm gücü", f"{tr_num(M2_PER_KWP_FLAT, 1)} m²/kWp", "Varsayım", "Plan alanı ÷ 6,5 (sıra aralığı dahil); PVGIS'in en uygun eğim ve yönü")
        ),
        ("Eğimli yüzler", f"{tr_num(M2_PER_KWP_PITCHED, 1)} m²/kWp", "Varsayım", f"Yüzey alanı (plan ÷ cos {DEFAULT_ROOF_PITCH_DEG}°) ÷ 5,5; eğim {DEFAULT_ROOF_PITCH_DEG}°, paneller çatıya paralel; eğimli alan yüzlere plan alanları oranında dağıtılır"),
        (
            "Mahya yönü", f"{tr_num(ridge, 0)}°", "Ölçüm/veri" if building is not None else "Varsayım",
            "Bina izinin uzun ekseni; yüzler mahyaya dik" if building is not None else "Bina izi yok: kuzey-güney mahya varsayıldı",
        ),
        ("Kurulabilir güç (bina)", format_range(r["kurulabilir_guc_kwp_aralik"], 1, "kWp"), "Varsayım", "Kullanılan çatı bölümlerinin toplamı"),
        (
            ("Düz bölüm eğimi / yönü", NO_FLAT_SECTION, "Varsayım", "Tamamen eğimli çatı")
            if layout_mode and r["cati_yapisi"] == "pitched" else
            ("Düz bölüm eğimi / yönü", f"{r['panel_egim_derece']}° / {r['panel_yon_derece']}°", "Varsayım", "PVGIS'in en uygun eğim ve yönü")
        ),
        ("Yıllık ışınım", f"{tr_num(r['yillik_isinim_kwh_m2'])} kWh/m²", "Ölçüm/veri", "PVGIS (JRC)"),
        ("Özgül verim (düz bölüm, referans)", f"{tr_num(r['golgeli_uretim_kwh_kwp'])} kWh/kWp", "Ölçüm/veri", "PVGIS (JRC), arazi gölgesi dahil; çatı yüzlerini elemek için referans"),
        *(
            [("Özgül verim (bina)", building_yield_text(r), "Ölçüm/veri", "Yıllık üretim ÷ kurulabilir güç")]
            if building_yield_text(r) else []
        ),
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
        y=alt.Y("kWh:Q", title="Aylık üretim (kWh)", axis=alt.Axis(labelExpr=TR_AXIS_LABEL_EXPR)),
        color=alt.Color("Seri:N", title=None),
        tooltip=["Ay", "Grup", "Seri", alt.Tooltip("kWh (metin):N", title="kWh")],
    )
    if len(groups) > 1:
        enc["xOffset"] = "Grup:N"
    for row in rows:
        row["kWh (metin)"] = tr_num(row["kWh"], 1)
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
                "Plan alanı (m²)": tr_num(seg["plan_m2"], 1),
                "Yüzey alanı (m²)": tr_num(seg.get("yuzey_m2", seg["plan_m2"]), 1),
                "kWp": tr_num(seg["kwp"], 1), "kWh/yıl": tr_num(sum(seg["aylik_kwh"])),
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
            if seg["tur"] == "face" and r["cati_yapisi"] == "flat":
                continue  # a fully flat roof has no pitched faces to report
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
    if (yield_text := building_yield_text(r)) is not None:  # hidden without installed kWp
        e2.metric("Özgül verim (bina)", yield_text)
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


def monthly_caption(r):
    """Chart caption: flat section only if one is used, pitched wording, plan area only outside layout mode."""
    segments = [seg for sc in r["senaryolar"] for seg in sc["segmentler"]]
    flat_used = any(seg["tur"] == "flat" and seg["durum"] == "kullanildi" for seg in segments)
    pitched = r["cati_yapisi"] != "flat"
    pitch = f"eğimli yüzler {DEFAULT_ROOF_PITCH_DEG}° eğimle mahyaya dik yönlerde varsayıldı"
    if flat_used and pitched:
        text = f"Düz bölüm en uygun eğim ({r['panel_egim_derece']}°) ve yönde; {pitch}."
    elif flat_used:
        text = f"Düz bölüm en uygun eğim ({r['panel_egim_derece']}°) ve yönde."
    elif pitched:
        text = pitch[0].upper() + pitch[1:] + "."
    else:
        text = ""
    if r["guc_kaynagi"] != "geometri":
        text = f"{text} Kullanılabilir plan alanı: {tr_num(r['kullanilabilir_alan_m2'], 1)} m².".strip()
    return text


def show_monthly(r):
    st.subheader("Aylık üretim")
    monthly_chart(r)
    st.dataframe(segment_table(r), hide_index=True, width="stretch")
    st.caption(monthly_caption(r))


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


def active_analysis(state, analysis, source):
    """The stored analysis seen through the active obstacle source.

    The photo analysis (detection, annotated photo, roof-type suggestion) is kept in `KEY_PHOTO_RESULT` and only
    joins the calculation while the photo source is selected; every other source sees a photo-less analysis.
    """
    photo = (state.get(KEY_PHOTO_RESULT) or {}) if source == SRC_PHOTO else {}
    return {**analysis, **{field: photo.get(field) for field in PHOTO_FIELDS}}


def photo_upload(state):
    """Uploader of the photo source. Returns {"name", "bytes"} of the photo to analyse, or None.

    The photo is kept in the session state (memory only) so it survives a trip to another source; removing it from
    the uploader while this source is visible forgets it.
    """
    st.write(PHOTO_GUIDANCE)
    uploaded = st.file_uploader(PHOTO_STEP_TITLE, type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.caption(PHOTO_BENEFIT)
    if uploaded is not None:
        state[KEY_PHOTO_LIVE] = True
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Dosya çok büyük; en fazla {MAX_UPLOAD_MB} MB yükleyebilirsiniz.")
            state.pop(KEY_PHOTO, None)
        else:
            state[KEY_PHOTO] = {"name": uploaded.name, "bytes": uploaded.getvalue()}
    elif state.pop(KEY_PHOTO_LIVE, False):  # the user removed the file from the uploader
        state.pop(KEY_PHOTO, None)
    photo = state.get(KEY_PHOTO)
    if photo is not None and uploaded is None:
        st.caption(f"Daha önce yüklenen fotoğraf kullanılıyor: {photo['name']}. Yeni bir fotoğraf yükleyebilirsiniz.")
    return photo


def show_demo_result(image_bytes, detection):
    """Annotated photo, obstacle table and roof-type remark of a sample analysis (display only)."""
    obstacles = detection["obstacles"]
    st.image(
        draw_overlay(image_bytes, detection, {o["id"] for o in obstacles}),
        caption="Yeşil: çatı sınırı · Kırmızı: YZ'nin bulduğu engeller",
    )
    if obstacles:
        st.dataframe(
            pd.DataFrame({
                "No": [o["id"] for o in obstacles],
                "Engel": [OBSTACLE_LABELS_TR[o["label"]] for o in obstacles],
                "Güven": [f"%{round(o['confidence'] * 100)}" for o in obstacles],
            }),
            hide_index=True, width="stretch",
        )
    else:
        st.write("Çatıda engel tespit edilmedi.")
    st.write(
        f"YZ bu fotoğraftaki çatıyı {DEMO_ROOF_TYPE_TR[detection['roof_type']]} olarak değerlendirdi. "
        "Bu yalnızca bir örnektir; çatı yapısı seçiminizi ve hesabınızı değiştirmez."
    )


def demo_section(state):
    """Sample photos: shown at once; the Gemini analysis runs only on 'YZ ile analiz et' and is cached per sample.

    Nothing here ever changes an input or a result.
    """
    choice = st.radio("Örnek fotoğraf", DEMO_LABELS, index=None, key=KEY_DEMO, horizontal=True)
    if choice is None:
        return
    filename = DEMO_SAMPLES[DEMO_LABELS.index(choice)]
    try:
        image_bytes = (DEMO_DIR / filename).read_bytes()
    except OSError:
        st.info("Örnek fotoğraf bulunamadı.")
        return
    st.image(image_bytes, caption=source_caption(filename))
    st.caption(f"Dosya: {filename}")
    st.info(DEMO_NOT_YOURS)
    cache = state.setdefault(KEY_DEMO_CACHE, {})
    if st.button(DEMO_ANALYZE_LABEL, key="demo_analyze"):
        api_key, key_error = resolve_api_key()
        if api_key is None:
            cache[filename] = {"error": key_error}
        else:
            with st.spinner(DEMO_SPINNER):
                try:
                    cache[filename] = {"detection": detect_roof_objects(image_bytes, api_key)}
                except VisionError as exc:
                    cache[filename] = {"error": str(exc)}
    entry = cache.get(filename)
    if entry is None:
        return
    if "error" in entry:
        st.warning(entry["error"])
    else:
        show_demo_result(image_bytes, entry["detection"])


def source_step(state, building):
    """The one obstacle source ('Çatı engelleri nereden gelsin?'). Returns (source, photo or None).

    Only the selected option's widgets are shown. Only the photo source returns a photo; the other sources'
    data stays in the state and is ignored by the calculation until their option is selected again.
    """
    st.subheader("Engel kaynağı")
    state.setdefault(KEY_SOURCE, SRC_DRAW)
    source = st.radio(SOURCE_LABEL, SOURCE_CHOICES, key=KEY_SOURCE)
    photo = None
    if source == SRC_PHOTO:
        photo = photo_upload(state)
        st.caption(PHOTO_MODE_CAPTION)
    else:
        state.pop(KEY_PHOTO_LIVE, None)  # a fresh uploader on return is not a removal
        if source == SRC_DEMO:
            demo_section(state)
        elif building is None:
            st.caption(NO_BUILDING_DRAW_CAPTION)
    return source, photo


def photo_section(state, applies):
    """Show detection messages, the confirmation table and the overlay ids; return the ticked obstacle ids or None."""
    detection = state.get("detection")
    if state.get("vision_error"):
        st.warning(f"{state['vision_error']} Fotoğrafsız hesap gösteriliyor; güven seviyesi düşük.")
    if detection is None:
        return None

    st.subheader("Fotoğraf analizi")
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

CROSSHAIR_CSS = "<style>.leaflet-container {cursor: crosshair !important;}</style>"


def build_roof_map(center, draw_tools=True, aligning=False):
    """Static part of the roof map: Esri World Imagery (DISPLAY ONLY, attribution kept) and the Draw control
    (rectangle and polygon only; editing off, deleting on). `draw_tools=False` (click-to-align mode) omits the
    toolbar so an alignment click can never add a vertex. `aligning` also turns off dragging, double-click zoom, box
    zoom and keyboard panning (a click must never become a pan) and shows a crosshair; they return when it ends."""
    # no scroll-wheel zoom: page scrolling must not be hijacked (the +/- buttons stay)
    m = folium.Map(
        location=center, zoom_start=ROOF_VIEW_ZOOM, max_zoom=21, tiles=None, scroll_wheel_zoom=False,
        dragging=not aligning, double_click_zoom=not aligning, box_zoom=not aligning, keyboard=not aligning,
    )
    if aligning:
        m.get_root().html.add_child(folium.Element(CROSSHAIR_CSS))
    folium.TileLayer(
        tiles=ESRI_TILES_URL, attr=ESRI_ATTRIBUTION, name="Uydu (yalnızca görüntüleme)", max_native_zoom=19, max_zoom=21,
    ).add_to(m)
    if draw_tools:
        Draw(
            export=False, position="topleft", show_geometry_on_click=False,
            draw_options={
                "polyline": False, "circle": False, "marker": False, "circlemarker": False,
                "rectangle": {"shapeOptions": {"color": "#f59e0b"}}, "polygon": {"shapeOptions": {"color": "#f59e0b"}},
            },
            edit_options={"edit": False, "remove": True},
        ).add_to(m)
    return m


def _metric_ring_latlon(coords, offset=(0.0, 0.0)):
    xs, ys = zip(*coords)
    xs, ys = [x + offset[0] for x in xs], [y + offset[1] for y in ys]
    lons, lats = drawings.to_wgs84(list(xs), list(ys))
    return [[float(la), float(lo)] for lo, la in zip(lons, lats)]


def roof_label(roof, result):
    """Label of the drawn scenario. Geometry mode: 'N panel · N×0,40 kWp · …'. Ratio modes: the kWp is not a panel
    count and no panels are drawn, so it reads '≈X kWp (<source>) · …' without the count."""
    sc = roof["scenario"]
    discount = format_range(sc["paylasim"]["indirim_ay_ortalama_tl"], 0)
    tail = f"{tr_num(sc['kwh'])} kWh/yıl · daire başına ~{discount} TL/ay"
    if roof["mode"] == "geometry":
        return f"{drawn_panel_count(roof)} panel · {tr_num(sc['kwp'], 1)} kWp · {tail}"
    source = NUMERIC_LABEL_SOURCE.get(result.get("guc_kaynagi"), NUMERIC_LABEL_SOURCE["ortalama"])
    return f"≈{tr_num(sc['kwp'], 1)} kWp ({source}) · {tail}"


def drawn_panels(roof):
    """{segment: (n, 4, 2) metric corners} actually drawn: all panels in geometry mode, else the numeric count."""
    out = {}
    for name, lay in roof["layouts"].items():
        limit = None if roof["limits"] is None else roof["limits"].get(name, 0)
        out[name] = lay["panels"] if limit is None else lay["panels"][:limit]
    return out


def drawn_panel_count(roof):
    return sum(len(p) for p in drawn_panels(roof).values())


def build_roof_overlay(footprint, setback, draw, roof=None, result=None, show_drawings=True):
    """Dynamic layers of the roof map: footprint, setback and (only for the drawing source) user shapes; with a
    calculated `roof` (after 'Hesapla') also the dropped faces and the label, and the panels in layout mode ONLY:
    ratio modes (average / photo) know a kWp, not a layout, so no panel rectangles are drawn."""
    fg = folium.FeatureGroup(name="Çatı")
    off = draw.get("offset", (0.0, 0.0))  # display-only imagery alignment: every overlay moves, the data does not
    for ring in drawings.polygon_to_latlon(footprint, off):
        folium.Polygon(ring, color="#ffffff", weight=2, fill=False, interactive=False).add_to(fg)
    if setback is not None:
        for ring in drawings.polygon_to_latlon(setback, off):
            folium.Polygon(ring, color="#ffffff", weight=1, dash_array="5", fill=False, interactive=False).add_to(fg)
    if roof is not None:
        for face in roof["model"]["faces"]:
            if layout.face_segment_name(face["name"]) in roof["skip"] and face["polygon"] is not None:
                for ring in drawings.polygon_to_latlon(face["polygon"], off):
                    folium.Polygon(
                        ring, color="#6b7280", weight=1, fill=True, fill_color="#6b7280", fill_opacity=0.35,
                        tooltip=DROPPED_FACE_TOOLTIP,
                    ).add_to(fg)
                for line in layout.hatch_lines(face["polygon"]):
                    folium.PolyLine(_metric_ring_latlon(line.coords, off), color="#374151", weight=1, interactive=False).add_to(fg)
        for panels in drawn_panels(roof).values() if roof["mode"] == "geometry" else ():
            for corners in panels:
                folium.Polygon(
                    _metric_ring_latlon(corners, off), color="#1d4ed8", weight=1, fill=True, fill_color="#3b82f6",
                    fill_opacity=0.75, interactive=False,
                ).add_to(fg)
    for shape in drawings.stored_shapes(draw) if show_drawings else ():
        color = "#16a34a" if shape["kind"] == drawings.TERRACE else "#dc2626"
        for ring in drawings.polygon_to_latlon(shape["geom"], off):
            folium.Polygon(ring, color=color, weight=2, fill=True, fill_color=color, fill_opacity=0.3, interactive=False).add_to(fg)
    if roof is not None:
        bounds = footprint.bounds
        lon, lat = drawings.to_wgs84((bounds[0] + bounds[2]) / 2 + off[0], bounds[3] + off[1])
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


def _move_offset(dx, dy):
    drawings.move_offset(st.session_state[KEY_DRAWINGS], dx, dy)


def _reset_offset():
    drawings.reset_offset(st.session_state[KEY_DRAWINGS])


def _start_align():
    drawings.start_align(st.session_state[KEY_DRAWINGS])


def _cancel_align():
    drawings.cancel_align(st.session_state[KEY_DRAWINGS])


def alignment_control(draw):
    """'Çerçeveyi çatıya hizala': click-to-align, arrow buttons (1 m and 5 m per click), reset and the current offset.
    Display only."""
    st.markdown("**Çerçeveyi çatıya hizala**")
    if draw.get("align_active"):
        st.button(ALIGN_CANCEL_LABEL, key="align_cancel", on_click=_cancel_align)
        st.caption(ALIGN_CLICK_CAPTION)
    else:
        st.button(ALIGN_CLICK_LABEL, key="align_click", on_click=_start_align)
    for step, prefix in ((drawings.OFFSET_STEP_M, ""), (drawings.OFFSET_STEP_BIG_M, "5 m ")):
        cols = st.columns([1, 1, 1, 1, 1, 3])
        deltas = ((0, step), (0, -step), (-step, 0), (step, 0))
        for col, arrow, delta in zip(cols, ("↑", "↓", "←", "→"), deltas):
            col.button(f"{prefix}{arrow}", key=f"align{prefix.strip()}_{arrow}", on_click=_move_offset, args=delta)
        if not prefix:
            cols[4].button("Sıfırla", key="align_reset", on_click=_reset_offset)
            cols[5].markdown(drawings.offset_text(draw["offset"]))
    st.caption(ALIGN_CAPTION)
    st.caption(ALIGN_APPROX_CAPTION)


def roof_view_section(state, building, roof=None, result=None, source=SRC_DRAW):
    """'Çatı görünümü': the one roof map (isolated from the building selection) and the drawing tools.

    The map is always rendered with the same key. Drawing tools, buttons and stored shapes belong to the drawing
    source only: with another source the toolbar is hidden, the stored drawings are neither rendered nor changed
    and whatever the map returns is ignored (an empty or missing `all_drawings` is never a deletion).

    Rendered in one fixed place above 'Hesapla'. Before 'Hesapla' (`roof` is None) it shows the footprint, the setback
    line and the user's shapes only; drawing and offset changes just update the stored state. After 'Hesapla' it also
    shows panels, dropped faces and the label. Only `all_drawings` is returned from this map (plus `last_clicked`
    while click-to-align is active; then the Draw toolbar is hidden and drawings are not read). Its clicks never
    change the selected building. A change to the stored drawings reruns once so the cards use it; a real
    deletion/replacement also bumps the map key.
    """
    st.subheader("Çatı görünümü")
    draw = drawing_state(state, building)
    footprint = shapely.from_wkb(building["footprint_wkb_m"])
    setback = roof["model"]["setback"] if roof is not None else layout.setback_polygon(footprint)
    structure = result["cati_yapisi"] if result is not None else STRUCTURE_FROM_CHOICE.get(state.get(KEY_STRUCTURE))
    drawing = source == SRC_DRAW
    mode_label = DRAW_OBSTACLE
    if drawing:
        modes = [DRAW_OBSTACLE, DRAW_TERRACE] if structure == "mixed" else [DRAW_OBSTACLE]
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
        if roof is None:
            st.caption(ROOF_VIEW_PRE_CAPTION)
    else:
        if source == SRC_DEMO:  # the photo-mode caption is already shown once, under the uploader
            st.caption(DEMO_MODE_CAPTION)
    alignment_control(draw)
    for note in (roof["notes"] if roof is not None else []):
        if note not in (NOTE_AVERAGE, NOTE_PHOTO):  # RATIO_NO_PANELS_CAPTION below replaces them
            st.info(note)
    if roof is not None and roof["mode"] != "geometry":
        st.caption(RATIO_NO_PANELS_CAPTION)
    aligning = bool(draw.get("align_active"))
    data = st_folium(
        build_roof_map(list(building["centroid"]), draw_tools=drawing and not aligning, aligning=aligning), key=drawings.map_key(draw),
        returned_objects=["all_drawings", "last_clicked"] if aligning else ["all_drawings"],
        feature_group_to_add=build_roof_overlay(footprint, setback, draw, roof, result, show_drawings=drawing),
        height=520, use_container_width=True,
    )
    if aligning:
        if drawings.apply_click(draw, (data or {}).get("last_clicked"), footprint):
            st.rerun()  # once: the mode is off and the map is rebuilt at the new offset
        change = {"changed": False}  # a click never creates, changes or deletes drawings
    elif drawing:
        mode = drawings.TERRACE if mode_label == DRAW_TERRACE else drawings.OBSTACLE
        change = drawings.sync(draw, (data or {}).get("all_drawings"), mode, footprint)
    else:
        change = {"changed": False}  # another source is active: the map's drawing output is ignored completely
    if change["changed"]:
        st.rerun()  # once: the cards above and the map key follow the new drawings (a notice waits in the state)
    if drawing:
        notice, draw["notice"] = draw.get("notice"), None
        if notice:
            st.warning(notice)
        st.caption(TRASH_NOTE)
    st.caption(ROOF_VIEW_CAPTION)


def breakdown_section(result, offset=None):
    """'Hesap dökümü' and the PVGIS verification links (all from the result dict, no LLM)."""
    with st.expander("Hesap dökümü"):
        st.dataframe(pd.DataFrame(report.breakdown_rows(result, offset)), hide_index=True, width="stretch")
        st.markdown("**PVGIS'te kendiniz doğrulayın**")
        st.caption(report.PVGIS_CAPTION)
        for label, url in report.pvgis_links(result):
            st.link_button(f"PVGIS'te kendiniz doğrulayın — {label}", url)


def download_section(result, assumptions, offset=None):
    """'Raporu indir': one Markdown file built in memory (no disk writes)."""
    st.download_button(
        "Raporu indir", data=report.report_markdown(result, assumptions, offset).encode("utf-8"),
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

    source, photo, pressed = state.get(KEY_SOURCE, SRC_DRAW), None, False
    image_bytes = None
    roof_slot = None  # the one fixed place of the roof map: above 'Hesapla', filled once per run further down
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
        source, photo = source_step(state, building)
        image_bytes = photo["bytes"] if photo is not None else None  # only the photo source feeds Gemini on Hesapla
        optional_inputs_step(state)
        if building is not None:
            roof_slot = st.container()
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
                roof_preselect, roof_note = photo_roof_choice(detection) if detection is not None else (None, None)
                if source == SRC_PHOTO:  # the photo analysis is replaced only while the photo source is active
                    if image_bytes is not None:
                        state[KEY_PHOTO_RESULT] = {
                            "detection": detection, "vision_error": vision_error, "roof_note": roof_note,
                            "roof_preselect": roof_preselect, "photo_bytes": image_bytes if detection else None,
                        }
                    else:
                        state.pop(KEY_PHOTO_RESULT, None)
                analysis = {
                    **bundle,
                    "key_error": key_error,
                    "building": building,
                    "inputs": {
                        "lat": point[0], "lon": point[1],
                        "area": building["area_m2"] if building else area,
                        "usable_base_area_m2": building["setback_area_m2"] if building else None,
                        "photo_applies": True,
                        "demo_photo": False,
                        **footprint_inputs,
                    },
                }
                state["analysis"] = analysis
                view = active_analysis(state, analysis, source)
                inputs = collect_inputs(state, view, override=roof_preselect)
                all_ids = {o["id"] for o in detection["obstacles"]} if detection else set()
                active_draw = drawing_state(state, building) if source == SRC_DRAW else None
                first, _, _ = evaluate(view, inputs, detection, all_ids, active_draw)
                eff = explained_detection(detection, all_ids, inputs)
                with st.spinner("YZ açıklaması hazırlanıyor…"):
                    state["explanation"] = generate_explanation(first, eff, False, api_key)
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
    if not analysis and roof_slot is not None:
        with roof_slot:  # before 'Hesapla': imagery, outline, alignment and drawing tools only; no evaluation
            roof_view_section(state, building, source=source)
    if analysis:
        analysis = active_analysis(state, analysis, source)  # a photo analysis counts only for the photo source
        detection = analysis["detection"]
        applies = analysis["inputs"]["photo_applies"]
        st.markdown("---")
        inputs = collect_inputs(state, analysis)
        if analysis.get("roof_preselect"):
            st.caption("Çatı yapısı fotoğraf analizine göre seçildi; kenar çubuğundan değiştirebilirsiniz.")
        ids = photo_section(analysis, applies)
        draw = drawing_state(state, inputs["building"])  # alignment offset lives here whatever the source is
        result, _ratio, roof = evaluate(
            analysis, inputs, detection, ids or set(), draw if source == SRC_DRAW else None
        )
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
            with roof_slot:
                roof_view_section(state, inputs["building"], roof, result, source=source)
        else:
            st.caption("Çatı görünümü yalnızca haritadan seçilen binalar için gösterilir (bina izi gerekir).")
        explanation_section(analysis, result, eff_detection)
        show_monthly(result)
        offset = draw.get("offset") if roof is not None else None
        breakdown_section(result, offset)
        st.subheader("Varsayımlar ve kaynaklar")
        assumptions = assumption_rows(result, inputs, detection, state.get("explanation"), index.report if index else None)
        st.dataframe(assumptions, hide_index=True, width="stretch")
        download_section(result, assumptions.itertuples(index=False, name=None), offset)



if __name__ == "__main__":
    main()
