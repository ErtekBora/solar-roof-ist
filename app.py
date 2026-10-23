"""Streamlit UI (Turkish): Istanbul rooftop solar suitability tool."""
import logging

import altair as alt
import folium
import pandas as pd
import shapely
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from core import buildings, building_mode, cost, drawings, explain, imagery_fx, layout, report
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
from core.vision import compute_obstacle_ratio

log = logging.getLogger(__name__)
MONTHS_TR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
ROOF_CHOICES = list(building_mode.STRUCTURE_LABELS_TR.values())
STRUCTURE_FROM_CHOICE = {label: key for key, label in building_mode.STRUCTURE_LABELS_TR.items()}
UNKNOWN_CHOICE = building_mode.STRUCTURE_LABELS_TR["unknown"]
MIXED_CHOICE = building_mode.STRUCTURE_LABELS_TR["mixed"]
# session keys: stable widget keys, plus bookkeeping for the per-building reset
KEY_BILL, KEY_FLATS, KEY_STRUCTURE, KEY_SHARE = "bill_input", "n_flats", "roof_structure", "flat_share"
KEY_BUILDING, KEY_FLATS_ESTIMATE = "_input_building", "_flats_estimate"
KEY_BILL_KEEP, KEY_MANUAL_AREA = "_bill_keep", "manual_area"
KEY_BILL_SHOWN = "_bill_shown"
KEY_PITCHED, KEY_PITCHED_KEEP, KEY_DRAWINGS, KEY_DRAW_MODE = "pitched_type", "_pitched_keep", "_roof_drawings", "draw_mode"
PITCHED_CHOICES = list(building_mode.PITCHED_TYPE_LABELS_TR.values())
PITCHED_FROM_CHOICE = {label: key for key, label in building_mode.PITCHED_TYPE_LABELS_TR.items()}
HIP_CHOICE = building_mode.PITCHED_TYPE_LABELS_TR["hip"]
DRAW_OBSTACLE, DRAW_TERRACE = "Engel çiz", "Teras çiz"
ALIGN_CLICK_LABEL, ALIGN_CANCEL_LABEL = "Çatıya tıklayarak hizala", "İptal"
ALIGN_CLICK_CAPTION = "Uydu görüntüsünde çatının ortasına bir kez tıklayın."
ROOF_VIEW_PRE_CAPTION = (
    "Uydu görüntüsüne bakarak çerçeveyi hizalayın ve engelleri şimdi çizebilirsiniz."
)
ALIGN_EXPANDER_LABEL = "Çerçeveyi çatıya hizala (yüksek binalarda gerekir)"  # fixed: a changing label would collapse it
KEY_ALIGN_EXPANDED = "_align_expanded"  # building id whose alignment expander stays open after an alignment click
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
NOTE_FLAT_ASSUMED = "Değer düz çatı varsayımıyla hesaplandı."
NOTE_FLAT_ASSUMED_GEOMETRY = (
    "Panel yerleşimi düz çatı varsayımıyla gösteriliyor; çatı yapısını seçerseniz sonuç tek değere iner."
)
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
    "3. Engelleri haritada çizin.\n"
    "4. Gerekirse çerçeveyi çatıya hizalayın.\n"
    "5. İsteğe bağlı bilgileri girip Hesapla'ya basın."
)
NOT_USED_LAYOUT = "kullanılmadı (panel yerleşimi)"
NO_FLAT_SECTION = "kullanılmadı (düz bölüm yok)"
NO_BUILDING_DRAW_CAPTION = "Çizim yalnızca haritadan seçilen binalar için yapılabilir; ortalama oran kullanılacak."
RATIO_NO_PANELS_CAPTION = (
    "Panel yerleşimini görmek için engelleri çizin veya 'Engel yok, çatı boş' deyin."
)
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


def reset_inputs(state):
    """'Sıfırla': clears the bill too and forces the per-building reset on the next run."""
    state[KEY_BILL] = None
    state[KEY_BILL_KEEP] = None
    state.pop(KEY_BUILDING, None)


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
        assumed.append("Engeller: çizim yok, ortalama kullanılabilir alan oranı")
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
        ("Çatı yapısı", structure_text(r), inputs["roof_tag"], "Kullanıcı seçimi veya 'Bilmiyorum' (iki uç durum hesaplanır)"),
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
        rows.append(("Engeller", "ortalama oran", "Varsayım", "Çizim yok: ortalama kullanılabilir alan oranı"))
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
        rows.append(("Kullanılabilir alan oranı", f"{USABLE_RATIO_NO_PHOTO}", "Varsayım", "Ember ortalaması (çizim yok)"))
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
        ("Güven seviyesi", r["guven_seviyesi"], "Varsayım", "Alan kaynağı, engel çizimi, çatı yapısı ve çatı yüzü verisine bağlı kural"),
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
# fractional zoom leaves a 1 px gap between tiles: 1 px of overlap hides it (over-zoom scales the level container,
# each tile stays 256 px); roof tile layer only
TILE_SEAM_CSS = "<style>.roof-esri .leaflet-tile { width: 257px !important; height: 257px !important; }</style>"
KEY_ROOF_BOUNDS = "_roof_view_bounds"  # {"building": id, "bounds": [[s, w], [n, e]]}: the fixed roof-map view
ROOF_FIT_MARGIN = 0.15  # share of the footprint bbox span added on every side
ROOF_MAX_ZOOM = 22
ESRI_NATIVE_ZOOM = 19  # deepest Esri level used; above it tiles are upscaled instead of disappearing
ROOF_MAP_HEIGHT = 620
ROOF_TILE_CLASS = "roof-esri"
FIT_VIEW_LABEL = "Görünümü çerçeveye ortala"
# display-only imagery enhancement (browser-side SVG filter); the plain key keeps the choice while the map is hidden
KEY_FX_ON, KEY_FX_ON_KEEP = "roof_fx_on", "_roof_fx_on_keep"


def build_roof_map(center, draw_tools=True, aligning=False, bounds=None, enhance=False):
    """Static part of the roof map: Esri World Imagery (DISPLAY ONLY, attribution kept) and the Draw control
    (rectangle and polygon only; editing off, deleting on). `draw_tools=False` (click-to-align mode) omits the
    toolbar so an alignment click can never add a vertex; `aligning` also shows a crosshair.

    The view is static: no dragging, zooming or zoom buttons. With `bounds` ([[s, w], [n, e]]) Leaflet fits the map
    to them on every build (zoomSnap 0.1, so the same fractional zoom each time). `enhance` adds the gentle
    browser-side SVG filter on the tile layer (core/imagery_fx); False adds nothing."""
    m = folium.Map(
        location=center, zoom_start=ROOF_VIEW_ZOOM, max_zoom=ROOF_MAX_ZOOM, tiles=None, zoom_control=False,
        scroll_wheel_zoom=False, dragging=False, double_click_zoom=False, touch_zoom=False, box_zoom=False,
        keyboard=False, zoom_snap=0.1, zoom_delta=0.1,
    )
    m.options["maxZoom"] = ROOF_MAX_ZOOM  # folium 0.20 does not pass Map(max_zoom=...) to the Leaflet map options
    if aligning:
        m.get_root().html.add_child(folium.Element(CROSSHAIR_CSS))
    m.get_root().html.add_child(folium.Element(TILE_SEAM_CSS))
    folium.TileLayer(
        tiles=ESRI_TILES_URL, attr=ESRI_ATTRIBUTION, name="Uydu (yalnızca görüntüleme)",
        max_native_zoom=ESRI_NATIVE_ZOOM, max_zoom=ROOF_MAX_ZOOM, class_name=ROOF_TILE_CLASS,
    ).add_to(m)
    if bounds is not None:
        m.fit_bounds(bounds, max_zoom=ROOF_MAX_ZOOM, padding=(10, 10))
    if enhance:  # after fit_bounds: the zoom script must run once the fitted view exists
        imagery_fx.add_to_map(m, ROOF_TILE_CLASS)
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


def _keep_alignment_open():
    """Alignment clicks keep 'Çerçeveyi çatıya hizala' open on the reruns that follow (for this building only)."""
    st.session_state[KEY_ALIGN_EXPANDED] = st.session_state[KEY_DRAWINGS]["building"]


def _move_offset(dx, dy):
    drawings.move_offset(st.session_state[KEY_DRAWINGS], dx, dy)
    _keep_alignment_open()


def _reset_offset(building_id=None, footprint_wkb=None):
    """'Sıfırla': offset back to zero and, for a building, the initial fitted view (footprint bbox + 15%) again."""
    draw = st.session_state[KEY_DRAWINGS]
    drawings.reset_offset(draw)
    if footprint_wkb is not None:
        initial = {"building": building_id, "bounds": roof_fit_bounds(shapely.from_wkb(footprint_wkb))}
        if st.session_state.get(KEY_ROOF_BOUNDS) != initial:
            st.session_state[KEY_ROOF_BOUNDS] = initial
            drawings.rebuild(draw)  # a new view is a new map build
    _keep_alignment_open()


def _start_align():
    drawings.start_align(st.session_state[KEY_DRAWINGS])
    _keep_alignment_open()


def _cancel_align():
    drawings.cancel_align(st.session_state[KEY_DRAWINGS])
    _keep_alignment_open()


def roof_fit_bounds(footprint, offset=(0.0, 0.0)):
    """[[s, w], [n, e]] of the (offset-shifted) metric footprint in WGS84, widened by ROOF_FIT_MARGIN of each span."""
    rings = drawings.polygon_to_latlon(footprint, offset)
    lats = [p[0] for ring in rings for p in ring]
    lons = [p[1] for ring in rings for p in ring]
    d_lat, d_lon = (max(lats) - min(lats)) * ROOF_FIT_MARGIN, (max(lons) - min(lons)) * ROOF_FIT_MARGIN
    return [[min(lats) - d_lat, min(lons) - d_lon], [max(lats) + d_lat, max(lons) + d_lon]]


def roof_view_bounds(state, building, footprint):
    """The fixed roof-map view of this building: set once from the footprint; only 'Görünümü çerçeveye ortala'
    changes it (offset, drawings and enhancement never do)."""
    stored = state.get(KEY_ROOF_BOUNDS)
    if stored is None or stored["building"] != building_key(building):
        stored = state[KEY_ROOF_BOUNDS] = {"building": building_key(building), "bounds": roof_fit_bounds(footprint)}
    return stored["bounds"]


def _fit_view_to_frame(building_id, footprint_wkb):
    """'Görünümü çerçeveye ortala': the view follows the frame shifted by the current offset (a new map is built)."""
    draw = st.session_state[KEY_DRAWINGS]
    bounds = roof_fit_bounds(shapely.from_wkb(footprint_wkb), draw["offset"])
    st.session_state[KEY_ROOF_BOUNDS] = {"building": building_id, "bounds": bounds}
    drawings.rebuild(draw)
    _keep_alignment_open()


def _fx_changed():
    """Enhancement switched on/off: keep the choice and rebuild the map (drawings and view stay)."""
    state = st.session_state
    state[KEY_FX_ON_KEEP] = state.get(KEY_FX_ON)
    drawings.rebuild(state[KEY_DRAWINGS])


def enhancement_control(state):
    """'Görüntü iyileştirme' toggle above the roof map (default on; off shows the original). Returns True when on."""
    if KEY_FX_ON not in state:  # hidden widgets lose their state: restore it first
        state[KEY_FX_ON] = state.get(KEY_FX_ON_KEEP, True) is not False
    on = st.toggle("Görüntü iyileştirme", key=KEY_FX_ON, on_change=_fx_changed)
    st.caption(imagery_fx.CAPTION)
    return on


def alignment_control(draw, building=None):
    """'Çerçeveyi çatıya hizala': click-to-align, arrow buttons (1 m and 5 m per click), reset and the current offset.
    Display only."""
    if not draw.get("align_active"):  # while active, 'İptal' and its caption live outside the expander
        st.button(ALIGN_CLICK_LABEL, key="align_click", on_click=_start_align)
    for step, prefix in ((drawings.OFFSET_STEP_M, ""), (drawings.OFFSET_STEP_BIG_M, "5 m ")):
        cols = st.columns([1, 1, 1, 1, 1, 3])
        deltas = ((0, step), (0, -step), (-step, 0), (step, 0))
        for col, arrow, delta in zip(cols, ("↑", "↓", "←", "→"), deltas):
            col.button(f"{prefix}{arrow}", key=f"align{prefix.strip()}_{arrow}", on_click=_move_offset, args=delta)
        if not prefix:
            reset_args = (building_key(building), building["footprint_wkb_m"]) if building is not None else ()
            cols[4].button("Sıfırla", key="align_reset", on_click=_reset_offset, args=reset_args)
            if building is not None:
                cols[5].button(
                    FIT_VIEW_LABEL, key="align_fit_view", on_click=_fit_view_to_frame,
                    args=(building_key(building), building["footprint_wkb_m"]),
                    help="Kaydırılmış çerçeve görünümün dışına çıktıysa haritayı ona göre yeniden ortalar.",
                )
    st.caption(ALIGN_CAPTION)
    st.caption(ALIGN_APPROX_CAPTION)


def roof_view_section(state, building, roof=None, result=None):
    """'Çatı görünümü': the one roof map (isolated from the building selection) and the drawing tools.

    The map is always rendered with the same key.

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
    if tuple(draw["offset"]) != (0.0, 0.0):
        st.caption(drawings.offset_text(draw["offset"]))  # outside the expander: its label never changes
    with st.expander(ALIGN_EXPANDER_LABEL, expanded=state.get(KEY_ALIGN_EXPANDED) == building_key(building)):
        alignment_control(draw, building)
    for note in (roof["notes"] if roof is not None else []):
        if note not in (NOTE_AVERAGE, NOTE_PHOTO):  # RATIO_NO_PANELS_CAPTION below replaces them
            st.info(note)
    if roof is not None and roof["mode"] != "geometry":
        st.caption(RATIO_NO_PANELS_CAPTION)
    aligning = bool(draw.get("align_active"))
    if aligning:  # always reachable, even when the expander is collapsed
        st.caption(ALIGN_CLICK_CAPTION)
        st.button(ALIGN_CANCEL_LABEL, key="align_cancel", on_click=_cancel_align)
    bounds = roof_view_bounds(state, building, footprint)
    enhance = enhancement_control(state)
    data = st_folium(
        build_roof_map(
            list(building["centroid"]), draw_tools=not aligning, aligning=aligning, bounds=bounds, enhance=enhance,
        ),
        key=drawings.map_key(draw),
        returned_objects=["all_drawings", "last_clicked"] if aligning else ["all_drawings"],
        feature_group_to_add=build_roof_overlay(footprint, setback, draw, roof, result),
        height=ROOF_MAP_HEIGHT, use_container_width=True,
    )
    if aligning:
        if drawings.apply_click(draw, (data or {}).get("last_clicked"), footprint):
            st.rerun()  # once: the mode is off and the map is rebuilt at the new offset
        change = {"changed": False}  # a click never creates, changes or deletes drawings
    else:
        mode = drawings.TERRACE if mode_label == DRAW_TERRACE else drawings.OBSTACLE
        change = drawings.sync(draw, (data or {}).get("all_drawings"), mode, footprint)
    if change["changed"]:
        st.rerun()  # once: the cards above and the map key follow the new drawings (a notice waits in the state)
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

    pressed = False
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
        ensure_defaults(state, bill_was_shown=state.pop(KEY_BILL_SHOWN, False))
        flats_step(state)
        if building is None:
            st.caption(NO_BUILDING_DRAW_CAPTION)
        optional_inputs_step(state)
        if building is not None:
            roof_slot = st.container()
        pressed = st.button("Hesapla", type="primary")

    if pressed:
        if not point:
            st.error("Önce haritadan bir konum seçmelisiniz.")
        else:
            api_key, key_error = resolve_api_key()  # main thread: workers never touch st.*
            footprint_inputs = {"footprint_wkb_m": building["footprint_wkb_m"] if building else None}
            try:
                with st.spinner("PVGIS verileri bekleniyor, lütfen bekleyin…"):
                    # flat + every unique face aspect (hip and gable), so later sub-type changes need no call
                    bundle = fetch_roof_yields(point[0], point[1], pvgis_aspects(footprint_inputs))
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
                inputs = collect_inputs(state, analysis)
                first, _, _ = evaluate(analysis, inputs, None, set(), drawing_state(state, building))
                with st.spinner("YZ açıklaması hazırlanıyor…"):
                    state["explanation"] = generate_explanation(first, None, False, api_key)
            except SolarDataError as exc:
                for k in ("analysis", "result", "explanation"):
                    state.pop(k, None)
                st.error(str(exc))
            except ValueError:
                for k in ("analysis", "result", "explanation"):
                    state.pop(k, None)
                st.error("Girdiler geçersiz görünüyor. Lütfen çatı alanını kontrol edin.")

    analysis = state.get("analysis")
    if not analysis and roof_slot is not None:
        with roof_slot:  # before 'Hesapla': imagery, outline, alignment and drawing tools only; no evaluation
            roof_view_section(state, building)
    if analysis:
        st.markdown("---")
        inputs = collect_inputs(state, analysis)
        draw = drawing_state(state, inputs["building"])
        result, _ratio, roof = evaluate(analysis, inputs, None, set(), draw)
        state["result"] = result
        st.subheader(f"Sonuçlar ({result['lat']:.4f}, {result['lon']:.4f})")
        if inputs["building"] is not None:
            # TODO(P2): building height joins the confidence and drives neighbour shading.
            st.caption(height_text(inputs["building"], index.report if index else None))
        show_cards(result, inputs, False, inputs["building"])
        if roof is not None:
            with roof_slot:
                roof_view_section(state, inputs["building"], roof, result)
        else:
            st.caption("Çatı görünümü yalnızca haritadan seçilen binalar için gösterilir (bina izi gerekir).")
        explanation_section(analysis, result, None)
        show_monthly(result)
        offset = draw.get("offset") if roof is not None else None
        breakdown_section(result, offset)
        st.subheader("Varsayımlar ve kaynaklar")
        assumptions = assumption_rows(result, inputs, None, state.get("explanation"), index.report if index else None)
        st.dataframe(assumptions, hide_index=True, width="stretch")
        download_section(result, assumptions.itertuples(index=False, name=None), offset)



if __name__ == "__main__":
    main()
