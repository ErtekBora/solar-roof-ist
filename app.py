"""Streamlit UI (Turkish): Istanbul rooftop solar suitability tool."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core import buildings, explain
from core.calc import compute_capacity
from core.config import (
    BUILDING_ZOOM,
    BUILDINGS_BBOX,
    CO2_FACTOR_T_PER_MWH,
    EDGE_FACTOR,
    EXPLAIN_MODEL,
    LEVEL_HEIGHT_M,
    M2_PER_KWP_FLAT,
    M2_PER_KWP_PITCHED,
    MAP_CENTER,
    MAP_ZOOM,
    MAX_UPLOAD_MB,
    SETBACK_M,
    TARIFF_SOURCE,
    USABLE_RATIO_NO_PHOTO,
    MissingApiKeyError,
    get_gemini_api_key,
)
from core.formatting import percent_from_ratio, round_half_up, tr_num
from core.solar import SolarDataError, get_solar_yield
from core.vision import (
    OBSTACLE_LABELS_TR,
    VisionError,
    compute_obstacle_ratio,
    detect_roof_objects,
    draw_overlay,
)

MONTHS_TR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
ROOF_CHOICES = ["Düz", "Eğimli", "Bilmiyorum"]
ROOF_FROM_CHOICE = {"Düz": "flat", "Eğimli": "pitched", "Bilmiyorum": None}
ROOF_TYPE_TR = {"flat": "Düz", "pitched": "Eğimli"}
PHOTO_NONE, PHOTO_UPLOAD, PHOTO_DEMO = "Fotoğraf yok", "Fotoğraf yükle", "Örnek fotoğraf seç"
DEMO_DIR = Path(__file__).parent / "data" / "demo_images"
DEMO_PHOTO_NOTE = "Bu örnek fotoğraf seçtiğiniz binaya ait değil; yalnızca YZ'nin çatı tespitini göstermek içindir."
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
    for key in ("analysis", "result", "explanation", "roof_choice", "lookup"):
        state.pop(key, None)


# ------------------------------------------------------------- calculation

def build_result(solar, inputs, obstacle_ratio=None, image_view=None):
    """Compute the full result dict from an already fetched PVGIS result (no network)."""
    ideal, shaded = solar["ideal"], solar["shaded"]
    result = compute_capacity(
        inputs["area"], obstacle_ratio, inputs["roof_type"], ideal["e_y"], shaded["e_y"], inputs.get("bill"),
        image_view=image_view, monthly_ideal=ideal["e_m"], monthly_shaded=shaded["e_m"],
        usable_base_area_m2=inputs.get("usable_base_area_m2"), roof_type_known=inputs.get("roof_known", True),
    )
    result.update(
        lat=inputs["lat"],
        lon=inputs["lon"],
        yillik_isinim_kwh_m2=shaded["h_i_y"],
        sicaklik_kaybi_yuzde=abs(shaded["l_tg"]),
        panel_egim_derece=solar["angle"],
        panel_yon_derece=solar["aspect"],
    )
    return result


def calculate(lat, lon, area, roof_type, bill=None):
    """Fetch PVGIS yields and compute the full result dict (no photo, terrain shading only)."""
    inputs = {"lat": lat, "lon": lon, "area": area, "roof_type": roof_type, "bill": bill}
    return build_result(get_solar_yield(lat, lon), inputs)


def resolve_roof_type(choice, preselected=None):
    """Return (roof_type, known, status tag) for the radio choice; 'Bilmiyorum' keeps the flat default."""
    roof_type = ROOF_FROM_CHOICE[choice]
    if roof_type is None:
        return "flat", False, "Varsayım"
    return roof_type, True, "Ölçüm/veri" if choice == preselected else "Kullanıcı girdisi"


def photo_roof_choice(detection):
    """(radio choice, note) for the roof type read from the photo; mixed roofs count as flat."""
    detected = detection["roof_type"]
    if detected == "mixed":
        return "Düz", "Çatı karma görünüyor; kapasite hesabı için düz çatı kabul edildi."
    return ROOF_TYPE_TR[detected], None


def clean_bill(value):
    """Monthly bill in TL, or None when empty, zero, negative or not a finite number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < float("inf") else None


def floors_prefill(building):
    """Integer floor count from the data (pre-fills the input), or None."""
    n = building.get("num_floors") if building else None
    return min(int(round(n)), 100) if n and n >= 1 else None


def effective_floors(value, building):
    """User floors, or None when empty or unchanged from the data (then the data height rule stays)."""
    if value is None or value == floors_prefill(building):
        return None
    return value


def resolve_inputs(analysis, choice, floors_value, bill_value):
    """Stored Hesapla inputs + the three result-panel widgets -> the inputs the calculation uses."""
    roof_type, known, tag = resolve_roof_type(choice, analysis.get("roof_preselect"))
    building = analysis.get("building")
    if building is not None:
        building = buildings.with_user_floors(building, effective_floors(floors_value, building))
    return {
        **analysis["inputs"], "roof_type": roof_type, "roof_known": known, "roof_tag": tag,
        "bill": clean_bill(bill_value), "building": building,
    }


def photo_applies(building, is_demo):
    """A bundled demo photo never describes a real clicked building; an uploaded photo is assumed to be of it."""
    return not (building is not None and is_demo)


def selected_ids(editor_df):
    """Obstacle ids ticked in the confirmation table."""
    return {int(i) for i in editor_df.index[editor_df["Dahil"]]}


def recompute(solar, inputs, detection, included_ids):
    """Recompute ratio and result from cached PVGIS + detection only (no Gemini, no PVGIS call).

    With a photo that applies to the building the ratio is a float (0.0 = clean roof); None means no usable photo.
    A bundled demo photo never applies to a real building (inputs["photo_applies"] is False).
    """
    applies = detection is not None and inputs.get("photo_applies", True)
    ratio = None
    if applies:
        boxes = [o["box_px"] for o in detection["obstacles"] if o["id"] in included_ids]
        ratio = compute_obstacle_ratio(detection["roof_polygon_px"], boxes, detection["width"], detection["height"])
    result = build_result(solar, inputs, obstacle_ratio=ratio, image_view=detection["image_view"] if applies else None)
    return result, ratio


def format_savings_short(lo, hi):
    """Card-sized savings range: thousands with one decimal ("27,5 – 55,0 bin TL")."""
    if hi < 1000:
        return f"{tr_num(lo)} – {tr_num(hi)} TL"
    return f"{tr_num(lo / 1000, 1)} – {tr_num(hi / 1000, 1)} bin TL"


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


def input_classes(inputs, has_photo_applied, building):
    """Which inputs were measured, estimated and assumed (Turkish bullet texts)."""
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
            measured.append("Bina yüksekliği: veri ya da girdiğiniz kat sayısı")
    else:
        assumed.append("Çatı alanı: elle girilen değer")
    if has_photo_applied:
        measured.append("Engel oranı: fotoğraf analizi")
    else:
        assumed.append("Engeller: fotoğraf yok, ortalama kullanılabilir alan oranı")
    if not inputs["roof_known"]:
        assumed.append("Çatı tipi: bilinmiyor, düz çatı varsayıldı")
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
        if b["match_distance_m"] > 0:
            st.caption(f"Tıkladığınız noktaya en yakın bina seçildi (~{tr_num(b['match_distance_m'], 1)} m).")
    else:
        st.info(lookup["message"])


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
    rows.append(("Çatı tipi", ROOF_TYPE_TR[inputs["roof_type"]], inputs["roof_tag"], "Fotoğraf analizi, kullanıcı seçimi veya varsayılan düz çatı"))
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
        rows.append(("Kullanılabilir alan oranı", f"{tr_num(USABLE_RATIO_NO_PHOTO / EDGE_FACTOR, 3)}", "Varsayım", "Ember ortalaması (0,276) ÷ 0,85; fotoğraf yok"))
    per_kwp = M2_PER_KWP_FLAT if inputs["roof_type"] == "flat" else M2_PER_KWP_PITCHED
    rows += [
        ("Çatı alanı başına güç", f"{tr_num(per_kwp, 1)} m²/kWp", "Varsayım", "Düz çatıda 6,5; eğimli çatıda 5,5"),
        ("Kullanılabilir alan", f"{tr_num(r['kullanilabilir_alan_m2'], 1)} m²", "Varsayım", "Alan × oran"),
        ("Kurulabilir güç", f"{tr_num(r['kurulabilir_guc_kwp'], 1)} kWp", "Varsayım", "Kullanılabilir alan ÷ m²/kWp"),
        ("Panel eğimi / yönü", f"{r['panel_egim_derece']}° / {r['panel_yon_derece']}°", "Varsayım", "Paneller PVGIS'in en uygun eğim ve yönünde varsayıldı"),
        ("Yıllık ışınım", f"{tr_num(r['yillik_isinim_kwh_m2'])} kWh/m²", "Ölçüm/veri", "PVGIS (JRC)"),
        ("Özgül verim", f"{tr_num(r['golgeli_uretim_kwh_kwp'])} kWh/kWp", "Ölçüm/veri", "PVGIS (JRC), arazi gölgesi dahil"),
        ("Sıcaklık kaybı", f"%{tr_num(r['sicaklik_kaybi_yuzde'], 1)}", "Ölçüm/veri", "PVGIS (JRC)"),
        ("Uygunluk (arazi gölgesi)", f"%{tr_num(r['uygunluk_puani_yuzde'], 1)}", "Ölçüm/veri", "PVGIS gölgeli ÷ gölgesiz verim"),
        ("Yıllık üretim", f"{tr_num(r['yillik_uretim_kwh'])} kWh", "Ölçüm/veri", "Güç × özgül verim"),
        ("Elektrik tarifesi", f"{tr_num(r['tarife_dusuk_tl_kwh'], 2)} / {tr_num(r['tarife_yuksek_tl_kwh'], 2)} TL/kWh", "Varsayım", f"{TARIFF_SOURCE}; vergiler dahil yaklaşık (×1,26)"),
    ]
    if r["tasarruf_yontemi"] == "fatura":
        rows += [
            ("Aylık fatura", f"{tr_num(r['fatura_tl_ay'])} TL", "Kullanıcı girdisi", f"Tüketim yaklaşık {tr_num(r['tuketim_kwh_ay'])} kWh/ay (iki kademeli tarifeden)"),
            ("Tasarruf aralığı", "fatura temelli", "Varsayım", "Alt sınır: üretimin yarısı kendi tüketiminizde; üst sınır: tamamı. Fazla üretim 0 TL"),
        ]
    else:
        rows.append(("Tasarruf aralığı", "geniş tahmin", "Varsayım", "Tüketim bilinmiyor: üretim × [0,5 × düşük kademe, 1,0 × yüksek kademe]"))
    rows += [
        ("Tasarruf kapsamı", "tek hane", "Varsayım", "Tek hanenin kendi sayacı için; çatıyı dairelerle paylaştırmak kapsam dışı"),
        ("CO₂ faktörü", f"{CO2_FACTOR_T_PER_MWH} tCO₂/MWh", "Varsayım", "ETKB TUESEF 2021"),
        ("Güven seviyesi", r["guven_seviyesi"], "Varsayım", "Alan kaynağı, fotoğraf ve çatı tipine bağlı kural"),
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
    rows = []
    for i, month in enumerate(MONTHS_TR):
        rows.append({"Ay": month, "Sıra": i, "Senaryo": "İdeal (gölgesiz)", "kWh": result["aylik_ideal_kwh"][i]})
        rows.append({"Ay": month, "Sıra": i, "Senaryo": "Arazi gölgeli", "kWh": result["aylik_uretim_kwh"][i]})
    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar()
        .encode(
            x=alt.X("Ay:N", sort=MONTHS_TR, title=None),
            xOffset="Senaryo:N",
            y=alt.Y("kWh:Q", title="Aylık üretim (kWh)"),
            color=alt.Color("Senaryo:N", title=None),
            tooltip=["Ay", "Senaryo", alt.Tooltip("kWh:Q", format=",.1f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def coverage_text(r):
    """'Faturanızın yaklaşık %X–%Y kadarını karşılar' (grammatical for any number), or None without a bill."""
    if r["tasarruf_yontemi"] != "fatura":
        return None
    lo, hi = (int(round_half_up(v, 0)) for v in r["fatura_karsilama_yuzde_aralik"])
    span = f"%{lo}" if lo == hi else f"%{lo}–%{hi}"
    return f"Faturanızın yaklaşık {span} kadarını karşılar."


def shared_roof_text(r):
    """Shared-roof warning (Turkish) when one flat cannot use the whole production, else None."""
    if not r["tek_daire_uyarisi"]:
        return None
    z = int(round_half_up(r["kendi_kullanim_orani_yuzde"], 0))
    tail = "artan kısım apartman ortak alanı veya diğer daireler için değerlendirilebilir."
    if r["fatura_karsilama_yuzde_aralik"][1] >= 99.99:
        return f"Tek daire faturanızın tamamını karşılar; üretimin yalnızca %{z} kadarı dairenizde kullanılır, {tail}"
    return f"Üretimin yalnızca %{z} kadarı dairenizde kullanılır; {tail}"


def show_cards(r, inputs=None, applied=False, building=None):
    lo, hi = r["yillik_tasarruf_tl_aralik"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Uygunluk (arazi gölgesi)", f"%{tr_num(r['uygunluk_puani_yuzde'], 1)}")
    c1.caption("Şu an yalnızca arazi (tepe) gölgesi hesaba katılıyor. Komşu bina gölgesi bir sonraki aşamada eklenecek.")
    c2.metric("Kurulabilir güç", f"{tr_num(r['kurulabilir_guc_kwp'], 1)} kWp")
    c3.metric("Yıllık üretim", f"{tr_num(r['yillik_uretim_kwh'])} kWh")

    c4, c5, c6 = st.columns(3)
    c4.metric(
        "Yıllık tasarruf (aralık)", format_savings_short(lo, hi), help=f"Kesin aralık: {tr_num(lo)} – {tr_num(hi)} TL"
    )
    if r["tasarruf_yontemi"] == "fatura":
        c4.caption("Faturanıza göre hesaplandı; vergiler dahil yaklaşık.")
    else:
        c4.caption("Geniş tahmin (tüketim bilinmiyor); vergiler dahil yaklaşık. Faturanızı girin.")
    c5.metric("CO₂ azaltımı", f"{tr_num(r['co2_ton_yil'], 1)} ton/yıl")
    c6.metric("Güven seviyesi", r["guven_seviyesi"])
    if r.get("guven_iyilestirme"):
        c6.caption("Sonucu iyileştirmek için: " + " ".join(r["guven_iyilestirme"]))

    c7, c8, c9 = st.columns(3)
    c7.metric("Yıllık ışınım", f"{tr_num(r['yillik_isinim_kwh_m2'])} kWh/m²")
    c8.metric("Özgül verim", f"{tr_num(r['golgeli_uretim_kwh_kwp'])} kWh/kWp")
    c9.metric("Sıcaklık kaybı", f"%{tr_num(r['sicaklik_kaybi_yuzde'], 1)}")

    if r["engel_orani"] is not None:
        c10, _, _ = st.columns(3)
        c10.metric("Engel oranı", f"%{tr_num(percent_from_ratio(r['engel_orani']), 1)}")

    text = coverage_text(r)
    if text:
        st.info(text)
    shared = shared_roof_text(r)
    if shared:
        st.warning(shared)
    st.caption(
        f"Tarife: {TARIFF_SOURCE}; vergiler dahil yaklaşık, kesin tutar değildir. Tek hanenin kendi sayacı için "
        "hesaplanır; fazla üretim 0 TL sayılır."
    )

    with st.expander("Güven seviyesinin nedenleri", expanded=True):
        for reason in r["guven_nedenleri"]:
            st.write(f"- {reason}")
        if inputs is not None:
            measured, estimated, assumed = input_classes(inputs, applied, building)
            for title, items in (("Ölçülen", measured), ("Tahmin edilen", estimated), ("Varsayılan", assumed)):
                if items:
                    st.write(f"**{title}:** " + "; ".join(items))


def show_monthly(r):
    st.subheader("Aylık üretim")
    monthly_chart(r)
    st.caption(
        f"Paneller en uygun eğim ({r['panel_egim_derece']}°) ve yönde varsayıldı. "
        f"Kullanılabilir alan: {tr_num(r['kullanilabilir_alan_m2'], 1)} m²."
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
    bullets = [ln for ln in lines[1:] if ln.startswith("-")]
    if bullets:
        st.markdown("\n".join(bullets))
    for ln in lines[1:]:
        if not ln.startswith("-"):
            st.warning(ln.replace("⚠️", "").strip())


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


def photo_sidebar():
    """Sidebar photo choice. Returns (image_bytes or None, filename or None, is_demo)."""
    st.header("Çatı fotoğrafı")
    mode = st.radio("Fotoğraf", [PHOTO_NONE, PHOTO_UPLOAD, PHOTO_DEMO], label_visibility="collapsed")
    if mode == PHOTO_UPLOAD:
        uploaded = st.file_uploader("JPG veya PNG yükleyin", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"Dosya çok büyük; en fazla {MAX_UPLOAD_MB} MB yükleyebilirsiniz.")
                return None, None, False
            return uploaded.getvalue(), uploaded.name, False
    elif mode == PHOTO_DEMO:
        files = sorted(p.name for p in DEMO_DIR.glob("roof_*.jpg"))
        if files:
            name = st.selectbox("Örnek fotoğraf", files)
            st.caption(source_caption(name))
            return (DEMO_DIR / name).read_bytes(), name, True
        st.info("Örnek fotoğraf bulunamadı.")
    return None, None, False


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


def input_widgets(analysis):
    """The three result-panel inputs (stable keys); returns (roof choice, floors value, bill value)."""
    building = analysis.get("building")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.setdefault("roof_choice", "Bilmiyorum")
        choice = st.radio("Çatı tipi", ROOF_CHOICES, key="roof_choice", horizontal=True)
        if analysis.get("roof_preselect"):
            st.caption("Fotoğraf analizine göre seçildi; değiştirebilirsiniz.")
    with col2:
        floors = None
        if building is not None:
            floors = st.number_input(
                "Bina kat sayısı (isteğe bağlı)", min_value=1, max_value=100, value=floors_prefill(building), step=1,
                key=f"floors_{building['id']}", help="Boş bırakırsanız veriden ya da YZ tahmininden gelen yükseklik kullanılır.",
            )
    with col3:
        bill = st.number_input(
            "Aylık elektrik faturanız yaklaşık kaç TL?", value=None, step=50.0, format="%.0f", key="bill_input",
            help="Hatırlamıyorsanız boş bırakın",
        )
    return choice, floors, bill


# ---------------------------------------------------------------------- main

def main():
    st.set_page_config(page_title="İstanbul Çatı GES Uygunluk Aracı", layout="wide")

    st.title("İstanbul Çatı GES Uygunluk Aracı")
    st.caption(
        "Haritadaki mavi çerçeve içinde (Erenköy / Suadiye / Göztepe) bir binaya tıklayın; "
        "çatınızın güneş enerjisi için uygun olup olmadığını öğrenin."
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

    with st.sidebar:
        st.header("Seçilen konum")
        if point:
            st.write(f"Enlem: {point[0]:.5f}")
            st.write(f"Boylam: {point[1]:.5f}")
        else:
            st.info("Haritaya tıklayarak bir konum seçin.")

        st.header("Çatı bilgileri")
        area = st.number_input("Bina bulunamazsa kullanılacak çatı alanı (m²)", min_value=1.0, value=150.0, step=10.0)
        image_bytes, photo_name, is_demo = photo_sidebar()
        pressed = st.button("Hesapla", type="primary")

    if pressed:
        if not point:
            st.error("Önce haritadan bir konum seçmelisiniz.")
        else:
            api_key, key_error = resolve_api_key()  # main thread: workers never touch st.*
            building = lookup["building"] if lookup and lookup["status"] == "ok" else None
            try:
                detection, vision_error = None, None
                with ThreadPoolExecutor(max_workers=2) as pool:
                    solar_future = pool.submit(get_solar_yield, point[0], point[1])
                    vision_future = None
                    if image_bytes is not None:
                        if api_key:
                            vision_future = pool.submit(detect_roof_objects, image_bytes, api_key)
                        else:
                            vision_error = key_error
                    with st.spinner("PVGIS verileri ve fotoğraf analizi bekleniyor, lütfen bekleyin…"):
                        solar = solar_future.result()
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
                state["roof_choice"] = roof_preselect or "Bilmiyorum"
                analysis = {
                    "solar": solar,
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
                    },
                }
                state["analysis"] = analysis
                inputs = resolve_inputs(
                    analysis, state["roof_choice"],
                    state.get(f"floors_{building['id']}") if building else None, state.get("bill_input"),
                )
                all_ids = {o["id"] for o in detection["obstacles"]} if detection else set()
                first, _ = recompute(solar, inputs, detection, all_ids)
                eff = explained_detection(detection, all_ids, inputs)
                with st.spinner("YZ açıklaması hazırlanıyor…"):
                    state["explanation"] = generate_explanation(first, eff, bool(inputs["demo_photo"] and eff), api_key)
            except SolarDataError as exc:
                for k in ("analysis", "result", "explanation"):
                    state.pop(k, None)
                st.error(str(exc))
            except ValueError:
                for k in ("analysis", "result", "explanation"):
                    state.pop(k, None)
                st.error("Girdiler geçersiz görünüyor. Lütfen çatı alanını kontrol edin.")

    analysis = state.get("analysis")
    if analysis:
        detection = analysis["detection"]
        applies = analysis["inputs"]["photo_applies"]
        st.markdown("---")
        choice, floors, bill_value = input_widgets(analysis)
        inputs = resolve_inputs(analysis, choice, floors, bill_value)
        ids = photo_section(analysis, applies)
        result, _ratio = recompute(analysis["solar"], inputs, detection, ids or set())
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
        explanation_section(analysis, result, eff_detection)
        show_monthly(result)
        st.subheader("Varsayımlar ve kaynaklar")
        st.dataframe(
            assumption_rows(result, inputs, detection, state.get("explanation"), index.report if index else None),
            hide_index=True,
            width="stretch",
        )

    st.markdown("---")
    st.caption(ATTRIBUTION)


if __name__ == "__main__":
    main()
