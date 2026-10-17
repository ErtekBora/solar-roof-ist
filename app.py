"""Streamlit UI (Turkish): Istanbul rooftop solar suitability tool."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core import explain
from core.calc import compute_capacity
from core.config import (
    CO2_FACTOR_T_PER_MWH,
    EDGE_FACTOR,
    M2_PER_KWP_FLAT,
    M2_PER_KWP_PITCHED,
    MAP_CENTER,
    MAP_ZOOM,
    EXPLAIN_MODEL,
    MAX_UPLOAD_MB,
    USABLE_RATIO_NO_PHOTO,
    MissingApiKeyError,
    default_tariff,
    get_gemini_api_key,
)
from core.formatting import percent_from_ratio, tr_num
from core.solar import SolarDataError, get_solar_yield
from core.vision import (
    OBSTACLE_LABELS_TR,
    VisionError,
    compute_obstacle_ratio,
    detect_roof_objects,
    draw_overlay,
)

MONTHS_TR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
AUTO_LABEL = "Otomatik (fotoğraftan)"
ROOF_TYPES = {AUTO_LABEL: None, "Düz": "flat", "Eğimli": "pitched"}
ROOF_TYPE_TR = {"flat": "Düz", "pitched": "Eğimli"}
PHOTO_NONE, PHOTO_UPLOAD, PHOTO_DEMO = "Fotoğraf yok", "Fotoğraf yükle", "Örnek fotoğraf seç"
DEMO_DIR = Path(__file__).parent / "data" / "demo_images"


def build_map():
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


def build_result(solar, lat, lon, area, roof_type, tariff, obstacle_ratio=None, image_view=None):
    """Compute the full result dict from an already fetched PVGIS result (no network)."""
    ideal, shaded = solar["ideal"], solar["shaded"]
    result = compute_capacity(
        area, obstacle_ratio, roof_type, ideal["e_y"], shaded["e_y"], tariff,
        image_view=image_view, monthly_ideal=ideal["e_m"], monthly_shaded=shaded["e_m"],
    )
    result.update(
        lat=lat,
        lon=lon,
        tarife_tl_kwh=tariff,
        yillik_isinim_kwh_m2=shaded["h_i_y"],
        sicaklik_kaybi_yuzde=abs(shaded["l_tg"]),
        panel_egim_derece=solar["angle"],
        panel_yon_derece=solar["aspect"],
    )
    return result


def calculate(lat, lon, area, roof_type, tariff):
    """Fetch PVGIS yields and compute the full result dict (no photo, terrain shading only)."""
    return build_result(get_solar_yield(lat, lon), lat, lon, area, roof_type, tariff)


def resolve_roof_type(choice, detection):
    """Return (roof_type, status tag, note) for the sidebar choice and the optional detection."""
    fixed = ROOF_TYPES[choice]
    if fixed:
        return fixed, "Kullanıcı girdisi", None
    if detection:
        detected = detection["roof_type"]
        if detected == "mixed":
            return "flat", "Ölçüm/veri", "Çatı karma görünüyor; kapasite hesabı için düz çatı kabul edildi."
        return detected, "Ölçüm/veri", None
    return "flat", "Varsayım", None


def selected_ids(editor_df):
    """Obstacle ids ticked in the confirmation table."""
    return {int(i) for i in editor_df.index[editor_df["Dahil"]]}


def recompute(solar, inputs, detection, included_ids):
    """Recompute ratio and result from cached PVGIS + detection only (no Gemini, no PVGIS call).

    With a photo the ratio is a float (0.0 = clean roof, EDGE_FACTOR applies); None means no photo.
    """
    ratio = None
    if detection is not None:
        boxes = [o["box_px"] for o in detection["obstacles"] if o["id"] in included_ids]
        ratio = compute_obstacle_ratio(detection["roof_polygon_px"], boxes, detection["width"], detection["height"])
    result = build_result(
        solar, inputs["lat"], inputs["lon"], inputs["area"], inputs["roof_type"], inputs["tariff"],
        obstacle_ratio=ratio, image_view=None if detection is None else detection["image_view"],
    )
    return result, ratio


def format_savings_short(lo, hi):
    """Card-sized savings range: thousands with one decimal ("27,5 – 55,0 bin TL")."""
    if hi < 1000:
        return f"{tr_num(lo)} – {tr_num(hi)} TL"
    return f"{tr_num(lo / 1000, 1)} – {tr_num(hi / 1000, 1)} bin TL"


def resolve_api_key():
    """Return (key, None) or (None, Turkish message). Main thread only: it may read st.secrets."""
    try:
        return get_gemini_api_key(), None
    except MissingApiKeyError as exc:
        return None, str(exc)


def explained_detection(detection, included_ids):
    """Detection restricted to the obstacles still ticked in the table (what the explanation may mention)."""
    if detection is None:
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


def assumption_rows(r, inputs, detection, explanation=None):
    """'Varsayımlar ve kaynaklar' table: one row per shown number (SPEC section 8)."""
    rows = [
        ("Çatı alanı", f"{tr_num(inputs['area'], 1)} m²", "Kullanıcı girdisi", "Kenar çubuğunda elle girilen değer"),
        ("Çatı tipi", ROOF_TYPE_TR[inputs["roof_type"]], inputs["roof_tag"], "Fotoğraf analizi, kullanıcı seçimi veya varsayılan düz çatı"),
        ("Elektrik tarifesi", f"{tr_num(inputs['tariff'], 2)} TL/kWh", "Kullanıcı girdisi", "Kenar çubuğundaki değer"),
    ]
    if detection is not None:
        view = "Kuşbakışı" if detection["image_view"] == "top-down" else "Açılı"
        rows += [
            ("Engel oranı", f"%{tr_num(percent_from_ratio(r['engel_orani']), 1)}", "Ölçüm/veri", "Gemini fotoğraf analizi; tabloda işaretlediğiniz engeller"),
            ("Görüntü türü", view, "Ölçüm/veri", "Gemini fotoğraf analizi"),
            ("Kenar payı çarpanı", f"{EDGE_FACTOR}", "Varsayım", "Çatı kenarlarında panel konmayan şerit"),
        ]
    else:
        rows.append(("Kullanılabilir alan oranı", f"{USABLE_RATIO_NO_PHOTO}", "Varsayım", "Ember ortalaması (fotoğraf yok)"))
    per_kwp = M2_PER_KWP_FLAT if inputs["roof_type"] == "flat" else M2_PER_KWP_PITCHED
    rows += [
        ("Çatı alanı başına güç", f"{tr_num(per_kwp, 1)} m²/kWp", "Varsayım", "Düz çatıda 6,5; eğimli çatıda 5,5"),
        ("Kullanılabilir alan", f"{tr_num(r['kullanilabilir_alan_m2'], 1)} m²", "Varsayım", "Çatı alanı × oran"),
        ("Kurulabilir güç", f"{tr_num(r['kurulabilir_guc_kwp'], 1)} kWp", "Varsayım", "Kullanılabilir alan ÷ m²/kWp"),
        ("Panel eğimi / yönü", f"{r['panel_egim_derece']}° / {r['panel_yon_derece']}°", "Varsayım", "Paneller PVGIS'in en uygun eğim ve yönünde varsayıldı"),
        ("Yıllık ışınım", f"{tr_num(r['yillik_isinim_kwh_m2'])} kWh/m²", "Ölçüm/veri", "PVGIS (JRC)"),
        ("Özgül verim", f"{tr_num(r['golgeli_uretim_kwh_kwp'])} kWh/kWp", "Ölçüm/veri", "PVGIS (JRC), arazi gölgesi dahil"),
        ("Sıcaklık kaybı", f"%{tr_num(r['sicaklik_kaybi_yuzde'], 1)}", "Ölçüm/veri", "PVGIS (JRC)"),
        ("Uygunluk (arazi gölgesi)", f"%{tr_num(r['uygunluk_puani_yuzde'], 1)}", "Ölçüm/veri", "PVGIS gölgeli ÷ gölgesiz verim"),
        ("Yıllık üretim", f"{tr_num(r['yillik_uretim_kwh'])} kWh", "Ölçüm/veri", "Güç × özgül verim"),
        ("Tasarruf aralığı", "üretimin %50–100'ü", "Varsayım", "Alt sınır: yarısı kendi tüketiminizde; üst sınır: tamamı"),
        ("CO₂ faktörü", f"{CO2_FACTOR_T_PER_MWH} tCO₂/MWh", "Varsayım", "ETKB TUESEF 2021"),
        ("Güven seviyesi", r["guven_seviyesi"], "Varsayım", "Demo aşamasında yalnızca fotoğrafa bağlı kural"),
        ("Komşu bina gölgesi", "—", "Sonraki aşamada", "OSM bina verisi ile"),
        ("Çatı alanı (OSM'den)", "—", "Sonraki aşamada", "OSM bina izi ve geometrik kenar boşluğu"),
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


def show_cards(r):
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
    c4.caption("Alt sınır: üretimin yarısı kendi tüketiminizde; üst sınır: tamamı.")
    c5.metric("CO₂ azaltımı", f"{tr_num(r['co2_ton_yil'], 1)} ton/yıl")
    c6.metric("Güven seviyesi", r["guven_seviyesi"])

    c7, c8, c9 = st.columns(3)
    c7.metric("Yıllık ışınım", f"{tr_num(r['yillik_isinim_kwh_m2'])} kWh/m²")
    c8.metric("Özgül verim", f"{tr_num(r['golgeli_uretim_kwh_kwp'])} kWh/kWp")
    c9.metric("Sıcaklık kaybı", f"%{tr_num(r['sicaklik_kaybi_yuzde'], 1)}")

    if r["engel_orani"] is not None:
        c10, _, _ = st.columns(3)
        c10.metric("Engel oranı", f"%{tr_num(percent_from_ratio(r['engel_orani']), 1)}")

    with st.expander("Güven seviyesinin nedenleri", expanded=True):
        for reason in r["guven_nedenleri"]:
            st.write(f"- {reason}")


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


def explanation_section(analysis, result, ids):
    """'Yapay Zekâ Değerlendirmesi': last explanation, a stale notice and a manual refresh (no auto Gemini call)."""
    explanation = st.session_state.get("explanation")
    if not explanation:
        return
    detection = explained_detection(analysis["detection"], ids or set())
    demo = analysis["inputs"].get("demo_photo", False)
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


def photo_section(state):
    """Show detection messages, the confirmation table and the overlay; return (ids, ratio) or None."""
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
    if obstacles:
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
        ids = selected_ids(edited)
    else:
        st.write("Çatıda engel tespit edilmedi.")
        ids = set()
    return ids


def main():
    st.set_page_config(page_title="İstanbul Çatı GES Uygunluk Aracı", layout="wide")

    st.title("İstanbul Çatı GES Uygunluk Aracı")
    st.caption("Haritadan bir konum seçin; çatınızın güneş enerjisi için uygun olup olmadığını öğrenin.")

    map_data = st_folium(build_map(), height=500, use_container_width=True, returned_objects=["last_clicked"], key="map")
    clicked = (map_data or {}).get("last_clicked")
    if clicked:
        st.session_state["point"] = (clicked["lat"], clicked["lng"])
    point = st.session_state.get("point")

    with st.sidebar:
        st.header("Seçilen konum")
        if point:
            st.write(f"Enlem: {point[0]:.5f}")
            st.write(f"Boylam: {point[1]:.5f}")
        else:
            st.info("Haritaya tıklayarak bir konum seçin.")

        st.header("Çatı bilgileri")
        area = st.number_input("Çatı alanı (m²)", min_value=1.0, value=150.0, step=10.0)
        roof_label = st.selectbox("Çatı tipi", list(ROOF_TYPES))
        tariff = st.number_input(
            "Elektrik tarifesi (TL/kWh)", min_value=0.0, value=default_tariff(), step=0.1, format="%.2f"
        )
        if default_tariff() is None and not tariff:
            st.warning("Lütfen güncel elektrik tarifenizi (TL/kWh) girin.")
        image_bytes, photo_name, is_demo = photo_sidebar()
        pressed = st.button("Hesapla", type="primary")

    if pressed:
        if not point:
            st.error("Önce haritadan bir konum seçmelisiniz.")
        elif not tariff or tariff <= 0:
            st.error("Lütfen elektrik tarifesini (TL/kWh) girin.")
        else:
            api_key, key_error = resolve_api_key()  # main thread: workers never touch st.*
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
                roof_type, roof_tag, roof_note = resolve_roof_type(roof_label, detection)
                inputs = {
                    "lat": point[0], "lon": point[1], "area": area, "tariff": tariff,
                    "roof_type": roof_type, "roof_tag": roof_tag,
                    "demo_photo": bool(detection) and is_demo,
                }
                st.session_state["analysis"] = {
                    "solar": solar,
                    "detection": detection,
                    "vision_error": vision_error,
                    "key_error": key_error,
                    "roof_note": roof_note,
                    "photo_bytes": image_bytes if detection else None,
                    "inputs": inputs,
                }
                all_ids = {o["id"] for o in detection["obstacles"]} if detection else set()
                first, _ = recompute(solar, inputs, detection, all_ids)
                with st.spinner("YZ açıklaması hazırlanıyor…"):
                    st.session_state["explanation"] = generate_explanation(
                        first, explained_detection(detection, all_ids), inputs["demo_photo"], api_key
                    )
            except SolarDataError as exc:
                for k in ("analysis", "result", "explanation"):
                    st.session_state.pop(k, None)
                st.error(str(exc))
            except ValueError:
                for k in ("analysis", "result", "explanation"):
                    st.session_state.pop(k, None)
                st.error("Girdiler geçersiz görünüyor. Lütfen çatı alanını ve tarifeyi kontrol edin.")

    analysis = st.session_state.get("analysis")
    if analysis:
        detection = analysis["detection"]
        ids = photo_section(analysis)
        result, _ratio = recompute(analysis["solar"], analysis["inputs"], detection, ids or set())
        st.session_state["result"] = result
        if detection is not None:
            st.image(
                draw_overlay(analysis["photo_bytes"], detection, ids),
                caption="Yeşil: çatı sınırı · Kırmızı: dahil edilen engeller · Gri: hariç tutulan engeller",
            )
        st.markdown("---")
        st.subheader(f"Sonuçlar ({result['lat']:.4f}, {result['lon']:.4f})")
        show_cards(result)
        explanation_section(analysis, result, ids)
        show_monthly(result)
        st.subheader("Varsayımlar ve kaynaklar")
        st.dataframe(
            assumption_rows(result, analysis["inputs"], detection, st.session_state.get("explanation")),
            hide_index=True,
            width="stretch",
        )

    st.markdown("---")
    st.caption("© OpenStreetMap katkıcıları · Güneş verisi: PVGIS (JRC)")


if __name__ == "__main__":
    main()
