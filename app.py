"""Streamlit UI (Turkish): Istanbul rooftop solar suitability tool."""
import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.calc import compute_capacity
from core.config import MAP_CENTER, MAP_ZOOM, default_tariff
from core.solar import SolarDataError, get_solar_yield

MONTHS_TR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
ROOF_TYPES = {"Düz": "flat", "Eğimli": "pitched"}


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


def calculate(lat, lon, area, roof_type, tariff):
    """Fetch PVGIS yields and compute the full result dict (Demo: no photo, terrain shading only)."""
    solar = get_solar_yield(lat, lon)
    ideal, shaded = solar["ideal"], solar["shaded"]
    result = compute_capacity(
        area, None, roof_type, ideal["e_y"], shaded["e_y"], tariff,
        monthly_ideal=ideal["e_m"], monthly_shaded=shaded["e_m"],
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


def tr_num(value, decimals=0):
    """Format with Turkish separators (1.234,5)."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


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


def show_result(r):
    lo, hi = r["yillik_tasarruf_tl_aralik"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Uygunluk (arazi gölgesi)", f"%{tr_num(r['uygunluk_puani_yuzde'], 1)}")
    c1.caption("Şu an yalnızca arazi (tepe) gölgesi hesaba katılıyor. Komşu bina gölgesi bir sonraki aşamada eklenecek.")
    c2.metric("Kurulabilir güç", f"{tr_num(r['kurulabilir_guc_kwp'], 1)} kWp")
    c3.metric("Yıllık üretim", f"{tr_num(r['yillik_uretim_kwh'])} kWh")

    c4, c5, c6 = st.columns(3)
    c4.metric("Yıllık tasarruf (aralık)", f"{tr_num(lo)} – {tr_num(hi)} TL")
    c4.caption("Alt sınır: üretimin yarısı kendi tüketiminizde; üst sınır: tamamı.")
    c5.metric("CO₂ azaltımı", f"{tr_num(r['co2_ton_yil'], 1)} ton/yıl")
    c6.metric("Güven seviyesi", r["guven_seviyesi"])

    c7, c8, c9 = st.columns(3)
    c7.metric("Yıllık ışınım", f"{tr_num(r['yillik_isinim_kwh_m2'])} kWh/m²")
    c8.metric("Özgül verim", f"{tr_num(r['golgeli_uretim_kwh_kwp'])} kWh/kWp")
    c9.metric("Sıcaklık kaybı", f"%{tr_num(r['sicaklik_kaybi_yuzde'], 1)}")

    with st.expander("Güven seviyesinin nedenleri", expanded=True):
        for reason in r["guven_nedenleri"]:
            st.write(f"- {reason}")

    st.subheader("Aylık üretim")
    monthly_chart(r)
    st.caption(
        f"Paneller en uygun eğim ({r['panel_egim_derece']}°) ve yönde varsayıldı. "
        f"Kullanılabilir alan: {tr_num(r['kullanilabilir_alan_m2'], 1)} m²."
    )


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
        pressed = st.button("Hesapla", type="primary")

    if pressed:
        if not point:
            st.error("Önce haritadan bir konum seçmelisiniz.")
        elif not tariff or tariff <= 0:
            st.error("Lütfen elektrik tarifesini (TL/kWh) girin.")
        else:
            try:
                with st.spinner("PVGIS verileri alınıyor, lütfen bekleyin…"):
                    st.session_state["result"] = calculate(point[0], point[1], area, ROOF_TYPES[roof_label], tariff)
            except SolarDataError as exc:
                st.session_state.pop("result", None)
                st.error(str(exc))
            except ValueError:
                st.session_state.pop("result", None)
                st.error("Girdiler geçersiz görünüyor. Lütfen çatı alanını ve tarifeyi kontrol edin.")

    result = st.session_state.get("result")
    if result:
        st.markdown("---")
        st.subheader(f"Sonuçlar ({result['lat']:.4f}, {result['lon']:.4f})")
        show_result(result)
    st.info("Fotoğraf analizi bir sonraki adımda eklenecek.")

    st.markdown("---")
    st.caption("© OpenStreetMap katkıcıları · Güneş verisi: PVGIS (JRC)")


if __name__ == "__main__":
    main()
