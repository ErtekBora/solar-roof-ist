"""Streamlit UI (Turkish): Istanbul rooftop solar suitability tool."""
import folium
import streamlit as st
from streamlit_folium import st_folium

from core.config import MAP_CENTER, MAP_ZOOM


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


def main():
    st.set_page_config(page_title="İstanbul Çatı GES Uygunluk Aracı", layout="wide")

    st.title("İstanbul Çatı GES Uygunluk Aracı")
    st.caption("Haritadan bir konum seçin; çatınızın güneş enerjisi için uygun olup olmadığını öğrenin.")

    map_data = st_folium(build_map(), height=600, use_container_width=True, returned_objects=["last_clicked"], key="map")

    with st.sidebar:
        st.header("Seçilen konum")
        clicked = (map_data or {}).get("last_clicked")
        if clicked:
            st.write(f"Enlem: {clicked['lat']:.5f}")
            st.write(f"Boylam: {clicked['lng']:.5f}")
        else:
            st.info("Haritaya tıklayarak bir konum seçin.")

    st.markdown("---")
    st.caption("© OpenStreetMap katkıcıları · Güneş verisi: PVGIS (JRC)")


if __name__ == "__main__":
    main()
