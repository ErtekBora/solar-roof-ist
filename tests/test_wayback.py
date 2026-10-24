"""Roof map imagery date selector (core/wayback): two fixed options, in-memory choice, never a network scan."""
import json
import re

import folium
import pytest
from streamlit_folium import _get_map_string

from core import wayback
from core.config import ESRI_TILES_URL

WB_56102 = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/"
    "56102/{z}/{y}/{x}"
)


def _js(key="b1"):
    m = folium.Map(location=[40.97, 29.07], tiles=None)
    layer = folium.TileLayer(tiles=ESRI_TILES_URL, attr="Esri", cross_origin="anonymous").add_to(m)
    wayback.add_to_map(m, layer, key, ESRI_TILES_URL)
    m.render()
    return re.sub(r"\s+", " ", _get_map_string(m)), layer


def test_two_fixed_options_live_2025_11_20_and_release_56102():
    assert wayback.WAYBACK_OPTIONS == (
        ("Güncel (2025-11-20)", "live", None),
        ("2023-12-07", "56102", WB_56102),
    )
    js, _ = _js()
    options = json.loads(re.search(r"var OPTIONS = (\[.*?\]);", js).group(1))
    assert options == [
        {"label": "Güncel (2025-11-20)", "release": "live", "url": None},
        {"label": "2023-12-07", "release": "56102", "url": WB_56102},
    ]
    assert f"LIVE_URL = {json.dumps(ESRI_TILES_URL)}" in js  # the live option uses the tile layer's own URL


def test_inputs_come_from_python_building_key_live_url_and_layer_name():
    js, layer = _js("b 42")
    assert 'var KEY = "b 42"' in js and f"tiles = {layer.get_name()}" in js


@pytest.mark.parametrize("gone", [
    "waybackconfig", "crypto.subtle", "digest(", "fetch(", "AbortController", "__roofWaybackList", "CONCURRENCY",
    "BUDGET_MS", "MIN_BYTES", "byteLength", "parseReleases", "groupReleases", "map.project(",
])
def test_no_release_list_fetch_tile_scan_or_hashing_remains(gone):
    js, _ = _js()
    assert gone not in js
    for name in ("CONFIG_URL", "GROUPING_JS", "MIN_TILE_BYTES", "CONCURRENCY", "BUDGET_MS", "TILE_TIMEOUT_MS"):
        assert not hasattr(wayback, name), name


def test_control_top_right_never_passes_clicks_or_scroll_to_the_map():
    js, _ = _js()
    assert 'position: "topright"' in js and "Görüntü tarihi" in js
    assert "L.DomEvent.disableClickPropagation(div)" in js and "L.DomEvent.disableScrollPropagation(div)" in js
    assert "OPTIONS.forEach(function (o) {" in js and "sel.value = chosen.release;" in js
    assert "yükleniyor" not in js  # nothing to load: the options are fixed


def test_change_switches_the_url_and_the_attribution_without_a_rerun():
    js, _ = _js()
    assert "tiles.setUrl(url)" in js
    assert "map.attributionControl.removeAttribution(currentAttr)" in js and "map.attributionControl.addAttribution(attr)" in js
    assert wayback.WAYBACK_ATTRIBUTION == (
        "Esri World Imagery Wayback ({date}) — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
    )
    assert f"var WB_ATTR = {json.dumps(wayback.WAYBACK_ATTRIBUTION)};" in js  # tojson escapes the dash as —
    assert 'WB_ATTR.replace("{date}", choice.label)' in js
    assert "LIVE_ATTR = tiles.getAttribution()" in js  # 'Güncel' restores the original attribution
    assert "Streamlit.setComponentValue" not in js


def test_imagery_url_is_published_on_the_map_and_a_change_fires_roofimagerychange_after_set_url():
    js, _ = _js()
    assert "map.roofImageryUrl = LIVE_URL;" in js  # on init, before a stored choice is applied
    assert js.index("map.roofImageryUrl = LIVE_URL;") < js.index("apply(chosen)")
    switch = js[js.index("if (url !== currentUrl) {"):js.index("if (attr !== currentAttr)")]
    assert switch.index("tiles.setUrl(url);") < switch.index("currentUrl = map.roofImageryUrl = url;")
    assert switch.index("currentUrl = map.roofImageryUrl = url;") < switch.index('map.fire("roofimagerychange", {url: url});')


def test_choice_lives_on_window_parent_inside_try_catch():
    js, _ = _js()
    store = re.search(r"function store\(\) \{.*?\} catch \(err\) \{ return null; \} \}", js).group()
    assert "try { var w = window.parent;" in store and "w.__roofWayback = new w.Map()" in store
    assert "var chosen = find(load()) || OPTIONS[0];" in js and "save(o.release);" in js
    load = re.search(r"function load\(\) \{.*?\}", js).group()
    assert "try {" in load


def test_stored_choice_is_applied_before_when_ready():
    js, _ = _js()
    assert "if (chosen.release !== LIVE) { try { apply(chosen); } catch (err) {} }" in js
    assert js.index("apply(chosen)") < js.index("map.whenReady(")


@pytest.mark.parametrize("forbidden", ["localStorage", "sessionStorage", "indexedDB", "sendBeacon", "WebSocket"])
def test_no_storage_or_upload_api(forbidden):
    js, _ = _js()
    assert forbidden.lower() not in js.lower()


def test_caption_text():
    assert wayback.CAPTION == (
        "Görüntü tarihi: haritanın sağ üstünden farklı yıllara ait çekimleri seçebilirsiniz; en net olanı seçin. "
        "Yüksek binalarda çerçeveyi yeniden hizalamanız gerekebilir."
    )
