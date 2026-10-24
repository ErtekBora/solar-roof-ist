"""Roof map imagery date selector (core/wayback): one browser script, in-memory only, never a Python network call."""
import json
import re
import shutil
import subprocess

import folium
import pytest
from streamlit_folium import _get_map_string

from core import wayback
from core.config import ESRI_TILES_URL


def _js(key="b1"):
    m = folium.Map(location=[40.97, 29.07], tiles=None)
    layer = folium.TileLayer(tiles=ESRI_TILES_URL, attr="Esri", cross_origin="anonymous").add_to(m)
    wayback.add_to_map(m, layer, key, (40.97, 29.07), ESRI_TILES_URL)
    m.render()
    return re.sub(r"\s+", " ", _get_map_string(m)), layer


def test_inputs_come_from_python_building_key_centre_live_url_and_layer_name():
    js, layer = _js("b 42")
    assert 'var KEY = "b 42"' in js and "CENTER = L.latLng([40.97, 29.07])" in js
    assert f"tiles = {layer.get_name()}" in js and json.dumps(ESRI_TILES_URL) in js
    assert "waybackconfig.json" in js and wayback.CONFIG_URL in js


def test_release_list_parses_the_date_from_the_title_and_the_level_row_col_template():
    js, _ = _js()
    assert "Wayback (\\d{4}-\\d{2}-\\d{2})" in js
    assert '.replace("{level}", "{z}").replace("{row}", "{y}").replace("{col}", "{x}")' in js


def test_tile_under_the_centre_at_z19_is_hashed_with_sha1_and_placeholders_are_skipped():
    js, _ = _js()
    assert "map.project(CENTER, NZ)" in js and "var NZ = 19" in js and "Math.floor(p.x / TILE)" in js
    assert 'crypto.subtle.digest("SHA-1", buf)' in js
    assert "var NZ = 19, TILE = 256, MIN_BYTES = 3000;" in js and "buf.byteLength < MIN_BYTES" in js
    assert wayback.MIN_TILE_BYTES == 3000


def test_concurrency_8_timeout_8s_and_overall_budget_20s():
    js, _ = _js()
    assert "var CONCURRENCY = 8, TIMEOUT_MS = 8000, BUDGET_MS = 20000;" in js
    assert "while (active < CONCURRENCY && next < jobs.length)" in js
    assert "setTimeout(function () { ctrl.abort(); }, TIMEOUT_MS)" in js
    assert "setTimeout(finish, BUDGET_MS)" in js and "budget.abort()" in js  # after the budget: what has arrived


def test_control_top_right_never_passes_clicks_or_scroll_to_the_map():
    js, _ = _js()
    assert 'position: "topright"' in js and "Görüntü tarihi" in js and "Görüntü tarihleri yükleniyor…" in js
    assert "L.DomEvent.disableClickPropagation(div)" in js and "L.DomEvent.disableScrollPropagation(div)" in js
    assert "options.length < 2) { hide(); return; }" in js  # fewer than two options: hidden


def test_change_switches_the_url_and_the_attribution_without_a_rerun():
    js, _ = _js()
    assert "tiles.setUrl(url)" in js
    assert "map.attributionControl.removeAttribution(currentAttr)" in js and "map.attributionControl.addAttribution(attr)" in js
    assert wayback.WAYBACK_ATTRIBUTION == (
        "Esri World Imagery Wayback ({date}) — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
    )
    assert f"var WB_ATTR = {json.dumps(wayback.WAYBACK_ATTRIBUTION)};" in js  # tojson escapes the dash as —
    assert "LIVE_ATTR = tiles.getAttribution()" in js  # 'Güncel' restores the original attribution
    assert "Streamlit.setComponentValue" not in js


def test_imagery_url_is_published_on_the_map_and_a_change_fires_roofimagerychange_after_set_url():
    js, _ = _js()
    assert "map.roofImageryUrl = LIVE_URL;" in js  # on init, before a stored choice is applied
    assert js.index("map.roofImageryUrl = LIVE_URL;") < js.index("apply(stored)")
    switch = js[js.index("if (url !== currentUrl) {"):js.index("if (attr !== currentAttr)")]
    assert switch.index("tiles.setUrl(url);") < switch.index("currentUrl = map.roofImageryUrl = url;")
    assert switch.index("currentUrl = map.roofImageryUrl = url;") < switch.index('map.fire("roofimagerychange", {url: url});')


def test_choice_and_list_live_on_window_parent_inside_try_catch():
    js, _ = _js()
    store = re.search(r"function store\(name\) \{.*?\} catch \(err\) \{ return null; \} \}", js).group()
    assert "try { var w = window.parent;" in store and "new w.Map()" in store
    assert 'load("__roofWayback")' in js and 'save("__roofWayback", o)' in js
    assert 'load("__roofWaybackList")' in js and 'save("__roofWaybackList", options)' in js


def test_stored_choice_is_applied_before_when_ready():
    js, _ = _js()
    assert 'if (stored) { try { apply(stored); } catch (err) {} }' in js
    assert js.index("apply(stored)") < js.index("map.whenReady(")
    assert js.index('load("__roofWaybackList")') > js.index("map.whenReady(")  # reused list: no refetch


@pytest.mark.parametrize("forbidden", ["localStorage", "sessionStorage", "indexedDB", "sendBeacon", "WebSocket"])
def test_no_storage_or_upload_api(forbidden):
    js, _ = _js()
    assert forbidden.lower() not in js.lower()


def test_caption_text():
    assert wayback.CAPTION == (
        "Görüntü tarihi: haritanın sağ üstünden farklı yıllara ait çekimleri seçebilirsiniz; en net olanı seçin. "
        "Yüksek binalarda çerçeveyi yeniden hizalamanız gerekebilir."
    )


# ------------------------------------------------------------------ grouping logic run in Node (no network)

STUB_CONFIG = {
    "63116": {"itemTitle": "World Imagery (Wayback 2026-09-24)", "itemURL": "https://wb/tile/63116/{level}/{row}/{col}"},
    "50000": {"itemTitle": "World Imagery (Wayback 2026-03-01)", "itemURL": "https://wb/tile/50000/{level}/{row}/{col}"},
    "40000": {"itemTitle": "World Imagery (Wayback 2024-06-01)", "itemURL": "https://wb/tile/40000/{level}/{row}/{col}"},
    "30000": {"itemTitle": "World Imagery (Wayback 2023-01-01)", "itemURL": "https://wb/tile/30000/{level}/{row}/{col}"},
    "20000": {"itemTitle": "World Imagery (Wayback 2021-05-01)", "itemURL": "https://wb/tile/20000/{level}/{row}/{col}"},
    "10000": {"itemTitle": "World Imagery (Wayback 2019-01-01)", "itemURL": "https://wb/tile/10000/{level}/{row}/{col}"},
    "99999": {"itemTitle": "no date here", "itemURL": "https://wb/tile/99999/{level}/{row}/{col}"},
}
# live == newest release (A, also in 2026-03-01), two older distinct captures (B twice, C), one placeholder (null)
STUB_HASHES = {"63116": "A", "50000": "A", "40000": "B", "30000": "B", "20000": "C", "10000": None}


def _node(live_hash):
    script = (
        wayback.GROUPING_JS
        + f"\nvar rel = parseReleases({json.dumps(STUB_CONFIG)});"
        + f"\nconsole.log(JSON.stringify(groupReleases(rel, {json.dumps(STUB_HASHES)}, {json.dumps(live_hash)})));"
    )
    out = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_grouping_keeps_the_oldest_release_per_capture_and_lists_the_live_one_as_guncel():
    assert _node("A") == [
        {"release": "live", "date": "2026-03-01", "label": "Güncel (2026-03-01)", "url": None},
        {"release": "30000", "date": "2023-01-01", "label": "2023-01-01", "url": "https://wb/tile/30000/{z}/{y}/{x}"},
        {"release": "20000", "date": "2021-05-01", "label": "2021-05-01", "url": "https://wb/tile/20000/{z}/{y}/{x}"},
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_failed_live_tile_gives_a_plain_guncel_and_every_capture():
    assert [o["label"] for o in _node(None)] == ["Güncel", "2026-03-01", "2023-01-01", "2021-05-01"]
