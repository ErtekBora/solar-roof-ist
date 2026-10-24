"""Browser-side imagery enhancement (core/imagery_fx): one canvas script, in-memory only, never a network upload."""
import re

import folium
import pytest
from streamlit_folium import _get_map_string

from core import imagery_fx as fx

CONSTANTS = (
    "GUIDED_RADIUS", "GUIDED_EPS", "CLAHE_GRID_X", "CLAHE_GRID_Y", "CLAHE_CLIP", "CLAHE_BLEND",
    "USM_SIGMA", "USM_AMOUNT", "USM_THRESHOLD", "OUTPUT_SCALE", "CHROMA_BOOST",
)


def _js(tile_layer=True):
    m = folium.Map(location=[40.97, 29.07], tiles=None)
    layer = folium.TileLayer(tiles=fx.ESRI_TILES_URL, attr="Esri", cross_origin="anonymous").add_to(m)
    fx.add_to_map(m, layer if tile_layer else None)
    m.render()
    return re.sub(r"\s+", " ", _get_map_string(m)), layer


def test_all_eleven_tunable_constants_are_declared_once_at_the_top():
    js, _ = _js()
    for name in CONSTANTS:
        assert len(re.findall(rf"var {name} = ", js)) == 1, name
    top = fx.PIPELINE_JS.index("function ")
    assert all(fx.PIPELINE_JS.index(f"var {n} = ") < top for n in CONSTANTS)
    assert "var GUIDED_RADIUS = 2;" in js and "var GUIDED_EPS = 0.02 * 0.02;" in js
    assert "var CLAHE_GRID_X = 8;" in js and "var CLAHE_GRID_Y = 4;" in js and "var CLAHE_CLIP = 2.5;" in js
    assert "var CLAHE_BLEND = 0.7;" in js and "var USM_SIGMA = 1.2;" in js and "var USM_AMOUNT = 1.0;" in js
    assert "var USM_THRESHOLD = 2 / 255;" in js and "var OUTPUT_SCALE = 2;" in js and "var CHROMA_BOOST = 1.05;" in js


def test_tiles_load_anonymously_at_native_zoom_19_from_the_tile_layer_url():
    js, _ = _js()
    assert re.search(r'img\.crossOrigin = "anonymous"', js) and "new Image()" in js
    assert "var NZ = 19;" in js and fx.NATIVE_ZOOM == 19
    assert '"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"' in js
    assert "L.Util.template(TILE_URL, {z: NZ, x: x, y: y})" in js


def test_overlay_bounds_come_from_leaflet_unproject_at_native_zoom():
    js, _ = _js()
    assert "map.unproject(L.point(xMin * TILE, yMin * TILE), NZ)" in js
    assert "map.unproject(L.point((xMax + 1) * TILE, (yMax + 1) * TILE), NZ)" in js
    assert "map.project(b.getNorthWest(), NZ)" in js and "- 1" in js and "+ 1" in js  # 1-tile margin


def test_overlay_is_in_a_non_interactive_pane_between_tiles_and_drawings():
    js, _ = _js()
    assert 'map.createPane("roofEnhanced")' in js and "pane.style.zIndex = 250;" in js
    assert 'pane.style.pointerEvents = "none";' in js
    assert 'L.imageOverlay(url, L.latLngBounds(nw, se), {pane: "roofEnhanced", interactive: false})' in js
    assert "URL.createObjectURL(blob)" in js and "URL.revokeObjectURL(url)" in js and ".toBlob(" in js
    assert "revoke(overlayUrl)" in js and "revoke(entry.url); entry.url = url;" in js  # previous URL revoked


def test_failure_path_removes_the_overlay_restores_the_tiles_and_warns_once():
    js, _ = _js()
    fail = re.search(r"function fail\(err\) \{.*?\} function loadTile", js).group()
    assert "removeOverlay();" in fail and "setTileOpacity(1)" in fail
    assert "function removeOverlay() { if (overlay) { map.removeLayer(overlay); overlay = null; }" in js
    assert fail.count("console.warn(") == 1 and "if (!failed)" in fail and js.count("console.warn(") == 1
    assert ".catch(fail)" in js and "catch (err) { fail(err); }" in js and 'overlay.on("error", onError)' in js
    assert js.count("console.info(") == 2  # processing time, or one cache-hit line


def test_passed_tile_layer_is_hidden_and_restored_else_every_tile_layer():
    js, layer = _js()
    assert f"var tiles = {layer.get_name()};" in js
    assert 'overlay.on("load", function () { if (!failed) setTileOpacity(0); })' in js
    js_none, _ = _js(tile_layer=False)
    assert "var tiles = null;" in js_none
    assert "map.eachLayer(function (l) { if (l instanceof L.TileLayer) l.setOpacity(v); });" in js_none


def test_runs_once_after_the_view_is_ready():
    js, _ = _js()
    assert js.count("map.whenReady(function () { setTimeout(roofEnhance, 0); });") == 1
    assert "zoomend" not in js and "moveend" not in js


def test_summed_area_tables_are_float64_with_the_reason():
    assert "new Float64Array(W1 * (H + 1))" in fx.PIPELINE_JS
    assert "Float32" in fx.PIPELINE_JS and "mean(Y^2) - mean(Y)^2" in fx.PIPELINE_JS
    integral = re.search(r"function integral\(.*?\n}\n", fx.PIPELINE_JS, re.S).group()
    assert "Float64Array" in integral and "Float32Array" not in integral
    guided = re.search(r"function guidedSelf\(.*?\n}\n", fx.PIPELINE_JS, re.S).group()
    assert guided.count("integral(Y, W, H,") == 2  # sum of Y and of Y^2


def test_pipeline_order_and_soft_threshold_unsharp_mask():
    js = fx.PIPELINE_JS
    native = re.search(r"function processNative\(.*?\n}\n", js, re.S).group()
    assert native.index("guidedSelf(") < native.index("clahe(") < native.index("CLAHE_BLEND * e[i] + (1 - CLAHE_BLEND) * q[i]")
    assert "mergeYCbCr(rgba, n, e, c.Cb, c.Cr, CHROMA_BOOST)" in native
    assert "0.299 * r + 0.587 * g + 0.114 * b" in js  # BT.601
    usm = re.search(r"function sharpenScaled\(.*?\n}\n", js, re.S).group()
    assert "Math.sign(d) * Math.max(0, Math.abs(d) - USM_THRESHOLD)" in usm
    assert "Y[i] + USM_AMOUNT * detail" in usm and "mergeYCbCr(rgba, n, Y, c.Cb, c.Cr, 1)" in usm
    script = _js()[0]
    assert script.index("processNative(native.data") < script.index('imageSmoothingQuality = "high"')
    assert script.index('imageSmoothingQuality = "high"') < script.index("sharpenScaled(scaled.data")


@pytest.mark.parametrize(
    "forbidden", ["fetch(", "XMLHttpRequest", "localStorage", "sessionStorage", "indexedDB", "sendBeacon", "WebSocket"],
)
def test_no_network_upload_or_storage_api(forbidden):
    js, _ = _js()
    assert forbidden not in js


def test_caption_says_browser_memory_only():
    assert fx.CAPTION == (
        "İyileştirme yalnızca tarayıcınızda, bellekte yapılır (gürültü azaltma, yerel kontrast, keskinleştirme). "
        "Görüntü sunucuya gönderilmez, saklanmaz ve yapay zekâya verilmez. Gerçek çözünürlüğü artırmaz."
    )


def test_results_are_cached_in_parent_window_memory_only():
    js, _ = _js()
    assert "window.parent" in js and "__roofFxCache" in js and "new w.Map()" in js
    assert "sessionStorage" not in js and "localStorage" not in js and "caches." not in js
    get_cache = re.search(r"function getCache\(\) \{.*?\} function cacheGet", js).group()
    assert "try {" in get_cache and "catch (err) { return null; }" in get_cache  # unreachable parent: no cache


def test_cache_key_is_the_tile_range_plus_a_hash_of_all_constants():
    js, _ = _js()
    assert 'var key = [NZ, xMin, xMax, yMin, yMax, PARAMS_HASH].join("|");' in js
    hashed = re.search(r"var PARAMS_HASH = hashString\(\[(.*?)\]\.join", js).group(1)
    assert [n.strip() for n in hashed.split(",")] == list(CONSTANTS)


def test_cache_keeps_at_most_four_entries_and_revokes_the_dropped_url():
    js, _ = _js()
    assert fx.CACHE_MAX == 4 and "CACHE_MAX = 4;" in js
    put = re.search(r"function cachePut\(key, blob\) \{.*?\} function cacheDrop", js).group()
    assert "while (c.size > CACHE_MAX)" in put and "c.keys().next().value" in put
    assert "revoke(c.get(oldest).url)" in put and "c.delete(oldest)" in put
    assert "new w.Blob([blob], {type: blob.type})" in put  # a parent-realm copy outlives the map iframe


def test_a_cache_hit_reuses_the_blob_and_a_stale_entry_is_processed_again():
    js, _ = _js()
    hit = js[js.index("var hit = cacheGet(key);"):js.index("map.whenReady(")]
    assert hit.index("show(hit.blob, nw, se, hit,") < hit.index("return;") < hit.rindex("build(key,")
    assert "cacheDrop(key); removeOverlay(); try { build(key, xMin, yMin, xMax, yMax, nw, se); }" in hit
