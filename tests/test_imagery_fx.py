"""Browser-side imagery enhancement (core/imagery_fx): the first canvas look, in-memory only, never a network upload."""
import re

import folium
import pytest
from streamlit_folium import _get_map_string

from core import imagery_fx as fx

CONSTANTS = {  # name -> exact JS value (the 1e89f84 look + TILE_MARGIN / MAX_MOSAIC_PIXELS); PARAMS_HASH order
    "TILE_MARGIN": "0", "MAX_MOSAIC_PIXELS": "1600 * 1200",
    "GUIDED_RADIUS": "2", "GUIDED_EPS": "0.02 * 0.02",
    "CLAHE_GRID_X": "8", "CLAHE_GRID_Y": "4", "CLAHE_CLIP": "2.5", "CLAHE_BLEND": "0.7",
    "USM_SIGMA": "1.2", "USM_AMOUNT": "1.0", "USM_THRESHOLD": "2 / 255",
    "OUTPUT_SCALE": "2", "CHROMA_BOOST": "1.05",
}


def _js(tile_layer=True):
    m = folium.Map(location=[40.97, 29.07], tiles=None)
    layer = folium.TileLayer(tiles=fx.ESRI_TILES_URL, attr="Esri", cross_origin="anonymous").add_to(m)
    fx.add_to_map(m, layer if tile_layer else None)
    m.render()
    return re.sub(r"\s+", " ", _get_map_string(m)), layer


def _fn(name):
    return re.search(rf"function {name}\(.*?\n}}\n", fx.PIPELINE_JS, re.S).group()


def test_first_look_constants_are_declared_once_at_the_top_with_their_values():
    js, _ = _js()
    top = fx.PIPELINE_JS.index("function ")
    for name, value in CONSTANTS.items():
        assert len(re.findall(rf"var {name} = ", js)) == 1, name
        assert f"var {name} = {value};" in js, name
        assert fx.PIPELINE_JS.index(f"var {name} = ") < top, name


@pytest.mark.parametrize("later", [
    "lanczos3", "upscale(", "PRE_USM", "USM_FINE", "USM_MED", "CLARITY", "clampRing", "minMax3", "envelope(",
    "RING_MARGIN", "boxBlur3", "MAX_OUTPUT_PIXELS", "UPSCALER", "MAX_TILES",
])
def test_later_pipeline_stages_are_gone(later):
    js, _ = _js()
    assert later not in js


def test_tiles_load_anonymously_at_native_zoom_19_from_the_current_imagery_url():
    js, _ = _js()
    assert re.search(r'img\.crossOrigin = "anonymous"', js) and "new Image()" in js
    assert "var NZ = 19;" in js and fx.NATIVE_ZOOM == 19
    assert '"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"' in js
    assert "L.Util.template(url, {z: NZ, x: x, y: y})" in js
    assert "return map.roofImageryUrl !== undefined ? map.roofImageryUrl : TILE_URL;" in js
    run = js[js.index("function roofEnhance()"):js.index("map.whenReady(")]
    assert "var url = imageryUrl();" in run and "build(key, run, url," in run  # read at run time


def test_tile_range_uses_the_margin_constant_and_overlay_bounds_come_from_unproject():
    js, _ = _js()
    assert "var xMin = Math.floor(p0.x / TILE) - TILE_MARGIN, yMin = Math.floor(p0.y / TILE) - TILE_MARGIN;" in js
    assert "var xMax = Math.floor(p1.x / TILE) + TILE_MARGIN, yMax = Math.floor(p1.y / TILE) + TILE_MARGIN;" in js
    assert "map.project(b.getNorthWest(), NZ)" in js
    assert "map.unproject(L.point(xMin * TILE, yMin * TILE), NZ)" in js
    assert "map.unproject(L.point((xMax + 1) * TILE, (yMax + 1) * TILE), NZ)" in js


def test_mosaic_cap_falls_back_to_the_plain_tiles():
    js, _ = _js()
    run = js[js.index("function roofEnhance()"):js.index("map.whenReady(")]
    assert 'if (W * H > MAX_MOSAIC_PIXELS) throw new Error("mosaic too large: "' in run
    assert run.index("MAX_MOSAIC_PIXELS") < run.index("cacheGet(key)")
    assert run.rstrip().endswith("} catch (err) { fail(err); } }")  # the throw ends in fail(): one warn


def test_pipeline_order_is_the_first_look():
    native = _fn("processNative")
    assert native.index("guidedSelf(") < native.index("clahe(") < native.index("CLAHE_BLEND * e[i] + (1 - CLAHE_BLEND) * q[i]")
    assert "mergeYCbCr(rgba, n, e, c.Cb, c.Cr, CHROMA_BOOST)" in native
    assert "0.299 * r + 0.587 * g + 0.114 * b" in fx.PIPELINE_JS  # BT.601
    usm = _fn("sharpenScaled")
    assert "gaussianBlur(c.Y, W, H, USM_SIGMA)" in usm
    assert "Math.sign(d) * Math.max(0, Math.abs(d) - USM_THRESHOLD)" in usm  # soft threshold
    assert "Y[i] + USM_AMOUNT * detail" in usm and "mergeYCbCr(rgba, n, Y, c.Cb, c.Cr, 1)" in usm
    js, _ = _js()
    assert js.index("processNative(native.data, W, H);") < js.index('octx.imageSmoothingQuality = "high";')
    assert js.index("out.width = W * OUTPUT_SCALE; out.height = H * OUTPUT_SCALE;") < js.index("octx.drawImage(mosaic, 0, 0, out.width, out.height);")
    assert js.index("octx.drawImage(mosaic, 0, 0, out.width, out.height);") < js.index("sharpenScaled(scaled.data, out.width, out.height);")


def test_summed_area_tables_are_float64_with_the_reason():
    assert "new Float64Array(W1 * (H + 1))" in fx.PIPELINE_JS
    assert "Float32" in fx.PIPELINE_JS and "mean(Y^2) - mean(Y)^2" in fx.PIPELINE_JS
    integral = _fn("integral")
    assert "Float64Array" in integral and "Float32Array" not in integral
    assert _fn("guidedSelf").count("integral(Y, W, H,") == 2  # sum of Y and of Y^2


def test_overlay_is_in_a_non_interactive_pane_between_tiles_and_drawings():
    js, _ = _js()
    assert 'map.createPane("roofEnhanced")' in js and "pane.style.zIndex = 250;" in js
    assert 'pane.style.pointerEvents = "none";' in js
    assert 'L.imageOverlay(url, L.latLngBounds(nw, se), {pane: "roofEnhanced", interactive: false})' in js
    assert "URL.createObjectURL(blob)" in js and "URL.revokeObjectURL(url)" in js and ".toBlob(" in js
    assert "revoke(overlayUrl)" in js and "revoke(entry.url); entry.url = url;" in js


def test_failure_path_removes_the_overlay_restores_the_tiles_and_warns_once():
    js, _ = _js()
    fail = re.search(r"function fail\(err\) \{.*?\} function loadTile", js).group()
    assert "removeOverlay();" in fail and "setTileOpacity(1)" in fail
    assert fail.count("console.warn(") == 1 and "if (!warned)" in fail and js.count("console.warn(") == 1
    assert ".catch(fail)" in js and "catch (err) { fail(err); }" in js and 'overlay.on("error", onError)' in js
    assert js.count("console.info(") == 2  # processing time on a miss, or one cache-hit line
    assert '" in " + Math.round(t2 - t0) + " ms (tiles "' in js


def test_passed_tile_layer_is_hidden_and_restored_else_every_tile_layer():
    js, layer = _js()
    assert f"var tiles = {layer.get_name()};" in js
    assert 'overlay.on("load", function () { if (!failed && overlay === mine) setTileOpacity(0); })' in js
    js_none, _ = _js(tile_layer=False)
    assert "var tiles = null;" in js_none
    assert "map.eachLayer(function (l) { if (l instanceof L.TileLayer) l.setOpacity(v); });" in js_none


def test_runs_after_the_fit_settles_and_again_on_zoomend_resize_or_imagery_change():
    js, _ = _js()
    assert js.count("map.whenReady(schedule);") == 1 and js.count('map.on("zoomend resize", schedule);') == 1
    assert "setTimeout(roofEnhance, 0)" not in js and "moveend" not in js
    assert "var SETTLE_MS = 300;" in js  # longer than Leaflet's 250 ms zoom animation
    assert "function schedule() { clearTimeout(settleTimer); settleTimer = setTimeout(roofEnhance, SETTLE_MS); }" in js
    change = js[js.index('map.on("roofimagerychange"'):]
    for step in ("runId++;", "currentKey = null;", "removeOverlay();", "setTileOpacity(1);", "schedule();"):
        assert step in change, step  # the old overlay never stays over other imagery


def test_an_unchanged_view_is_skipped_and_a_superseded_build_is_discarded():
    js, _ = _js()
    assert "if (key === currentKey) return;" in js and "currentKey = key; var run = ++runId;" in js
    assert "if (run !== runId) return; // a newer view superseded this one" in js
    assert "cachePut(key, blob); if (run !== runId) return;" in js


def test_results_are_cached_in_parent_window_memory_only_under_their_own_name():
    js, _ = _js()
    assert fx.CACHE_NAME == "__roofFxCache" and 'CACHE_NAME = "__roofFxCache"' in js
    assert "window.parent" in js and "w[CACHE_NAME] = new w.Map()" in js
    get_cache = re.search(r"function getCache\(\) \{.*?\} function cacheGet", js).group()
    assert "try {" in get_cache and "catch (err) { return null; }" in get_cache  # unreachable parent: no cache


def test_cache_key_is_the_imagery_url_the_tile_range_the_scale_and_a_hash_of_all_constants():
    js, _ = _js()
    assert 'var key = [url, NZ, xMin, xMax, yMin, yMax, OUTPUT_SCALE, PARAMS_HASH].join("|");' in js
    hashed = re.search(r"var PARAMS_HASH = hashString\(\[(.*?)\]\.join", js).group(1)
    assert [n.strip() for n in hashed.split(",")] == list(CONSTANTS)


def test_cache_keeps_at_most_four_entries_and_revokes_the_dropped_url():
    js, _ = _js()
    assert fx.CACHE_MAX == 4 and "CACHE_MAX = 4," in js
    put = re.search(r"function cachePut\(key, blob\) \{.*?\} function cacheDrop", js).group()
    assert "while (c.size > CACHE_MAX)" in put and "c.keys().next().value" in put
    assert "revoke(c.get(oldest).url)" in put and "c.delete(oldest)" in put
    assert "new w.Blob([blob], {type: blob.type})" in put  # a parent-realm copy outlives the map iframe


def test_a_cache_hit_reuses_the_blob_and_a_stale_entry_is_processed_again():
    js, _ = _js()
    hit = js[js.index("var hit = cacheGet(key);"):js.index("map.whenReady(")]
    assert hit.index("show(hit.blob, nw, se, hit,") < hit.index("return;") < hit.rindex("build(key,")
    assert "cacheDrop(key); removeOverlay(); try { build(key, run, url, xMin, yMin, xMax, yMax, nw, se); }" in hit


@pytest.mark.parametrize(
    "forbidden", ["fetch(", "XMLHttpRequest", "localStorage", "sessionStorage", "indexedDB", "sendBeacon", "WebSocket"],
)
def test_no_network_upload_or_storage_api(forbidden):
    js, _ = _js()
    assert forbidden not in js


def test_caption_is_the_first_one():
    assert fx.CAPTION == (
        "İyileştirme yalnızca tarayıcınızda, bellekte yapılır (gürültü azaltma, yerel kontrast, keskinleştirme). "
        "Görüntü sunucuya gönderilmez, saklanmaz ve yapay zekâya verilmez. Gerçek çözünürlüğü artırmaz."
    )
