"""Browser-side imagery enhancement (core/imagery_fx): one canvas script, in-memory only, never a network upload."""
import re

import folium
import pytest
from streamlit_folium import _get_map_string

from core import imagery_fx as fx

CONSTANTS = {  # name -> exact JS value; the order is the PARAMS_HASH order
    "TILE_MARGIN": "0", "MAX_MOSAIC_PIXELS": "1600 * 1200", "MAX_OUTPUT_PIXELS": "2048 * 2048",
    "GUIDED_RADIUS": "1", "GUIDED_EPS": "0.01 * 0.01",
    "CLAHE_GRID_X": "8", "CLAHE_GRID_Y": "4", "CLAHE_CLIP": "1.8", "CLAHE_BLEND": "0.35",
    "CLARITY_SIGMA": "10", "CLARITY_AMOUNT": "0.15",
    "PRE_USM_SIGMA": "0.75", "PRE_USM_AMOUNT": "0.5", "PRE_USM_THRESHOLD": "1 / 255",
    "UPSCALER": '"lanczos3"',
    "USM_FINE_SIGMA": "0.5", "USM_FINE_AMOUNT": "0.7", "USM_MED_SIGMA": "1.4", "USM_MED_AMOUNT": "0.3",
    "USM_THRESHOLD": "1 / 255", "RING_MARGIN": "2 / 255", "CHROMA_BOOST": "1.05",
}


def _js(tile_layer=True):
    m = folium.Map(location=[40.97, 29.07], tiles=None)
    layer = folium.TileLayer(tiles=fx.ESRI_TILES_URL, attr="Esri", cross_origin="anonymous").add_to(m)
    fx.add_to_map(m, layer if tile_layer else None)
    m.render()
    return re.sub(r"\s+", " ", _get_map_string(m)), layer


def _fn(name):
    return re.search(rf"function {name}\(.*?\n}}\n", fx.PIPELINE_JS, re.S).group()


def test_tunable_constants_are_declared_once_at_the_top_with_their_values():
    js, _ = _js()
    top = fx.PIPELINE_JS.index("function ")
    for name, value in CONSTANTS.items():
        assert len(re.findall(rf"var {name} = ", js)) == 1, name
        assert f"var {name} = {value};" in js, name
        assert fx.PIPELINE_JS.index(f"var {name} = ") < top, name


@pytest.mark.parametrize("old", ["USM_SIGMA", "USM_AMOUNT", "OUTPUT_SCALE", "MAX_TILES", "processNative", "sharpenScaled"])
def test_removed_constants_and_functions_are_gone(old):
    js, _ = _js()
    assert not re.search(rf"\b{old}\b", js)


def test_tiles_load_anonymously_at_native_zoom_19_from_the_tile_layer_url():
    js, _ = _js()
    assert re.search(r'img\.crossOrigin = "anonymous"', js) and "new Image()" in js
    assert "var NZ = 19;" in js and fx.NATIVE_ZOOM == 19
    assert '"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"' in js
    assert "L.Util.template(TILE_URL, {z: NZ, x: x, y: y})" in js


def test_tile_range_uses_the_margin_constant_and_overlay_bounds_come_from_unproject():
    js, _ = _js()
    assert "var xMin = Math.floor(p0.x / TILE) - TILE_MARGIN, yMin = Math.floor(p0.y / TILE) - TILE_MARGIN;" in js
    assert "var xMax = Math.floor(p1.x / TILE) + TILE_MARGIN, yMax = Math.floor(p1.y / TILE) + TILE_MARGIN;" in js
    assert "map.project(b.getNorthWest(), NZ)" in js
    assert "map.unproject(L.point(xMin * TILE, yMin * TILE), NZ)" in js
    assert "map.unproject(L.point((xMax + 1) * TILE, (yMax + 1) * TILE), NZ)" in js


def test_mosaic_cap_falls_back_to_the_plain_tiles():
    js, _ = _js()
    enhance = js[js.index("function roofEnhance()"):js.index("map.whenReady(")]
    assert 'if (W * H > MAX_MOSAIC_PIXELS) throw new Error("mosaic too large: "' in enhance
    assert enhance.index("MAX_MOSAIC_PIXELS") < enhance.index("cacheGet(key)")
    assert enhance.rstrip().endswith("} catch (err) { fail(err); } }")  # the throw ends in fail(): one warn


def test_output_scale_is_a_capped_power_of_two_from_the_zoom():
    js, _ = _js()
    assert "var S = Math.min(4, Math.max(1, Math.pow(2, Math.ceil(map.getZoom()) - NZ)));" in js
    assert "while (S > 1 && W * H * S * S > MAX_OUTPUT_PIXELS) S = S >> 1;" in js
    assert js.index("S = S >> 1;") < js.index("var key = ")


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
    assert js.count("console.info(") == 2  # total time on a miss, or one cache-hit line
    assert '") in " + Math.round(t2 - t0) + " ms (tiles "' in js


def test_passed_tile_layer_is_hidden_and_restored_else_every_tile_layer():
    js, layer = _js()
    assert f"var tiles = {layer.get_name()};" in js
    assert 'overlay.on("load", function () { if (!failed) setTileOpacity(0); })' in js
    js_none, _ = _js(tile_layer=False)
    assert "var tiles = null;" in js_none
    assert "map.eachLayer(function (l) { if (l instanceof L.TileLayer) l.setOpacity(v); });" in js_none


def test_s_is_computed_after_the_fit_settles_and_again_on_zoomend_or_resize():
    js, _ = _js()
    assert js.count("map.whenReady(schedule);") == 1 and js.count('map.on("zoomend resize", schedule);') == 1
    assert "setTimeout(roofEnhance, 0)" not in js and "moveend" not in js
    assert "var SETTLE_MS = 300;" in js  # longer than Leaflet's 250 ms zoom animation
    assert "function schedule() { clearTimeout(settleTimer); settleTimer = setTimeout(roofEnhance, SETTLE_MS); }" in js
    run = js[js.index("function roofEnhance()"):js.index("map.whenReady(schedule);")]
    assert "Math.ceil(map.getZoom())" in run and run.index("map.getBounds()") < run.index("var S = ")
    assert js.count("map.getZoom()") == 1  # S never comes from a zoom read at script time


def test_an_unchanged_view_is_skipped_and_a_superseded_build_is_discarded():
    js, _ = _js()
    assert "if (key === currentKey) return;" in js and "currentKey = key; var run = ++runId;" in js
    assert "if (run !== runId) return; // a newer view superseded this one" in js
    assert "cachePut(key, blob); if (run !== runId) return;" in js
    assert "if (failed) return;" in js[js.index("function roofEnhance()"):]


def test_summed_area_tables_are_float64_with_the_reason():
    assert "new Float64Array(W1 * (H + 1))" in fx.PIPELINE_JS
    assert "Float32" in fx.PIPELINE_JS and "mean(Y^2) - mean(Y)^2" in fx.PIPELINE_JS
    integral = _fn("integral")
    assert "Float64Array" in integral and "Float32Array" not in integral
    assert _fn("guidedSelf").count("integral(Y, W, H,") == 2  # sum of Y and of Y^2


def test_one_conversion_in_and_out_in_float32_no_intermediate_8_bit():
    enhance = _fn("enhance")
    assert enhance.count("splitYCbCr(") == 1 and enhance.count("mergeYCbCr(") == 1
    assert enhance.rstrip().endswith("return {data: mergeYCbCr(Y, Cb, Cr, n2, CHROMA_BOOST), width: W2, height: H2};\n}")
    assert "Uint8ClampedArray" in _fn("mergeYCbCr") and "Uint8ClampedArray" not in enhance
    assert "0.299 * r + 0.587 * g + 0.114 * b" in fx.PIPELINE_JS  # BT.601
    js, _ = _js()
    assert "imageSmoothingQuality" not in js and "drawImage(mosaic" not in js  # no canvas resampling
    assert "var r = enhance(native.data, W, H, S);" in js and "img.data.set(r.data);" in js


@pytest.mark.parametrize("banned", ["gamma", "saturat", "Math.pow(Y", "curve"])
def test_no_tone_curve_gamma_or_saturation(banned):
    assert banned not in _fn("enhance") and banned not in _fn("mergeYCbCr")


def test_native_stages_run_in_order_guided_clahe_clarity_pre_sharpen():
    e = _fn("enhance")
    order = [
        "guidedSelf(c.Y, W, H, GUIDED_RADIUS, GUIDED_EPS)",
        "clahe(Y, W, H, CLAHE_GRID_X, CLAHE_GRID_Y, CLAHE_CLIP)",
        "CLAHE_BLEND * e[i] + (1 - CLAHE_BLEND) * Y[i]",
        "blur(Y, W, H, CLARITY_SIGMA)",
        "Y[i] += CLARITY_AMOUNT * (Y[i] - cl[i])",
        "blur(Y, W, H, PRE_USM_SIGMA)",
        "PRE_USM_AMOUNT * Math.sign(d) * Math.max(0, Math.abs(d) - PRE_USM_THRESHOLD)",
        "upscale(Y, W, H, S, lanczos3, 3)",
        "blur(Y, W2, H2, USM_FINE_SIGMA * S)",
        "blur(Y, W2, H2, USM_MED_SIGMA * S)",
    ]
    positions = [e.index(token) for token in order]
    assert positions == sorted(positions)


def test_lanczos3_for_y_bilinear_for_chroma():
    e = _fn("enhance")
    flat = re.sub(r"\s+", " ", e)
    assert 'var Yb = upscale(Y, W, H, S, bilinear, 1); if (UPSCALER === "bilinear") { Y = Yb; } else { Y = upscale(Y, W, H, S, lanczos3, 3);' in flat
    assert "Cb = upscale(Cb, W, H, S, bilinear, 1);" in e and "Cr = upscale(Cr, W, H, S, bilinear, 1);" in e
    assert "3 * Math.sin(a) * Math.sin(a / 3) / (a * a)" in _fn("lanczos3")
    upscale = _fn("upscale")
    assert "Float32Array" in upscale and "Math.min(1" not in upscale and "Uint8" not in upscale  # no clamping


def test_two_band_soft_threshold_unsharp_mask_at_output_scale():
    e = _fn("enhance")
    assert "var fine = blur(Y, W2, H2, USM_FINE_SIGMA * S);" in e and "var med = blur(Y, W2, H2, USM_MED_SIGMA * S);" in e
    assert "Y[i] += USM_FINE_AMOUNT * Math.sign(d) * Math.max(0, Math.abs(d) - USM_THRESHOLD);" in e
    assert "Y[i] += USM_MED_AMOUNT * Math.sign(d) * Math.max(0, Math.abs(d) - USM_THRESHOLD);" in e
    assert e.index("USM_FINE_AMOUNT") < e.index("var med = ")  # the medium band works on the clamped fine result


def test_anti_ringing_clamp_after_clarity_pre_sharpen_lanczos_fine_and_medium():
    e = _fn("enhance")
    stages = [  # (stage marker, the clamp right after it)
        ("Y[i] += CLARITY_AMOUNT * (Y[i] - cl[i]);", "clampRing(Y, env, RING_MARGIN, env);"),
        ("PRE_USM_AMOUNT * Math.sign(d)", "clampRing(Y, mm, RING_MARGIN, env);"),
        ("Y = upscale(Y, W, H, S, lanczos3, 3);", "clampRing(Y, mm, RING_MARGIN, env); // removes the Lanczos3 overshoot"),
        ("USM_FINE_AMOUNT * Math.sign(d)", "clampRing(Y, mm, RING_MARGIN, env);"),
        ("USM_MED_AMOUNT * Math.sign(d)", "clampRing(Y, mm, RING_MARGIN, env);"),
    ]
    flat = re.sub(r"\s+", " ", e)
    pos = 0
    for marker, clamp in stages:
        pos = flat.index(marker, pos)
        nxt = flat.index("clampRing(", pos)
        assert flat.startswith(clamp, nxt), (marker, flat[nxt:nxt + 80])
        pos = nxt + 1
    assert flat.count("clampRing(") == 5
    # each stage's 3x3 range is taken from that stage's INPUT, before the stage changes Y
    assert flat.index("env = envelope(ref, W, H)") < flat.index("blur(Y, W, H, CLARITY_SIGMA)")
    assert flat.index("mm = minMax3(Y, W, H, null); // d. pre-sharpen") < flat.index("blur(Y, W, H, PRE_USM_SIGMA)")
    assert flat.index("var Yb = upscale(Y, W, H, S, bilinear, 1);") < flat.index("mm = minMax3(Yb, W2, H2, null);")
    assert flat.index("mm = minMax3(Y, W2, H2, mm); // fine band") < flat.index("var fine = ")
    assert flat.index("mm = minMax3(Y, W2, H2, mm); // medium band") < flat.index("var med = ")
    assert "env = envelope(upscale(ref, W, H, S, bilinear, 1), W2, H2);" in flat  # the S x anchor


def test_clamp_and_3x3_min_max_helpers():
    clamp = re.sub(r"\s+", " ", _fn("clampRing"))
    assert "(mn[i] > en[i] ? mn[i] : en[i]) - margin" in clamp and "(mx[i] < ex[i] ? mx[i] : ex[i]) + margin" in clamp
    assert "Y[i] = v < lo ? lo : (v > hi ? hi : v);" in clamp
    mm = _fn("minMax3")
    assert "// rows" in mm and "// columns" in mm and "Float32Array" in mm  # separable 1D passes, typed arrays
    assert "src[o + (x > 0 ? x - 1 : 0)]" in mm and "(y > 0 ? y - 1 : 0) * W" in mm


def test_blur_uses_gaussian_below_sigma_1_5_and_a_three_pass_box_from_1_5():
    assert "return sigma < 1.5 ? gaussianBlur(src, W, H, sigma) : boxBlur3(src, W, H, sigma);" in _fn("blur")
    box3 = _fn("boxBlur3")
    assert "boxesForGauss(sigma, 3)" in box3 and "boxPassH(a, b, W, H, r);" in box3 and "boxPassV(b, a, W, H, r);" in box3
    assert "acc += src[o + Math.min(W - 1, x + r + 1)] - src[o + Math.max(0, x - r)];" in _fn("boxPassH")  # sliding
    assert "Math.sqrt(12 * sigma * sigma / n + 1)" in _fn("boxesForGauss")


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


def test_cache_key_is_the_tile_range_the_scale_and_a_hash_of_all_constants():
    js, _ = _js()
    assert 'var key = [NZ, xMin, xMax, yMin, yMax, S, PARAMS_HASH].join("|");' in js
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
    assert "cacheDrop(key); removeOverlay(); try { build(key, run, xMin, yMin, xMax, yMax, S, nw, se); }" in hit
