"""AI super-resolution (core/superres): Real-ESRGAN x4 in the browser on WebGPU only, in memory, never uploaded."""
import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import folium
import pytest
from streamlit_folium import _get_map_string

from core import imagery_fx, superres as sr
from core.config import ESRI_TILES_URL

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "static" / "models"
BBOX = [[40.9719, 29.0761], [40.9723, 29.0766]]


def _js():
    m = folium.Map(location=[40.97, 29.07], tiles=None)
    layer = folium.TileLayer(tiles=ESRI_TILES_URL, attr="Esri", cross_origin="anonymous").add_to(m)
    sr.add_to_map(m, layer, BBOX)
    m.render()
    return re.sub(r"\s+", " ", _get_map_string(m)), layer


# ------------------------------------------------------------------ runtime and model

def test_pinned_onnxruntime_web_webgpu_bundle_with_wasm_paths_and_one_thread():
    js, _ = _js()
    assert sr.ORT_SCRIPT == "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.js"
    assert json.dumps(sr.ORT_SCRIPT) in js and '"wasmPaths": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/"' in js
    assert "ort.env.wasm.wasmPaths = cfg.wasmPaths;" in sr.PARENT_JS and "ort.env.wasm.numThreads = 1;" in sr.PARENT_JS


def test_webgpu_only_no_wasm_provider_and_the_unsupported_notice():
    js, _ = _js()
    assert 'executionProviders: ["webgpu"]' in sr.PARENT_JS
    assert '"wasm"' not in js and "'wasm'" not in js and "WASM_MAX_PATCHES" not in js and not hasattr(sr, "WASM_MAX_PATCHES")
    assert sr.NO_WEBGPU_TEXT == "Bu tarayıcıda YZ netleştirme desteklenmiyor (WebGPU gerekli); orijinal görüntü gösteriliyor."
    assert json.dumps(sr.NO_WEBGPU_TEXT) in js and "setStatus(TEXT.noWebGPU);" in js
    assert 'if (!navigator.gpu) { unsupported("navigator.gpu missing"); return; }' in js  # before any download
    assert 'throw noWebGPU("no WebGPU adapter")' in sr.PARENT_JS
    assert "if (err && err.noWebGPU) unsupported(err.message); else fail(err);" in js


def test_static_model_paths_and_external_data():
    js, _ = _js()
    assert '"modelUrl": "/app/static/models/real_esrgan_general_x4v3.onnx"' in js
    assert '"dataUrl": "/app/static/models/real_esrgan_general_x4v3.data"' in js
    assert '"dataName": "real_esrgan_general_x4v3.data"' in js
    assert "externalData: [{path: cfg.dataName, data: files[1]}]" in sr.PARENT_JS


def test_static_serving_is_enabled_and_the_model_files_match_sources_md():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["enableStaticServing"] is True and config["server"]["headless"] is True
    sources = (MODELS / "SOURCES.md").read_text(encoding="utf-8")
    for name in ("real_esrgan_general_x4v3.onnx", "real_esrgan_general_x4v3.data"):
        data = (MODELS / name).read_bytes()
        row = re.search(rf"\| {re.escape(name)} \| (\d+) \| ([0-9a-f]{{64}}) \|", sources)
        assert row and int(row.group(1)) == len(data) and row.group(2) == hashlib.sha256(data).hexdigest(), name
    assert sr.MODEL_SHA256 == hashlib.sha256((MODELS / "real_esrgan_general_x4v3.onnx").read_bytes()).hexdigest()
    assert "BSD-3-Clause" in sources and "v0.65.0" in sources


# ------------------------------------------------------------------ pipeline settings

def test_patch_128_overlap_16_scale_4_and_edge_replicate_padding():
    js, _ = _js()
    assert "var NZ = 19, TILE = 256, S = 4, P = 128, OVERLAP = 16;" in js
    patch = re.search(r"function extractPatch\(.*?\n}\n", sr.PATCH_JS, re.S).group()
    assert "Math.min(H - 1, y0 + y)" in patch and "Math.min(W - 1, x0 + x)" in patch  # edge replicate
    assert "out[o] = rgba[j] / 255; out[plane + o] = rgba[j + 1] / 255; out[2 * plane + o] = rgba[j + 2] / 255;" in patch
    assert "if (ty >= OH) break;" in sr.PATCH_JS and "if (tx >= OW) break;" in sr.PATCH_JS  # padded part cropped
    assert "v < 0 ? 0 : (v > 255 ? 255 : v)" in sr.PATCH_JS  # clamp to [0, 255]


def test_crop_cap_timeouts_and_debounce():
    js, _ = _js()
    assert "var MAX_CROP_PX = 512, CACHE_MAX = 4;" in js
    assert "var MODEL_TIMEOUT_MS = 45000, INFER_TIMEOUT_MS = 30000;" in js
    assert 'withTimeout(sr.load(CFG), MODEL_TIMEOUT_MS, "model load")' in js
    assert 'INFER_TIMEOUT_MS, "inference")' in js and "cancelled = true;" in js  # no further patches after it
    assert "if (W > MAX_CROP_PX || H > MAX_CROP_PX) {" in js and json.dumps(sr.TOO_BIG_TEXT) in js
    assert "var DEBOUNCE_MS = 400," in js and "timer = setTimeout(run, DEBOUNCE_MS);" in js
    assert "runId++; // a run in progress is superseded" in js


def test_imagery_comes_from_map_roof_imagery_url_and_reruns_on_imagery_change():
    js, _ = _js()
    assert "return map.roofImageryUrl !== undefined ? map.roofImageryUrl : tiles._url;" in js
    assert 'map.on("roofimagerychange", function () { removeOverlay(); schedule(); });' in js
    assert 'var url = imageryUrl(), key = [url, NZ, x0, y0, W, H, MODEL_KEY].join("|");' in js
    assert f'MODEL_KEY = "{sr.MODEL_SHA256[:12]}"' in js


def test_overlay_covers_the_crop_without_hiding_the_tiles():
    js, _ = _js()
    assert "map.unproject(L.point(x0, y0), NZ)" in js and "map.unproject(L.point(x0 + W, y0 + H), NZ)" in js
    assert 'L.imageOverlay(url, L.latLngBounds(nw, se), {pane: "roofSuperRes", interactive: false})' in js
    assert 'pane.style.pointerEvents = "none";' in js and "setOpacity" not in js
    assert "L.DomEvent.disableClickPropagation(div);" in js and json.dumps(sr.PROGRESS_TEXT) in js


def test_session_and_cache_live_on_window_parent_inside_try_catch_with_their_own_cache_name():
    js, _ = _js()
    host = re.search(r"function host\(\) \{.*?return window\.__roofSr; \}", js).group()
    assert "try { w = window.parent; void w.document.head; } catch (err) { w = null; }" in host
    assert "s.textContent = PARENT_JS;" in host
    assert sr.CACHE_NAME == "__roofSrCache" and "w.__roofSrCache = new w.Map()" in sr.PARENT_JS
    assert sr.CACHE_NAME != imagery_fx.CACHE_NAME and "__roofFxCache" not in js
    assert "while (c.size > max)" in sr.PARENT_JS and "URL.revokeObjectURL(e.url)" in sr.PARENT_JS
    assert "new w.Blob([blob], {type: blob.type})" in sr.PARENT_JS


def test_provider_and_inference_time_are_logged():
    js, _ = _js()
    assert '"roofSuperRes: provider " + provider + ", crop "' in js and '" patches, inference "' in js


@pytest.mark.parametrize("forbidden", ["localStorage", "sessionStorage", "indexedDB", "sendBeacon", "WebSocket"])
def test_no_storage_or_upload_api(forbidden):
    js, _ = _js()
    assert forbidden.lower() not in js.lower()


def test_texts():
    assert sr.TOGGLE_LABEL == "Netleştir (YZ)"
    assert sr.CAPTION == (
        "Netleştirme, çatı görüntüsünü tarayıcınızda bir yapay zekâ modeliyle 4 kat büyütür; küçük ayrıntılar modelin "
        "tahmini olabilir. Engelleri çizerken kapatıp orijinalle karşılaştırabilirsiniz. Görüntü sunucuya gönderilmez "
        "ve saklanmaz."
    )
    assert sr.PROGRESS_TEXT == "YZ ile netleştiriliyor…" and sr.TOO_BIG_TEXT == "Bu çatı netleştirme için çok büyük"


# ------------------------------------------------------------------ patch logic run in Node (no network)

NODE_SCRIPT = """
function rnd(seed) { return function () { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; }; }
function nearest4(patch, P) {  // stand-in model: exact 4x nearest-neighbour upscale (NCHW)
    var PS = P * 4, out = new Float32Array(3 * PS * PS);
    for (var c = 0; c < 3; c++) for (var y = 0; y < PS; y++) for (var x = 0; x < PS; x++)
        out[c * PS * PS + y * PS + x] = patch[c * P * P + ((y >> 2) * P + (x >> 2))];
    return Promise.resolve(out);
}
var results = {starts: {}, cases: []};
[100, 128, 129, 158, 177, 240, 512].forEach(function (n) { results.starts[n] = patchStarts(n, 128, 16); });
var sizes = [[158, 177], [100, 60], [300, 250]], chain = Promise.resolve();
sizes.forEach(function (s) {
    chain = chain.then(function () {
        var W = s[0], H = s[1], r = rnd(W * 1000 + H), rgba = new Uint8ClampedArray(W * H * 4);
        for (var i = 0; i < rgba.length; i++) rgba[i] = (i % 4 === 3) ? 255 : Math.floor(r() * 256);
        var calls = 0;
        return upscaleCrop(rgba, W, H, 128, 16, 4, function (p) { calls++; return nearest4(p, 128); }, function () { return true; })
            .then(function (out) {
                var maxDiff = 0;
                for (var y = 0; y < out.height; y++) for (var x = 0; x < out.width; x++) for (var c = 0; c < 4; c++) {
                    var d = Math.abs(out.data[(y * out.width + x) * 4 + c] - rgba[((y >> 2) * W + (x >> 2)) * 4 + c]);
                    if (d > maxDiff) maxDiff = d;
                }
                results.cases.push({crop: [W, H], out: [out.width, out.height], patches: calls,
                    patchCount: patchCount(W, H, 128, 16), maxDiff: maxDiff});
            });
    });
});
chain.then(function () { console.log(JSON.stringify(results)); });
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_patching_padding_and_feathered_stitching_reproduce_an_exact_model_exactly():
    out = subprocess.run(
        ["node", "-"], input=sr.PATCH_JS + NODE_SCRIPT, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    res = json.loads(out.stdout)
    assert res["starts"] == {
        "100": [0], "128": [0], "129": [0, 112], "158": [0, 112], "177": [0, 112], "240": [0, 112],
        "512": [0, 112, 224, 336, 448],
    }
    assert res["cases"] == [
        {"crop": [158, 177], "out": [632, 708], "patches": 4, "patchCount": 4, "maxDiff": 0},
        {"crop": [100, 60], "out": [400, 240], "patches": 1, "patchCount": 1, "maxDiff": 0},
        {"crop": [300, 250], "out": [1200, 1000], "patches": 9, "patchCount": 9, "maxDiff": 0},
    ]
