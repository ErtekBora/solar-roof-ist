"""Roof map AI super-resolution ("Netleştir (YZ)"): Real-ESRGAN General x4v3 run in the user's browser, DISPLAY ONLY.

Privacy/licence rule: Python never fetches, stores or reads a tile. The script runs in the user's browser, in memory
only: it loads the z19 tiles of the current imagery (live or the chosen Wayback release, `map.roofImageryUrl`), crops
the roof area, runs the model with onnxruntime-web on WebGPU and shows the 4x result as an image overlay over exactly
that crop. Without WebGPU nothing runs (no WASM fallback: far too slow, ~8 s per patch) and a notice is shown. Nothing
is uploaded, stored or sent to a server or a remote model. The tile layer is never hidden; on any failure the overlay
is removed and the plain tiles stay.

Model: static/models (see SOURCES.md): input "image" float32 NCHW [1, 3, 128, 128] RGB in 0..1, output
"upscaled_image" [1, 3, 512, 512] in 0..1 (not clamped by the graph). Weights are external data
(ONNX Runtime "externalData" session option).

Realms: every map rebuild is a new iframe, and promise jobs of a removed iframe never run again. So onnxruntime-web,
the session and the inference queue live in the PARENT window (window.parent.__roofSr, a script injected once into the
parent document; falls back to the iframe itself when the parent is unreachable), and so does the result cache
(window.parent.__roofSrCache, never shared with the classical enhancement's __roofFxCache). The iframe only passes
plain arrays and awaits the results.

Docs: https://onnxruntime.ai/docs/tutorials/web/  https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
https://onnxruntime.ai/docs/tutorials/web/large-models.html (externalData)
https://leafletjs.com/reference.html#imageoverlay  https://leafletjs.com/reference.html#map-project
https://docs.streamlit.io/develop/concepts/configuration/serving-static-files
"""
from branca.element import MacroElement
from jinja2 import Template

ORT_VERSION = "1.30.0"
ORT_DIST = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/"
ORT_SCRIPT = ORT_DIST + "ort.webgpu.min.js"  # the WebGPU build; its runtime is a .wasm loaded from wasmPaths
MODEL_URL = "/app/static/models/real_esrgan_general_x4v3.onnx"  # Streamlit static serving (static/ -> /app/static/)
MODEL_DATA_URL = "/app/static/models/real_esrgan_general_x4v3.data"
MODEL_DATA_NAME = "real_esrgan_general_x4v3.data"  # the external-data location recorded in the graph
MODEL_SHA256 = "09e6b675a0a18a97e1057c8dd7e984ef878a0a02ce104430da525b6241c858e7"  # of the .onnx, see SOURCES.md
NATIVE_ZOOM = 19  # must match app.ESRI_NATIVE_ZOOM
SCALE = 4
PATCH = 128  # the model's fixed input size
OVERLAP = 16  # native px shared by neighbouring patches (linear feathering)
MAX_CROP_PX = 512  # native px per side; a larger crop is skipped
MODEL_TIMEOUT_MS = 45000  # loading onnxruntime-web, the model and the session
INFER_TIMEOUT_MS = 30000  # all patches of one crop
DEBOUNCE_MS = 400  # quiet time before a run (offset clicks, imagery changes)
CACHE_MAX = 4  # results kept in browser memory (window.parent.__roofSrCache)
CACHE_NAME = "__roofSrCache"  # never shared with the classical enhancement cache (core/imagery_fx)
PANE = "roofSuperRes"

TOGGLE_LABEL = "Netleştir (YZ)"
CAPTION = (
    "Netleştirme, çatı görüntüsünü tarayıcınızda bir yapay zekâ modeliyle 4 kat büyütür; küçük ayrıntılar modelin "
    "tahmini olabilir. Engelleri çizerken kapatıp orijinalle karşılaştırabilirsiniz. Görüntü sunucuya gönderilmez ve "
    "saklanmaz."
)
PROGRESS_TEXT = "YZ ile netleştiriliyor…"
TOO_BIG_TEXT = "Bu çatı netleştirme için çok büyük"
NO_WEBGPU_TEXT = "Bu tarayıcıda YZ netleştirme desteklenmiyor (WebGPU gerekli); orijinal görüntü gösteriliyor."

# Pure patch functions (no DOM, no Leaflet, no Jinja syntax), so the same text can be run outside the map (Node).
PATCH_JS = """
function patchStarts(len, P, overlap) {  // patch origins along one axis; the last patch may run past the end (padded)
    var step = P - overlap, out = [0];
    while (out[out.length - 1] + P < len) out.push(out[out.length - 1] + step);
    return out;
}

function extractPatch(rgba, W, H, x0, y0, P) {  // NCHW RGB float32 in 0..1, edges replicated outside the crop
    var plane = P * P, out = new Float32Array(3 * plane);
    for (var y = 0; y < P; y++) {
        var sy = Math.min(H - 1, y0 + y), row = sy * W;
        for (var x = 0; x < P; x++) {
            var j = (row + Math.min(W - 1, x0 + x)) * 4, o = y * P + x;
            out[o] = rgba[j] / 255; out[plane + o] = rgba[j + 1] / 255; out[2 * plane + o] = rgba[j + 2] / 255;
        }
    }
    return out;
}

function featherWeights(len, ramp, hasPrev, hasNext) {  // linear ramps only on sides shared with a neighbour
    var w = new Float32Array(len);
    for (var u = 0; u < len; u++) {
        var a = hasPrev ? Math.min(1, (u + 0.5) / ramp) : 1, b = hasNext ? Math.min(1, (len - u - 0.5) / ramp) : 1;
        w[u] = a * b;
    }
    return w;
}

// Adds one upscaled patch (NCHW, PS x PS) at output origin (ox, oy) into acc (3 planes of OW x OH) and wsum; pixels
// outside the output (the padded part of an edge patch) are cropped away.
function accumulate(acc, wsum, OW, OH, out, PS, ox, oy, wx, wy) {
    var plane = PS * PS, oplane = OW * OH;
    for (var y = 0; y < PS; y++) {
        var ty = oy + y;
        if (ty >= OH) break;
        for (var x = 0; x < PS; x++) {
            var tx = ox + x;
            if (tx >= OW) break;
            var w = wx[x] * wy[y], i = ty * OW + tx, s = y * PS + x;
            acc[i] += w * out[s]; acc[oplane + i] += w * out[plane + s]; acc[2 * oplane + i] += w * out[2 * plane + s];
            wsum[i] += w;
        }
    }
}

function toRGBA(acc, wsum, OW, OH) {  // weighted mean, clamped to [0, 255]; Uint8ClampedArray rounds on assignment
    var n = OW * OH, rgba = new Uint8ClampedArray(n * 4);
    for (var i = 0; i < n; i++) {
        var k = wsum[i] > 0 ? 255 / wsum[i] : 0;
        for (var c = 0; c < 3; c++) {
            var v = acc[c * n + i] * k;
            rgba[i * 4 + c] = v < 0 ? 0 : (v > 255 ? 255 : v);
        }
        rgba[i * 4 + 3] = 255;
    }
    return rgba;
}

// Runs `infer(patch) -> Promise<Float32Array NCHW>` over the crop, one patch at a time; `alive()` false aborts.
function upscaleCrop(rgba, W, H, P, overlap, S, infer, alive) {
    var xs = patchStarts(W, P, overlap), ys = patchStarts(H, P, overlap), PS = P * S, OW = W * S, OH = H * S;
    var acc = new Float32Array(3 * OW * OH), wsum = new Float32Array(OW * OH), jobs = [];
    ys.forEach(function (y0, j) { xs.forEach(function (x0, i) { jobs.push([x0, y0, i, j]); }); });
    var chain = Promise.resolve();
    jobs.forEach(function (job) {
        chain = chain.then(function () {
            if (alive && !alive()) throw new Error("superseded");
            return infer(extractPatch(rgba, W, H, job[0], job[1], P)).then(function (out) {
                var wx = featherWeights(PS, overlap * S, job[2] > 0, job[2] < xs.length - 1);
                var wy = featherWeights(PS, overlap * S, job[3] > 0, job[3] < ys.length - 1);
                accumulate(acc, wsum, OW, OH, out, PS, job[0] * S, job[1] * S, wx, wy);
            });
        });
    });
    return chain.then(function () { return {data: toRGBA(acc, wsum, OW, OH), width: OW, height: OH}; });
}

function patchCount(W, H, P, overlap) { return patchStarts(W, P, overlap).length * patchStarts(H, P, overlap).length; }
"""

# Runs in the PARENT window (or the iframe as a fallback) as `(function (w) {...})(window)`: onnxruntime-web, one
# WebGPU session per page, a serial inference queue and the result cache (w.__roofSrCache). No Jinja syntax inside
# (values come via cfg).
PARENT_JS = """
(function (w) {
    if (w.__roofSr) return;
    var state = {ready: null, queue: Promise.resolve()};
    function cache() {  // the AI result cache; never the classical enhancement's cache
        if (!w.__roofSrCache) w.__roofSrCache = new w.Map();
        return w.__roofSrCache;
    }
    function noWebGPU(msg) { var e = new Error(msg); e.noWebGPU = true; return e; }
    function adapter() {  // WebGPU or nothing: no WASM fallback
        if (!w.navigator.gpu) return Promise.reject(noWebGPU("navigator.gpu missing"));
        return w.navigator.gpu.requestAdapter().then(function (a) {
            if (!a) throw noWebGPU("no WebGPU adapter");
            return a;
        }, function () { throw noWebGPU("WebGPU adapter request failed"); });
    }
    function loadOrt(cfg) {
        if (w.ort && w.ort.InferenceSession) return Promise.resolve(w.ort);
        return new Promise(function (resolve, reject) {
            var s = w.document.createElement("script");
            s.src = cfg.ortScript; s.crossOrigin = "anonymous";
            s.onload = function () { w.ort ? resolve(w.ort) : reject(new Error("ort missing")); };
            s.onerror = function () { reject(new Error("onnxruntime-web load failed")); };
            w.document.head.appendChild(s);
        });
    }
    function fetchBytes(url) {
        return w.fetch(url, {credentials: "same-origin"}).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status + " " + url);
            return r.arrayBuffer();
        }).then(function (b) { return new Uint8Array(b); });
    }
    function load(cfg) {  // one session per page; a failure lets the next call try again
        if (state.ready) return state.ready;
        state.ready = adapter().then(function () { return loadOrt(cfg); }).then(function (ort) {
            ort.env.wasm.wasmPaths = cfg.wasmPaths;  // the WebGPU build's runtime is a .wasm module
            ort.env.wasm.numThreads = 1;  // not cross-origin isolated: no SharedArrayBuffer
            return Promise.all([fetchBytes(cfg.modelUrl), fetchBytes(cfg.dataUrl)]).then(function (files) {
                return ort.InferenceSession.create(files[0], {
                    executionProviders: ["webgpu"], externalData: [{path: cfg.dataName, data: files[1]}],
                });
            }).then(function (session) { return {ort: ort, session: session, provider: "webgpu"}; });
        });
        state.ready.catch(function () { state.ready = null; });
        return state.ready;
    }
    function run(cfg, patch, P) {  // one patch, serialised: a session runs one inference at a time
        var job = state.queue.then(function () {
            return load(cfg).then(function (s) {
                var feeds = {};
                feeds[s.session.inputNames[0]] = new s.ort.Tensor("float32", new Float32Array(patch), [1, 3, P, P]);
                return s.session.run(feeds).then(function (res) {
                    var t = res[s.session.outputNames[0]];
                    return t.location && t.location !== "cpu" ? t.getData(true) : t.data;
                });
            });
        });
        state.queue = job.catch(function () {});
        return job;
    }
    function cacheGet(key) {
        var c = cache();
        if (!c.has(key)) return null;
        var e = c.get(key);
        c.delete(key); c.set(key, e);
        return e;
    }
    function cachePut(key, blob, max) {
        var c = cache();
        c.delete(key);
        c.set(key, {blob: new w.Blob([blob], {type: blob.type}), url: null});
        while (c.size > max) {
            var oldest = c.keys().next().value, e = c.get(oldest);
            if (e.url) { try { URL.revokeObjectURL(e.url); } catch (err) {} }
            c.delete(oldest);
        }
    }
    w.__roofSr = {load: load, run: run, cacheGet: cacheGet, cachePut: cachePut};
})(window);
"""


class SuperResScript(MacroElement):
    """Upscale the roof crop (`bbox`: [[s, w], [n, e]], the offset-shifted footprint bbox + margin) and show it over
    exactly that crop. Runs DEBOUNCE_MS after whenReady and after every "roofimagerychange" (fired by core/wayback);
    a superseded run is discarded."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            {{ this.patch_js }}
            var map = {{ this._parent.get_name() }}, tiles = {{ this.tile_layer_name }};
            var BBOX = {{ this.bbox|tojson }}, CFG = {{ this.cfg|tojson }};
            var PARENT_JS = {{ this.parent_js|tojson }};
            var NZ = {{ this.native }}, TILE = 256, S = {{ this.scale }}, P = {{ this.patch }}, OVERLAP = {{ this.overlap }};
            var MAX_CROP_PX = {{ this.max_crop }}, CACHE_MAX = {{ this.cache_max }};
            var MODEL_TIMEOUT_MS = {{ this.model_timeout }}, INFER_TIMEOUT_MS = {{ this.infer_timeout }};
            var DEBOUNCE_MS = {{ this.debounce }}, MODEL_KEY = {{ this.model_key|tojson }};
            var TEXT = {{ this.texts|tojson }};
            var overlay = null, overlayUrl = null, status = null, timer = null, runId = 0, warned = false;

            function host() {  // the parent window when reachable (the session outlives this iframe), else this one
                var w = null;
                try { w = window.parent; void w.document.head; } catch (err) { w = null; }
                w = w || window;
                try {
                    if (!w.__roofSr) {
                        var s = w.document.createElement("script");
                        s.textContent = PARENT_JS;
                        w.document.head.appendChild(s);
                    }
                    if (w.__roofSr) return w.__roofSr;
                } catch (err) {}
                if (w !== window) {
                    var s2 = document.createElement("script");
                    s2.textContent = PARENT_JS;
                    document.head.appendChild(s2);
                }
                return window.__roofSr;
            }
            function imageryUrl() { return map.roofImageryUrl !== undefined ? map.roofImageryUrl : tiles._url; }
            function withTimeout(promise, ms, what) {
                return new Promise(function (resolve, reject) {
                    var t = setTimeout(function () { reject(new Error(what + " timed out after " + ms + " ms")); }, ms);
                    promise.then(function (v) { clearTimeout(t); resolve(v); }, function (e) { clearTimeout(t); reject(e); });
                });
            }
            var Status = L.Control.extend({
                options: {position: "topright"},
                onAdd: function () {
                    var div = L.DomUtil.create("div", "leaflet-bar roof-superres");
                    div.style.cssText = "background:#fff;padding:4px 6px;font:12px/1.4 sans-serif;color:#111;max-width:240px;";
                    L.DomEvent.disableClickPropagation(div);
                    L.DomEvent.disableScrollPropagation(div);
                    return div;
                }
            });
            function setStatus(text) {
                if (!text) { if (status) { map.removeControl(status); status = null; } return; }
                if (!status) { status = new Status(); status.addTo(map); }
                status.getContainer().textContent = text;
            }
            function revoke(url) { if (url) { try { URL.revokeObjectURL(url); } catch (err) {} } }
            function removeOverlay() {
                if (overlay) { map.removeLayer(overlay); overlay = null; }
                revoke(overlayUrl); overlayUrl = null;
            }
            function fail(err) {  // back to the plain tiles
                removeOverlay();
                setStatus(null);
                if (!warned) { warned = true; console.warn("roofSuperRes: off, original tiles shown:", err); }
            }
            function unsupported(reason) {  // no WebGPU: notice, plain tiles, nothing else runs
                removeOverlay();
                setStatus(TEXT.noWebGPU);
                console.info("roofSuperRes: WebGPU unavailable (" + reason + "), original tiles shown");
            }
            function show(blob, entry, nw, se) {  // `entry`: the cache entry that owns the URL, or null
                removeOverlay();
                var url = URL.createObjectURL(blob);
                if (entry) { revoke(entry.url); entry.url = url; } else { overlayUrl = url; }
                var pane = map.getPane("{{ this.pane }}") || map.createPane("{{ this.pane }}");
                pane.style.zIndex = 250;  // above the tiles (200), under the drawings (400)
                pane.style.pointerEvents = "none";
                overlay = L.imageOverlay(url, L.latLngBounds(nw, se), {pane: "{{ this.pane }}", interactive: false});
                overlay.on("error", function (e) { fail(new Error("overlay image failed")); });
                overlay.addTo(map);
            }
            function loadTile(url, x, y) {
                return new Promise(function (resolve, reject) {
                    var img = new Image();
                    img.crossOrigin = "anonymous";
                    img.onload = function () { resolve({img: img, x: x, y: y}); };
                    img.onerror = function () { reject(new Error("tile load failed " + NZ + "/" + y + "/" + x)); };
                    img.src = L.Util.template(url, {z: NZ, x: x, y: y});
                });
            }
            function loadCrop(url, x0, y0, W, H) {  // RGBA of the crop from the stitched z19 tiles
                var tx0 = Math.floor(x0 / TILE), ty0 = Math.floor(y0 / TILE);
                var tx1 = Math.floor((x0 + W - 1) / TILE), ty1 = Math.floor((y0 + H - 1) / TILE), jobs = [];
                for (var ty = ty0; ty <= ty1; ty++) for (var tx = tx0; tx <= tx1; tx++) jobs.push(loadTile(url, tx, ty));
                return Promise.all(jobs).then(function (loaded) {
                    var c = document.createElement("canvas");
                    c.width = W; c.height = H;
                    var ctx = c.getContext("2d");
                    loaded.forEach(function (t) { ctx.drawImage(t.img, t.x * TILE - x0, t.y * TILE - y0); });
                    return ctx.getImageData(0, 0, W, H).data;  // throws on a tainted canvas
                });
            }
            function toBlob(r) {
                var c = document.createElement("canvas");
                c.width = r.width; c.height = r.height;
                var ctx = c.getContext("2d"), img = ctx.createImageData(r.width, r.height);
                img.data.set(r.data);
                ctx.putImageData(img, 0, 0);
                return new Promise(function (resolve, reject) {
                    c.toBlob(function (b) { b ? resolve(b) : reject(new Error("toBlob returned null")); }, "image/png");
                });
            }
            function schedule() {
                clearTimeout(timer);
                runId++;  // a run in progress is superseded
                timer = setTimeout(run, DEBOUNCE_MS);
            }
            function run() {
                var me = runId, cancelled = false, alive = function () { return me === runId && !cancelled; };
                try {
                    var p0 = map.project(L.latLng(BBOX[1][0], BBOX[0][1]), NZ), p1 = map.project(L.latLng(BBOX[0][0], BBOX[1][1]), NZ);
                    var x0 = Math.floor(p0.x), y0 = Math.floor(p0.y), W = Math.ceil(p1.x) - x0, H = Math.ceil(p1.y) - y0;
                    if (W > MAX_CROP_PX || H > MAX_CROP_PX) {
                        removeOverlay();
                        setStatus(TEXT.tooBig);
                        console.warn("roofSuperRes: crop " + W + "x" + H + " px is above " + MAX_CROP_PX + " px, skipped");
                        return;
                    }
                    var url = imageryUrl(), key = [url, NZ, x0, y0, W, H, MODEL_KEY].join("|");
                    var nw = map.unproject(L.point(x0, y0), NZ), se = map.unproject(L.point(x0 + W, y0 + H), NZ);
                    var sr = host();
                    var hit = sr.cacheGet(key);
                    if (hit) { setStatus(null); show(hit.blob, hit, nw, se); return; }
                    removeOverlay();  // never show an overlay of other imagery
                    if (!navigator.gpu) { unsupported("navigator.gpu missing"); return; }  // no WASM fallback
                    setStatus(TEXT.progress);
                    var t0 = performance.now(), t1, provider;
                    Promise.all([loadCrop(url, x0, y0, W, H), withTimeout(sr.load(CFG), MODEL_TIMEOUT_MS, "model load")])
                        .then(function (res) {
                            if (!alive()) return null;
                            provider = res[1].provider;
                            t1 = performance.now();
                            var infer = function (patch) { return sr.run(CFG, patch, P); };
                            return withTimeout(upscaleCrop(res[0], W, H, P, OVERLAP, S, infer, alive), INFER_TIMEOUT_MS, "inference")
                                .catch(function (err) { cancelled = true; throw err; });  // no further patches
                        })
                        .then(function (r) {
                            if (!r || !alive()) return null;
                            var t2 = performance.now();
                            console.info("roofSuperRes: provider " + provider + ", crop " + W + "x" + H + " -> " + r.width + "x"
                                + r.height + ", " + patchCount(W, H, P, OVERLAP) + " patches, inference " + Math.round(t2 - t1)
                                + " ms, total " + Math.round(t2 - t0) + " ms");
                            return toBlob(r).then(function (blob) {
                                if (!alive()) return;
                                sr.cachePut(key, blob, CACHE_MAX);
                                var entry = sr.cacheGet(key);
                                setStatus(null);
                                show(entry ? entry.blob : blob, entry, nw, se);
                            });
                        })
                        .catch(function (err) {
                            if (me !== runId) return;
                            if (err && err.noWebGPU) unsupported(err.message); else fail(err);
                        });
                } catch (err) {
                    fail(err);
                }
            }
            map.whenReady(schedule);
            map.on("roofimagerychange", function () { removeOverlay(); schedule(); });
        })();
        {% endmacro %}
        """
    )

    def __init__(self, tile_layer, bbox):
        super().__init__()
        self._name = "SuperResScript"
        self.patch_js, self.parent_js, self.tile_layer_name = PATCH_JS, PARENT_JS, tile_layer.get_name()
        self.bbox = [[float(v) for v in corner] for corner in bbox]
        self.cfg = {
            "ortScript": ORT_SCRIPT, "wasmPaths": ORT_DIST, "modelUrl": MODEL_URL, "dataUrl": MODEL_DATA_URL,
            "dataName": MODEL_DATA_NAME,
        }
        self.native, self.scale, self.patch, self.overlap = NATIVE_ZOOM, SCALE, PATCH, OVERLAP
        self.max_crop, self.cache_max = MAX_CROP_PX, CACHE_MAX
        self.model_timeout, self.infer_timeout, self.debounce = MODEL_TIMEOUT_MS, INFER_TIMEOUT_MS, DEBOUNCE_MS
        self.model_key, self.pane = MODEL_SHA256[:12], PANE
        self.texts = {"progress": PROGRESS_TEXT, "tooBig": TOO_BIG_TEXT, "noWebGPU": NO_WEBGPU_TEXT}


def add_to_map(m, tile_layer, bbox):
    """Add the super-resolution script to the folium map `m`. `tile_layer`: the Esri folium.TileLayer (its URL is the
    fallback when core/wayback has not set map.roofImageryUrl); `bbox`: [[s, w], [n, e]] of the crop in WGS84."""
    m.add_child(SuperResScript(tile_layer, bbox))
