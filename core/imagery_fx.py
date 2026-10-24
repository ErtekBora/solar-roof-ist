"""Browser-side display enhancement of the roof map's Esri tiles: one canvas script (DISPLAY ONLY).

Privacy/licence rule: Python never fetches, stores or reads a tile. The script runs in the user's browser, in memory
only: it loads the native-zoom tiles covering the static view (crossOrigin "anonymous", same URL as the tile layer),
stitches them into an offscreen canvas, processes the pixels and shows the result as an image overlay (blob URL)
under the drawings. Nothing is uploaded, stored or sent to any model. The normal tile layer stays as the fallback and
the OFF state; on any failure the overlay is removed and the tiles are shown again.

Pipeline (luminance only, colour from the original):
  1x native: RGB -> YCbCr (BT.601); Y: self-guided filter (edge-preserving denoise/deblock), then CLAHE blended with
  the denoised Y; back to RGB with Cb/Cr x CHROMA_BOOST.
  OUTPUT_SCALE x: drawImage upscale (imageSmoothingQuality "high"), then a soft-threshold unsharp mask on Y.
The tunable constants are at the top of PIPELINE_JS.
Docs: https://leafletjs.com/reference.html#tilelayer-crossorigin  https://leafletjs.com/reference.html#map-unproject
https://leafletjs.com/reference.html#map-createpane  https://leafletjs.com/reference.html#imageoverlay
https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/getImageData
"""
from branca.element import MacroElement
from jinja2 import Template

from core.config import ESRI_TILES_URL

NATIVE_ZOOM = 19  # must match app.ESRI_NATIVE_ZOOM
PANE = "roofEnhanced"
CACHE_MAX = 4  # processed views kept in browser memory (window.parent.__roofFxCache)
CAPTION = (
    "İyileştirme yalnızca tarayıcınızda, bellekte yapılır (gürültü azaltma, yerel kontrast, keskinleştirme). Görüntü "
    "sunucuya gönderilmez, saklanmaz ve yapay zekâya verilmez. Gerçek çözünürlüğü artırmaz."
)

# Pure pixel functions (no DOM, no Leaflet): processNative(rgba, W, H) and sharpenScaled(rgba, W, H) work in place on
# ImageData.data. No Jinja syntax inside, so the same text can be run outside the map (e.g. with Node).
PIPELINE_JS = """
var GUIDED_RADIUS = 2;            // guided filter box radius (native px)
var GUIDED_EPS = 0.02 * 0.02;     // guided filter regulariser (Y on 0..1)
var CLAHE_GRID_X = 8;             // CLAHE cells across the mosaic
var CLAHE_GRID_Y = 4;             // CLAHE cells down the mosaic
var CLAHE_CLIP = 2.5;             // histogram clip limit (x the uniform bin height)
var CLAHE_BLEND = 0.7;            // share of CLAHE in the result (rest: denoised Y)
var USM_SIGMA = 1.2;              // unsharp-mask Gaussian sigma (output px)
var USM_AMOUNT = 1.0;             // unsharp-mask strength
var USM_THRESHOLD = 2 / 255;      // unsharp-mask soft threshold (Y on 0..1)
var OUTPUT_SCALE = 2;             // overlay resolution / native resolution
var CHROMA_BOOST = 1.05;          // Cb/Cr scale around 0 (keep <= 1.1)

function clamp01(v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

function splitYCbCr(rgba, n) {  // BT.601 full range, all channels on 0..1, Cb/Cr centred at 0
    var Y = new Float32Array(n), Cb = new Float32Array(n), Cr = new Float32Array(n);
    for (var i = 0, j = 0; i < n; i++, j += 4) {
        var r = rgba[j] / 255, g = rgba[j + 1] / 255, b = rgba[j + 2] / 255;
        Y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        Cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b;
        Cr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b;
    }
    return {Y: Y, Cb: Cb, Cr: Cr};
}

function mergeYCbCr(rgba, n, Y, Cb, Cr, chroma) {
    for (var i = 0, j = 0; i < n; i++, j += 4) {
        var y = Y[i], cb = Cb[i] * chroma, cr = Cr[i] * chroma;
        rgba[j] = Math.round(clamp01(y + 1.402 * cr) * 255);
        rgba[j + 1] = Math.round(clamp01(y - 0.344136 * cb - 0.714136 * cr) * 255);
        rgba[j + 2] = Math.round(clamp01(y + 1.772 * cb) * 255);
        rgba[j + 3] = 255;
    }
}

// Summed-area tables are Float64Array on purpose: the guided filter needs var = mean(Y^2) - mean(Y)^2, a small
// difference of two large running sums. In Float32 (24-bit mantissa) the sums over a ~1000 x 600 mosaic lose the
// low digits, the subtraction cancels them and var turns into noise or goes negative. Pixel buffers stay Float32.
function integral(src, W, H, squared) {
    var W1 = W + 1, S = new Float64Array(W1 * (H + 1));
    for (var y = 0; y < H; y++) {
        var row = 0, o = (y + 1) * W1, p = y * W1, s = y * W;
        for (var x = 0; x < W; x++) {
            var v = src[s + x];
            row += squared ? v * v : v;
            S[o + x + 1] = S[p + x + 1] + row;
        }
    }
    return S;
}

function boxSum(S, W1, x0, y0, x1, y1) { return S[y1 * W1 + x1] - S[y0 * W1 + x1] - S[y1 * W1 + x0] + S[y0 * W1 + x0]; }

function boxMean(src, W, H, r) {  // O(N) box mean with a window clipped at the borders
    var S = integral(src, W, H, false), W1 = W + 1, out = new Float32Array(W * H);
    for (var y = 0; y < H; y++) {
        var y0 = Math.max(0, y - r), y1 = Math.min(H, y + r + 1);
        for (var x = 0; x < W; x++) {
            var x0 = Math.max(0, x - r), x1 = Math.min(W, x + r + 1);
            out[y * W + x] = boxSum(S, W1, x0, y0, x1, y1) / ((y1 - y0) * (x1 - x0));
        }
    }
    return out;
}

function guidedSelf(Y, W, H, r, eps) {  // He et al. guided filter with I = p = Y
    var S1 = integral(Y, W, H, false), S2 = integral(Y, W, H, true), W1 = W + 1, n = W * H;
    var A = new Float32Array(n), B = new Float32Array(n);
    for (var y = 0; y < H; y++) {
        var y0 = Math.max(0, y - r), y1 = Math.min(H, y + r + 1);
        for (var x = 0; x < W; x++) {
            var x0 = Math.max(0, x - r), x1 = Math.min(W, x + r + 1), cnt = (y1 - y0) * (x1 - x0);
            var mean = boxSum(S1, W1, x0, y0, x1, y1) / cnt;  // doubles until here
            var v = Math.max(0, boxSum(S2, W1, x0, y0, x1, y1) / cnt - mean * mean);
            var a = v / (v + eps);
            A[y * W + x] = a;
            B[y * W + x] = mean - a * mean;
        }
    }
    var mA = boxMean(A, W, H, r), mB = boxMean(B, W, H, r), q = new Float32Array(n);
    for (var i = 0; i < n; i++) q[i] = clamp01(mA[i] * Y[i] + mB[i]);
    return q;
}

function clahe(Y, W, H, gx, gy, clip) {  // 256-bin CLAHE, bilinear interpolation between cell LUTs
    var cw = W / gx, ch = H / gy, luts = [], cx, cy;
    for (cy = 0; cy < gy; cy++) {
        for (cx = 0; cx < gx; cx++) {
            var xa = Math.floor(cx * cw), xb = Math.floor((cx + 1) * cw), ya = Math.floor(cy * ch), yb = Math.floor((cy + 1) * ch);
            var hist = new Float64Array(256), count = (xb - xa) * (yb - ya);
            for (var y = ya; y < yb; y++) for (var x = xa; x < xb; x++) hist[Math.min(255, Math.floor(Y[y * W + x] * 256))]++;
            var limit = Math.max(1, clip * count / 256), excess = 0, k;
            for (k = 0; k < 256; k++) if (hist[k] > limit) { excess += hist[k] - limit; hist[k] = limit; }
            var lut = new Float32Array(256), cdf = 0, add = excess / 256;
            for (k = 0; k < 256; k++) { cdf += hist[k] + add; lut[k] = count > 0 ? cdf / count : k / 255; }
            luts.push(lut);
        }
    }
    function axis(len, cell, cells) {  // per-coordinate lower cell index and weight
        var i0 = new Int32Array(len), f = new Float32Array(len);
        for (var t = 0; t < len; t++) {
            var g = (t + 0.5) / cell - 0.5, i = Math.floor(g), w = g - i;
            if (i < 0) { i = 0; w = 0; }
            if (i >= cells - 1) { i = cells - 1; w = 0; }
            i0[t] = i; f[t] = w;
        }
        return {i0: i0, f: f};
    }
    var ax = axis(W, cw, gx), ay = axis(H, ch, gy), out = new Float32Array(W * H);
    for (var py = 0; py < H; py++) {
        var r0 = ay.i0[py], r1 = Math.min(r0 + 1, gy - 1), fy = ay.f[py];
        for (var px = 0; px < W; px++) {
            var c0 = ax.i0[px], c1 = Math.min(c0 + 1, gx - 1), fx = ax.f[px];
            var b = Math.min(255, Math.floor(Y[py * W + px] * 256));
            var top = luts[r0 * gx + c0][b] * (1 - fx) + luts[r0 * gx + c1][b] * fx;
            var bot = luts[r1 * gx + c0][b] * (1 - fx) + luts[r1 * gx + c1][b] * fx;
            out[py * W + px] = top * (1 - fy) + bot * fy;
        }
    }
    return out;
}

function gaussianBlur(src, W, H, sigma) {  // separable, edges clamped
    var rad = Math.max(1, Math.ceil(3 * sigma)), k = new Float32Array(2 * rad + 1), sum = 0, i;
    for (i = -rad; i <= rad; i++) { k[i + rad] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + rad]; }
    for (i = 0; i < k.length; i++) k[i] /= sum;
    var tmp = new Float32Array(W * H), out = new Float32Array(W * H), x, y, t, acc;
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            acc = 0;
            for (t = -rad; t <= rad; t++) acc += k[t + rad] * src[y * W + Math.min(W - 1, Math.max(0, x + t))];
            tmp[y * W + x] = acc;
        }
    }
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            acc = 0;
            for (t = -rad; t <= rad; t++) acc += k[t + rad] * tmp[Math.min(H - 1, Math.max(0, y + t)) * W + x];
            out[y * W + x] = acc;
        }
    }
    return out;
}

function processNative(rgba, W, H) {  // 1x: denoise + local contrast on Y, chroma from the original
    var n = W * H, c = splitYCbCr(rgba, n);
    var q = guidedSelf(c.Y, W, H, GUIDED_RADIUS, GUIDED_EPS);
    var e = clahe(q, W, H, CLAHE_GRID_X, CLAHE_GRID_Y, CLAHE_CLIP);
    for (var i = 0; i < n; i++) e[i] = CLAHE_BLEND * e[i] + (1 - CLAHE_BLEND) * q[i];
    mergeYCbCr(rgba, n, e, c.Cb, c.Cr, CHROMA_BOOST);
}

function sharpenScaled(rgba, W, H) {  // OUTPUT_SCALE x: soft-threshold unsharp mask on Y, Cb/Cr unchanged
    var n = W * H, c = splitYCbCr(rgba, n), blur = gaussianBlur(c.Y, W, H, USM_SIGMA), Y = c.Y;
    for (var i = 0; i < n; i++) {
        var d = Y[i] - blur[i];
        var detail = Math.sign(d) * Math.max(0, Math.abs(d) - USM_THRESHOLD);
        Y[i] = Y[i] + USM_AMOUNT * detail;
    }
    mergeYCbCr(rgba, n, Y, c.Cb, c.Cr, 1);
}
"""


class CanvasEnhanceScript(MacroElement):
    """After fitBounds has been applied (whenReady + setTimeout 0): show the enhanced overlay once for the static
    view, from the in-memory cache when the same tile range and constants were processed before. Falls back to the
    plain tiles (one console.warn) on any failure.

    Cache: a Map on window.parent (`__roofFxCache`, at most CACHE_MAX entries, oldest dropped), plain JS memory only,
    gone when the page is closed. Every map rebuild is a new iframe and a blob URL dies with the document that made
    it, so the cache keeps a parent-realm copy of the PNG Blob and each rebuild makes its own URL from it."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            {{ this.pipeline }}
            var NZ = {{ this.native }};
            var TILE = 256, MAX_TILES = 400, CACHE_MAX = {{ this.cache_max }};
            var TILE_URL = {{ this.tile_url|tojson }};
            var map = {{ this._parent.get_name() }};
            var tiles = {{ this.tile_layer_name }};
            var overlay = null, overlayUrl = null, failed = false;
            var PARAMS_HASH = hashString([GUIDED_RADIUS, GUIDED_EPS, CLAHE_GRID_X, CLAHE_GRID_Y, CLAHE_CLIP, CLAHE_BLEND,
                USM_SIGMA, USM_AMOUNT, USM_THRESHOLD, OUTPUT_SCALE, CHROMA_BOOST].join(","));

            function hashString(s) {  // djb2-xor, enough to tell constant sets apart
                var h = 5381;
                for (var i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0;
                return h.toString(16);
            }
            function revoke(url) { if (url) { try { URL.revokeObjectURL(url); } catch (err) {} } }
            function getCache() {  // null when window.parent is unreachable: then no cache
                try {
                    var w = window.parent;
                    if (!w.__roofFxCache) w.__roofFxCache = new w.Map();
                    return w.__roofFxCache;
                } catch (err) { return null; }
            }
            function cacheGet(key) {  // a hit becomes the newest entry
                try {
                    var c = getCache();
                    if (!c || !c.has(key)) return null;
                    var entry = c.get(key);
                    c.delete(key); c.set(key, entry);
                    return entry;
                } catch (err) { return null; }
            }
            function cachePut(key, blob) {
                try {
                    var c = getCache(), w = window.parent;
                    if (!c) return;
                    c.delete(key);
                    c.set(key, {blob: new w.Blob([blob], {type: blob.type}), url: null});  // outlives this iframe
                    while (c.size > CACHE_MAX) {
                        var oldest = c.keys().next().value;
                        revoke(c.get(oldest).url);
                        c.delete(oldest);
                    }
                } catch (err) {}
            }
            function cacheDrop(key) {
                try {
                    var c = getCache();
                    if (c && c.has(key)) { revoke(c.get(key).url); c.delete(key); }
                } catch (err) {}
            }
            function setTileOpacity(v) {
                if (tiles) { tiles.setOpacity(v); return; }
                map.eachLayer(function (l) { if (l instanceof L.TileLayer) l.setOpacity(v); });
            }
            function removeOverlay() {
                if (overlay) { map.removeLayer(overlay); overlay = null; }
                revoke(overlayUrl); overlayUrl = null;
            }
            function fail(err) {  // back to the original tiles
                removeOverlay();
                setTileOpacity(1);
                if (!failed) { failed = true; console.warn("roofEnhanced: enhancement off, original tiles shown:", err); }
            }
            function loadTile(x, y) {
                return new Promise(function (resolve, reject) {
                    var img = new Image();
                    img.crossOrigin = "anonymous";
                    img.onload = function () { resolve({img: img, x: x, y: y}); };
                    img.onerror = function () { reject(new Error("tile load failed " + NZ + "/" + y + "/" + x)); };
                    img.src = L.Util.template(TILE_URL, {z: NZ, x: x, y: y});
                });
            }
            function show(blob, nw, se, entry, onError) {  // `entry`: the cache entry that owns the URL, or null
                if (!blob) throw new Error("toBlob returned null");
                removeOverlay();
                var url = URL.createObjectURL(blob);
                if (entry) { revoke(entry.url); entry.url = url; } else { overlayUrl = url; }
                var pane = map.getPane("{{ this.pane }}") || map.createPane("{{ this.pane }}");
                pane.style.zIndex = 250;
                pane.style.pointerEvents = "none";
                overlay = L.imageOverlay(url, L.latLngBounds(nw, se), {pane: "{{ this.pane }}", interactive: false});
                overlay.on("load", function () { if (!failed) setTileOpacity(0); });
                overlay.on("error", onError);
                overlay.addTo(map);
            }
            function build(key, xMin, yMin, xMax, yMax, nw, se) {
                var nx = xMax - xMin + 1, ny = yMax - yMin + 1;
                if (nx * ny > MAX_TILES) throw new Error("too many tiles: " + nx * ny);
                var jobs = [];
                for (var ty = yMin; ty <= yMax; ty++) for (var tx = xMin; tx <= xMax; tx++) jobs.push(loadTile(tx, ty));
                Promise.all(jobs).then(function (loaded) {
                    var t0 = performance.now(), W = nx * TILE, H = ny * TILE;
                    var mosaic = document.createElement("canvas");
                    mosaic.width = W; mosaic.height = H;
                    var mctx = mosaic.getContext("2d");
                    loaded.forEach(function (t) { mctx.drawImage(t.img, (t.x - xMin) * TILE, (t.y - yMin) * TILE); });
                    var native = mctx.getImageData(0, 0, W, H);  // throws on a tainted canvas
                    processNative(native.data, W, H);
                    mctx.putImageData(native, 0, 0);
                    var out = document.createElement("canvas");
                    out.width = W * OUTPUT_SCALE; out.height = H * OUTPUT_SCALE;
                    var octx = out.getContext("2d");
                    octx.imageSmoothingEnabled = true;
                    octx.imageSmoothingQuality = "high";
                    octx.drawImage(mosaic, 0, 0, out.width, out.height);
                    var scaled = octx.getImageData(0, 0, out.width, out.height);
                    sharpenScaled(scaled.data, out.width, out.height);
                    octx.putImageData(scaled, 0, 0);
                    console.info("roofEnhanced: " + W + "x" + H + " -> " + out.width + "x" + out.height + " in "
                        + Math.round(performance.now() - t0) + " ms");
                    out.toBlob(function (blob) {
                        try {
                            if (!blob) throw new Error("toBlob returned null");
                            cachePut(key, blob);
                            var entry = cacheGet(key);
                            show(entry ? entry.blob : blob, nw, se, entry, fail);
                        } catch (err) { fail(err); }
                    }, "image/png");
                }).catch(fail);
            }
            function roofEnhance() {
                try {
                    var b = map.getBounds(), p0 = map.project(b.getNorthWest(), NZ), p1 = map.project(b.getSouthEast(), NZ);
                    var xMin = Math.floor(p0.x / TILE) - 1, yMin = Math.floor(p0.y / TILE) - 1;
                    var xMax = Math.floor(p1.x / TILE) + 1, yMax = Math.floor(p1.y / TILE) + 1;
                    var nw = map.unproject(L.point(xMin * TILE, yMin * TILE), NZ);
                    var se = map.unproject(L.point((xMax + 1) * TILE, (yMax + 1) * TILE), NZ);
                    var key = [NZ, xMin, xMax, yMin, yMax, PARAMS_HASH].join("|");
                    var hit = cacheGet(key);
                    if (hit) {
                        try {
                            show(hit.blob, nw, se, hit, function () {  // a stale entry: drop it and process again
                                cacheDrop(key);
                                removeOverlay();
                                try { build(key, xMin, yMin, xMax, yMax, nw, se); } catch (err) { fail(err); }
                            });
                            console.info("roofEnhanced: cached result reused (" + key + ")");
                            return;
                        } catch (err) {
                            cacheDrop(key);
                        }
                    }
                    build(key, xMin, yMin, xMax, yMax, nw, se);
                } catch (err) {
                    fail(err);
                }
            }
            map.whenReady(function () { setTimeout(roofEnhance, 0); });
        })();
        {% endmacro %}
        """
    )

    def __init__(self, tile_layer=None):
        super().__init__()
        self._name = "CanvasEnhanceScript"
        self.pipeline, self.native, self.tile_url, self.pane = PIPELINE_JS, NATIVE_ZOOM, ESRI_TILES_URL, PANE
        self.cache_max = CACHE_MAX
        self.tile_layer_name = tile_layer.get_name() if tile_layer is not None else "null"


def add_to_map(m, tile_layer=None):
    """Add the canvas enhancement script to the folium map `m` (call AFTER fit_bounds). `tile_layer` is the Esri
    folium.TileLayer to hide under the overlay; without it every Leaflet tile layer of the map is hidden/restored."""
    m.add_child(CanvasEnhanceScript(tile_layer))
