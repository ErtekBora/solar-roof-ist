"""Browser-side display enhancement of the roof map's Esri tiles: one canvas script (DISPLAY ONLY).

Privacy/licence rule: Python never fetches, stores or reads a tile. The script runs in the user's browser, in memory
only: it loads the native-zoom tiles covering the static view (crossOrigin "anonymous", same URL as the tile layer),
stitches them into an offscreen canvas, processes the pixels and shows the result as an image overlay (blob URL)
under the drawings. Nothing is uploaded, stored or sent to any model. The normal tile layer stays as the fallback and
the OFF state; on any failure the overlay is removed and the tiles are shown again.

Pipeline, all in Float32 YCbCr (BT.601): RGB -> YCbCr once at the start, YCbCr -> RGB once at the end.
  1x native, on Y: light self-guided filter (softens JPEG block edges), CLAHE blended with it, local clarity
  (large-sigma unsharp), soft-threshold pre-sharpen.
  S x (power of two from the fitted zoom, capped by MAX_OUTPUT_PIXELS): Y by separable Lanczos3 (or bilinear),
  Cb/Cr by separable bilinear; then a fine and a medium soft-threshold unsharp band on Y; Cb/Cr x CHROMA_BOOST.
Anti-ringing: after every sharpening stage and after Lanczos3, Y is clamped to the 3x3 min/max of that stage's input
(for Lanczos3: of a bilinear upscale) +- RING_MARGIN, capped by the 3x3 min/max of the unsharpened Y, so edges get
steeper without dark/bright halos.
No S-curve, gamma or saturation. The tunable constants are at the top of PIPELINE_JS.
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

# Pure pixel functions (no DOM, no Leaflet): enhance(rgba, W, H, S) returns the S x RGBA result. No Jinja syntax
# inside, so the same text can be run outside the map (e.g. with Node).
PIPELINE_JS = """
var TILE_MARGIN = 0;                  // native tiles added around the view on every side
var MAX_MOSAIC_PIXELS = 1600 * 1200;  // 1x mosaic cap; above it the plain tiles stay
var MAX_OUTPUT_PIXELS = 2048 * 2048;  // output cap; S is halved until W * H * S * S fits
var GUIDED_RADIUS = 1;                // guided filter box radius (native px)
var GUIDED_EPS = 0.01 * 0.01;         // guided filter regulariser (Y on 0..1)
var CLAHE_GRID_X = 8;                 // CLAHE cells across the mosaic
var CLAHE_GRID_Y = 4;                 // CLAHE cells down the mosaic
var CLAHE_CLIP = 1.8;                 // histogram clip limit (x the uniform bin height)
var CLAHE_BLEND = 0.35;               // share of CLAHE in the result (rest: guided Y)
var CLARITY_SIGMA = 10;               // local clarity blur sigma (native px)
var CLARITY_AMOUNT = 0.15;            // local clarity strength
var PRE_USM_SIGMA = 0.75;             // pre-sharpen sigma (native px)
var PRE_USM_AMOUNT = 0.5;             // pre-sharpen strength
var PRE_USM_THRESHOLD = 1 / 255;      // pre-sharpen soft threshold
var UPSCALER = "lanczos3";            // Y upscaler: "lanczos3" or "bilinear" (Cb/Cr always bilinear)
var USM_FINE_SIGMA = 0.5;             // fine band sigma, x S (output px)
var USM_FINE_AMOUNT = 0.7;            // fine band strength
var USM_MED_SIGMA = 1.4;              // medium band sigma, x S (output px)
var USM_MED_AMOUNT = 0.3;             // medium band strength
var USM_THRESHOLD = 1 / 255;          // soft threshold of both bands
var RING_MARGIN = 2 / 255;            // anti-ringing: allowed excursion beyond the input's 3x3 min/max
var CHROMA_BOOST = 1.05;              // Cb/Cr scale around 0 (keep <= 1.1)

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

function mergeYCbCr(Y, Cb, Cr, n, chroma) {  // the only conversion back to 8 bit (clamped, alpha 255)
    var rgba = new Uint8ClampedArray(n * 4);
    for (var i = 0, j = 0; i < n; i++, j += 4) {
        var y = Y[i], cb = Cb[i] * chroma, cr = Cr[i] * chroma;
        rgba[j] = (y + 1.402 * cr) * 255 + 0.5;
        rgba[j + 1] = (y - 0.344136 * cb - 0.714136 * cr) * 255 + 0.5;
        rgba[j + 2] = (y + 1.772 * cb) * 255 + 0.5;
        rgba[j + 3] = 255;
    }
    return rgba;
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
    for (var i = 0; i < n; i++) q[i] = mA[i] * Y[i] + mB[i];
    return q;
}

function bin256(v) { var b = Math.floor(v * 256); return b < 0 ? 0 : (b > 255 ? 255 : b); }

function clahe(Y, W, H, gx, gy, clip) {  // 256-bin CLAHE, bilinear interpolation between cell LUTs
    var cw = W / gx, ch = H / gy, luts = [], cx, cy;
    for (cy = 0; cy < gy; cy++) {
        for (cx = 0; cx < gx; cx++) {
            var xa = Math.floor(cx * cw), xb = Math.floor((cx + 1) * cw), ya = Math.floor(cy * ch), yb = Math.floor((cy + 1) * ch);
            var hist = new Float64Array(256), count = (xb - xa) * (yb - ya);
            for (var y = ya; y < yb; y++) for (var x = xa; x < xb; x++) hist[bin256(Y[y * W + x])]++;
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
            var b = bin256(Y[py * W + px]);
            var top = luts[r0 * gx + c0][b] * (1 - fx) + luts[r0 * gx + c1][b] * fx;
            var bot = luts[r1 * gx + c0][b] * (1 - fx) + luts[r1 * gx + c1][b] * fx;
            out[py * W + px] = top * (1 - fy) + bot * fy;
        }
    }
    return out;
}

function gaussianBlur(src, W, H, sigma) {  // separable 1D Gaussian kernels, edges clamped (sigma < 1.5)
    var rad = Math.max(1, Math.ceil(3 * sigma)), k = new Float32Array(2 * rad + 1), sum = 0, i;
    for (i = -rad; i <= rad; i++) { k[i + rad] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + rad]; }
    for (i = 0; i < k.length; i++) k[i] /= sum;
    var tmp = new Float32Array(W * H), out = new Float32Array(W * H), klen = k.length, x, y, t, acc, o;
    for (y = 0; y < H; y++) {
        o = y * W;
        for (x = 0; x < W; x++) {
            acc = 0;
            if (x >= rad && x < W - rad) {  // interior: no clamping
                var base = o + x - rad;
                for (t = 0; t < klen; t++) acc += k[t] * src[base + t];
            } else {
                for (t = -rad; t <= rad; t++) acc += k[t + rad] * src[o + Math.min(W - 1, Math.max(0, x + t))];
            }
            tmp[o + x] = acc;
        }
    }
    for (y = 0; y < H; y++) {
        o = y * W;
        for (t = -rad; t <= rad; t++) {
            var kt = k[t + rad], row = Math.min(H - 1, Math.max(0, y + t)) * W;
            for (x = 0; x < W; x++) out[o + x] += kt * tmp[row + x];
        }
    }
    return out;
}

function boxesForGauss(sigma, n) {  // n box widths whose repeated box blur approximates a Gaussian of sigma
    var wIdeal = Math.sqrt(12 * sigma * sigma / n + 1), wl = Math.floor(wIdeal);
    if (wl % 2 === 0) wl--;
    var wu = wl + 2, m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
    var sizes = [];
    for (var i = 0; i < n; i++) sizes.push(i < m ? wl : wu);
    return sizes;
}

function boxPassH(src, dst, W, H, r) {  // O(N) sliding window along rows, edges clamped
    var norm = 1 / (2 * r + 1);
    for (var y = 0; y < H; y++) {
        var o = y * W, acc = 0, k;
        for (k = -r; k <= r; k++) acc += src[o + Math.min(W - 1, Math.max(0, k))];
        for (var x = 0; x < W; x++) {
            dst[o + x] = acc * norm;
            acc += src[o + Math.min(W - 1, x + r + 1)] - src[o + Math.max(0, x - r)];
        }
    }
}

function boxPassV(src, dst, W, H, r) {  // O(N) sliding window along columns, edges clamped
    var norm = 1 / (2 * r + 1), acc = new Float64Array(W), x, k, o;
    for (k = -r; k <= r; k++) { o = Math.min(H - 1, Math.max(0, k)) * W; for (x = 0; x < W; x++) acc[x] += src[o + x]; }
    for (var y = 0; y < H; y++) {
        var add = Math.min(H - 1, y + r + 1) * W, sub = Math.max(0, y - r) * W;
        o = y * W;
        for (x = 0; x < W; x++) { dst[o + x] = acc[x] * norm; acc[x] += src[add + x] - src[sub + x]; }
    }
}

function boxBlur3(src, W, H, sigma) {  // 3-pass box blur approximating a Gaussian (sigma >= 1.5)
    var sizes = boxesForGauss(sigma, 3), a = Float32Array.from(src), b = new Float32Array(W * H);
    for (var i = 0; i < 3; i++) {
        var r = (sizes[i] - 1) / 2;
        boxPassH(a, b, W, H, r);
        boxPassV(b, a, W, H, r);
    }
    return a;
}

function blur(src, W, H, sigma) { return sigma < 1.5 ? gaussianBlur(src, W, H, sigma) : boxBlur3(src, W, H, sigma); }

function minMax3(src, W, H, buf) {  // 3x3 local min and max as separable 1D passes, edges clamped; reuses `buf`
    var n = W * H;
    if (!buf || buf.min.length !== n) {
        buf = {min: new Float32Array(n), max: new Float32Array(n), tMin: new Float32Array(n), tMax: new Float32Array(n)};
    }
    var mn = buf.min, mx = buf.max, tMin = buf.tMin, tMax = buf.tMax, x, y, o, a, b, c, lo, hi;
    for (y = 0; y < H; y++) {  // rows
        o = y * W;
        for (x = 0; x < W; x++) {
            a = src[o + (x > 0 ? x - 1 : 0)]; b = src[o + x]; c = src[o + (x < W - 1 ? x + 1 : x)];
            lo = a < b ? a : b; hi = a > b ? a : b;
            tMin[o + x] = lo < c ? lo : c; tMax[o + x] = hi > c ? hi : c;
        }
    }
    for (y = 0; y < H; y++) {  // columns
        var up = (y > 0 ? y - 1 : 0) * W, dn = (y < H - 1 ? y + 1 : y) * W;
        o = y * W;
        for (x = 0; x < W; x++) {
            a = tMin[up + x]; b = tMin[o + x]; c = tMin[dn + x];
            lo = a < b ? a : b; mn[o + x] = lo < c ? lo : c;
            a = tMax[up + x]; b = tMax[o + x]; c = tMax[dn + x];
            hi = a > b ? a : b; mx[o + x] = hi > c ? hi : c;
        }
    }
    return buf;
}

// Anti-ringing: each pixel stays within its stage input's 3x3 [min, max] +- margin. Clamping against each stage's
// own input alone lets the margin pile up over the five stages (~8/255 on a hard edge), so the bounds are also capped
// by `env`, the 3x3 [min, max] of the unsharpened Y: the total excursion stays within one margin.
function clampRing(Y, mm, margin, env) {
    var mn = mm.min, mx = mm.max, en = env.min, ex = env.max;
    for (var i = 0; i < Y.length; i++) {
        var lo = (mn[i] > en[i] ? mn[i] : en[i]) - margin, hi = (mx[i] < ex[i] ? mx[i] : ex[i]) + margin, v = Y[i];
        if (lo > hi) lo = hi = 0.5 * (lo + hi);
        Y[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}

function envelope(Y, W, H) {  // 3x3 [min, max] kept as the anchor (its temporary buffers are dropped)
    var env = minMax3(Y, W, H, null);
    env.tMin = env.tMax = null;
    return env;
}

function lanczos3(t) {
    if (t === 0) return 1;
    if (t <= -3 || t >= 3) return 0;
    var a = Math.PI * t;
    return 3 * Math.sin(a) * Math.sin(a / 3) / (a * a);
}

function bilinear(t) { t = Math.abs(t); return t < 1 ? 1 - t : 0; }

function resampleAxis(len, S, kernel, radius) {  // taps and normalised weights for each output coordinate
    var outLen = len * S, taps = 2 * radius, idx = new Int32Array(outLen * taps), w = new Float32Array(outLen * taps);
    for (var o = 0; o < outLen; o++) {
        var u = (o + 0.5) / S - 0.5, start = Math.floor(u) - radius + 1, sum = 0, k;
        for (k = 0; k < taps; k++) {
            var i = start + k, wk = kernel(u - i);
            idx[o * taps + k] = Math.min(len - 1, Math.max(0, i));
            w[o * taps + k] = wk;
            sum += wk;
        }
        for (k = 0; k < taps; k++) w[o * taps + k] /= sum;
    }
    return {idx: idx, w: w, taps: taps, outLen: outLen};
}

function upscale(src, W, H, S, kernel, radius) {  // separable: rows, then columns; Float32 in/out, no clamping
    var ax = resampleAxis(W, S, kernel, radius), ay = resampleAxis(H, S, kernel, radius);
    var W2 = ax.outLen, H2 = ay.outLen, tmp = new Float32Array(W2 * H), out = new Float32Array(W2 * H2);
    var t = ax.taps, x, y, k, b, o, acc;
    for (y = 0; y < H; y++) {
        var row = y * W;
        o = y * W2;
        for (x = 0; x < W2; x++) {
            acc = 0; b = x * t;
            for (k = 0; k < t; k++) acc += ax.w[b + k] * src[row + ax.idx[b + k]];
            tmp[o + x] = acc;
        }
    }
    t = ay.taps;
    for (y = 0; y < H2; y++) {
        o = y * W2; b = y * t;
        for (k = 0; k < t; k++) {
            var wk = ay.w[b + k], src2 = ay.idx[b + k] * W2;
            for (x = 0; x < W2; x++) out[o + x] += wk * tmp[src2 + x];
        }
    }
    return out;
}

function enhance(rgba, W, H, S) {  // 1x RGBA in, {data, width, height} at S x out
    var n = W * H, c = splitYCbCr(rgba, n), i, d, mm;
    var Y = guidedSelf(c.Y, W, H, GUIDED_RADIUS, GUIDED_EPS);  // a. soften JPEG block edges only
    var e = clahe(Y, W, H, CLAHE_GRID_X, CLAHE_GRID_Y, CLAHE_CLIP);  // b. local contrast
    for (i = 0; i < n; i++) Y[i] = CLAHE_BLEND * e[i] + (1 - CLAHE_BLEND) * Y[i];
    var ref = Float32Array.from(Y), env = envelope(ref, W, H);  // the unsharpened Y and its 3x3 anchor
    var cl = blur(Y, W, H, CLARITY_SIGMA);  // c. local clarity (its input is `ref`, so its 3x3 range is `env`)
    for (i = 0; i < n; i++) Y[i] += CLARITY_AMOUNT * (Y[i] - cl[i]);
    clampRing(Y, env, RING_MARGIN, env);
    mm = minMax3(Y, W, H, null);  // d. pre-sharpen
    var pb = blur(Y, W, H, PRE_USM_SIGMA);
    for (i = 0; i < n; i++) {
        d = Y[i] - pb[i];
        Y[i] += PRE_USM_AMOUNT * Math.sign(d) * Math.max(0, Math.abs(d) - PRE_USM_THRESHOLD);
    }
    clampRing(Y, mm, RING_MARGIN, env);
    var Cb = c.Cb, Cr = c.Cr, W2 = W * S, H2 = H * S, n2 = W2 * H2;
    if (S > 1) {
        env = envelope(upscale(ref, W, H, S, bilinear, 1), W2, H2);  // the anchor at S x
        var Yb = upscale(Y, W, H, S, bilinear, 1);
        if (UPSCALER === "bilinear") {
            Y = Yb;
        } else {
            Y = upscale(Y, W, H, S, lanczos3, 3);
            mm = minMax3(Yb, W2, H2, null);
            clampRing(Y, mm, RING_MARGIN, env);  // removes the Lanczos3 overshoot
        }
        Yb = null;
        Cb = upscale(Cb, W, H, S, bilinear, 1);
        Cr = upscale(Cr, W, H, S, bilinear, 1);
    }
    mm = minMax3(Y, W2, H2, mm);  // fine band
    var fine = blur(Y, W2, H2, USM_FINE_SIGMA * S);
    for (i = 0; i < n2; i++) {
        d = Y[i] - fine[i];
        Y[i] += USM_FINE_AMOUNT * Math.sign(d) * Math.max(0, Math.abs(d) - USM_THRESHOLD);
    }
    clampRing(Y, mm, RING_MARGIN, env);
    fine = null;
    mm = minMax3(Y, W2, H2, mm);  // medium band
    var med = blur(Y, W2, H2, USM_MED_SIGMA * S);
    for (i = 0; i < n2; i++) {
        d = Y[i] - med[i];
        Y[i] += USM_MED_AMOUNT * Math.sign(d) * Math.max(0, Math.abs(d) - USM_THRESHOLD);
    }
    clampRing(Y, mm, RING_MARGIN, env);
    return {data: mergeYCbCr(Y, Cb, Cr, n2, CHROMA_BOOST), width: W2, height: H2};
}
"""


class CanvasEnhanceScript(MacroElement):
    """Show the enhanced overlay for the fitted view, from the in-memory cache when the same tile range, scale and
    constants were processed before. Falls back to the plain tiles (one console.warn) on any failure.

    Timing: folium's fitBounds runs on an already loaded map, so Leaflet animates the zoom: the new zoom arrives in a
    later animation frame and zoomend after the animation. whenReady + setTimeout 0 raced that frame and could read
    the initial zoom (19, so S = 1, and the pre-fit bounds). Now whenReady, zoomend and resize all (re)schedule the
    run SETTLE_MS later, the run skips an unchanged view and a superseded build is discarded.

    Cache: a Map on window.parent (`__roofFxCache`, at most CACHE_MAX entries, oldest dropped), plain JS memory only,
    gone when the page is closed. Every map rebuild is a new iframe and a blob URL dies with the document that made
    it, so the cache keeps a parent-realm copy of the PNG Blob and each rebuild makes its own URL from it."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            {{ this.pipeline }}
            var NZ = {{ this.native }};
            var TILE = 256, CACHE_MAX = {{ this.cache_max }};
            var TILE_URL = {{ this.tile_url|tojson }};
            var map = {{ this._parent.get_name() }};
            var tiles = {{ this.tile_layer_name }};
            var overlay = null, overlayUrl = null, failed = false;
            var PARAMS_HASH = hashString([TILE_MARGIN, MAX_MOSAIC_PIXELS, MAX_OUTPUT_PIXELS, GUIDED_RADIUS, GUIDED_EPS,
                CLAHE_GRID_X, CLAHE_GRID_Y, CLAHE_CLIP, CLAHE_BLEND, CLARITY_SIGMA, CLARITY_AMOUNT, PRE_USM_SIGMA,
                PRE_USM_AMOUNT, PRE_USM_THRESHOLD, UPSCALER, USM_FINE_SIGMA, USM_FINE_AMOUNT, USM_MED_SIGMA, USM_MED_AMOUNT,
                USM_THRESHOLD, RING_MARGIN, CHROMA_BOOST].join(","));
            var SETTLE_MS = 300;  // > Leaflet's 250 ms zoom animation: the fitted zoom and bounds are final by then
            var settleTimer = null, currentKey = null, runId = 0;

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
            function build(key, run, xMin, yMin, xMax, yMax, S, nw, se) {
                var t0 = performance.now(), nx = xMax - xMin + 1, ny = yMax - yMin + 1, W = nx * TILE, H = ny * TILE;
                var jobs = [];
                for (var ty = yMin; ty <= yMax; ty++) for (var tx = xMin; tx <= xMax; tx++) jobs.push(loadTile(tx, ty));
                Promise.all(jobs).then(function (loaded) {
                    if (run !== runId) return;  // a newer view superseded this one
                    var t1 = performance.now();
                    var mosaic = document.createElement("canvas");
                    mosaic.width = W; mosaic.height = H;
                    var mctx = mosaic.getContext("2d");
                    loaded.forEach(function (t) { mctx.drawImage(t.img, (t.x - xMin) * TILE, (t.y - yMin) * TILE); });
                    var native = mctx.getImageData(0, 0, W, H);  // throws on a tainted canvas
                    var r = enhance(native.data, W, H, S);
                    var out = document.createElement("canvas");
                    out.width = r.width; out.height = r.height;
                    var octx = out.getContext("2d"), img = octx.createImageData(r.width, r.height);
                    img.data.set(r.data);
                    octx.putImageData(img, 0, 0);
                    var t2 = performance.now();
                    console.info("roofEnhanced: " + W + "x" + H + " -> " + r.width + "x" + r.height + " (S=" + S + ") in "
                        + Math.round(t2 - t0) + " ms (tiles " + Math.round(t1 - t0) + " ms, processing "
                        + Math.round(t2 - t1) + " ms)");
                    out.toBlob(function (blob) {
                        try {
                            if (!blob) throw new Error("toBlob returned null");
                            cachePut(key, blob);
                            if (run !== runId) return;
                            var entry = cacheGet(key);
                            show(entry ? entry.blob : blob, nw, se, entry, fail);
                        } catch (err) { fail(err); }
                    }, "image/png");
                }).catch(fail);
            }
            function schedule() {
                clearTimeout(settleTimer);
                settleTimer = setTimeout(roofEnhance, SETTLE_MS);
            }
            function roofEnhance() {  // runs after the fit has settled: zoom and bounds are the fitted ones
                if (failed) return;
                try {
                    var b = map.getBounds(), p0 = map.project(b.getNorthWest(), NZ), p1 = map.project(b.getSouthEast(), NZ);
                    var xMin = Math.floor(p0.x / TILE) - TILE_MARGIN, yMin = Math.floor(p0.y / TILE) - TILE_MARGIN;
                    var xMax = Math.floor(p1.x / TILE) + TILE_MARGIN, yMax = Math.floor(p1.y / TILE) + TILE_MARGIN;
                    var W = (xMax - xMin + 1) * TILE, H = (yMax - yMin + 1) * TILE;
                    if (W * H > MAX_MOSAIC_PIXELS) throw new Error("mosaic too large: " + W + "x" + H);
                    var S = Math.min(4, Math.max(1, Math.pow(2, Math.ceil(map.getZoom()) - NZ)));
                    while (S > 1 && W * H * S * S > MAX_OUTPUT_PIXELS) S = S >> 1;
                    var nw = map.unproject(L.point(xMin * TILE, yMin * TILE), NZ);
                    var se = map.unproject(L.point((xMax + 1) * TILE, (yMax + 1) * TILE), NZ);
                    var key = [NZ, xMin, xMax, yMin, yMax, S, PARAMS_HASH].join("|");
                    if (key === currentKey) return;  // same view: nothing to redo
                    currentKey = key;
                    var run = ++runId;
                    var hit = cacheGet(key);
                    if (hit) {
                        try {
                            show(hit.blob, nw, se, hit, function () {  // a stale entry: drop it and process again
                                cacheDrop(key);
                                removeOverlay();
                                try { build(key, run, xMin, yMin, xMax, yMax, S, nw, se); } catch (err) { fail(err); }
                            });
                            console.info("roofEnhanced: cached result reused (" + key + ")");
                            return;
                        } catch (err) {
                            cacheDrop(key);
                        }
                    }
                    build(key, run, xMin, yMin, xMax, yMax, S, nw, se);
                } catch (err) {
                    fail(err);
                }
            }
            map.whenReady(schedule);
            map.on("zoomend resize", schedule);
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
