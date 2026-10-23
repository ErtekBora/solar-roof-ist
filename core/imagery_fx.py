"""Browser-side display enhancement of the roof map's Esri tiles: inline SVG filters + CSS (DISPLAY ONLY).

Licence rule: the tiles are never fetched, cached, rasterised or read here or in the browser script. Everything is
a CSS `filter: url(#id)` on the tile layer's container (Leaflet GridLayer `className`), so the browser filters the
composited layer (no seams between 256 px tiles). The only script reads `map.getZoom()` to scale the fine blur.

Encoding (SVG clamps every intermediate result to [0, 1]): a signed difference a - b is stored as
0.5 + 0.5 (a - b) = feComposite arithmetic k2 = 0.5 (a), k3 = 0.5 (1 - b), k4 = 0. Its alpha is k2 + k3 + k4 = 1,
so no premultiplied colour is clipped. Two such differences combine into ONE detail image centred at 0.5 with
weights summing to 1 (alpha stays 1); the detail is added once and alpha is locked to 1 afterwards. Sharpening runs
on BT.709 luminance only. Sobel uses feConvolveMatrix with bias 0.5 and |x - 0.5| * 2 via a "1 0 1" table.
Docs: https://drafts.fxtf.org/filter-effects/  https://leafletjs.com/reference.html#gridlayer-classname
"""
from branca.element import MacroElement
from jinja2 import Template

ORIGINAL, STANDARD, SHADOW, EDGES = "Orijinal", "Standart (önerilen)", "Gölgeli çatı", "Kenar modu"
PRESETS = (ORIGINAL, STANDARD, SHADOW, EDGES)
DEFAULT_PRESET = STANDARD
STRENGTH_MIN, STRENGTH_MAX, STRENGTH_DEFAULT = 0.5, 2.0, 1.0
FILTER_IDS = {STANDARD: "roof-fx-standard", SHADOW: "roof-fx-shadow", EDGES: "roof-fx-edges"}
DEFS_ID = "roof-fx-defs"
NATIVE_ZOOM = 19  # must match app.ESRI_NATIVE_ZOOM
SIGMA_FINE_BASE, SIGMA_FINE_MIN, SIGMA_FINE_MAX = 0.8, 0.6, 4.0
PRE_DENOISE_SHARE = 0.4  # pre-denoise sigma = 0.4 x fine sigma
CLARITY_SIGMA = 25.0
CORING_DEAD_ZONE = 0.01  # in the 0.5-encoded difference: |a - b| < 0.02 counts as noise
S_CURVE = (0.0, 0.05, 0.15, 0.32, 0.5, 0.68, 0.85, 0.95, 1.0)
LUMA = (0.2126, 0.7152, 0.0722)
CAPTION = (
    "İyileştirme yalnızca tarayıcıda görüntüye uygulanan bir filtredir; görüntü indirilmez, saklanmaz veya analiz "
    "edilmez. Gerçek çözünürlüğü artırmaz, kenarları belirginleştirir."
)
SOBEL_KERNELS = (
    "-1 0 1 -2 0 2 -1 0 1",  # x
    "-1 -2 -1 0 0 0 1 2 1",  # y
    "0 1 2 -1 0 1 -2 -1 0",  # diagonal /
    "-2 -1 0 -1 0 1 0 1 2",  # diagonal \
)
_ALPHA_LOCK = '<feComponentTransfer result="{out}" in="{inp}"><feFuncA type="linear" slope="0" intercept="1"/></feComponentTransfer>'


def clamp_strength(strength):
    try:
        value = float(strength)
    except (TypeError, ValueError):
        return STRENGTH_DEFAULT
    return min(STRENGTH_MAX, max(STRENGTH_MIN, value)) if value == value else STRENGTH_DEFAULT


def fine_sigma(zoom):
    """Fine-detail blur sigma for a map zoom: 0.8 x 2^(zoom - native), clamped to [0.6, 4] (mirrors the JS)."""
    return min(SIGMA_FINE_MAX, max(SIGMA_FINE_MIN, SIGMA_FINE_BASE * 2 ** (zoom - NATIVE_ZOOM)))


def s_curve_values(strength):
    """Midtone S-curve table: blended toward identity below strength 1, the curve as given from 1 on."""
    t = min(1.0, clamp_strength(strength))
    n = len(S_CURVE) - 1
    return [round(i / n + t * (v - i / n), 4) for i, v in enumerate(S_CURVE)]


def coring_values(dead_zone=CORING_DEAD_ZONE, n=100):
    """Soft-threshold table on a 0.5-encoded difference: |d - 0.5| <= dead_zone -> 0.5, the rest rescaled to [0, 1]."""
    out = []
    for i in range(n + 1):
        x = i / n - 0.5
        mag = max(0.0, abs(x) - dead_zone) * 0.5 / (0.5 - dead_zone)
        out.append(round(0.5 + (mag if x > 0 else -mag), 4))
    return out


def _fmt(values):
    return " ".join(f"{v:g}" for v in values)


def _rgb_transfer(inp, out, func):
    return (
        f'<feComponentTransfer in="{inp}" result="{out}">'
        + "".join(f"<feFunc{c} {func}/>" for c in "RGB")
        + "</feComponentTransfer>"
    )


def _luminance(inp, out):
    row = " ".join(f"{w}" for w in LUMA) + " 0 0"
    return f'<feColorMatrix in="{inp}" type="matrix" values="{row} {row} {row} 0 0 0 1 0" result="{out}"/>'


def _half_difference(a, blurred, out):
    """0.5 + 0.5 (a - blurred): invert the blur, then arithmetic k2 = k3 = 0.5 (alpha stays 1)."""
    inv = f"{out}-inv"
    return (
        _rgb_transfer(blurred, inv, 'type="linear" slope="-1" intercept="1"')
        + f'<feComposite in="{a}" in2="{inv}" operator="arithmetic" k1="0" k2="0.5" k3="0.5" k4="0" result="{out}"/>'
    )


def _detail(base, s, fine_amount, clarity_amount, out):
    """Luminance detail of `base` added once: pre-denoise -> cored fine difference (+ clarity) -> one detail image."""
    parts = [
        _luminance(base, "L"),
        f'<feGaussianBlur class="fx-pre" in="L" stdDeviation="{PRE_DENOISE_SHARE * SIGMA_FINE_BASE:g}" result="Lp"/>',
        f'<feGaussianBlur class="fx-fine" in="Lp" stdDeviation="{SIGMA_FINE_BASE:g}" result="Lb"/>',
        _half_difference("Lp", "Lb", "dF"),
        _rgb_transfer("dF", "dFc", f'type="table" tableValues="{_fmt(coring_values())}"'),
    ]
    a_f, a_c = fine_amount * s, clarity_amount * s
    detail = "dFc"
    if a_c > 0:
        w_f = a_f / (a_f + a_c)
        parts += [
            f'<feGaussianBlur in="L" stdDeviation="{CLARITY_SIGMA:g}" result="Lc"/>',
            _half_difference("L", "Lc", "dC"),
            # weights sum to 1: still centred at 0.5, alpha 1
            f'<feComposite in="dFc" in2="dC" operator="arithmetic" k1="0" k2="{w_f:.4f}" k3="{1 - w_f:.4f}" k4="0" result="D"/>',
        ]
        detail = "D"
    k = 2 * (a_f + a_c)  # base + a_f dF + a_c dC = base + k (D - 0.5)
    parts += [
        f'<feComposite in="{base}" in2="{detail}" operator="arithmetic" k1="0" k2="1" k3="{k:.4f}" k4="{-k / 2:.4f}" result="{out}-raw"/>',
        _ALPHA_LOCK.format(inp=f"{out}-raw", out=out),
    ]
    return "".join(parts)


def _s_curve(inp, out, s):
    return _rgb_transfer(inp, out, f'type="table" tableValues="{_fmt(s_curve_values(s))}"')


def _saturate(inp, out, value):
    return f'<feColorMatrix in="{inp}" type="saturate" values="{value:.3f}" result="{out}"/>'


def _edges(inp, out, s):
    """Thin cyan Sobel edges (x, y, two diagonals) composited over `inp`; never replaces the image."""
    parts = [_luminance(inp, "E0")]
    for i, kernel in enumerate(SOBEL_KERNELS):
        parts.append(
            f'<feConvolveMatrix in="E0" order="3" kernelMatrix="{kernel}" divisor="8" bias="0.5" '
            f'edgeMode="duplicate" preserveAlpha="true" result="S{i}"/>'
        )
        parts.append(_rgb_transfer(f"S{i}", f"A{i}", 'type="table" tableValues="1 0 1"'))
    parts += [  # averages keep alpha at 1
        '<feComposite in="A0" in2="A1" operator="arithmetic" k1="0" k2="0.5" k3="0.5" k4="0" result="A01"/>',
        '<feComposite in="A2" in2="A3" operator="arithmetic" k1="0" k2="0.5" k3="0.5" k4="0" result="A23"/>',
        '<feComposite in="A01" in2="A23" operator="arithmetic" k1="0" k2="0.5" k3="0.5" k4="0" result="Esum"/>',
        f'<feColorMatrix in="Esum" type="matrix" values="0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 {6 * s:.3f} 0 0 0 -0.15" result="Ecyan"/>',
        f'<feComposite in="Ecyan" in2="{inp}" operator="over" result="{out}-raw"/>',
        _ALPHA_LOCK.format(inp=f"{out}-raw", out=out),
    ]
    return "".join(parts)


def _filter(fid, body):
    return (
        f'<filter id="{fid}" color-interpolation-filters="sRGB" x="0" y="0" width="100%" height="100%">{body}</filter>'
    )


def filter_body(preset, strength):
    """The primitives of one preset (no <filter> wrapper)."""
    s = clamp_strength(strength)
    if preset == STANDARD:
        return (
            _detail("SourceGraphic", s, 0.6, 0.25, "sharp") + _s_curve("sharp", "curve", s)
            + _saturate("curve", "sat", 1 + 0.15 * s) + _ALPHA_LOCK.format(inp="sat", out="final")
        )
    if preset == SHADOW:
        lift = max(0.45, 1 - 0.3 * s)  # gamma exponent < 1 lifts the darks, 1 stays 1 (no clipped highlights)
        return (
            _rgb_transfer("SourceGraphic", "lifted", f'type="gamma" amplitude="1" exponent="{lift:.3f}" offset="0"')
            + _detail("lifted", s, 0.6, 0.3, "sharp") + _s_curve("sharp", "curve", s)
            + _saturate("curve", "sat", 1 + 0.25 * s) + _ALPHA_LOCK.format(inp="sat", out="final")
        )
    if preset == EDGES:
        c = min(2.5, 1 + 0.5 * s)
        return (
            _saturate("SourceGraphic", "gray", 0)
            + _rgb_transfer("gray", "contrast", f'type="linear" slope="{c:.3f}" intercept="{0.5 - 0.5 * c:.3f}"')
            + _detail("contrast", s, 0.8, 0.0, "sharp") + _edges("sharp", "edged", s)
            + _ALPHA_LOCK.format(inp="edged", out="final")
        )
    return ""


def svg_defs(preset, strength=STRENGTH_DEFAULT):
    """Hidden inline <svg> with the active preset's filter, or "" for Orijinal / unknown presets.

    Not display:none (Firefox does not apply filters from an undisplayed SVG); a 0x0 absolutely positioned box."""
    if preset not in FILTER_IDS:
        return ""
    return (
        f'<svg id="{DEFS_ID}" xmlns="http://www.w3.org/2000/svg" width="0" height="0" aria-hidden="true" '
        f'style="position:absolute;width:0;height:0;overflow:hidden">'
        f"{_filter(FILTER_IDS[preset], filter_body(preset, strength))}</svg>"
    )


def css(preset, tile_class):
    """CSS putting the preset's filter on the roof tile layer container only (no other map, no other layer)."""
    if preset not in FILTER_IDS:
        return ""
    return f"<style>.{tile_class} {{ filter: url(#{FILTER_IDS[preset]}); }}</style>"


class ZoomSigmaScript(MacroElement):
    """After fitBounds has been applied (whenReady + setTimeout 0) and on every zoomend: set the fine / pre-denoise
    blur sigma from map.getZoom(). Reads the zoom number only, never pixels."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var roofFxMap = {{ this._parent.get_name() }};
            function roofFxSigma() {
                var z = roofFxMap.getZoom();
                var s = Math.min({{ this.hi }}, Math.max({{ this.lo }}, {{ this.base }} * Math.pow(2, z - {{ this.native }})));
                document.querySelectorAll("#{{ this.defs_id }} .fx-fine").forEach(function (n) {
                    n.setAttribute("stdDeviation", s.toFixed(3));
                });
                document.querySelectorAll("#{{ this.defs_id }} .fx-pre").forEach(function (n) {
                    n.setAttribute("stdDeviation", ({{ this.pre }} * s).toFixed(3));
                });
            }
            roofFxMap.whenReady(function () { setTimeout(roofFxSigma, 0); });
            roofFxMap.on("zoomend", roofFxSigma);
        })();
        {% endmacro %}
        """
    )

    def __init__(self):
        super().__init__()
        self._name = "ZoomSigmaScript"
        self.lo, self.hi, self.base, self.native = SIGMA_FINE_MIN, SIGMA_FINE_MAX, SIGMA_FINE_BASE, NATIVE_ZOOM
        self.pre, self.defs_id = PRE_DENOISE_SHARE, DEFS_ID


def add_to_map(m, preset, strength, tile_class):
    """Add the preset's SVG defs, CSS and zoom script to the folium map `m` (call AFTER fit_bounds). Orijinal adds nothing."""
    from folium import Element

    if preset not in FILTER_IDS:
        return
    m.get_root().html.add_child(Element(svg_defs(preset, strength) + css(preset, tile_class)))
    m.add_child(ZoomSigmaScript())
