"""Browser-side display enhancement of the roof map's Esri tiles: one gentle inline SVG filter + CSS (DISPLAY ONLY).

Licence rule: the tiles are never fetched, cached, rasterised or read here or in the browser script. Everything is
a CSS `filter: url(#id)` on the tile layer's container (Leaflet GridLayer `className`), so the browser filters the
composited layer (no seams between 256 px tiles). The only script reads `map.getZoom()` to scale the blurs.

The filter works on BT.709 luminance L only and never changes colour (no saturation, curve or gamma):
  Ls  = blur(L, sigma_pre)                      deblock: hides upscaled JPEG block edges
  out = Src + (Ls - L) + 0.5 (Ls - blur(Ls, sigma_us)) + 0.12 (Ls - blur(Ls, 12 px))
with s = 2^(zoom - native) the upscale factor, sigma_pre = clamp(0.35 s, 0.4, 1.6), sigma_us = clamp(0.7 s, 0.8, 3).

Encoding (SVG clamps every intermediate result to [0, 1]): a signed difference a - b is stored as
0.5 + 0.5 (a - b) = feComposite arithmetic k2 = 0.5 (a), k3 = 0.5 (1 - b), k4 = 0. Its alpha is k2 + k3 + k4 = 1,
so no premultiplied colour is clipped. The three differences are averaged into ONE detail image centred at 0.5
(weights sum to 1, alpha stays 1), added once to SourceGraphic, and alpha is locked to 1 at the end.
Docs: https://drafts.fxtf.org/filter-effects/  https://leafletjs.com/reference.html#gridlayer-classname
"""
from branca.element import MacroElement
from jinja2 import Template

FILTER_ID = "roof-fx"
DEFS_ID = "roof-fx-defs"
NATIVE_ZOOM = 19  # must match app.ESRI_NATIVE_ZOOM
PRE_FACTOR, PRE_MIN, PRE_MAX = 0.35, 0.4, 1.6
US_FACTOR, US_MIN, US_MAX = 0.7, 0.8, 3.0
US_AMOUNT = 0.5
CLARITY_SIGMA, CLARITY_AMOUNT = 12.0, 0.12
DEBLOCK_AMOUNT = 1.0  # Ls replaces L
LUMA = (0.2126, 0.7152, 0.0722)
CAPTION = (
    "İyileştirme yalnızca tarayıcıda görüntüye uygulanan bir filtredir; görüntü indirilmez, saklanmaz veya analiz "
    "edilmez. Gerçek çözünürlüğü artırmaz, kenarları belirginleştirir."
)
_ALPHA_LOCK = '<feComponentTransfer in="{inp}" result="{out}"><feFuncA type="linear" slope="0" intercept="1"/></feComponentTransfer>'


def upscale(zoom):
    """Tile upscale factor at a map zoom: 2^(zoom - native)."""
    return 2 ** (zoom - NATIVE_ZOOM)


def sigma_pre(zoom):
    return min(PRE_MAX, max(PRE_MIN, PRE_FACTOR * upscale(zoom)))


def sigma_us(zoom):
    return min(US_MAX, max(US_MIN, US_FACTOR * upscale(zoom)))


def _luminance(inp, out):
    row = " ".join(f"{w}" for w in LUMA) + " 0 0"
    return f'<feColorMatrix in="{inp}" type="matrix" values="{row} {row} {row} 0 0 0 1 0" result="{out}"/>'


def _half_difference(a, b, out):
    """0.5 + 0.5 (a - b): invert b, then arithmetic k2 = k3 = 0.5 (alpha stays 1)."""
    inv = f"{out}-inv"
    return (
        f'<feComponentTransfer in="{b}" result="{inv}">'
        + "".join(f'<feFunc{c} type="linear" slope="-1" intercept="1"/>' for c in "RGB")
        + "</feComponentTransfer>"
        + f'<feComposite in="{a}" in2="{inv}" operator="arithmetic" k1="0" k2="0.5" k3="0.5" k4="0" result="{out}"/>'
    )


def _average(a, b, weight_a, out):
    """weight_a * a + (1 - weight_a) * b: two 0.5-centred images stay centred at 0.5, alpha stays 1."""
    return (
        f'<feComposite in="{a}" in2="{b}" operator="arithmetic" k1="0" k2="{weight_a:.4f}" '
        f'k3="{1 - weight_a:.4f}" k4="0" result="{out}"/>'
    )


def filter_body():
    """The primitives of the gentle filter (no <filter> wrapper)."""
    z = NATIVE_ZOOM  # initial sigmas; the zoom script sets the real ones once the fitted view exists
    a_p, a_u, a_c = DEBLOCK_AMOUNT, US_AMOUNT, CLARITY_AMOUNT
    total = a_p + a_u + a_c
    k = 2 * total  # Src + a_p dP + a_u dU + a_c dC = Src + k (D - 0.5)
    return "".join([
        _luminance("SourceGraphic", "L"),
        f'<feGaussianBlur class="fx-pre" in="L" stdDeviation="{sigma_pre(z):g}" result="Ls"/>',
        f'<feGaussianBlur class="fx-us" in="Ls" stdDeviation="{sigma_us(z):g}" result="Lb"/>',
        f'<feGaussianBlur in="Ls" stdDeviation="{CLARITY_SIGMA:g}" result="Lc"/>',
        _half_difference("Ls", "L", "dP"),   # deblock
        _half_difference("Ls", "Lb", "dU"),  # unsharp (plain, no threshold)
        _half_difference("Ls", "Lc", "dC"),  # local contrast
        _average("dP", "dU", a_p / (a_p + a_u), "dPU"),
        _average("dPU", "dC", (a_p + a_u) / total, "D"),
        f'<feComposite in="SourceGraphic" in2="D" operator="arithmetic" k1="0" k2="1" k3="{k:.4f}" '
        f'k4="{-k / 2:.4f}" result="out"/>',
        _ALPHA_LOCK.format(inp="out", out="final"),
    ])


def svg_defs():
    """Hidden inline <svg> with the filter. Not display:none (Firefox does not apply filters from an undisplayed
    SVG); a 0x0 absolutely positioned box."""
    return (
        f'<svg id="{DEFS_ID}" xmlns="http://www.w3.org/2000/svg" width="0" height="0" aria-hidden="true" '
        f'style="position:absolute;width:0;height:0;overflow:hidden">'
        f'<filter id="{FILTER_ID}" color-interpolation-filters="sRGB" x="0" y="0" width="100%" height="100%">'
        f"{filter_body()}</filter></svg>"
    )


def css(tile_class):
    """CSS putting the filter on the roof tile layer container only (no other map, no other layer)."""
    return f"<style>.{tile_class} {{ filter: url(#{FILTER_ID}); }}</style>"


class ZoomSigmaScript(MacroElement):
    """After fitBounds has been applied (whenReady + setTimeout 0) and on every zoomend: set the deblock and unsharp
    blur sigmas from map.getZoom(). Reads the zoom number only, never pixels."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var roofFxMap = {{ this._parent.get_name() }};
            function setSigma(cls, value) {
                document.querySelectorAll("#{{ this.defs_id }} ." + cls).forEach(function (n) {
                    n.setAttribute("stdDeviation", value.toFixed(3));
                });
            }
            function roofFxSigma() {
                var s = Math.pow(2, roofFxMap.getZoom() - {{ this.native }});
                setSigma("fx-pre", Math.min({{ this.pre_max }}, Math.max({{ this.pre_min }}, {{ this.pre_factor }} * s)));
                setSigma("fx-us", Math.min({{ this.us_max }}, Math.max({{ this.us_min }}, {{ this.us_factor }} * s)));
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
        self.defs_id, self.native = DEFS_ID, NATIVE_ZOOM
        self.pre_factor, self.pre_min, self.pre_max = PRE_FACTOR, PRE_MIN, PRE_MAX
        self.us_factor, self.us_min, self.us_max = US_FACTOR, US_MIN, US_MAX


def add_to_map(m, tile_class):
    """Add the filter's SVG defs, CSS and zoom script to the folium map `m` (call AFTER fit_bounds)."""
    from folium import Element

    m.get_root().html.add_child(Element(svg_defs() + css(tile_class)))
    m.add_child(ZoomSigmaScript())
