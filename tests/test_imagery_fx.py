"""Browser-side imagery enhancement (core/imagery_fx): one gentle SVG filter, strings only, never pixel access."""
import xml.dom.minidom as minidom

import folium
import pytest

from core import imagery_fx as fx


def _filter():
    [f] = minidom.parseString(fx.svg_defs()).getElementsByTagName("filter")
    return f


def _prims():
    return [n for n in _filter().childNodes if n.nodeType == n.ELEMENT_NODE]


def test_one_filter_with_srgb_and_css_on_the_roof_tile_layer_only():
    f = _filter()
    assert f.getAttribute("id") == fx.FILTER_ID and f.getAttribute("color-interpolation-filters") == "sRGB"
    assert fx.css("roof-esri") == "<style>.roof-esri { filter: url(#roof-fx); }</style>"


def test_no_colour_changes_no_curves_no_coring():
    svg = fx.svg_defs()
    assert 'type="saturate"' not in svg and 'type="hueRotate"' not in svg
    assert 'type="table"' not in svg and 'type="gamma"' not in svg  # no S-curve, no coring table, no gamma
    assert "feConvolveMatrix" not in svg and "luminanceToAlpha" not in svg
    matrices = [p for p in _prims() if p.tagName == "feColorMatrix"]
    row = "0.2126 0.7152 0.0722 0 0"
    assert [m.getAttribute("values") for m in matrices] == [f"{row} {row} {row} 0 0 0 1 0"]  # BT.709 luminance only


def test_chain_ends_with_an_alpha_lock():
    last = _prims()[-1]
    [func] = [n for n in last.childNodes if n.nodeType == n.ELEMENT_NODE]
    assert last.tagName == "feComponentTransfer"
    assert (func.tagName, func.getAttribute("type"), func.getAttribute("slope"), func.getAttribute("intercept")) == (
        "feFuncA", "linear", "0", "1"
    )


def test_detail_is_one_half_encoded_image_added_once_to_the_source():
    prims = _prims()
    by_result = {p.getAttribute("result"): p for p in prims}
    for name, a, b in (("dP", "Ls", "L"), ("dU", "Ls", "Lb"), ("dC", "Ls", "Lc")):
        diff = by_result[name]
        assert (diff.getAttribute("in"), diff.getAttribute("in2")) == (a, f"{name}-inv")
        assert (diff.getAttribute("k2"), diff.getAttribute("k3"), diff.getAttribute("k4")) == ("0.5", "0.5", "0")
        assert by_result[f"{name}-inv"].getAttribute("in") == b
    for avg in (by_result["dPU"], by_result["D"]):
        assert float(avg.getAttribute("k2")) + float(avg.getAttribute("k3")) == pytest.approx(1.0)  # alpha stays 1
    added = [p for p in prims if p.getAttribute("in2") == "D"]
    assert len(added) == 1 and added[0].getAttribute("in") == "SourceGraphic"
    k = float(added[0].getAttribute("k3"))
    assert k == pytest.approx(2 * (1.0 + 0.5 + 0.12)) and float(added[0].getAttribute("k4")) == pytest.approx(-k / 2)


def test_blurs_deblock_then_unsharp_and_light_local_contrast():
    blurs = {p.getAttribute("result"): p for p in _prims() if p.tagName == "feGaussianBlur"}
    assert blurs["Ls"].getAttribute("class") == "fx-pre" and blurs["Ls"].getAttribute("in") == "L"
    assert blurs["Lb"].getAttribute("class") == "fx-us" and blurs["Lb"].getAttribute("in") == "Ls"
    assert blurs["Lc"].getAttribute("in") == "Ls" and blurs["Lc"].getAttribute("stdDeviation") == "12"


@pytest.mark.parametrize("zoom, pre, us", [(19, 0.4, 0.8), (20, 0.7, 1.4), (21, 1.4, 2.8), (21.5, 1.6, 3.0), (17, 0.4, 0.8)])
def test_sigmas_follow_the_upscale_factor_and_are_clamped(zoom, pre, us):
    assert fx.sigma_pre(zoom) == pytest.approx(pre) and fx.sigma_us(zoom) == pytest.approx(us)


def test_hidden_svg_is_not_display_none():
    svg = fx.svg_defs()
    assert 'width="0" height="0"' in svg and "display:none" not in svg.replace(" ", "")


def test_zoom_script_sets_both_sigmas_and_never_reads_pixels_or_the_network():
    m = folium.Map()
    fx.add_to_map(m, "roof-esri")
    html = m.get_root().render()
    assert "getZoom()" in html and "whenReady" in html and '"zoomend"' in html and "setTimeout(roofFxSigma, 0)" in html
    assert 'setSigma("fx-pre"' in html and 'setSigma("fx-us"' in html
    for forbidden in ("getImageData", "toDataURL", "fetch(", "XMLHttpRequest", "getContext", "canvas"):
        assert forbidden not in html
