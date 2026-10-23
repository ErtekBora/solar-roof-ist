"""Browser-side imagery enhancement (core/imagery_fx): SVG/CSS/JS strings only, never pixel access."""
import re
import xml.dom.minidom as minidom

import pytest

from core import imagery_fx as fx

ACTIVE = (fx.STANDARD, fx.SHADOW, fx.EDGES)


def _filters(preset, strength=1.0):
    return minidom.parseString(fx.svg_defs(preset, strength)).getElementsByTagName("filter")


def _children(node):
    return [n for n in node.childNodes if n.nodeType == n.ELEMENT_NODE]


@pytest.mark.parametrize("preset", ACTIVE)
def test_each_preset_has_its_filter_id_and_srgb_on_every_filter(preset):
    filters = _filters(preset)
    assert [f.getAttribute("id") for f in filters] == [fx.FILTER_IDS[preset]]
    assert all(f.getAttribute("color-interpolation-filters") == "sRGB" for f in filters)
    assert f"filter: url(#{fx.FILTER_IDS[preset]})" in fx.css(preset, "roof-esri")
    assert fx.css(preset, "roof-esri").startswith("<style>.roof-esri ")  # the roof tile layer container only


def test_orijinal_injects_no_filter():
    assert fx.svg_defs(fx.ORIGINAL) == "" and fx.css(fx.ORIGINAL, "roof-esri") == ""
    import folium

    m = folium.Map()
    fx.add_to_map(m, fx.ORIGINAL, 1.0, "roof-esri")
    html = m.get_root().render()
    assert "<filter" not in html and "roofFxSigma" not in html


@pytest.mark.parametrize("preset", ACTIVE)
def test_every_chain_ends_with_an_alpha_lock(preset):
    last = _children(_filters(preset)[0])[-1]
    assert last.tagName == "feComponentTransfer"
    [func] = _children(last)
    assert (func.tagName, func.getAttribute("type"), func.getAttribute("slope"), func.getAttribute("intercept")) == (
        "feFuncA", "linear", "0", "1"
    )


@pytest.mark.parametrize("preset", ACTIVE)
def test_sharpening_runs_on_bt709_luminance_without_luminance_to_alpha(preset):
    svg = fx.svg_defs(preset)
    assert "luminanceToAlpha" not in svg
    row = "0.2126 0.7152 0.0722 0 0"
    assert f'values="{row} {row} {row} 0 0 0 1 0"' in svg
    assert 'class="fx-fine"' in svg and 'class="fx-pre"' in svg


def test_detail_is_combined_once_and_added_in_a_single_composite():
    prims = _children(_filters(fx.STANDARD)[0])
    added = [p for p in prims if p.getAttribute("in2") == "D"]
    assert len(added) == 1 and added[0].getAttribute("in") == "SourceGraphic"
    combine = next(p for p in prims if p.getAttribute("result") == "D")
    assert float(combine.getAttribute("k2")) + float(combine.getAttribute("k3")) == pytest.approx(1.0)  # alpha 1
    for diff in (p for p in prims if p.getAttribute("result") in ("dF", "dC")):
        assert (diff.getAttribute("k2"), diff.getAttribute("k3"), diff.getAttribute("k4")) == ("0.5", "0.5", "0")


def test_s_curve_only_in_standard_and_shadow_and_blends_toward_identity():
    curve = 'tableValues="0 0.05 0.15 0.32 0.5 0.68 0.85 0.95 1"'
    assert curve in fx.svg_defs(fx.STANDARD, 1.0) and curve in fx.svg_defs(fx.SHADOW, 2.0)  # no extrapolation
    assert curve not in fx.svg_defs(fx.EDGES)
    half = fx.s_curve_values(0.5)
    identity = [i / 8 for i in range(9)]
    assert half == pytest.approx([(a + b) / 2 for a, b in zip(identity, fx.S_CURVE)], abs=1e-4)
    for s in (0.5, 0.7, 1.0, 1.5, 2.0):
        assert all(0 <= v <= 1 for v in fx.s_curve_values(s))


def test_coring_table_has_a_dead_zone_around_half_and_stays_in_range():
    values = fx.coring_values()
    assert values[0] == 0 and values[-1] == 1 and values[50] == 0.5
    assert values[49] == values[51] == 0.5  # |d - 0.5| <= 0.01 -> 0.5
    assert all(0 <= v <= 1 for v in values) and values == sorted(values)


def test_edge_mode_uses_four_sobel_kernels_with_bias_and_abs_table():
    svg = fx.svg_defs(fx.EDGES)
    convs = re.findall(r"<feConvolveMatrix[^>]*>", svg)
    assert len(convs) == 4 and all('bias="0.5"' in c and 'preserveAlpha="true"' in c for c in convs)
    assert {re.search(r'kernelMatrix="([^"]+)"', c).group(1) for c in convs} == set(fx.SOBEL_KERNELS)
    assert svg.count('tableValues="1 0 1"') == 12  # R, G, B of each of the 4 responses
    assert 'operator="over"' in svg  # the edges are drawn over the image, never instead of it


def test_strength_scales_the_amounts_and_is_clamped():
    weak, strong = fx.svg_defs(fx.STANDARD, 0.5), fx.svg_defs(fx.STANDARD, 2.0)
    assert weak != strong
    assert fx.clamp_strength(5) == fx.STRENGTH_MAX and fx.clamp_strength(0) == fx.STRENGTH_MIN
    assert fx.clamp_strength("x") == fx.clamp_strength(float("nan")) == fx.STRENGTH_DEFAULT


def test_fine_sigma_follows_zoom_and_is_clamped():
    assert fx.fine_sigma(19) == pytest.approx(0.8)
    assert fx.fine_sigma(21) == pytest.approx(3.2)
    assert fx.fine_sigma(16) == 0.6 and fx.fine_sigma(22) == 4.0


def test_hidden_svg_is_not_display_none():
    svg = fx.svg_defs(fx.STANDARD)
    assert 'width="0" height="0"' in svg and "display:none" not in svg.replace(" ", "")


def test_zoom_script_reads_only_the_zoom_and_never_pixels_or_network():
    import folium

    m = folium.Map()
    fx.add_to_map(m, fx.STANDARD, 1.0, "roof-esri")
    html = m.get_root().render()
    assert "getZoom()" in html and "whenReady" in html and '"zoomend"' in html and "setTimeout(roofFxSigma, 0)" in html
    for forbidden in ("getImageData", "toDataURL", "fetch(", "XMLHttpRequest", "getContext", "canvas"):
        assert forbidden not in html
