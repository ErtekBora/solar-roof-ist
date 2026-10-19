import math

import pytest

from core import config, cost
from core.cost import combine_cost_scenarios, estimate_install_cost, per_flat_cost


@pytest.mark.parametrize(
    "kwp, band",
    [
        (7.4, (24000, 36000)),
        (7.5, (22000, 32000)),
        (12.4, (22000, 32000)),
        (12.5, (17500, 36700)),
        (20, (17500, 36700)),
        (25, (17500, 36700)),
    ],
)
def test_band_boundaries(kwp, band):
    low, high, got, _ = estimate_install_cost(kwp)
    assert got == band
    assert low == pytest.approx(kwp * band[0]) and high == pytest.approx(kwp * band[1])


def test_cost_low_never_above_cost_high():
    for kwp in (0.5, 3, 7.4, 7.5, 10, 12.4, 12.5, 20, 25, 80):
        low, high, _, _ = estimate_install_cost(kwp)
        assert low <= high


def test_published_totals_are_inside_the_band_estimate():
    low, high, _, _ = estimate_install_cost(5)
    assert (low, high) == (120_000, 180_000)  # 5 kWp: 120,000-180,000 TL
    low, high, _, _ = estimate_install_cost(10)
    assert (low, high) == (220_000, 320_000)  # 10 kWp: 220,000-320,000 TL


@pytest.mark.parametrize("kwp", [0, -1, None, float("nan"), "5", True])
def test_no_panels_gives_zero_and_a_flag(kwp):
    low, high, band, flags = estimate_install_cost(kwp)
    assert (low, high, band) == (0, 0, None) and flags == ["kurulabilir panel yok"]


def test_out_of_sourced_range_flag_only_above_20_kwp():
    assert estimate_install_cost(20)[3] == []
    assert estimate_install_cost(20.1)[3] == ["kaynak aralığı dışında, tahmin daha belirsiz"]
    assert estimate_install_cost(25)[3] == ["kaynak aralığı dışında, tahmin daha belirsiz"]


def test_numbers_come_from_config_only(monkeypatch):
    monkeypatch.setattr(cost, "COST_BANDS_TL_PER_KWP", ((math.inf, 1000, 2000),))
    assert estimate_install_cost(3)[:2] == (3000, 6000)


@pytest.mark.parametrize("n, expected", [(4, (1000, 2000)), (1, (4000, 8000)), (0, (4000, 8000)), (None, (4000, 8000)), (-3, (4000, 8000))])
def test_per_flat_division_never_divides_by_zero(n, expected):
    assert per_flat_cost((4000, 8000), n) == expected


def test_scenarios_combine_to_the_same_range_in_both_orders():
    flat_more = estimate_install_cost(20)  # flat has more kWp than pitched
    pitched_less = estimate_install_cost(9)
    a = combine_cost_scenarios([flat_more[:2], pitched_less[:2]])
    b = combine_cost_scenarios([pitched_less[:2], flat_more[:2]])
    assert a == b
    assert a == (min(flat_more[0], pitched_less[0]), max(flat_more[1], pitched_less[1]))


def test_scenario_combination_with_a_dropped_north_face_pitched_smaller_than_flat():
    # all-pitched with a dropped north face has fewer kWp than all-flat: the range must not assume an order
    flat, pitched = estimate_install_cost(12.6), estimate_install_cost(7.4)
    for order in ([flat, pitched], [pitched, flat]):
        low, high = combine_cost_scenarios([c[:2] for c in order])
        assert low == pytest.approx(7.4 * 24000) and high == pytest.approx(12.6 * 36700)


def test_pitched_larger_than_flat_gives_the_same_formula():
    flat, pitched = estimate_install_cost(6), estimate_install_cost(14)
    assert combine_cost_scenarios([flat[:2], pitched[:2]]) == (min(flat[0], pitched[0]), max(flat[1], pitched[1]))


def test_empty_scenarios_and_zero_kwp_scenario():
    assert combine_cost_scenarios([]) == (0.0, 0.0)
    zero = estimate_install_cost(0)[:2]
    ten = estimate_install_cost(10)[:2]
    assert combine_cost_scenarios([zero, ten]) == (0, ten[1])


def test_exclusions_and_source_are_listed():
    text = " ".join(cost.EXCLUSIONS_TR)
    for word in ("güçlendirme", "inverter", "Bakım", "verim", "karar"):
        assert word in text
    assert "KDV ve kurulum dahil" in config.COST_SOURCE and config.COST_BAND_MAX_SOURCED_KWP == 20
