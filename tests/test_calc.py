import pytest

from core import calc
from core.calc import building_confidence, run_scenario, usable_plan_area

MONTHLY = [50, 60, 90, 110, 140, 160, 170, 160, 120, 90, 55, 45]


def test_run_scenario_is_stub():
    with pytest.raises(NotImplementedError):
        run_scenario({}, "x")


# ------------------------------------------------------------------ usable plan area

def test_usable_base_with_photo_uses_ratio_only_and_no_edge_factor():
    assert usable_plan_area(200, 0.25, 100.0) == pytest.approx(75.0)  # 100 x (1 - 0.25); EDGE_FACTOR not applied again


def test_usable_base_without_photo_uses_ratio_over_edge_factor():
    assert usable_plan_area(200, None, 100.0) == pytest.approx(100.0 * 0.276 / 0.85)


def test_zero_usable_base_gives_zero_area():
    assert usable_plan_area(200, None, 0.0) == 0


def test_manual_area_rules():
    assert usable_plan_area(200, 0.25) == pytest.approx(127.5)  # 200 x 0.85 x 0.75
    assert usable_plan_area(200, None) == pytest.approx(55.2)  # 200 x 0.276
    assert usable_plan_area(200, 0) == pytest.approx(170.0)


# ------------------------------------------------------------------ building confidence

def test_polygon_or_top_down_photo_is_orta_otherwise_dusuk():
    assert building_confidence(None, True, True)[0] == "Orta"
    assert building_confidence("top-down", False, True)[0] == "Orta"
    assert building_confidence("oblique", False, True)[0] == "Düşük"
    assert building_confidence(None, False, True)[0] == "Düşük"


def test_unknown_structure_and_unavailable_face_each_drop_one_level_never_below_dusuk():
    assert building_confidence("top-down", True, False)[0] == "Düşük"
    assert building_confidence("top-down", True, True, n_unavailable_faces=1)[0] == "Düşük"
    assert building_confidence("top-down", True, False, n_unavailable_faces=1)[0] == "Düşük"
    assert building_confidence(None, False, False)[0] == "Düşük"


def test_confidence_ignores_the_bill_and_flat_count():
    base = building_confidence(None, True, True)
    assert building_confidence(None, True, True, has_bill=True, flats_estimated=True)[:2] == base[:2]


def test_improvement_hints_list_only_what_would_help():
    assert building_confidence("top-down", True, True, has_bill=True)[2] == []
    worst = building_confidence(None, False, False, flats_estimated=True)[2]
    assert worst == ["Daire sayısını kontrol edin.", "Ortalama daire faturasını girin.", "Çatı yapısını seçin.", "Engelleri haritada çizin."]


# ------------------------------------------------------------------ two-tier tariff

LOW = 2.918965 * 1.26
HIGH = 4.320708 * 1.26
LIMIT = 240
AT_LIMIT = LIMIT * LOW


def test_tariff_constants_are_the_computed_values():
    assert calc.LOW == pytest.approx(LOW) and calc.HIGH == pytest.approx(HIGH)
    assert calc.LIMIT == LIMIT == 8 * 30
    assert calc.BILL_AT_LIMIT == pytest.approx(AT_LIMIT)  # computed constant (882.70 TL), not a hard-coded 883


@pytest.mark.parametrize("kwh", [0, -5, None, float("nan")])
def test_bill_tl_zero_for_non_positive_or_invalid(kwh):
    assert calc.bill_tl(kwh) == 0


def test_bill_tl_low_tier_limit_and_high_tier():
    assert calc.bill_tl(100) == pytest.approx(100 * LOW)
    assert calc.bill_tl(240) == pytest.approx(AT_LIMIT)
    assert calc.bill_tl(340) == pytest.approx(AT_LIMIT + 100 * HIGH)  # only the excess is billed at HIGH


@pytest.mark.parametrize("bill", [None, 0, -10, float("nan")])
def test_kwh_from_bill_zero_for_no_bill(bill):
    assert calc.kwh_from_bill(bill) == 0


@pytest.mark.parametrize("kwh", [1, 100, 239.9, 240, 240.1, 353.4, 1000])
def test_bill_round_trip_on_both_sides_of_the_limit(kwh):
    assert calc.kwh_from_bill(calc.bill_tl(kwh)) == pytest.approx(kwh)


def test_known_bill_to_kwh_values():
    assert calc.kwh_from_bill(calc.BILL_AT_LIMIT) == pytest.approx(240)
    assert calc.kwh_from_bill(AT_LIMIT) == pytest.approx(calc.LIMIT)
    assert calc.kwh_from_bill(883) == pytest.approx(240, abs=0.5)  # "883 TL is about 240 kWh"
    assert calc.kwh_from_bill(1500) == pytest.approx(240 + (1500 - AT_LIMIT) / HIGH)
    assert calc.kwh_from_bill(1500) == pytest.approx(353, abs=0.5)


def test_monthly_savings_is_bounded_by_zero_and_the_bill():
    for c in (50, 240, 400):
        for prod in (0, 10, 100, 1000):
            for share in (0.5, 1.0):
                v = calc.monthly_savings(c, prod, share)
                assert 0 <= v <= calc.bill_tl(c) + 1e-9


def test_surplus_beyond_consumption_is_worth_zero():
    assert calc.monthly_savings(300, 10_000, 1.0) == pytest.approx(calc.bill_tl(300))
    assert calc.monthly_savings(300, 600, 1.0) == pytest.approx(calc.monthly_savings(300, 300, 1.0))


def test_pv_offsets_the_high_tier_first():
    # C = 340 kWh, PV 100 kWh: the 100 kWh above the limit are removed -> saving = 100 x HIGH
    assert calc.monthly_savings(340, 100, 1.0) == pytest.approx(100 * HIGH)
    # PV 150 kWh: 100 at HIGH, then 50 at LOW
    assert calc.monthly_savings(340, 150, 1.0) == pytest.approx(100 * HIGH + 50 * LOW)


def test_pv_offsets_with_netting_share_one_is_the_whole_month():
    assert calc.monthly_savings(240, 20, calc.SAVINGS_HIGH_FACTOR) == pytest.approx(20 * LOW)


@pytest.mark.parametrize("annual", [0, 1200, 5000])
def test_fallback_savings_range_has_tax_applied_once(annual):
    lo, hi = calc.fallback_savings(annual)
    assert lo == pytest.approx(annual * 2.918965 * 1.26) and hi == pytest.approx(annual * 4.320708 * 1.26)
    assert lo != pytest.approx(annual * 2.918965 * 1.26 * 1.26, rel=0.01) or annual == 0
    assert calc.fallback_savings(annual)[0] <= calc.fallback_savings(annual)[1]


def test_removed_single_flat_api_is_gone():
    for name in ("compute_capacity", "savings_from_bill", "_confidence"):
        assert not hasattr(calc, name)
