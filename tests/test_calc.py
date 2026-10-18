import pytest

from core import calc
from core.calc import compute_capacity, run_scenario

MONTHLY = [50, 60, 90, 110, 140, 160, 170, 160, 120, 90, 55, 45]
MONTHLY_IDEAL = [m + 1 for m in MONTHLY]


def cap(**overrides):
    args = dict(
        roof_area_m2=200, obstacle_ratio=0.25, roof_type="flat",
        yield_ideal=1350, yield_shaded=1300,
    )
    args.update(overrides)
    return compute_capacity(**args)


def test_reference_case():
    r = cap()
    assert r["kullanilabilir_alan_m2"] == pytest.approx(127.5)
    assert r["kurulabilir_guc_kwp"] == pytest.approx(19.615, abs=1e-3)
    assert r["yillik_uretim_kwh"] == pytest.approx(25500)
    assert r["uygunluk_puani_yuzde"] == pytest.approx(96.30)
    assert r["golge_kaybi_yuzde"] == pytest.approx(3.70)
    assert r["co2_ton_yil"] == pytest.approx(16.18)
    # no bill: 25500 kWh x [0.5 x 3.677896, 1.0 x 5.444092] (tax-included two-tier tariff)
    assert r["yillik_tasarruf_tl_aralik"] == pytest.approx([46893.17, 138824.35], abs=0.01)
    assert r["tasarruf_yontemi"] == "genis_tahmin"
    assert r["engel_alani_m2"] == pytest.approx(50.0)
    assert r["cati_tipi"] == "flat"


def test_pitched_uses_5_5():
    r = cap(roof_type="pitched")
    assert r["kurulabilir_guc_kwp"] == pytest.approx(127.5 / 5.5, abs=1e-3)


def test_no_photo_case():
    r = cap(obstacle_ratio=None)
    assert r["kullanilabilir_alan_m2"] == pytest.approx(55.2)
    assert r["engel_alani_m2"] is None
    assert r["engel_orani"] is None


def test_engel_alani_rounded_to_2_decimals():
    r = cap(roof_area_m2=123.456, obstacle_ratio=0.333)
    assert r["engel_alani_m2"] == pytest.approx(41.11)


def test_confidence_levels():
    assert cap(obstacle_ratio=None)["guven_seviyesi"] == "Düşük"
    assert cap(image_view="oblique")["guven_seviyesi"] == "Düşük"
    top = cap(image_view="top-down")
    assert top["guven_seviyesi"] == "Orta"
    assert top["guven_nedenleri"]
    assert all(isinstance(x, str) for x in top["guven_nedenleri"])


def test_pitched_adds_orientation_reason():
    flat = cap(image_view="top-down")["guven_nedenleri"]
    pitched = cap(image_view="top-down", roof_type="pitched")["guven_nedenleri"]
    assert len(pitched) == len(flat) + 1


def test_shaded_above_ideal_is_capped():
    r = cap(yield_shaded=1400)
    assert r["uygunluk_puani_yuzde"] == 100.0
    assert r["golge_kaybi_yuzde"] == 0.0


def test_fully_shaded_roof_does_not_raise():
    r = cap(yield_shaded=0)
    assert r["uygunluk_puani_yuzde"] == 0.0
    assert r["golge_kaybi_yuzde"] == 100.0
    assert r["yillik_uretim_kwh"] == 0
    assert r["co2_ton_yil"] == 0
    assert r["yillik_tasarruf_tl_aralik"] == [0, 0]


def test_monthly_scaled_to_roof_totals():
    r = cap(monthly_ideal=MONTHLY_IDEAL, monthly_shaded=MONTHLY)
    kwp = r["kurulabilir_guc_kwp"]
    assert r["aylik_uretim_kwh"] == pytest.approx([kwp * m for m in MONTHLY], abs=0.2)
    assert r["aylik_ideal_kwh"] == pytest.approx([kwp * m for m in MONTHLY_IDEAL], abs=0.2)
    assert len(r["aylik_uretim_kwh"]) == 12


def test_monthly_sum_matches_annual():
    r = cap(yield_shaded=sum(MONTHLY), yield_ideal=sum(MONTHLY) + 12,
            monthly_ideal=MONTHLY_IDEAL, monthly_shaded=MONTHLY)
    assert sum(r["aylik_uretim_kwh"]) == pytest.approx(r["yillik_uretim_kwh"], abs=0.6)


def test_monthly_none_by_default():
    r = cap()
    assert r["aylik_uretim_kwh"] is None and r["aylik_ideal_kwh"] is None


@pytest.mark.parametrize("key", ["monthly_ideal", "monthly_shaded"])
@pytest.mark.parametrize("bad", [
    [1] * 11,
    [1] * 13,
    [1] * 11 + [-1],
    [1] * 11 + [None],
    [1] * 11 + ["a"],
    [1] * 11 + [float("nan")],
    [1] * 11 + [True],
])
def test_monthly_validation(key, bad):
    with pytest.raises(ValueError):
        cap(**{key: bad})


@pytest.mark.parametrize("overrides", [
    {"roof_area_m2": 0},
    {"roof_area_m2": -5},
    {"obstacle_ratio": -0.1},
    {"obstacle_ratio": 1.0},
    {"usable_base_area_m2": -1},
    {"usable_base_area_m2": float("nan")},
    {"roof_type": "dome"},
    {"yield_ideal": 0},
    {"yield_shaded": -1},
    {"image_view": "side"},
])
def test_input_validation(overrides):
    with pytest.raises(ValueError):
        cap(**overrides)


def test_ratio_zero_is_valid():
    assert cap(obstacle_ratio=0)["kullanilabilir_alan_m2"] == pytest.approx(170.0)


def test_run_scenario_is_stub():
    with pytest.raises(NotImplementedError):
        run_scenario({}, "x")


# ------------------------------------------------------------------ usable base area (building polygon)

def test_usable_base_with_photo_uses_ratio_only_and_no_edge_factor():
    r = cap(usable_base_area_m2=100.0, obstacle_ratio=0.25)
    assert r["kullanilabilir_alan_m2"] == pytest.approx(75.0)  # 100 x (1 - 0.25); EDGE_FACTOR not applied again


def test_usable_base_without_photo_uses_ratio_over_edge_factor():
    r = cap(usable_base_area_m2=100.0, obstacle_ratio=None)
    assert r["kullanilabilir_alan_m2"] == pytest.approx(100.0 * 0.276 / 0.85, abs=0.01)


def test_zero_usable_base_gives_zero_capacity():
    r = cap(usable_base_area_m2=0.0, obstacle_ratio=None)
    assert r["kullanilabilir_alan_m2"] == 0 and r["kurulabilir_guc_kwp"] == 0 and r["yillik_uretim_kwh"] == 0


def test_without_usable_base_behaviour_is_unchanged():
    assert cap()["kullanilabilir_alan_m2"] == pytest.approx(127.5)
    assert cap(obstacle_ratio=None)["kullanilabilir_alan_m2"] == pytest.approx(55.2)


# ------------------------------------------------------------------ confidence (P1 rule)

def test_confidence_polygon_without_photo_is_orta_and_manual_without_photo_is_dusuk():
    assert cap(usable_base_area_m2=100.0, obstacle_ratio=None)["guven_seviyesi"] == "Orta"
    assert cap(obstacle_ratio=None)["guven_seviyesi"] == "Düşük"


def test_confidence_polygon_with_top_down_photo_is_orta_not_yuksek():
    assert cap(usable_base_area_m2=100.0, image_view="top-down")["guven_seviyesi"] == "Orta"


def test_confidence_unknown_roof_type_drops_one_level_but_not_below_dusuk():
    assert cap(usable_base_area_m2=100.0, obstacle_ratio=None, roof_type_known=False)["guven_seviyesi"] == "Düşük"
    assert cap(obstacle_ratio=None, roof_type_known=False)["guven_seviyesi"] == "Düşük"
    assert cap(image_view="top-down", roof_type_known=False)["guven_seviyesi"] == "Düşük"


def test_confidence_ignores_the_bill():
    base = cap(usable_base_area_m2=100.0, obstacle_ratio=None)
    with_bill = cap(usable_base_area_m2=100.0, obstacle_ratio=None, monthly_bill_tl=1500)
    assert base["guven_seviyesi"] == with_bill["guven_seviyesi"]
    assert base["guven_nedenleri"] == with_bill["guven_nedenleri"]


def test_improvement_actions_list_only_what_would_help():
    best = cap(usable_base_area_m2=100.0, image_view="top-down")
    assert best["guven_iyilestirme"] == []
    worst = cap(obstacle_ratio=None, roof_type_known=False)
    text = " ".join(worst["guven_iyilestirme"])
    assert "fotoğrafını yükleyin" in text and "Çatı tipini seçin" in text and "binayı seçin" in text
    assert "Faturanızı" not in text  # the bill hint belongs to the TL card, not to the confidence level


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


def test_savings_from_bill_uses_the_given_monthly_production():
    sv = calc.savings_from_bill(AT_LIMIT, [20.0] * 12)  # C = 240 kWh, all LOW tier
    assert sv["range"][0] == pytest.approx(12 * 10 * LOW)  # share 0.5
    assert sv["range"][1] == pytest.approx(12 * 20 * LOW)  # share 1.0
    assert sv["coverage_pct"] == pytest.approx([100 * 10 / 240, 100 * 20 / 240])


def test_savings_scale_with_kwp_and_raw_e_m_is_never_used():
    kwargs = dict(roof_area_m2=500, obstacle_ratio=None, roof_type="flat", yield_ideal=1300, yield_shaded=sum(MONTHLY),
                  monthly_shaded=MONTHLY, usable_base_area_m2=100.0, monthly_bill_tl=1_000_000)
    one = compute_capacity(**kwargs)
    two = compute_capacity(**{**kwargs, "usable_base_area_m2": 200.0})
    assert two["kurulabilir_guc_kwp"] == pytest.approx(2 * one["kurulabilir_guc_kwp"], abs=0.002)
    ratio = two["yillik_tasarruf_tl_aralik"][1] / one["yillik_tasarruf_tl_aralik"][1]
    assert ratio == pytest.approx(2.0, rel=0.05)  # a huge bill: nothing is capped, savings follow kWp
    # raw per-kWp E_m would give `raw`; the roof-scaled saving is exactly kWp times larger
    raw = sum(calc.monthly_savings(calc.kwh_from_bill(1_000_000), m, 1.0) for m in MONTHLY)
    assert one["yillik_tasarruf_tl_aralik"][1] == pytest.approx(raw * one["kurulabilir_guc_kwp"], rel=0.01)


def test_monthly_production_sums_to_displayed_annual_kwh():
    r = cap(yield_ideal=sum(MONTHLY) + 10, yield_shaded=sum(MONTHLY), monthly_shaded=MONTHLY, monthly_ideal=MONTHLY)
    assert sum(r["aylik_uretim_kwh"]) == pytest.approx(r["yillik_uretim_kwh"], abs=0.6)


def test_savings_never_exceed_the_annual_bill():
    for bill in (300, 883, 1500, 5000):
        r = cap(monthly_bill_tl=bill, usable_base_area_m2=300.0, obstacle_ratio=None, monthly_shaded=MONTHLY)
        assert r["yillik_tasarruf_tl_aralik"][1] <= 12 * bill + 0.01
        assert max(r["fatura_karsilama_yuzde_aralik"]) <= 100.0


def test_coverage_display_is_capped_at_100():
    r = cap(monthly_bill_tl=300, usable_base_area_m2=300.0, obstacle_ratio=None, monthly_shaded=MONTHLY)
    assert r["fatura_karsilama_yuzde_aralik"] == [100.0, 100.0]


def test_small_bill_on_a_large_roof_triggers_the_shared_roof_flag():
    r = cap(monthly_bill_tl=300, usable_base_area_m2=300.0, obstacle_ratio=None, monthly_shaded=MONTHLY)
    assert r["tek_daire_uyarisi"] is True and r["kendi_kullanim_orani_yuzde"] < 50


def test_a_bill_matching_the_roof_does_not_trigger_the_flag():
    # tiny roof, big bill: all production is used and coverage stays low
    r = cap(monthly_bill_tl=6000, usable_base_area_m2=20.0, obstacle_ratio=None, monthly_shaded=MONTHLY)
    assert r["tek_daire_uyarisi"] is False and r["kendi_kullanim_orani_yuzde"] == pytest.approx(100.0)
    assert r["fatura_karsilama_yuzde_aralik"][1] < 100


@pytest.mark.parametrize("bill", [None, 0, -50, float("nan")])
def test_empty_zero_or_negative_bill_uses_the_fallback_range(bill):
    r = cap(monthly_bill_tl=bill)
    assert r["tasarruf_yontemi"] == "genis_tahmin" and r["fatura_tl_ay"] is None
    kwh = r["yillik_uretim_kwh"]
    assert r["yillik_tasarruf_tl_aralik"] == pytest.approx([kwh * 0.5 * LOW, kwh * HIGH], abs=0.02)
    assert r["fatura_karsilama_yuzde_aralik"] is None and r["tek_daire_uyarisi"] is False


def test_tax_factor_is_applied_exactly_once():
    r = cap(usable_base_area_m2=100.0, obstacle_ratio=None)
    once = r["yillik_uretim_kwh"] * 0.5 * 2.918965 * 1.26
    assert r["yillik_tasarruf_tl_aralik"][0] == pytest.approx(once, rel=1e-5)
    assert r["yillik_tasarruf_tl_aralik"][0] != pytest.approx(once * 1.26, rel=0.01)
    assert r["tarife_dusuk_tl_kwh"] == 3.68 and r["tarife_yuksek_tl_kwh"] == 5.44
    assert "EPDK" in r["tarife_kaynagi"]


def test_old_tariff_parameter_is_gone():
    with pytest.raises(TypeError):
        compute_capacity(100, None, "flat", 1300, 1300, tariff=3.0)
