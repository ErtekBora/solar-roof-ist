import pytest

from core.calc import compute_capacity, run_scenario

MONTHLY = [50, 60, 90, 110, 140, 160, 170, 160, 120, 90, 55, 45]
MONTHLY_IDEAL = [m + 1 for m in MONTHLY]


def cap(**overrides):
    args = dict(
        roof_area_m2=200, obstacle_ratio=0.25, roof_type="flat",
        yield_ideal=1350, yield_shaded=1300, tariff=3.0,
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
    assert r["yillik_tasarruf_tl_aralik"] == pytest.approx([38250, 76500])
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
    {"tariff": 0},
    {"tariff": -1},
    {"tariff": None},
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
