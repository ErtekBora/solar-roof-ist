import math

import pytest

from core import building_mode as bm
from core import buildings, calc, config, explain
from core.building_mode import (
    allocate_equal_share,
    analyze_building,
    compute_segments,
    estimate_flat_count,
    face_azimuths,
    face_specs,
    payback_range,
    to_pvgis_aspect,
)
from core.cost import clamp_flats, estimate_install_cost

E_M = [60, 75, 105, 130, 150, 160, 170, 160, 130, 100, 70, 55]


def pv(e_y):
    """A PVcalc-like result; e_m scaled so that it sums to e_y."""
    k = e_y / sum(E_M)
    return {"e_y": float(e_y), "e_m": [m * k for m in E_M], "h_i_y": 1700.0, "l_tg": -6.5, "l_total": -21.0, "angle": 33, "aspect": 8}


def flat_solar(shaded=1400.0, ideal=1410.0):
    return {"ideal": pv(ideal), "shaded": pv(shaded), "angle": 33, "aspect": 8}


def faces_for(ridge, ys):
    return [{**spec, "solar": None if y is None else pv(y)} for spec, y in zip(face_specs(ridge), ys)]


def analyze(**kw):
    base = dict(
        roof_area_m2=300.0, usable_base_area_m2=240.0, obstacle_ratio=0.0, image_view="top-down", structure="flat",
        flat_share_pct=50, n_flats=6, n_flats_estimated=False, bill_tl=None, flat_solar=flat_solar(),
        faces=faces_for(0, [1250, 1250]),
    )
    base.update(kw)
    return analyze_building(**base)


# ------------------------------------------------------------------ flat count

def test_flat_count_from_num_floors():
    assert estimate_flat_count(5, None, 240) == 10  # 5 x 240 / 120
    assert estimate_flat_count(4.0, 99, 120) == 4  # num_floors wins over the height


def test_flat_count_from_height_when_floors_unknown():
    assert estimate_flat_count(None, 15.0, 240) == 10  # round(15 / 3) = 5 floors
    assert estimate_flat_count(0, 15.0, 240) == 10


def test_height_of_one_metre_gives_one_floor_and_at_least_one_flat():
    assert estimate_flat_count(None, 1.0, 120) == 1
    assert estimate_flat_count(None, 1.0, 5) >= 1


def test_small_and_large_footprints_give_different_counts():
    assert estimate_flat_count(5, None, 100) < estimate_flat_count(5, None, 400)


@pytest.mark.parametrize(
    "floors, height, area",
    [(None, None, 0), (None, None, None), (0, 0, 10), (-2, -5, -10), (3, float("nan"), float("nan"))],
)
def test_flat_count_is_never_zero(floors, height, area):
    assert estimate_flat_count(floors, height, area) >= 1


@pytest.mark.parametrize("n", [0, -1, None, float("nan"), "x"])
def test_clamp_flats(n):
    assert clamp_flats(n) == 1


def test_user_override_is_used_and_clamped():
    assert analyze(n_flats=12, n_flats_estimated=False)["daire_sayisi"] == 12
    r = analyze(n_flats=0)
    assert r["daire_sayisi"] == 1 and r["daire_basi_maliyet_tl_aralik"] == r["toplam_maliyet_tl_aralik"]


# ----------------------------------------------------------------------- faces

@pytest.mark.parametrize("north, aspect", [(180, 0), (90, -90), (270, 90), (0, 180), (135, -45), (225, 45)])
def test_north_azimuth_to_pvgis_aspect(north, aspect):
    assert to_pvgis_aspect(north) == aspect


def test_ns_ridge_gives_east_and_west_faces():
    specs = face_specs(0)
    assert sorted(s["name"] for s in specs) == ["Batı", "Doğu"]
    assert sorted(s["aspect"] for s in specs) == [-90, 90]


def test_ew_ridge_gives_north_and_south_faces_and_north_is_dropped():
    specs = face_specs(90)
    assert sorted(s["name"] for s in specs) == ["Güney", "Kuzey"]
    south = next(s for s in specs if s["name"] == "Güney")
    assert south["aspect"] == 0 and sorted(face_azimuths(90)) == [0, 180]
    faces = faces_for(90, [1400 if s["name"] == "Güney" else 800 for s in specs])
    seg = compute_segments(100, 0.0, flat_solar(1400), faces)
    by = {s["yon"]: s for s in seg["segmentler"] if s["tur"] == "face"}
    assert by["Kuzey"]["durum"] == "elendi" and by["Güney"]["durum"] == "kullanildi"
    assert "Kuzeye bakan çatı yüzü verimsiz olduğu için hesaba katılmadı" in seg["uyarilar"]


def test_dropped_face_contributes_nothing_and_costs_nothing():
    faces = faces_for(90, [1400 if s["name"] == "Güney" else 800 for s in face_specs(90)])
    seg = compute_segments(100, 0.0, flat_solar(1400), faces)
    north = next(s for s in seg["segmentler"] if s.get("yon") == "Kuzey")
    assert north["kwp"] == 0 and sum(north["aylik_kwh"]) == 0
    south = next(s for s in seg["segmentler"] if s.get("yon") == "Güney")
    assert seg["kwp"] == pytest.approx(south["kwp"]) and seg["kwh"] == pytest.approx(sum(south["aylik_kwh"]))
    assert estimate_install_cost(seg["kwp"])[:2] == estimate_install_cost(south["kwp"])[:2]  # its area is simply unused


def test_eighty_percent_rule_boundary():
    ref = flat_solar(1000)
    seg = compute_segments(100, 0.0, ref, faces_for(0, [799.9, 800.0]))
    assert [s["durum"] for s in seg["segmentler"][1:]] == ["elendi", "kullanildi"]  # exactly 80 % stays
    assert seg["elenen_yuz"] == 1


def test_plan_versus_surface_area_rules():
    seg = compute_segments(100.0, 0.0, flat_solar(), faces_for(0, [1300, 1300]))
    assert seg["kwp"] == pytest.approx(100 / math.cos(math.radians(30)) / 5.5)
    for face in seg["segmentler"][1:]:
        assert face["plan_m2"] == pytest.approx(50) and face["yuzey_m2"] == pytest.approx(50 / math.cos(math.radians(30)))
        assert face["kwp"] == pytest.approx(face["yuzey_m2"] / 5.5)
    flat = compute_segments(100.0, 1.0, flat_solar(), faces_for(0, [1300, 1300]))
    assert flat["kwp"] == pytest.approx(100 / 6.5)
    assert all(s["durum"] == "kullanilmadi" for s in flat["segmentler"][1:])  # no pitched area, faces unused


def test_segment_sums_and_monthly_values():
    seg = compute_segments(100.0, 0.4, flat_solar(1400), faces_for(0, [1300, 1200]))
    flat, east, west = seg["segmentler"]
    assert flat["plan_m2"] == pytest.approx(40) and east["plan_m2"] == pytest.approx(30)
    assert seg["kwp"] == pytest.approx(flat["kwp"] + east["kwp"] + west["kwp"])
    assert seg["kwh"] == pytest.approx(sum(seg["aylik_kwh"]))
    assert seg["kwh"] == pytest.approx(flat["kwp"] * 1400 + east["kwp"] * 1300 + west["kwp"] * 1200)
    assert seg["aylik_kwh"][6] == pytest.approx(sum(s["aylik_kwh"][6] for s in seg["segmentler"]))


def test_face_failure_continues_with_the_other_segments_and_warns():
    r = analyze(structure="mixed", flat_share_pct=50, faces=faces_for(0, [1250, None]))
    sc = r["senaryolar"][0]
    assert sc["alinamayan_yuz"] == 1 and sc["kwp"] > 0
    assert "Bir çatı yüzü için güneş verisi alınamadı; sonuç eksik olabilir" in r["uyarilar"]
    assert [s["durum"] for s in sc["segmentler"]] == ["kullanildi", "kullanildi", "alinamadi"]
    ok = analyze(structure="mixed", flat_share_pct=50)
    assert ok["guven_seviyesi"] == "Orta" and r["guven_seviyesi"] == "Düşük"  # an unavailable face drops one level


def test_unavailable_face_does_not_matter_for_a_fully_flat_roof():
    r = analyze(structure="flat", faces=faces_for(0, [None, None]))
    assert r["uyarilar"] == [] and r["guven_seviyesi"] == "Orta"


def test_unknown_structure_computes_both_extremes_as_ranges():
    r = analyze(structure="unknown", bill_tl=1500.0)
    flat, pitched = r["senaryolar"]
    assert (flat["duz_pay"], pitched["duz_pay"]) == (1.0, 0.0)
    lo, hi = r["kurulabilir_guc_kwp_aralik"]
    assert lo == pytest.approx(min(flat["kwp"], pitched["kwp"])) and hi == pytest.approx(max(flat["kwp"], pitched["kwp"]))
    assert lo < hi
    for key in ("yillik_uretim_kwh_aralik", "toplam_maliyet_tl_aralik", "daire_basi_tasarruf_tl_aralik"):
        assert r[key][0] <= r[key][1]
    assert r["guven_seviyesi"] == "Düşük"  # Orta base, one level down
    assert analyze(structure="mixed")["guven_seviyesi"] == "Orta"  # a user-set Karma counts as known


def test_unknown_range_is_correct_when_a_north_face_makes_pitched_smaller_than_flat():
    faces = faces_for(90, [1300 if s["name"] == "Güney" else 700 for s in face_specs(90)])
    r = analyze(structure="unknown", faces=faces)
    flat, pitched = r["senaryolar"]
    assert pitched["kwp"] < flat["kwp"]
    low, high = r["toplam_maliyet_tl_aralik"]
    assert low == pytest.approx(min(flat["maliyet_tl"][0], pitched["maliyet_tl"][0]))
    assert high == pytest.approx(max(flat["maliyet_tl"][1], pitched["maliyet_tl"][1]))


def test_totals_are_sums_over_used_segments_and_co2_follows_kwh():
    r = analyze(structure="mixed", flat_share_pct=30)
    sc = r["senaryolar"][0]
    assert r["yillik_uretim_kwh_aralik"][0] == pytest.approx(sum(sum(s["aylik_kwh"]) for s in sc["segmentler"]))
    assert r["co2_ton_yil_aralik"][1] == pytest.approx(sc["kwh"] * config.CO2_FACTOR_T_PER_MWH / 1000)


def test_manual_area_without_polygon_works():
    r = analyze(usable_base_area_m2=None, obstacle_ratio=None, image_view=None)
    assert r["kullanilabilir_alan_m2"] == pytest.approx(300 * 0.276) and r["guven_seviyesi"] == "Düşük"


def test_invalid_inputs_raise_value_error():
    for kw in ({"roof_area_m2": 0}, {"obstacle_ratio": 1.0}, {"structure": "weird"}):
        with pytest.raises(ValueError):
            analyze(**kw)


def test_zero_usable_area_gives_zero_everything_without_errors():
    r = analyze(usable_base_area_m2=0.0, obstacle_ratio=0.0, bill_tl=1500.0)
    assert r["kurulabilir_guc_kwp_aralik"] == [0, 0] and r["toplam_maliyet_tl_aralik"] == [0, 0]
    assert r["geri_odeme_yil_aralik"] == [None, None]
    assert "kurulabilir panel yok" in r["maliyet_bayraklari"]


# --------------------------------------------------------------------- sharing

def test_equal_split_among_flats():
    monthly = [100.0] * 12
    one, four = allocate_equal_share(monthly, 1, 2000.0), allocate_equal_share(monthly, 4, 2000.0)
    assert four["daire_uretim_kwh"] == pytest.approx(one["daire_uretim_kwh"] / 4) == pytest.approx(300)
    assert four["tasarruf_tl"][1] < one["tasarruf_tl"][1]


def test_per_flat_savings_never_exceed_the_flat_bill_and_coverage_is_capped():
    out = allocate_equal_share([5000.0] * 12, 2, 300.0)  # production far above consumption
    assert out["tasarruf_tl"][1] <= 12 * 300.0 + 1e-6
    assert out["karsilama_yuzde"][1] <= 100.0
    assert out["kalan_fatura_tl_ay"][0] >= 0


def test_surplus_beyond_consumption_is_valued_at_zero():
    big = allocate_equal_share([5000.0] * 12, 1, 300.0)["tasarruf_tl"]
    bigger = allocate_equal_share([50000.0] * 12, 1, 300.0)["tasarruf_tl"]
    assert big[1] == pytest.approx(bigger[1])


def test_savings_use_the_existing_mesken_functions():
    c = calc.kwh_from_bill(1500.0)
    expected = sum(calc.monthly_savings(c, 100.0 / 5, 1.0) for _ in range(12))
    assert allocate_equal_share([100.0] * 12, 5, 1500.0)["tasarruf_tl"][1] == pytest.approx(expected)


def test_empty_bill_fallback_range_is_labelled_wide():
    out = allocate_equal_share([100.0] * 12, 4, None)
    annual = 1200 / 4
    assert out["yontem"] == "genis_tahmin" and out["karsilama_yuzde"] is None and out["varsayim"] is None
    assert out["tasarruf_tl"] == pytest.approx([annual * calc.LOW, annual * calc.HIGH])
    assert allocate_equal_share([100.0] * 12, 4, 0)["yontem"] == "genis_tahmin"


def test_assumption_text_present_only_with_a_bill():
    with_bill = allocate_equal_share([100.0] * 12, 3, 1000.0)["varsayim"]
    assert with_bill == bm.ASSUMPTION_CONSTANT_CONSUMPTION
    assert "her ayı için ortalama tüketim" in with_bill and "0 TL sayıldığı için tasarruf ihtiyatlı" in with_bill
    assert analyze(bill_tl=1000.0)["tuketim_varsayimi"] == with_bill and analyze()["tuketim_varsayimi"] is None


@pytest.mark.parametrize("n", [0, -4, None])
def test_sharing_clamps_the_flat_count(n):
    assert allocate_equal_share([100.0] * 12, n, 1000.0)["daire_sayisi"] == 1


def test_coverage_and_remaining_bill_in_the_result():
    r = analyze(bill_tl=1500.0, n_flats=10)
    lo, hi = r["fatura_karsilama_yuzde_aralik"]
    assert 0 <= lo <= hi <= 100
    rem_lo, rem_hi = r["kalan_fatura_tl_ay_aralik"]
    assert rem_hi == pytest.approx(1500 * (1 - lo / 100)) and rem_lo == pytest.approx(1500 * (1 - hi / 100))
    assert analyze()["fatura_karsilama_yuzde_aralik"] is None


# -------------------------------------------------------------------- payback

def test_payback_range_ordering_and_formula():
    lo, hi = payback_range(100_000, 150_000, 10_000, 20_000)
    assert (lo, hi) == (5.0, 15.0) and lo <= hi


def test_larger_savings_shorten_the_payback():
    slow = payback_range(100_000, 150_000, 8_000, 12_000)
    fast = payback_range(100_000, 150_000, 16_000, 24_000)
    assert fast[0] < slow[0] and fast[1] < slow[1]


@pytest.mark.parametrize(
    "sav_low, sav_high, expected",
    [(0, 0, (None, None)), (0.5, 0.9, (None, None)), (0, 20_000, (5.0, None)), (0.99, 20_000, (5.0, None))],
)
def test_low_savings_give_none_not_an_exception(sav_low, sav_high, expected):
    assert payback_range(100_000, 150_000, sav_low, sav_high) == expected


def test_payback_absurd_years_and_bad_inputs_give_none():
    assert payback_range(100, 100, 1.0, 1.0) == (100.0, 100.0) or payback_range(100, 100, 1.0, 1.0) == (None, None)
    assert payback_range(10_000_000, 12_000_000, 1000, 2000) == (None, None)  # beyond PAYBACK_MAX_YEARS
    assert payback_range(0, 0, 1000, 1000) == (None, None)
    assert payback_range(None, 1, float("nan"), 5) == (None, None)


def test_payback_in_the_result():
    assert analyze(bill_tl=0.05)["geri_odeme_yil_aralik"] == [None, None]
    assert analyze(bill_tl=3000.0)["geri_odeme_yil_aralik"][0] is not None


# ------------------------------------------------------------------- explainer

def test_forbidden_terms_follow_the_payload():
    base = explain.forbidden_terms({})
    assert set(base) == set(explain.FINANCIAL_TERMS)
    with_cost = explain.forbidden_terms({"kurulum_maliyeti_tl": [100, 200]})
    assert "maliyet" not in with_cost and "yatırım" not in with_cost and "geri ödeme" in with_cost and "amorti" in with_cost
    with_pb = explain.forbidden_terms({"geri_odeme_yil": [5.0, 9.0]})
    assert "amorti" not in with_pb and "geri ödeme" not in with_pb and "maliyet" in with_pb
    assert explain.forbidden_terms({"geri_odeme_yil": "hesaplanamaz"}) == base
    both = explain.forbidden_terms({"kurulum_maliyeti_tl": 5, "geri_odeme_yil": [1, 2]})
    assert set(both) == {"geri dönüş süresi", "teşvik", "kredi", "taksit", "kazanç süresi"}


def _banned_line(prompt):
    return prompt.split("Şu sözcükleri hiç kullanma:")[1].split("\n")[0]


def test_payback_words_allowed_only_with_a_numeric_payback_in_the_payload():
    p = explain.build_payload(analyze(bill_tl=3000.0))
    pb = p["geri_odeme_yil"]
    assert isinstance(pb, (list, int, float))
    lo = pb[0] if isinstance(pb, list) else pb
    text = f"Geri ödeme süresi yaklaşık {explain.tr_num(lo, 1)} yıl."
    assert explain.validate_explanation_numbers(text, p)[0] is True
    p_none = explain.build_payload(analyze(bill_tl=0.05))
    assert p_none["geri_odeme_yil"] == "hesaplanamaz"
    assert explain.validate_explanation_numbers(text, p_none)[0] is False
    assert "geri ödeme" in _banned_line(explain.system_prompt_for(p_none))
    assert "geri ödeme" not in _banned_line(explain.system_prompt_for(p))


def test_cost_words_allowed_only_with_a_cost_range_in_the_payload():
    p = explain.build_payload(analyze())
    cost_lo = p["kurulum_maliyeti_tl"][0]
    text = f"Kurulum maliyeti yaklaşık {explain.tr_num(cost_lo)} TL."
    assert explain.validate_explanation_numbers(text, p)[0] is True
    without = {k: v for k, v in p.items() if k != "kurulum_maliyeti_tl"}
    without["x"] = cost_lo
    assert explain.validate_explanation_numbers(text, without)[0] is False
    assert "maliyet" in _banned_line(explain.system_prompt_for(without))
    assert explain.validate_explanation_numbers("Teşvik 5 yıl sürer, kredi alınır.", p)[0] is False  # others stay forbidden


def test_payload_carries_building_fields_and_no_hidden_numbers():
    r = analyze(
        structure="mixed", flat_share_pct=40, bill_tl=1500.0, n_flats=8, n_flats_estimated=True,
        faces=faces_for(0, [1250, None]),
    )
    p = explain.build_payload(r)
    for key in (
        "cati_yapisi", "duz_pay_yuzde", "daire_sayisi", "daire_sayisi_tahmini", "kurulabilir_guc_kwp", "yillik_uretim_kwh",
        "co2_ton_yil", "kurulum_maliyeti_tl", "daire_basi_maliyet_tl", "daire_basi_tasarruf_tl_yil", "geri_odeme_yil",
        "fatura_karsilama_yuzde_aralik", "tuketim_varsayimi", "maliyet_bandi_tl_kwp", "senaryolar", "uyarilar",
    ):
        assert key in p, key
    assert p["daire_sayisi"] == 8 and p["daire_sayisi_tahmini"] is True and p["duz_pay_yuzde"] == 40
    statuses = {seg["durum"] for sc in p["senaryolar"] for seg in sc["cati_yuzleri"]}
    assert statuses == {"kullanıldı", "güneş verisi alınamadı"}
    assert "aylik_kwh" not in str(p)


def test_template_passes_the_number_check_for_every_structure():
    for structure in ("flat", "pitched", "mixed", "unknown"):
        for bill in (None, 1500.0, 0.05):
            r = analyze(structure=structure, bill_tl=bill, n_flats_estimated=True)
            text = explain.build_fallback_explanation(r)
            ok, invalid = explain.validate_explanation_numbers(text, explain.build_payload(r))
            assert ok, (structure, bill, invalid, text)
            assert "tek daire" not in text.lower() and "faturanızın" not in text


def test_template_mentions_payback_only_when_computable():
    assert "geri ödeme süresi" in explain.build_fallback_explanation(analyze(bill_tl=3000.0))
    assert "geri ödeme" not in explain.build_fallback_explanation(analyze(bill_tl=0.05))


def test_prompt_is_building_framed():
    prompt = explain.SYSTEM_PROMPT
    assert "EŞİT paylaştır" in prompt and "daire başına" in prompt.lower()
    assert "faturanızın %60" not in prompt


def test_ridge_azimuth_of_footprints():
    from shapely.affinity import rotate
    from shapely.geometry import box

    assert buildings.ridge_azimuth_deg(box(0, 0, 10, 30)) == pytest.approx(0, abs=1e-6)  # long side N-S
    assert buildings.ridge_azimuth_deg(box(0, 0, 30, 10)) == pytest.approx(90, abs=1e-6)  # long side E-W
    rotated = rotate(box(0, 0, 10, 30), -30, origin=(5, 15))  # shapely angles are counter-clockwise
    assert buildings.ridge_azimuth_deg(rotated) == pytest.approx(30, abs=1e-6)


# ------------------------------------------------- equal TL discount per flat

def test_equal_tl_discount_times_n_equals_the_building_total():
    for n in (1, 4, 10):
        out = allocate_equal_share([120.0, 150, 200, 260, 320, 380, 400, 380, 300, 220, 150, 110], n, 1500.0)
        for i in (0, 1):
            assert out["indirim_yillik_tl"][i] * n == pytest.approx(out["bina_toplam_tasarruf_tl"][i])
        c = calc.kwh_from_bill(1500.0)
        direct = [n * calc.monthly_savings(c, m / n, 1.0) for m in [120.0, 150, 200, 260, 320, 380, 400, 380, 300, 220, 150, 110]]
        assert out["bina_toplam_tasarruf_tl"][1] == pytest.approx(sum(direct))


def test_discount_never_exceeds_the_average_flat_bill_even_monthly():
    out = allocate_equal_share([9000.0] * 12, 2, 300.0)
    assert out["indirim_yillik_tl"][1] <= 12 * 300.0 + 1e-6
    assert all(v <= 300.0 + 1e-6 for v in out["indirim_aylik_tl"][1])
    assert out["indirim_ay_ortalama_tl"][1] <= 300.0 + 1e-6


def test_monthly_discount_varies_with_production():
    monthly = [60.0, 75, 105, 130, 150, 160, 170, 160, 130, 100, 70, 55]
    low, high = allocate_equal_share(monthly, 1, 1500.0)["indirim_aylik_tl"]
    assert len(set(round(v, 6) for v in high)) > 1
    assert high[6] > high[0] and all(h >= lo for h, lo in zip(high, low))
    assert sum(high) / 12 == pytest.approx(allocate_equal_share(monthly, 1, 1500.0)["indirim_ay_ortalama_tl"][1])


def test_both_shares_are_returned_and_the_fallback_is_divided_per_flat():
    out = allocate_equal_share([100.0] * 12, 4, 1500.0)
    assert out["daire_aylik_kwh"] == [25.0] * 12 and out["daire_uretim_kwh"] == pytest.approx(300)
    wide = allocate_equal_share([100.0] * 12, 4, None)
    assert wide["indirim_aylik_tl"] is None
    assert wide["indirim_yillik_tl"] == pytest.approx(calc.fallback_savings(300))
    assert wide["bina_toplam_tasarruf_tl"][0] == pytest.approx(4 * wide["indirim_yillik_tl"][0])


def test_payback_uses_the_per_flat_discount_and_notes_are_in_the_result():
    r = analyze(bill_tl=3000.0, n_flats=5)
    lo, hi = r["daire_basi_maliyet_tl_aralik"]
    s_lo, s_hi = r["daire_basi_tasarruf_tl_aralik"]
    assert r["geri_odeme_yil_aralik"] == list(payback_range(lo, hi, s_lo, s_hi))
    assert r["bina_toplam_tasarruf_tl_aralik"][1] == pytest.approx(5 * s_hi)
    assert r["daire_basi_indirim_tl_ay_aralik"][1] == pytest.approx(s_hi / 12)
    assert r["indirim_notu"] == "Yıllık ortalamadır; kışın daha az, yazın daha fazla indirim olur ve indirim faturayı aşamaz."
    assert r["paylasim_notu"].startswith("Fiziksel paylaşımda (her daireye eşit kWh)") and "aidattan eşit düşülmesi" in r["paylasim_notu"]


def test_payload_carries_the_discount_fields_and_template_validates():
    r = analyze(bill_tl=3000.0, n_flats=5)
    p = explain.build_payload(r)
    assert "daire_basi_indirim_tl_ay" in p and "bina_toplam_tasarruf_tl_yil" in p
    text = explain.build_fallback_explanation(r)
    assert "sabit indirim" in text and explain.validate_explanation_numbers(text, p)[0]


# ------------------------------------------------- monthly netting (share 1.0 only)

def test_bill_case_uses_share_one_only_so_savings_have_no_lower_bound_spread():
    monthly = [60.0, 75, 105, 130, 150, 160, 170, 160, 130, 100, 70, 55]
    out = allocate_equal_share(monthly, 3, 1500.0)
    c = calc.kwh_from_bill(1500.0)
    expected = sum(calc.monthly_savings(c, m / 3, 1.0) for m in monthly)
    assert out["indirim_yillik_tl"] == [pytest.approx(expected)] * 2
    half = sum(calc.monthly_savings(c, m / 3, 0.5) for m in monthly)
    assert out["indirim_yillik_tl"][0] > half  # the 0.5 lower bound is gone
    assert out["karsilama_yuzde"][0] == out["karsilama_yuzde"][1]


def test_single_scenario_bill_range_comes_only_from_cost_bands():
    r = analyze(bill_tl=3000.0, n_flats=5, structure="flat")
    s_lo, s_hi = r["daire_basi_tasarruf_tl_aralik"]
    assert s_lo == pytest.approx(s_hi)
    c_lo, c_hi = r["daire_basi_maliyet_tl_aralik"]
    pb_lo, pb_hi = r["geri_odeme_yil_aralik"]
    assert pb_hi / pb_lo == pytest.approx(c_hi / c_lo)  # payback spread = cost spread only
    wide = analyze(structure="flat", n_flats=5)  # no bill: wide fallback keeps its savings spread
    assert wide["daire_basi_tasarruf_tl_aralik"][0] < wide["daire_basi_tasarruf_tl_aralik"][1]


def test_unknown_structure_range_comes_from_scenarios_and_costs():
    r = analyze(bill_tl=3000.0, structure="unknown", n_flats=5)
    assert r["daire_basi_tasarruf_tl_aralik"][0] <= r["daire_basi_tasarruf_tl_aralik"][1]
    assert r["geri_odeme_yil_aralik"][0] <= r["geri_odeme_yil_aralik"][1]


def test_netting_note_only_with_a_bill_and_in_the_payload():
    assert analyze(bill_tl=1500.0)["mahsuplasma_notu"] == bm.NETTING_NOTE
    assert bm.NETTING_NOTE.startswith("Aylık mahsuplaşma varsayımıyla hesaplanmıştır.")
    assert analyze()["mahsuplasma_notu"] is None
    assert explain.build_payload(analyze(bill_tl=1500.0))["mahsuplasma_varsayimi"] == bm.NETTING_ASSUMPTION
    assert "mahsuplasma_varsayimi" not in explain.build_payload(analyze())
    assert bm.NETTING_NOTE not in explain.build_fallback_explanation(analyze(bill_tl=1500.0))  # already on screen


# ----------------------------------------------------------- hip roofs, weights, layout capacity

def test_hip_face_specs_have_four_faces_with_quarter_weights():
    specs = face_specs(0, "hip")
    assert [s["name"] for s in specs] == ["Doğu", "Batı", "Kuzey", "Güney"]
    assert [s["aspect"] for s in specs] == [-90, 90, 180, 0] and all(s["weight"] == 0.25 for s in specs)
    assert all(s["weight"] == 0.5 for s in face_specs(0, "gable"))
    with pytest.raises(ValueError):
        face_specs(0, "mansard")


def test_pitched_plan_area_is_split_by_face_weights():
    faces = [{**s, "solar": pv(1300), "weight": w} for s, w in zip(face_specs(0, "hip"), [0.4, 0.4, 0.1, 0.1])]
    seg = compute_segments(100.0, 0.0, flat_solar(1400), faces)
    plans = [s["plan_m2"] for s in seg["segmentler"][1:]]
    assert plans == pytest.approx([40, 40, 10, 10]) and sum(plans) == pytest.approx(100)


def test_north_facing_hip_face_is_dropped_and_listed_as_unusable():
    faces = [{**s, "solar": pv(800 if s["name"] == "Kuzey" else 1300)} for s in face_specs(0, "hip")]
    seg = compute_segments(100.0, 0.0, flat_solar(1400), faces)
    north = next(s for s in seg["segmentler"] if s["ad"] == "Kuzey yüzü")
    assert north["durum"] == "elendi" and north["kwp"] == 0 and north["panel_sayisi"] == 0
    assert bm.unusable_faces(flat_solar(1400), faces) == {"Kuzey yüzü"}


def test_geometry_mode_kwp_is_panel_count_times_0_40():
    faces = [{**s, "solar": pv(1300)} for s in face_specs(0, "gable")]
    geometry = {"Doğu yüzü": {"plan_m2": 50.0, "panels": 20}, "Batı yüzü": {"plan_m2": 50.0, "panels": 17}}
    seg = compute_segments(999.0, 0.0, flat_solar(1400), faces, geometry)
    assert seg["kwp"] == pytest.approx(37 * config.PANEL_KWP) and seg["panel_sayisi"] == 37
    assert seg["kullanilabilir_plan_m2"] == pytest.approx(100.0)  # the usable share is not used
    assert seg["kwh"] == pytest.approx(37 * config.PANEL_KWP * 1300)


def test_numeric_panel_count_is_the_floor_of_kwp_over_0_40():
    seg = compute_segments(100.0, 1.0, flat_solar(1400), [])
    assert seg["segmentler"][0]["panel_sayisi"] == math.floor(100 / 6.5 / 0.4)
    assert bm.numeric_panel_count(0.79) == 1 and bm.numeric_panel_count(0.8) == 2 and bm.numeric_panel_count(0) == 0


def test_unknown_with_geometry_gives_a_panel_range():
    faces = [{**s, "solar": pv(1300)} for s in face_specs(0, "gable")]
    geo = [
        {"Düz teras": {"plan_m2": 200.0, "panels": 30}},
        {"Doğu yüzü": {"plan_m2": 100.0, "panels": 40}, "Batı yüzü": {"plan_m2": 100.0, "panels": 40}},
    ]
    r = analyze(structure="unknown", faces=faces, geometry=geo, pitched_type="gable")
    assert r["panel_sayisi_aralik"] == [30, 80] and r["kurulabilir_guc_kwp_aralik"] == pytest.approx([12.0, 32.0])
    assert r["guc_kaynagi"] == "geometri" and r["kullanilabilir_oran"] is None


def test_drawn_obstacles_replace_the_reason_but_never_raise_confidence_above_orta():
    r = analyze(structure="pitched", obstacle_ratio=None, image_view=None, drawn={"count": 1, "area_m2": 2.0, "marked": True})
    assert r["guven_seviyesi"] == "Orta" and "Engeller kullanıcı tarafından işaretlendi." in r["guven_nedenleri"]
    assert not any("ortalama orana göre" in x for x in r["guven_nedenleri"])
    assert "Çatı fotoğrafı yükleyin." not in r["guven_iyilestirme"]
    assert r["cizilen_engel_sayisi"] == 1 and r["cizilen_engel_alani_m2"] == 2.0
    plain = analyze(structure="pitched", obstacle_ratio=None, image_view=None)
    assert plain["cizilen_engel_sayisi"] is None and plain["guc_kaynagi"] == "ortalama"


def test_high_flat_share_flag_only_without_a_bill():
    r = analyze(n_flats=1, bill_tl=None)
    assert r["daire_payi_kwh_ay_aralik"][1] == pytest.approx(r["yillik_uretim_kwh_aralik"][1] / 12)
    assert r["daire_payi_yuksek"] and bm.high_share_note(r).startswith("Daire payı (≈")
    assert "Netleştirmek için ortalama daire faturasını girin." in bm.high_share_note(r)
    assert not analyze(n_flats=1, bill_tl=1500.0)["daire_payi_yuksek"]
    assert not analyze(n_flats=60)["daire_payi_yuksek"] and bm.high_share_note(analyze(n_flats=60)) is None


def test_usable_ratio_in_the_result_matches_the_rule():
    assert analyze(obstacle_ratio=None)["kullanilabilir_oran"] == pytest.approx(0.276 / 0.85)
    assert analyze(obstacle_ratio=0.2)["kullanilabilir_oran"] == pytest.approx(0.8)
    r = analyze(obstacle_ratio=0.0, capacity_source="engel_yok")
    assert r["guc_kaynagi"] == "engel_yok" and r["kullanilabilir_alan_m2"] == pytest.approx(240.0)
