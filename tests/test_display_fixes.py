"""Part A display fixes: nothing that was calculated before may change; only new display fields and texts."""
import json
from pathlib import Path

import pytest

import app
from core import building_mode, explain
from test_app import (  # noqa: F401  (fixtures and helpers shared with the app tests)
    _centre_obstacle, _draw_state, _fake_face, _fake_yield, _roof_analysis, _roof_inputs, building_point, mocked,
    sample_index,
)
from test_explain import make_result

GOLDEN = Path(__file__).parent / "fixtures" / "display_golden.json"


def _pvgis_calls(monkeypatch, inputs):
    """Number of PVGIS requests one 'Hesapla' fetch makes for these inputs (the fakes only count)."""
    from core import solar

    calls = []
    monkeypatch.setattr(solar, "get_solar_yield", lambda *a, **k: calls.append("flat") or _fake_yield(*a, **k))
    monkeypatch.setattr(solar, "get_face_yield", lambda *a, **k: calls.append("face") or _fake_face(*a, **k))
    app.fetch_roof_yields(inputs["lat"], inputs["lon"], app.pvgis_aspects(inputs))
    return len(calls)


def _snapshot(monkeypatch, sample_index, building_point):
    """Existing, pre-change result fields for the three reference cases."""
    out = {}
    cases = {
        "flat_numeric": dict(structure="flat", draw={}),
        "flat_engel_yok": dict(structure="flat", draw={"empty": True}),
        "gable_layout": dict(structure="pitched", pitched="gable", draw={"obstacles": True}),
    }
    for name, case in cases.items():
        inputs = _roof_inputs(sample_index, building_point, structure=case["structure"], pitched=case.get("pitched", "hip"), bill=1500)
        draw_kw = {"empty": True} if case["draw"].get("empty") else {}
        obstacles = [_centre_obstacle(inputs)] if case["draw"].get("obstacles") else ()
        r, _, _ = app.evaluate(_roof_analysis(inputs), inputs, None, set(), _draw_state(inputs, obstacles, **draw_kw))
        out[name] = {
            "kwp": r["kurulabilir_guc_kwp_aralik"], "kwh": r["yillik_uretim_kwh_aralik"],
            "tl_flat": r["daire_basi_tasarruf_tl_aralik"], "payback": r["geri_odeme_yil_aralik"],
            "golgeli": r["golgeli_uretim_kwh_kwp"], "source": r["guc_kaynagi"], "panels": r["panel_sayisi_aralik"],
            "pvgis_calls": _pvgis_calls(monkeypatch, inputs),
        }
    return out


def test_existing_results_and_pvgis_calls_are_unchanged(mocked, monkeypatch, sample_index, building_point):
    """Golden values were produced at the commit before this change (kWp, kWh, per-flat TL, payback, PVGIS calls)."""
    now = _snapshot(monkeypatch, sample_index, building_point)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert now.keys() == golden.keys()
    for name, values in golden.items():
        assert now[name] == pytest.approx(values), name


def _eval(sample_index, building_point, structure="pitched", pitched="hip", obstacle=False, empty=False):
    inputs = _roof_inputs(sample_index, building_point, structure=structure, pitched=pitched)
    obstacles = [_centre_obstacle(inputs)] if obstacle else ()
    r, _, _ = app.evaluate(_roof_analysis(inputs), inputs, None, set(), _draw_state(inputs, obstacles, empty=empty))
    return r, inputs


def _rows(r, inputs):
    return {row[0]: row for row in app.assumption_rows(r, inputs, None).itertuples(index=False, name=None)}


# ------------------------------------------------------------------ A1 building specific yield

def test_building_yield_is_kwh_over_kwp_per_scenario_and_the_flat_value_is_untouched(mocked, sample_index, building_point):
    r, _ = _eval(sample_index, building_point, structure="unknown")
    for sc in r["senaryolar"]:
        assert sc["ozgul_verim_bina"] == pytest.approx(sc["kwh"] / sc["kwp"])
    assert r["ozgul_verim_bina_aralik"] == pytest.approx(sorted(sc["ozgul_verim_bina"] for sc in r["senaryolar"]))
    assert r["golgeli_uretim_kwh_kwp"] == pytest.approx(1369.8)  # the flat PVGIS value keeps its meaning
    flat, _ = _eval(sample_index, building_point, structure="flat")
    kwp, kwh = flat["kurulabilir_guc_kwp_aralik"][0], flat["yillik_uretim_kwh_aralik"][0]
    assert flat["ozgul_verim_bina_aralik"] == pytest.approx([kwh / kwp] * 2)


def test_yield_text_range_single_value_or_hidden():
    make = lambda bounds: {"ozgul_verim_bina_aralik": bounds}
    assert app.building_yield_text(make([1300.2, 1400.7])) == "1.300 – 1.401 kWh/kWp"
    assert app.building_yield_text(make([None, 1400.0])) == "1.400 kWh/kWp"
    assert app.building_yield_text(make([1400.0, None])) == "1.400 kWh/kWp"
    assert app.building_yield_text(make([None, None])) is None and app.building_yield_text({}) is None


def _zero_kwp_result():
    flat = _fake_yield(0, 0)
    solar = {"ideal": flat["ideal"], "shaded": flat["shaded"], "angle": 33, "aspect": 8}
    faces = [{**spec, "solar": None} for spec in building_mode.face_specs(0)]  # every face failed: 0 kWp
    r = building_mode.analyze_building(
        roof_area_m2=150.0, usable_base_area_m2=None, obstacle_ratio=None, image_view=None, structure="pitched",
        flat_share_pct=50, n_flats=4, n_flats_estimated=False, bill_tl=None, flat_solar=solar, faces=faces,
    )
    r.update(lat=41.0, lon=29.0)
    return r


def test_zero_kwp_hides_the_card_and_nothing_breaks():
    r = _zero_kwp_result()
    assert r["kurulabilir_guc_kwp_aralik"] == [0, 0] and r["ozgul_verim_bina_aralik"] == [None, None]
    assert app.building_yield_text(r) is None
    inputs = {"area": 150.0, "building": None, "roof_tag": "x", "structure": "pitched"}
    names = [row[0] for row in app.assumption_rows(r, inputs, None).itertuples(index=False, name=None)]
    assert "Özgül verim (bina)" not in names and "Özgül verim (düz bölüm, referans)" in names
    assert "ozgul_verim_kwh_kwp" not in explain.build_payload(r)
    assert explain.build_fallback_explanation(r) and app.monthly_caption(r) is not None


def test_assumption_rows_show_both_yields(mocked, sample_index, building_point):
    r, inputs = _eval(sample_index, building_point, structure="flat")
    rows = _rows(r, inputs)
    assert rows["Özgül verim (düz bölüm, referans)"][1] == "1.370 kWh/kWp"
    assert rows["Özgül verim (bina)"][1] == app.building_yield_text(r)


def test_explainer_payload_uses_the_building_yield_and_the_validator_accepts_it():
    r = make_result()
    p = explain.build_payload(r)
    assert "golgeli_uretim_kwh_kwp" not in p
    kwp, kwh = r["kurulabilir_guc_kwp_aralik"][0], r["yillik_uretim_kwh_aralik"][0]
    assert p["ozgul_verim_kwh_kwp"] == round(kwh / kwp)
    ok, bad = explain.validate_explanation_numbers(f"Her kWp yılda {app.tr_num(p['ozgul_verim_kwh_kwp'])} kWh üretir.", p)
    assert ok and not bad
    assert not explain.validate_explanation_numbers("Her kWp yılda 1.234 kWh üretir.", p)[0]


# ---------------------------------------------------------- A2 layout-mode breakdown rows

def test_layout_mode_marks_unused_rows(mocked, sample_index, building_point):
    r, inputs = _eval(sample_index, building_point, structure="flat", obstacle=True)
    assert r["guc_kaynagi"] == "geometri"
    rows = _rows(r, inputs)
    for name in ("Kullanılabilir alan oranı", "Kullanılabilir alan (plan)", "Düz bölüm gücü"):
        assert rows[name][1] == "kullanılmadı (panel yerleşimi)"
    assert rows["Kenar boşluğu sonrası alan"][1].endswith("m²")
    assert rows["Düz bölüm eğimi / yönü"][1].endswith("°")  # a flat section exists


def test_fully_pitched_layout_has_no_flat_slope(mocked, sample_index, building_point):
    r, inputs = _eval(sample_index, building_point, structure="pitched", empty=True)
    assert r["guc_kaynagi"] == "geometri"
    assert _rows(r, inputs)["Düz bölüm eğimi / yönü"][1] == "kullanılmadı (düz bölüm yok)"


def test_numeric_modes_keep_the_numeric_rows(mocked, sample_index, building_point):
    karma, inputs = _eval(sample_index, building_point, structure="mixed", empty=True)  # Karma without terrace: numeric
    assert karma["guc_kaynagi"] == "engel_yok"
    average, inputs_avg = _eval(sample_index, building_point, structure="flat")
    assert average["guc_kaynagi"] == "ortalama"
    for r, i in ((karma, inputs), (average, inputs_avg)):
        rows = _rows(r, i)
        assert rows["Düz bölüm gücü"][1].endswith("m²/kWp") and rows["Kullanılabilir alan (plan)"][1].endswith("m²")
        assert rows["Düz bölüm eğimi / yönü"][1].endswith("°")
        assert "Kenar boşluğu sonrası alan" not in rows


# ------------------------------------------------------------------------ A3 chart caption

def test_chart_caption_follows_the_roof_structure_and_mode(mocked, sample_index, building_point):
    pitched, _ = _eval(sample_index, building_point, structure="pitched")
    text = app.monthly_caption(pitched)
    assert text.startswith("Eğimli yüzler 30° eğimle mahyaya dik yönlerde varsayıldı.") and "Düz bölüm" not in text
    assert "Kullanılabilir plan alanı" in text
    flat, _ = _eval(sample_index, building_point, structure="flat")
    assert "Düz bölüm en uygun eğim" in app.monthly_caption(flat) and "eğimli" not in app.monthly_caption(flat)
    mixed, _ = _eval(sample_index, building_point, structure="mixed")
    assert "Düz bölüm" in app.monthly_caption(mixed) and "eğimli yüzler 30°" in app.monthly_caption(mixed)
    layout_r, _ = _eval(sample_index, building_point, structure="pitched", empty=True)
    assert "Kullanılabilir plan alanı" not in app.monthly_caption(layout_r)


# ------------------------------------------------------------------------- A4 Turkish text

def test_report_and_breakdown_use_proper_turkish(mocked, sample_index, building_point):
    from core import report

    r, _ = _eval(sample_index, building_point, structure="pitched")
    rows = report.breakdown_rows(r, (3, 5))
    assert {"Üretim", "İndirim", "Geri ödeme"} <= {row["Adım"].split(" · ")[-1] for row in rows}
    text = " ".join(f"{row['Adım']} {row['Hesap']}" for row in rows)
    assert "1 m içeri" in text and "yalnızca gösterim" in text and "m2" not in text and "m²" in text
    md = report.report_markdown(r, offset=(3, 5))
    assert "m2" not in md and "Uretim" not in md and "Indirim" not in md
    dropped = [line for line in app.face_status_lines(r) if "verim oranı" in line]
    assert dropped and all("verim oranı %" in line and "% " not in line for line in dropped)


# --------------------------------------------------------------------- A5 explainer guards

def test_prompt_says_the_user_draws_obstacles_not_panels():
    prompt = explain.SYSTEM_PROMPT
    assert "yalnızca engelleri ve terası çizer" in prompt and "panelleri uygulama yerleştirir" in prompt
    assert "gelecek_ozellikler" in prompt


def test_payload_carries_the_future_feature_allow_list():
    assert explain.build_payload(make_result())["gelecek_ozellikler"] == ["komşu bina gölgesi"]


def test_future_sentences_are_removed_whole_and_numbers_with_dots_are_not_split():
    text = (
        "Her kWp yılda 1.402 kWh üretir.\n"
        "Bina yılda 17.678 kWh üretir. Kaydırma ve ısı kaybı sonraki aşamada eklenecek. Güven orta.\n"
        "- Komşu bina gölgesi sonraki aşamada eklenecek.\n"
        "- İleride yağmur modeli eklenecek.\n"
        + explain.WARNING_LINE
    )
    out, meaningful = explain.drop_unlisted_future_sentences(text)
    assert meaningful
    assert "Bina yılda 17.678 kWh üretir. Güven orta." in out
    assert "Komşu bina gölgesi sonraki aşamada eklenecek." in out  # the allowed feature stays
    assert "yağmur" not in out and "Kaydırma" not in out and out.endswith(explain.WARNING_LINE)
    assert out.splitlines()[0] == "Her kWp yılda 1.402 kWh üretir."


def test_nothing_meaningful_left_falls_back_to_the_template(monkeypatch):
    result = make_result()
    monkeypatch.setattr(explain, "_get_client", lambda api_key=None: object())
    monkeypatch.setattr(explain, "_generate", lambda client, payload, hint=None: "Sonuçlar ileride eklenecek.\n" + explain.WARNING_LINE)
    explain.clear_cache()
    out = explain.explain_result(result)
    assert out["source"] == "template" and "eklenecek" not in out["text"]

    def good(client, payload, hint=None):
        yield_text = app.tr_num(payload["ozgul_verim_kwh_kwp"])
        return f"Her kWp yılda yaklaşık {yield_text} kWh üretir. Isı modeli ileride eklenecek.\n- Daire sayısını girin.\n" + explain.WARNING_LINE

    monkeypatch.setattr(explain, "_generate", good)
    explain.clear_cache()
    out = explain.explain_result(result)
    assert out["source"] == "gemini" and "ileride" not in out["text"] and "Daire sayısını girin" in out["text"]
