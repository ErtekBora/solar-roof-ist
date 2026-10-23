import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import calc, config, explain
from core.building_mode import analyze_building, face_specs
from core.formatting import round_half_up
from core.explain import (
    WARNING_LINE,
    build_fallback_explanation,
    build_payload,
    explain_result,
    extract_numbers,
    validate_explanation_numbers,
)

FIXTURES = Path(__file__).parent / "fixtures"


E_M = [60, 75, 105, 130, 150, 160, 170, 160, 130, 100, 70, 55]


def _pv(e_y):
    k = e_y / sum(E_M)
    return {"e_y": float(e_y), "e_m": [m * k for m in E_M], "h_i_y": 1738.72, "l_tg": -6.49, "l_total": -21.0, "angle": 33, "aspect": 8}


def make_result(bill=None, structure="flat", n_flats=6, estimated=False):
    solar = {"ideal": _pv(1372.68), "shaded": _pv(1369.8), "angle": 33, "aspect": 8}
    faces = [{**spec, "solar": _pv(1250)} for spec in face_specs(0)]
    r = analyze_building(
        roof_area_m2=150.0, usable_base_area_m2=None, obstacle_ratio=0.2715, image_view="oblique", structure=structure,
        flat_share_pct=50, n_flats=n_flats, n_flats_estimated=estimated, bill_tl=bill, flat_solar=solar, faces=faces,
    )
    r.update(lat=41.015, lon=28.979)
    return r


def make_detection(labels=("skylight", "skylight", "satellite dish", "air conditioner unit", "chimney")):
    return {"image_view": "oblique", "roof_type": "flat", "roof_polygon_px": [(0, 0)] * 4,
            "obstacles": [{"label": label, "box_px": (0, 0, 1, 1)} for label in labels]}


@pytest.fixture
def result():
    return make_result()


@pytest.fixture
def detection():
    return make_detection()


@pytest.fixture
def payload(result, detection):
    return build_payload(result, detection, True)


@pytest.fixture
def good_text(result, detection):
    """A valid Gemini-like answer (the deterministic template passes the validator by construction)."""
    return build_fallback_explanation(result, detection, True)


# ----------------------------------------------------------------- extraction

def test_extraction_ignores_digits_in_units():
    text = "Yılda 1.726 kWh/m² ışınım, %41,8 engel oranı, 127,5 m² alan ve 16,2 ton CO₂ (CO2)"
    assert extract_numbers(text) == [1726.0, 41.8, 127.5, 16.2]


def test_range_dash_is_a_separator_not_a_minus():
    assert extract_numbers("27.515–55.031 TL") == [27515.0, 55031.0]
    assert extract_numbers("27.515-55.031 TL") == [27515.0, 55031.0]
    assert extract_numbers("27.515—55.031 TL") == [27515.0, 55031.0]
    assert all(n >= 0 for n in extract_numbers("- 5–10 arası"))


@pytest.mark.parametrize(
    "text, expected",
    [("1.726", 1726.0), ("38.250", 38250.0), ("25.500,5", 25500.5), ("41,8", 41.8), ("%41,8", 41.8), ("6.3", 6.3)],
)
def test_turkish_number_formats(text, expected):
    assert extract_numbers(text) == [expected]


def test_warning_line_and_list_markers_are_ignored():
    assert extract_numbers(f"- 5 madde\n{WARNING_LINE}") == [5.0]


# ----------------------------------------------------------------- matching

def test_rounding_rule():
    p = {"sicaklik_kaybi_yuzde": 6.3}
    assert validate_explanation_numbers("%6,3", p) == (True, [])
    assert validate_explanation_numbers("%6", p) == (True, [])
    assert validate_explanation_numbers("%7", p) == (False, [7.0])
    assert validate_explanation_numbers("%6,35", p)[0] is False  # more precise than the payload


def test_bin_form_for_thousands():
    p = {"yillik_tasarruf_tl_aralik": [27515, 55031]}
    assert validate_explanation_numbers("27,5 bin TL", p) == (True, [])
    assert validate_explanation_numbers("55,0 bin TL", p) == (True, [])
    assert validate_explanation_numbers("27.500 TL", p)[0] is True  # 0.5% thousands rounding
    assert validate_explanation_numbers("28.500 TL", p)[0] is False


def test_ratio_percentage_only_for_engel_orani():
    assert validate_explanation_numbers("%41,8", {"engel_orani": 0.418}) == (True, [])
    # golge_kaybi_yuzde is already a percentage: 0.2 must not allow 20 (the live-sample bug)
    assert validate_explanation_numbers("gölge kaybı yüzde 20", {"golge_kaybi_yuzde": 0.2}) == (False, [20.0])


# ----------------------------------------------------------------- forbidden context

def test_forbidden_payback_sentence_fails_even_when_number_matches():
    p = {"sicaklik_kaybi_yuzde": 6.3}
    assert validate_explanation_numbers("Yatırım 6 yılda kendini amorti eder", p) == (False, [6.0])
    assert validate_explanation_numbers("Yaklaşık 6 yıl içinde geri döner", p) == (False, [6.0])
    # "fatura" is no longer a forbidden word: a bill sentence is judged by its numbers only
    assert validate_explanation_numbers("Faturanızda %6 azalma olur", p)[0] is True
    assert validate_explanation_numbers("Faturanızda %7 azalma olur", p)[0] is False


def test_bill_coverage_sentence_needs_payload_numbers():
    p = {"fatura_karsilama_yuzde_aralik": [60, 85], "fatura_tl_ay": 1500}
    assert validate_explanation_numbers("Faturanızın %60–%85'ini karşılar.", p) == (True, [])
    assert validate_explanation_numbers("Faturanızın %60–%85'ini karşılar.", {"co2_ton_yil": 12.4})[0] is False
    assert validate_explanation_numbers("Faturanızın %60–%90'ını karşılar.", p) == (False, [90.0])
    assert validate_explanation_numbers("Aylık 1.500 TL faturanız var.", p) == (True, [])


def test_forbidden_panel_count_fails():
    assert validate_explanation_numbers("Çatıya 28 adet panel sığar", {"x": 28}) == (False, [28.0])
    assert validate_explanation_numbers("Çatıya 28 panel sığar", {"x": 28}) == (False, [28.0])


def test_plain_annual_phrasing_passes():
    assert validate_explanation_numbers("Bu konum yılda 1.726 kWh/m² güneş alır", {"yillik_isinim_kwh_m2": 1726}) == (True, [])
    assert validate_explanation_numbers("Yıllık 16,2 ton CO₂ azalır", {"co2_ton_yil": 16.2}) == (True, [])
    assert validate_explanation_numbers("Çatıda 2 adet uydu çanağı var", {"engel_sayilari": {"Uydu çanağı": 2}}) == (True, [])


def test_financial_term_only_taints_its_own_sentence():
    p = {"co2_ton_yil": 12.4}
    assert validate_explanation_numbers("Maliyet hesaplanmadı. Yıllık 12,4 ton CO₂ azalır.", p) == (True, [])


# ----------------------------------------------------------------- live sample

def test_real_sample_is_valid_and_a_wrong_number_is_caught():
    sample = json.loads((FIXTURES / "gemini_explain_sample.json").read_text(encoding="utf-8"))
    assert validate_explanation_numbers(sample["text"], sample["payload"]) == (True, [])
    wrong = sample["text"].replace("sıcaklık kaybı %6,5", "sıcaklık kaybı %5")  # an earlier live run made this error
    assert wrong != sample["text"]
    assert validate_explanation_numbers(wrong, sample["payload"]) == (False, [5.0])


# ----------------------------------------------------------------- payload + prompt

def test_payload_rounding():
    p = build_payload(make_result(), None, False)
    assert p["yillik_isinim_kwh_m2"] == 1739 and isinstance(p["yillik_isinim_kwh_m2"], int)
    assert p["ozgul_verim_kwh_kwp"] == 1370 and "golgeli_uretim_kwh_kwp" not in p and isinstance(p["yillik_uretim_kwh"], int)
    assert p["sicaklik_kaybi_yuzde"] == 6.5 and p["kurulabilir_guc_kwp"] == 14.3  # 150 x .85 x (1 - .2715) / 6.5
    assert p["daire_basi_tasarruf_tl_yil"] == [round_half_up(v, 0) for v in make_result()["daire_basi_tasarruf_tl_aralik"]]
    assert all(isinstance(v, int) for v in p["daire_basi_tasarruf_tl_yil"]) and isinstance(p["daire_sayisi"], int)
    assert round_half_up(27.15, 1) == 27.2 and round_half_up(1738.72, 0) == 1739
    assert "engel_orani" not in p  # no photo: obstacles are unknown, not zero
    assert build_payload(make_result(), make_detection(), False)["engel_orani"] == 0.272


def test_payload_without_bill_has_tariff_and_method_but_no_coverage(result):
    p = build_payload(result, None, False)
    assert p["tasarruf_yontemi"] == "genis_tahmin"
    assert p["tarife_dusuk_tl_kwh"] == 3.68 and p["tarife_yuksek_tl_kwh"] == 5.44
    assert "fatura_karsilama_yuzde_aralik" not in p and "tarife_tl_kwh" not in p


def test_payload_with_bill_carries_coverage_and_self_use():
    r = make_result(bill=1500, n_flats=6)
    p = build_payload(r, None, False)
    assert p["tasarruf_yontemi"] == "fatura" and p["fatura_tl_ay"] == 1500
    assert 0 < p["fatura_karsilama_yuzde_aralik"] <= 100 and "kendi_kullanim_orani_yuzde" not in p
    assert p["tuketim_kwh_ay"] == 353
    assert p["mahsuplasma_varsayimi"] == "Aylık mahsuplaşma varsayımıyla hesaplanmıştır."  # explainer gets only the short assumption
    text = f"Daire başına faturanın yaklaşık %{p['fatura_karsilama_yuzde_aralik']} kadarı karşılanır."
    assert validate_explanation_numbers(text, p) == (True, [])


def test_forbidden_words_no_longer_include_fatura_but_keep_payback_terms():
    assert "fatura" not in explain.FINANCIAL_TERMS and "amorti" in explain.FINANCIAL_TERMS
    assert "geri ödeme" in explain.FINANCIAL_TERMS
    assert "kredi, taksit, fatura" not in explain.SYSTEM_PROMPT


def test_payload_strips_large_fields_and_counts_obstacles(payload):
    for key in ("ufuk_cizgisi_72", "aylik_uretim_kwh", "aylik_ideal_kwh", "roof_polygon_px", "lat", "lon"):
        assert key not in payload
    assert payload["engel_sayilari"] == {"Çatı penceresi": 2, "Uydu çanağı": 1, "Klima": 1, "Baca": 1}
    assert payload["goruntu_turu"] == "oblique" and payload["cati_yapisi"] == "Tamamen düz"
    assert payload["demo_photo"] is True and payload["fotograf_var"] is True


def test_payload_without_photo(result):
    p = build_payload(result, None, True)
    assert p["fotograf_var"] is False and p["demo_photo"] is False  # demo flag only counts with a photo
    assert "engel_sayilari" not in p


def test_prompt_contains_the_required_rules():
    prompt = explain.SYSTEM_PROMPT
    assert "arazi gölgesi açısından" in prompt and "asla genel uygunluk hükmü" in prompt
    assert "su deposu, klima ve uydu çanağı" in prompt and "ASLA kaldırma veya taşıma önerme" in prompt
    assert "kWh/m², kWh/kWp, kWp" in prompt and "yazıyla açma" in prompt
    assert "AYRI bir cümle" in prompt and "Değerlendirici sıfat kullanma" in prompt and "tasarruf" in prompt
    assert "Koşullu anlatım kullanma" in prompt and '"varsa"' in prompt
    assert "kontrol edin" in prompt and "bakın" in prompt and "Dilbilgisi" in prompt
    assert "adetleriyle adlandırarak" in prompt and "ozgul_verim_kwh_kwp" in prompt
    assert WARNING_LINE in prompt


# ----------------------------------------------------------------- explain_result

def test_valid_first_response_is_used(fake_explain, result, detection, good_text):
    client = fake_explain([good_text])
    out = explain_result(result, detection, api_key="k", demo_photo=True)
    assert out == {"text": good_text, "source": "gemini", "retried": False, "invalid_numbers": []}
    assert client.calls == 1


def test_invented_number_then_clean_retries_once(fake_explain, result, detection, good_text):
    client = fake_explain([good_text + "\n- Ek: %77 daha iyi olur.", good_text])
    out = explain_result(result, detection, api_key="k", demo_photo=True)
    assert out["source"] == "gemini" and out["retried"] is True and client.calls == 2


def test_retry_hint_names_the_invalid_numbers(fake_explain, result, detection, good_text, monkeypatch):
    seen = []
    real = explain._generate
    monkeypatch.setattr(explain, "_generate", lambda c, p, hint=None: seen.append(hint) or real(c, p, hint))
    fake_explain(["Sıcaklık kaybı %77", good_text])
    explain_result(result, detection, api_key="k", demo_photo=True)
    assert seen[0] is None and "77" in seen[1]


def test_invented_numbers_twice_fall_back_to_template(fake_explain, result, detection):
    client = fake_explain(["Sıcaklık kaybı %77", "Sıcaklık kaybı %78"])
    out = explain_result(result, detection, api_key="k", demo_photo=True)
    assert out["source"] == "template" and out["retried"] is True and client.calls == 2
    assert out["invalid_numbers"] == [78.0]
    assert out["text"] == build_fallback_explanation(result, detection, True)


def test_empty_response_is_retried(fake_explain, result, detection, good_text):
    client = fake_explain(["", good_text])
    out = explain_result(result, detection, api_key="k", demo_photo=True)
    assert out["source"] == "gemini" and out["retried"] is True and client.calls == 2


@pytest.mark.parametrize(
    "error",
    [RuntimeError("boom"), RuntimeError("429 RESOURCE_EXHAUSTED"), RuntimeError("503 UNAVAILABLE"), ConnectionError("net")],
)
def test_api_errors_give_template_without_raising(fake_explain, result, detection, error):
    client = fake_explain([error])
    out = explain_result(result, detection, api_key="k", demo_photo=True)
    assert out["source"] == "template" and out["retried"] is False and client.calls == 1


def test_missing_key_gives_template(monkeypatch, result, detection):
    def no_key():
        raise config.MissingApiKeyError("anahtar yok")

    explain.clear_cache()
    monkeypatch.setattr(explain, "get_gemini_api_key", no_key)
    out = explain_result(result, detection)
    assert out["source"] == "template" and WARNING_LINE in out["text"]


def test_api_key_parameter_is_used_instead_of_lookup(monkeypatch, result, detection, good_text):
    import google.genai

    explain.clear_cache()
    monkeypatch.setattr(explain, "get_gemini_api_key", lambda: pytest.fail("key lookup called"))
    created = []

    class SpyClient:
        def __init__(self, api_key):
            created.append(api_key)
            self.models = SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text=good_text))

    monkeypatch.setattr(google.genai, "Client", SpyClient)
    out = explain_result(result, detection, api_key="worker-key", demo_photo=True)
    assert created == ["worker-key"] and out["source"] == "gemini"
    explain.clear_cache()


def test_cache_hit_makes_no_second_call(fake_explain, result, detection, good_text):
    client = fake_explain([good_text])
    first = explain_result(result, detection, api_key="k", demo_photo=True)
    second = explain_result(result, detection, api_key="k", demo_photo=True)
    assert first == second and client.calls == 1
    explain.clear_cache()
    fake_explain([good_text])
    assert explain_result(result, detection, api_key="k", demo_photo=True)["source"] == "gemini"


def test_template_results_are_not_cached(fake_explain, result, detection, good_text):
    fake_explain([RuntimeError("503")])
    assert explain_result(result, detection, api_key="k", demo_photo=True)["source"] == "template"
    client = fake_explain([good_text])
    assert explain_result(result, detection, api_key="k", demo_photo=True)["source"] == "gemini"
    assert client.calls == 1


# ----------------------------------------------------------------- template

@pytest.mark.parametrize("with_photo", [True, False])
@pytest.mark.parametrize("demo", [True, False])
def test_fallback_template_passes_validation(result, detection, with_photo, demo):
    det = detection if with_photo else None
    text = build_fallback_explanation(result, det, demo)
    ok, invalid = validate_explanation_numbers(text, build_payload(result, det, demo))
    assert ok, invalid
    lines = text.splitlines()
    assert lines[-1] == WARNING_LINE and 1 <= sum(ln.startswith("-") for ln in lines) <= 3
    body = [ln for ln in lines[1:-1] if not ln.startswith("-")]
    assert len(body) == 1 and len(re.findall(r"\.(?:\s|$)", body[0])) <= 5  # one short paragraph (at most 5 sentences)
    assert ("örnektir" in lines[0]) == (with_photo and demo)


def test_template_verdict_is_factual_without_labels(result, detection):
    verdict = build_fallback_explanation(result, detection, True).splitlines()[0]
    for label in ("çok uygun", "sınırlı uygun", "uygun"):
        assert label not in verdict.lower()
    assert "yılda yaklaşık 1.370 kWh" in verdict and "14,3 kWp" in verdict and "güven: Düşük" in verdict


def test_template_names_relocatable_obstacles_with_counts_and_never_others(result):
    det = make_detection(("skylight", "chimney", "solar panel", "satellite dish", "satellite dish", "water tank"))
    text = build_fallback_explanation(result, det, False)
    assert "Çatıda 2 uydu çanağı, 1 su deposu var;" in text
    for blocked in ("çatı penceresi", "baca", "güneş paneli"):
        assert blocked not in text.lower()


@pytest.mark.parametrize(
    "labels, expected, absent",
    [
        (("skylight", "chimney", "chimney", "chimney"), "Çatınızdaki 3 bacanın çevresindeki boşluğu", "taşı"),
        (("skylight", "solar panel"), "Ortalama daire faturasını girin", "baca"),
        ((), "Ortalama daire faturasını girin", "taşı"),
    ],
)
def test_template_without_relocatable_obstacles_gives_another_recommendation(result, labels, expected, absent):
    text = build_fallback_explanation(result, make_detection(labels), False)
    tips = [ln for ln in text.splitlines() if ln.startswith("-")]
    assert 1 <= len(tips) <= 3 and any(expected in t for t in tips) and not any(absent in t.lower() for t in tips)
    assert "taşınabilir" not in text


@pytest.mark.parametrize("with_photo", [True, False])
def test_template_has_no_conditional_phrasing(result, detection, with_photo):
    det = detection if with_photo else None
    text = build_fallback_explanation(result, det, False).lower()
    import re

    assert not re.search(r"varsa|olması durumunda|[ae]biliyorsa", text)  # "varsayıldı" is fine


def test_chimney_tip_states_the_count_and_prompt_has_no_conditional_chimney_wording(result):
    det = make_detection(("chimney",) * 5)
    text = build_fallback_explanation(result, det, False)
    assert "- Çatınızdaki 5 bacanın çevresindeki boşluğu bir kurulum firmasına kontrol ettirin." in text
    assert validate_explanation_numbers(text, build_payload(result, det, False)) == (True, [])
    assert "baca" not in build_fallback_explanation(result, make_detection(("skylight",)), False).lower()
    assert "bacalar varsa" not in explain.SYSTEM_PROMPT and "baca varsa" not in explain.SYSTEM_PROMPT
    assert "Çatınızdaki 5 bacanın" in explain.SYSTEM_PROMPT


# ------------------------------------------------- field-name leaks, no-photo payload, terrain, conciseness

def test_field_names_in_the_text_are_detected():
    p = build_payload(make_result(bill=1500), None, False)
    leak = "Bina toplamı (kurulabilir_guc_kwp 21,0 kWp ve yillik_uretim_kwh 28.590 kWh) bellidir."
    assert explain.find_field_names(leak, p) == ["kurulabilir_guc_kwp", "yillik_uretim_kwh"]
    assert explain.find_field_names("Sabit indirim yıllık 5.000 TL, kWh/kWp 1.370; ⚠️ not.", p) == []
    assert explain.find_field_names("Bu daire_sayisi ve senaryolar alanları.", p) == ["daire_sayisi", "senaryolar"]
    assert explain.find_field_names("", p) == [] and explain.find_field_names(WARNING_LINE, p) == []


def test_a_leaked_field_name_triggers_the_retry_then_the_template(fake_explain):
    r = make_result(bill=1500)
    p = build_payload(r, None, False)
    clean = build_fallback_explanation(r)
    leaky = clean.replace("Bina toplamı", "Bina toplamı (yillik_uretim_kwh)")
    client = fake_explain([leaky, clean])
    out = explain_result(r, None, api_key="k")
    assert out["source"] == "gemini" and out["retried"] is True and client.calls == 2
    assert validate_explanation_numbers(out["text"], p)[0]
    explain.clear_cache()
    client = fake_explain([leaky, leaky])
    out = explain_result(r, None, api_key="k")
    assert out["source"] == "template" and client.calls == 2 and not explain.find_field_names(out["text"], p)


def test_retry_hint_names_the_leaked_fields():
    hint = explain._retry_hint([], ["yillik_uretim_kwh"])
    assert "yillik_uretim_kwh" in hint and "alan adları" in hint


def test_prompt_forbids_field_names_and_asks_for_a_short_answer():
    prompt = explain.SYSTEM_PROMPT
    assert "alan (anahtar) adlarını ASLA yazma" in prompt
    assert "en fazla yaklaşık 6 cümle" in prompt and "EN FAZLA 3 pratik öneri" in prompt
    assert "Ekranda zaten gösterilen notları tekrar etme" in prompt and "mahsuplaşma" in prompt


def test_no_photo_payload_has_no_obstacle_numbers_even_if_the_result_carries_zero():
    r = make_result()
    r["engel_orani"], r["engel_alani_m2"] = 0.0, 0.0  # a stray zero must not become "obstacles: 0 %"
    p = build_payload(r, None, False)
    assert "engel_orani" not in p and "engel_alani_m2" not in p and p["fotograf_var"] is False
    text = build_fallback_explanation(r)
    assert "Engel oranı" not in text and "engellerle kaplı" not in text and "%0" not in text
    with_photo = build_payload(r, make_detection(), False)
    assert with_photo["engel_orani"] == 0.0 and with_photo["engel_alani_m2"] == 0.0


def test_zero_terrain_loss_is_said_as_no_terrain_shading():
    r = make_result()
    r.update(golge_kaybi_yuzde=0.0, uygunluk_puani_yuzde=100.0)
    p = build_payload(r, None, False)
    assert "golge_kaybi_yuzde" not in p and p["arazi_golgesi"] == "yok" and p["uygunluk_puani_yuzde"] == 100
    text = build_fallback_explanation(r)
    assert "arazi gölgesi yok (%100)" in text and "%0" not in text
    assert validate_explanation_numbers(text, p)[0]
    shaded = build_payload(make_result(), None, False)
    assert shaded["golge_kaybi_yuzde"] > 0 and "arazi_golgesi" not in shaded
    assert "arazi gölgesi açısından uygunluk" in build_fallback_explanation(make_result())


def test_template_is_concise_and_does_not_repeat_screen_notes():
    from core import building_mode as bm

    text = build_fallback_explanation(make_result(bill=1500))
    lines = text.splitlines()
    assert len(lines) <= 6 and sum(ln.startswith("-") for ln in lines) <= 3
    for note in (bm.NETTING_NOTE, bm.SEASONAL_NOTE, bm.PHYSICAL_SHARING_NOTE, bm.LEGAL_NOTE, config.COST_SOURCE):
        assert note not in text
    for word in ("mahsuplaşma", "aidat", "KDV", "güçlendirme", "inverter"):
        assert word not in text


# --------------------------------------------------------- panel layout fields (roof view)

def test_panel_count_is_valid_only_when_it_is_the_layout_count():
    payload = {"panel_sayisi": 42}
    assert validate_explanation_numbers("Çatıya 42 panel yerleşti", payload) == (True, [])
    assert validate_explanation_numbers("Çatıya 43 panel yerleşti", {**payload, "x": 43}) == (False, [43.0])
    assert validate_explanation_numbers("30 panel", {"panel_sayisi": [30, 80]})[0]


def test_payload_carries_layout_fields_and_template_validates(result):
    r = {
        **result, "panel_sayisi_aralik": [52, 52], "guc_kaynagi": "geometri", "egimli_tip": "hip",
        "cizilen_engel_sayisi": 2, "cizilen_engel_alani_m2": 6.54, "teras_cizildi": False,
        "daire_payi_yuksek": True, "daire_payi_kwh_ay_aralik": [251.2, 251.2],
    }
    p = build_payload(r)
    assert p["panel_sayisi"] == 52 and p["guc_kaynagi"] == "kullanıcı çizimi (panel yerleşimi)"
    assert p["cizilen_engel_sayisi"] == 2 and p["cizilen_engel_alani_m2"] == 6.5
    assert p["daire_payi_kwh_ay"] == 251 and p["daire_payi_tipik_tuketimden_yuksek"] is True
    if r["cati_yapisi"] != "flat":
        assert p["egimli_cati_tipi"] == "Kırma (4 yüz)"
    text = build_fallback_explanation(r)
    assert validate_explanation_numbers(text, p) == (True, [])
    assert "52 panel" in text and "251 kWh/ay" in text
    from core.explain import find_field_names

    assert find_field_names(text, p) == []


def test_no_drawing_fields_without_a_drawing(result):
    p = build_payload({**result, "cizilen_engel_sayisi": None, "daire_payi_yuksek": False})
    assert "cizilen_engel_sayisi" not in p and "cizilen_engel_alani_m2" not in p and "daire_payi_kwh_ay" not in p


@pytest.mark.parametrize("tip", [
    "- Kuşbakışı (yukarıdan çekilmiş) bir fotoğraf deneyin.",
    "- Uydu görüntüsü yükleyin.",
    "- Bir hava fotoğrafı deneyin.",
    "- KUŞBAKIŞI fotoğraf deneyin.",
])
def test_imagery_tips_are_replaced_by_the_standard_tip(tip):
    out = explain.sanitize_tips(f"Karar\nParagraf\n{tip}\n- Daire sayısını girin.")
    assert tip not in out and "- " + explain.PHOTO_OR_DRAW_TIP in out and "- Daire sayısını girin." in out
    assert "uydu çanağı" in explain.sanitize_tips("- 2 uydu çanağını taşıyın.")


def test_photo_upload_tips_are_replaced_by_the_drawing_tip():
    for tip in (
        "- Çatıya çıkıp fotoğraf yükleyin.",
        "- Çatınızın fotoğrafını yükleyin.",
        "- Bize bir çatı fotoğrafı gönderin.",
        "- Daha net bir fotoğraf ekleyin.",
    ):
        out = explain.sanitize_tips(f"Karar\nParagraf\n{tip}\n- Daire sayısını girin.")
        assert tip not in out and "- " + explain.PHOTO_OR_DRAW_TIP in out and "- Daire sayısını girin." in out, tip
    assert "fotoğraf" not in explain.PHOTO_OR_DRAW_TIP and "haritaya çizin" in explain.PHOTO_OR_DRAW_TIP
    assert "Çatıda fotoğraf" in explain.sanitize_tips("Çatıda fotoğraf yükleyin.")  # only '-' tips are filtered


def test_template_fallback_and_gemini_text_never_suggest_imagery(result, detection):
    text = build_fallback_explanation(result, None, False)
    assert not any(explain._BAD_TIP_RE.search(explain._tr_lower(ln)) for ln in text.splitlines() if ln.startswith("-"))
    assert explain.PHOTO_OR_DRAW_TIP in text
