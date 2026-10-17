import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import calc, config, explain
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


def make_result():
    r = calc.compute_capacity(150.0, 0.2715, "flat", 1372.68, 1369.8, 3.0, image_view="oblique")
    r.update(
        lat=41.015, lon=28.979, tarife_tl_kwh=3.0, yillik_isinim_kwh_m2=1738.72, sicaklik_kaybi_yuzde=6.49,
        panel_egim_derece=33, panel_yon_derece=8, ufuk_cizgisi_72=[0.0] * 72,
        aylik_uretim_kwh=[1.0] * 12, aylik_ideal_kwh=[1.0] * 12,
    )
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
    assert validate_explanation_numbers("Faturanızda %6 azalma olur", p)[0] is False


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
    assert p["golgeli_uretim_kwh_kwp"] == 1370 and isinstance(p["yillik_uretim_kwh"], int)
    assert p["sicaklik_kaybi_yuzde"] == 6.5 and p["kurulabilir_guc_kwp"] == 14.3
    assert p["yillik_tasarruf_tl_aralik"] == [29361, 58723]  # 29361.x / 58723.x from the result, whole TL
    assert all(isinstance(v, int) for v in p["yillik_tasarruf_tl_aralik"])
    assert round_half_up(27.15, 1) == 27.2 and round_half_up(1738.72, 0) == 1739
    assert p["engel_orani"] == 0.272


def test_payload_strips_large_fields_and_counts_obstacles(payload):
    for key in ("ufuk_cizgisi_72", "aylik_uretim_kwh", "aylik_ideal_kwh", "roof_polygon_px", "lat", "lon"):
        assert key not in payload
    assert payload["engel_sayilari"] == {"Çatı penceresi": 2, "Uydu çanağı": 1, "Klima": 1, "Baca": 1}
    assert payload["goruntu_turu"] == "oblique" and payload["cati_tipi"] == "flat"
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
    assert "AYRI bir cümle" in prompt and "Değerlendirici sıfat kullanma" in prompt and "fatura" in prompt and "tasarruf" in prompt
    assert "Koşullu anlatım kullanma" in prompt and '"varsa"' in prompt
    assert "kontrol edin" in prompt and "bakın" in prompt and "Dilbilgisi" in prompt
    assert "adetleriyle adlandırarak" in prompt and "golgeli_uretim_kwh_kwp" in prompt
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
    assert lines[-1] == WARNING_LINE and sum(ln.startswith("-") for ln in lines) == 5
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
        (("skylight", "solar panel"), "kurulum firmasından görüş alın", "baca"),
        ((), "kurulum firmasından görüş alın", "taşı"),
    ],
)
def test_template_without_relocatable_obstacles_gives_another_recommendation(result, labels, expected, absent):
    text = build_fallback_explanation(result, make_detection(labels), False)
    recs = [ln for ln in text.splitlines() if ln.startswith("-")][3:]
    assert len(recs) == 2 and expected in recs[1] and absent not in recs[1]
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
