"""Grounded Turkish explanation with number check and template fallback.

Gemini docs: https://ai.google.dev/gemini-api/docs/text-generation (system_instruction in GenerateContentConfig).

The LLM only phrases numbers it is given. Every number in its text is checked against the payload
(`validate_explanation_numbers`); on failure it is retried once, then the deterministic template is used.
"""
import hashlib
import json
import re
from decimal import ROUND_HALF_UP, Decimal

from core.config import EXPLAIN_MODEL, get_gemini_api_key

WARNING_LINE = "⚠️ Kurulumdan önce çatının statik taşıma kapasitesini yetkili bir mühendise kontrol ettirin."

# Result keys sent to the model, grouped by the rounding they get before sending (and before validation).
_INT_KEYS = (
    "yillik_uretim_kwh", "yillik_isinim_kwh_m2", "ideal_uretim_kwh_kwp", "golgeli_uretim_kwh_kwp",
    "panel_egim_derece", "panel_yon_derece",
)
_ONE_DECIMAL_KEYS = (
    "toplam_cati_alani_m2", "engel_alani_m2", "kullanilabilir_alan_m2", "kurulabilir_guc_kwp",
    "golge_kaybi_yuzde", "uygunluk_puani_yuzde", "sicaklik_kaybi_yuzde", "co2_ton_yil",
)
_PASSTHROUGH_KEYS = ("cati_tipi", "goruntu_turu", "guven_seviyesi", "guven_nedenleri", "tarife_tl_kwh")
_RELOCATABLE_LABELS = ("Su deposu", "Klima", "Uydu çanağı")

FINANCIAL_TERMS = (
    "amorti", "geri ödeme", "geri dönüş süresi", "maliyet", "yatırım", "teşvik", "kredi", "taksit",
    "fatura", "kazanç süresi",
)

SYSTEM_PROMPT = f"""Sen İstanbul'da bir apartmanda yaşayan birine konuşan, sade Türkçe kullanan bir çatı güneş enerjisi danışmanısın.
Girdi, hesaplanmış sonuçları içeren bir JSON'dur. Yanıtın TAM OLARAK şu yapıda olsun:
1) İlk satır: TEK cümlelik genel hüküm. Değerlendirici sıfat kullanma (elverişli, uygun, ideal, mükemmel vb. yok); yalnızca özgül verimi (kWh/kWp) ve kurulabilir gücü (kWp) sayılarıyla aktar ve güven seviyesini belirt. Cümle nokta ile bitsin.
2) "-" ile başlayan 3 madde (nedenler); her biri girdideki sayılara dayansın:
   a) bu konumdaki yıllık ışınım ve özgül verim;
   b) engel oranı ve kullanılabilir alan. "Ortalama orana göre varsayım" ifadesini YALNIZCA fotograf_var false ise kullan; fotograf_var true ise alanın fotoğraftan ölçüldüğünü söyle;
   c) arazi gölgesi, sıcaklık kaybı veya güven seviyesi.
3) "-" ile başlayan 2 pratik öneri (ör. fotoğraf açılıysa veya yoksa kuşbakışı fotoğraf deneyin; engel_sayilari içinde su deposu, klima veya uydu çanağı varsa onları başka yere almanın mümkün olup olmadığına bakın).
4) Son satır AYNEN şu olsun: {WARNING_LINE}

Kesin kurallar:
- YALNIZCA girdideki sayıları kullan. Geri ödeme süresi, panel adedi, maliyet, teşvik, kredi koşulu veya girdide olmayan hiçbir yüzdeyi hesaplama ya da uydurma. Sayıları değiştirme, yeniden toplama veya bölme.
- Şu sözcükleri hiç kullanma: amorti, geri ödeme, geri dönüş süresi, maliyet, yatırım, teşvik, kredi, taksit, fatura, kazanç süresi. Parasal kazancı anlatırken "tasarruf" de.
- "uygunluk_puani_yuzde" YALNIZCA arazi (tepe) gölgesini yansıtır. Onu sadece "arazi gölgesi açısından" diyerek an; asla genel uygunluk hükmü olarak kullanma. Genel hükmü özgül verim, kullanılabilir alan / kWp ve güven seviyesine dayandır.
- Yalnızca su deposu, klima ve uydu çanağı başka yere taşınabilir. Çatı penceresi, baca, çatı katı / çatıdaki yapılar ve mevcut güneş panelleri için ASLA kaldırma veya taşıma önerme; bunlardan hiç söz etme (ne öneri ne uyarı olarak).
- Komşu bina gölgesi ve İstanbul ortalamasıyla karşılaştırma hakkında iddiada bulunma; bunlar "sonraki aşamada" eklenecek diyebilirsin.
- demo_photo true ise, fotoğrafın örnek olduğunu ve kullanıcının kendi çatısı olmadığını BİR KEZ, ilk satırda hüküm cümlesinden sonra, noktayla ayrılmış AYRI bir cümle olarak söyle (noktalı virgülle bağlama).
- Birimleri sembolle yaz: kWh/m², kWh/kWp, kWp, kWh, m², %. Birimleri yazıyla açma ("kilovatsaat", "metrekare", "yüzde" yazma).
- Sayıları Türkçe biçimde yaz (1.739 ; 41,8). Büyük TL tutarları için "27,5 bin TL" biçimi de kabul edilir.
- Yalnızca "-" madde işareti kullan; numaralı liste ve Markdown başlık kullanma."""

_cache = {}


def clear_cache():
    """Drop all cached explanations."""
    _cache.clear()


# --------------------------------------------------------------------- payload

def _round(value, decimals):
    """Round half up on the decimal repr (27.15 -> 27.2); returns int when decimals == 0."""
    q = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(q, ROUND_HALF_UP)
    return int(rounded) if decimals == 0 else float(rounded)


def build_payload(result, detection=None, demo_photo=False):
    """Compact, rounded payload for the model; the same dict is used for validation.

    kWh, kWh/m² and angles -> integers; %, m², kWp and tons -> 1 decimal; TL range -> integers.
    engel_orani stays a ratio (3 decimals) so its percentage form has 1 decimal.
    """
    payload = {}
    for key in _PASSTHROUGH_KEYS:
        if result.get(key) is not None:
            payload[key] = result[key]
    for key in _INT_KEYS:
        if result.get(key) is not None:
            payload[key] = _round(result[key], 0)
    for key in _ONE_DECIMAL_KEYS:
        if result.get(key) is not None:
            payload[key] = _round(result[key], 1)
    if result.get("engel_orani") is not None:
        payload["engel_orani"] = _round(result["engel_orani"], 3)
    if result.get("yillik_tasarruf_tl_aralik") is not None:
        payload["yillik_tasarruf_tl_aralik"] = [_round(v, 0) for v in result["yillik_tasarruf_tl_aralik"]]
    payload["fotograf_var"] = detection is not None
    payload["demo_photo"] = bool(demo_photo) and detection is not None
    if detection is not None:
        from core.vision import OBSTACLE_LABELS_TR

        counts = {}
        for obstacle in detection["obstacles"]:
            label = OBSTACLE_LABELS_TR[obstacle["label"]]
            counts[label] = counts.get(label, 0) + 1
        payload["engel_sayilari"] = counts
        payload["goruntu_turu"] = detection["image_view"]
    return payload


def payload_key(payload):
    """sha256 of the canonical (sorted-keys) payload JSON; the cache key and the staleness marker."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ validation

_UNIT_RE = re.compile(r"kWh/kWp|kWh/m²|kWh/m2|kWp|kWh|CO₂|CO2|m²|m2")
_NUM_RE = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?(?!\d)|\d+(?:[.,]\d+)?")
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")
_DURATION_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:yıl|yılda|ay|ayda)\s*(?:içinde|sonra|sonunda|zarfında)")
_PANEL_RE = re.compile(r"\d+\s*(?:adet\s+)?panel")
_SENTENCE_SPLIT_RE = re.compile(r"([!?]+|\.(?=\s|$)|\n)")
_THOUSANDS_TOLERANCE = 0.005


def _tr_lower(text):
    return text.replace("İ", "i").replace("I", "ı").lower()


def _parse_number(token):
    """Turkish number text -> (value, decimals): '25.500,5' -> (25500.5, 1), '1.726' -> (1726, 0)."""
    if _THOUSANDS_RE.fullmatch(token):
        token = token.replace(".", "")
    token = token.replace(",", ".")
    decimals = len(token.split(".")[1]) if "." in token else 0
    return float(token), decimals


def _collect_numbers(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_numbers(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_numbers(value, out)


def _allowed_values(payload):
    """Every payload number, the % form of engel_orani only, and the 'bin' (/1000) form of values >= 1000."""
    values = []
    _collect_numbers(payload, values)
    allowed = set(values)
    for v in values:
        if abs(v) >= 1000:
            allowed.add(v / 1000)
    ratio = payload.get("engel_orani")
    if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
        allowed.add(ratio * 100)
    return allowed


def _quantize(value, decimals):
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-decimals), ROUND_HALF_UP)


def _matches(value, decimals, allowed):
    """A number with k decimals is valid if an allowed value rounded to k decimals equals it exactly."""
    target = _quantize(value, decimals)
    for a in allowed:
        if _quantize(a, decimals) == target:
            return True
        if decimals == 0 and value >= 1000 and a and abs(value - a) / abs(a) <= _THOUSANDS_TOLERANCE:
            return True
    return False


def _clean_text(text):
    """Drop the fixed warning line, list markers and unit tokens that contain digits (CO2, m², kWh/m² ...)."""
    cleaned = (text or "").replace(WARNING_LINE, "")
    cleaned = re.sub(r"(?m)^\s*[-•*]\s+", "", cleaned)
    return _UNIT_RE.sub(" ", cleaned)


def extract_numbers(text):
    """All numbers in `text` as floats (Turkish formats; '-'/'–' between numbers is a separator, never a sign)."""
    return [_parse_number(m.group())[0] for m in _NUM_RE.finditer(_clean_text(text))]


def validate_explanation_numbers(text, payload):
    """Return (all_valid, invalid_numbers) for every number in `text` against `payload`."""
    cleaned = _clean_text(text)
    allowed = _allowed_values(payload)

    forbidden_starts = set()
    for rx in (_DURATION_RE, _PANEL_RE):
        forbidden_starts.update(m.start() for m in rx.finditer(cleaned))
    financial_spans = []
    pos = 0
    for part in _SENTENCE_SPLIT_RE.split(cleaned):
        if any(term in _tr_lower(part) for term in FINANCIAL_TERMS):
            financial_spans.append((pos, pos + len(part)))
        pos += len(part)

    invalid = []
    for m in _NUM_RE.finditer(cleaned):
        value, decimals = _parse_number(m.group())
        in_financial_sentence = any(start <= m.start() < end for start, end in financial_spans)
        if in_financial_sentence or m.start() in forbidden_starts or not _matches(value, decimals, allowed):
            invalid.append(value)
    return not invalid, invalid


# -------------------------------------------------------------------- template

def _tr(value, decimals=0):
    """Turkish number format: 1.234,5."""
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _relocatable_phrase(payload):
    parts = [f"{n} {label.lower()}" for label, n in payload.get("engel_sayilari", {}).items() if label in _RELOCATABLE_LABELS]
    return ", ".join(parts)


def build_fallback_explanation(result, detection=None, demo_photo=False):
    """Deterministic Turkish template with the same 4 sections; only payload numbers, no qualitative verdict."""
    p = build_payload(result, detection, demo_photo)
    verdict = (
        f"Bu konumda her kWp panel yılda yaklaşık {_tr(p['golgeli_uretim_kwh_kwp'])} kWh üretir; "
        f"çatınıza yaklaşık {_tr(p['kurulabilir_guc_kwp'], 1)} kWp sığabilir (güven: {p['guven_seviyesi']})."
    )
    if p["demo_photo"]:
        verdict += " Gösterilen fotoğraf örnektir, sizin çatınız değildir."

    if p["fotograf_var"] and p.get("engel_orani") is not None:
        area_line = (
            f"- Fotoğrafa göre çatının %{_tr(p['engel_orani'] * 100, 1)} kadarı engellerle kaplı; "
            f"{_tr(p['toplam_cati_alani_m2'], 1)} m² çatıdan {_tr(p['kullanilabilir_alan_m2'], 1)} m² kullanılabilir."
        )
    else:
        area_line = (
            f"- Fotoğraf olmadığı için kullanılabilir alan ortalama orana göre varsayıldı: "
            f"{_tr(p['toplam_cati_alani_m2'], 1)} m² çatıdan {_tr(p['kullanilabilir_alan_m2'], 1)} m²."
        )
    bullets = [
        f"- Bu konumda yıllık ışınım {_tr(p['yillik_isinim_kwh_m2'])} kWh/m²; panel başına özgül verim "
        f"{_tr(p['golgeli_uretim_kwh_kwp'])} kWh/kWp.",
        area_line,
        f"- Arazi gölgesi açısından uygunluk %{_tr(p['uygunluk_puani_yuzde'], 1)}, sıcaklık kaybı "
        f"%{_tr(p['sicaklik_kaybi_yuzde'], 1)}; güven seviyesi {p['guven_seviyesi']}.",
    ]

    if p.get("goruntu_turu") != "top-down":
        photo_tip = "- Kuşbakışı (yukarıdan çekilmiş) bir fotoğraf deneyin; alan ve engel ölçümü daha güvenilir olur."
    else:
        photo_tip = "- Engel tablosunu kontrol edin; çatıda olmayan engellerin işaretini kaldırarak sonucu güncelleyin."
    relocatable = _relocatable_phrase(p)
    if relocatable:
        tip2 = f"- Çatıdaki taşınabilir engeller ({relocatable}) başka yere alınabiliyorsa kullanılabilir alan artabilir."
    else:
        tip2 = "- Komşu bina gölgesi sonraki aşamada eklenecek; şimdilik yalnızca arazi gölgesi hesaba katıldı."
    return "\n".join([verdict, *bullets, photo_tip, tip2, WARNING_LINE])


# ---------------------------------------------------------------------- Gemini

def _get_client(api_key=None):
    """Create the Gemini client lazily (heavy import); patched in tests. A given api_key skips the lookup."""
    from google import genai

    return genai.Client(api_key=api_key or get_gemini_api_key())


def _generate(client, payload, hint=None):
    """One generate_content call; returns the response text (may be None/empty)."""
    from google.genai import types

    contents = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if hint:
        contents += "\n\n" + hint
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2)
    return client.models.generate_content(model=EXPLAIN_MODEL, contents=contents, config=config).text


def _retry_hint(invalid):
    if not invalid:
        return "Önceki yanıt boştu. Kurallara uyarak yanıtı yeniden yaz."
    numbers = ", ".join(_tr(v, 0 if float(v).is_integer() else 2) for v in invalid)
    return (
        f"Önceki yanıtta girdide bulunmayan veya yasak bağlamda geçen şu sayılar vardı: {numbers}. "
        "Bu sayıları kaldır; yalnızca girdideki sayıları, olduğu gibi kullan."
    )


def explain_result(result, detection=None, api_key=None, demo_photo=False):
    """Return {text, source: "gemini"|"template", retried, invalid_numbers}. Never raises."""
    payload = build_payload(result, detection, demo_photo)
    key = payload_key(payload)
    if key in _cache:
        return dict(_cache[key])

    retried, invalid = False, []
    try:
        client = _get_client(api_key)
        hint = None
        for attempt in range(2):
            text = (_generate(client, payload, hint) or "").strip()
            ok, invalid = validate_explanation_numbers(text, payload) if text else (False, [])
            if ok:
                out = {"text": text, "source": "gemini", "retried": retried, "invalid_numbers": []}
                _cache[key] = out
                return dict(out)
            if attempt == 0:
                retried, hint = True, _retry_hint(invalid)
    except Exception:  # missing key, network, 429/503, SDK errors: degrade to the template
        pass
    return {
        "text": build_fallback_explanation(result, detection, demo_photo),
        "source": "template",
        "retried": retried,
        "invalid_numbers": invalid,
    }
