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
from core.formatting import percent_from_ratio, round_half_up, tr_num

WARNING_LINE = "⚠️ Kurulumdan önce çatının statik taşıma kapasitesini yetkili bir mühendise kontrol ettirin."

# Result keys sent to the model, grouped by the rounding they get before sending (and before validation).
_INT_KEYS = ("yillik_isinim_kwh_m2", "ideal_uretim_kwh_kwp", "golgeli_uretim_kwh_kwp", "panel_egim_derece", "panel_yon_derece")
_ONE_DECIMAL_KEYS = (
    "toplam_cati_alani_m2", "engel_alani_m2", "kullanilabilir_alan_m2", "golge_kaybi_yuzde", "uygunluk_puani_yuzde",
    "sicaklik_kaybi_yuzde",
)
_PASSTHROUGH_KEYS = ("goruntu_turu", "guven_seviyesi", "guven_nedenleri", "tasarruf_yontemi")
# bill fields: the model may quote them, never compute with them
_SAVINGS_INT_KEYS = ("fatura_tl_ay", "tuketim_kwh_ay")
_TARIFF_KEYS = ("tarife_dusuk_tl_kwh", "tarife_yuksek_tl_kwh")
_RELOCATABLE_LABELS = ("Su deposu", "Klima", "Uydu çanağı")

FINANCIAL_TERMS = (
    "amorti", "geri ödeme", "geri dönüş süresi", "maliyet", "yatırım", "teşvik", "kredi", "taksit",
    "kazanç süresi",
)
COST_TERMS = ("maliyet", "yatırım")  # allowed only when a cost range is in the payload
PAYBACK_TERMS = ("amorti", "geri ödeme")  # allowed only when a numeric payback range is in the payload


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _has_numbers(value):
    """True for a number or a list of numbers (a 'hesaplanamaz' string or None does not count)."""
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(_is_num(v) for v in value)
    return _is_num(value)


def forbidden_terms(payload=None):
    """FINANCIAL_TERMS minus the words the payload's numbers make legitimate (cost range, numeric payback range)."""
    payload = payload or {}
    allowed = set()
    if _has_numbers(payload.get("kurulum_maliyeti_tl")):
        allowed.update(COST_TERMS)
    if _has_numbers(payload.get("geri_odeme_yil")):
        allowed.update(PAYBACK_TERMS)
    return tuple(t for t in FINANCIAL_TERMS if t not in allowed)


def system_prompt_for(payload=None):
    """System prompt; the forbidden-word list follows the payload (cost and payback words only with their numbers)."""
    banned = ", ".join(t for t in forbidden_terms(payload) if t != "kazanç süresi")
    return f"""Sen İstanbul'da bir apartmanda yaşayan kat maliklerine konuşan, sade Türkçe kullanan bir çatı güneş enerjisi danışmanısın.
Amaç: apartmandaki dairelerin birlikte tek bir çatı sistemi kurması, tek seferde ödemesi ve üretimi dairelere EŞİT paylaştırarak elektrik faturalarını düşürmesidir.
Girdi, hesaplanmış sonuçları içeren bir JSON'dur. Yanıt KISA olsun: toplam en fazla yaklaşık 6 cümle ve en fazla 3 madde. TAM OLARAK şu yapıda yaz:
1) İlk satır: TEK cümlelik genel hüküm. Değerlendirici sıfat kullanma (elverişli, uygun, ideal, mükemmel vb. yok); yalnızca özgül verimi (kWh/kWp) ve kurulabilir gücü (kWp) sayılarıyla aktar ve güven seviyesini belirt. Cümle nokta ile bitsin.
2) Ayrı bir satırda en fazla 4 cümlelik kısa bir paragraf. Odak: binanın kurulabilir gücü ve yıllık üretimi (bina toplamı); daire başına sabit indirim (yıllık ve aylık ortalama) ve varsa geri ödeme süresi; en büyük belirsizlik ve onu azaltacak TEK adım. Yıllık ışınım ve arazi gölgesine en fazla bir cümle ayır. "Ortalama orana göre varsayım" ifadesini YALNIZCA fotograf_var false ise kullan; fotograf_var true ise alanın fotoğraftan ölçüldüğünü söyle. daire_sayisi_tahmini true ise "tahmini" de.
3) "-" ile başlayan EN FAZLA 3 pratik öneri (ör. fotoğraf açılıysa veya yoksa kuşbakışı fotoğraf deneyin; daire_sayisi_tahmini true ise gerçek daire sayısını girmelerini söyle; yalnızca engel_sayilari gerçekten su deposu, klima veya uydu çanağı içeriyorsa onları adetleriyle adlandırarak ("2 uydu çanağı") başka yere taşımayı öner; içermiyorsa taşıma önerme, bunun yerine başka yararlı bir öneri ver: kuşbakışı fotoğraf deneyin veya engel_sayilari "Baca" içeriyorsa tespit edilen baca sayısını yazarak şöyle öner: "Çatınızdaki 5 bacanın çevresindeki boşluğu bir kurulum firmasına kontrol ettirin.").
4) Son satır AYNEN şu olsun: {WARNING_LINE}

Kesin kurallar:
- YALNIZCA girdideki sayıları kullan. Panel adedi, girdide olmayan süre veya tutar, teşvik, kredi koşulu veya girdide olmayan hiçbir yüzdeyi hesaplama ya da uydurma. Sayıları değiştirme, yeniden toplama veya bölme.
- Şu sözcükleri hiç kullanma: {banned}, kazanç süresi. Parasal kazancı anlatırken "tasarruf" de.
- Tasarruf her daireye EŞİT TL indirim olarak verilir (herkes kurulumu eşit öder): daire_basi_tasarruf_tl_yil yıllık, daire_basi_indirim_tl_ay yıllık ortalama aylık indirimdir; "sabit indirim" de. Yıllık ortalama olduğunu, kışın az yazın çok olduğunu ve indirimin faturayı aşamayacağını belirt. Bireysel daire faturası isteme.
- Girdideki alan (anahtar) adlarını ASLA yazma: alt çizgili hiçbir ad (örneğin kurulabilir_guc_kwp) metinde geçmesin; her şeyi doğal Türkçe ile söyle.
- Ekranda zaten gösterilen notları tekrar etme: aylık mahsuplaşma, yıllık ortalama / mevsim notu, eşit paylaşım ve aidat notu, maliyet kaynağı, maliyete dahil olmayan kalemler ve hukuki not. Yalnızca gerekiyorsa tek kelimeyle an.
- fotograf_var false ise engel oranından ve engellerden hiç söz etme (engeller bilinmiyor, sıfır değil).
- arazi_golgesi "yok" ise "arazi gölgesi yok (%100)" de; gölge kaybı veya sıfır yüzdesi yazma.
- Sonuçlar bina için ve dairelere eşit paylaştırma esasıyla verilir: "tek daire" veya "tek hane" anlatımı kullanma, "daire başına" de.
- tasarruf_yontemi "fatura" ise tasarrufu daire başına fatura ile ilişkilendir: faturanın yüzde kaçının karşılandığını fatura_karsilama_yuzde_aralik ile ("Daire başına faturanın yaklaşık %20–%30'u karşılanır") anlat ve tuketim_varsayimi ile mahsuplasma_varsayimi metinlerini kısaca aktar. "genis_tahmin" ise tüketim (tarife kademesi) bilinmediği için tasarruf aralığının geniş olduğunu söyle. Tarife değerlerini (tarife_dusuk_tl_kwh, tarife_yuksek_tl_kwh) anabilirsin; ama bunlarla ASLA hesap yapma, yeni TL tutarı üretme.
- Kurulum maliyeti: kurulum_maliyeti_tl bina toplamıdır, daire_basi_maliyet_tl daire başınadır; ikisini olduğu gibi aktar ve gerçek fiyatın teklifle netleşeceğini söyle. maliyet_bayraklari doluysa içeriğini aktar.
- geri_odeme_yil bir sayı aralığıysa geri ödeme süresini YALNIZCA bu aralıkla ve "bozulma, bakım ve tarife değişimi dikkate alınmadı" notuyla aktar. "hesaplanamaz" ise geri ödeme süresinden hiç söz etme.
- cati_yapisi "Bilmiyorum" ise sonuçların tamamen düz ve tamamen eğimli çatı için verilen aralıklar olduğunu söyle.
- "uygunluk_puani_yuzde" YALNIZCA arazi (tepe) gölgesini yansıtır. Onu sadece "arazi gölgesi açısından" diyerek an; asla genel uygunluk hükmü olarak kullanma. Genel hükmü özgül verim, kullanılabilir alan / kWp ve güven seviyesine dayandır.
- Yalnızca su deposu, klima ve uydu çanağı başka yere taşınabilir. Çatı penceresi, baca, çatı katı / çatıdaki yapılar ve mevcut güneş panelleri için ASLA kaldırma veya taşıma önerme; bunlardan söz etme (tek istisna: engel_sayilari "Baca" içerdiğinde, adedini belirterek çevresindeki boşluğu kurulum firmasına kontrol ettirmeyi önerebilirsin).
- Koşullu anlatım kullanma ("varsa", "olması durumunda", "-abiliyorsa" gibi). Öneriyi kesin cümleyle yaz.
- Dilbilgisi açısından doğru Türkçe yaz. "Check" karşılığı emir kipi "kontrol edin" veya "bakın"dır ("kontrol et", "kontrol etmeyi deneyin" yazma).
- "Özgül verim" derken HER ZAMAN golgeli_uretim_kwh_kwp değerini kullan (ideal_uretim_kwh_kwp değil).
- Komşu bina gölgesi ve İstanbul ortalamasıyla karşılaştırma hakkında iddiada bulunma; bunlar "sonraki aşamada" eklenecek diyebilirsin.
- demo_photo true ise, fotoğrafın örnek olduğunu ve kullanıcının kendi çatısı olmadığını BİR KEZ, ilk satırda hüküm cümlesinden sonra, noktayla ayrılmış AYRI bir cümle olarak söyle (noktalı virgülle bağlama).
- Birimleri sembolle yaz: kWh/m², kWh/kWp, kWp, kWh, m², %. Birimleri yazıyla açma ("kilovatsaat", "metrekare", "yüzde" yazma).
- Sayıları Türkçe biçimde yaz (1.739 ; 41,8). Büyük TL tutarları için "27,5 bin TL" biçimi de kabul edilir.
- Yalnızca "-" madde işareti kullan; numaralı liste ve Markdown başlık kullanma."""


SYSTEM_PROMPT = system_prompt_for(None)

_cache = {}


def clear_cache():
    """Drop all cached explanations."""
    _cache.clear()


# --------------------------------------------------------------------- payload

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
            payload[key] = round_half_up(result[key], 0)
    for key in _ONE_DECIMAL_KEYS:
        if key == "engel_alani_m2" and detection is None:
            continue  # no photo: obstacles are unknown, not zero
        if result.get(key) is not None:
            payload[key] = round_half_up(result[key], 1)
    if payload.get("golge_kaybi_yuzde") == 0:
        del payload["golge_kaybi_yuzde"]  # a 0 % loss is said as "arazi gölgesi yok (%100)"
        payload["arazi_golgesi"] = "yok"
    if result.get("engel_orani") is not None and detection is not None:
        payload["engel_orani"] = round_half_up(result["engel_orani"], 3)
    for key in _SAVINGS_INT_KEYS:
        if result.get(key) is not None:
            payload[key] = round_half_up(result[key], 0)
    if result.get("fatura_karsilama_yuzde_aralik") is not None:
        payload["fatura_karsilama_yuzde_aralik"] = [round_half_up(v, 0) for v in result["fatura_karsilama_yuzde_aralik"]]
    for key in _TARIFF_KEYS:
        if result.get(key) is not None:
            payload[key] = round_half_up(result[key], 2)
    _add_building_payload(payload, result)
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


def _collapse(values, decimals):
    """[lo, hi] rounded; a single number when both ends are equal after rounding (so no '5–5')."""
    lo, hi = (round_half_up(v, decimals) for v in values)
    if decimals == 0:
        lo, hi = int(lo), int(hi)
    return lo if lo == hi else [lo, hi]


_SEGMENT_STATUS_TR = {
    "kullanildi": "kullanıldı",
    "elendi": "verimsiz olduğu için hesaba katılmadı",
    "alinamadi": "güneş verisi alınamadı",
}


def _add_building_payload(payload, r):
    """Building-mode fields: structure, flats, ranges, cost, sharing, payback, assumption, segments."""
    from core.building_mode import STRUCTURE_LABELS_TR

    payload["cati_yapisi"] = STRUCTURE_LABELS_TR[r["cati_yapisi"]]
    if r.get("duz_pay_yuzde") is not None:
        payload["duz_pay_yuzde"] = int(r["duz_pay_yuzde"])
    payload["daire_sayisi"] = int(r["daire_sayisi"])
    payload["daire_sayisi_tahmini"] = bool(r["daire_sayisi_tahmini"])
    payload["kurulabilir_guc_kwp"] = _collapse(r["kurulabilir_guc_kwp_aralik"], 1)
    payload["yillik_uretim_kwh"] = _collapse(r["yillik_uretim_kwh_aralik"], 0)
    payload["co2_ton_yil"] = _collapse(r["co2_ton_yil_aralik"], 1)
    payload["kurulum_maliyeti_tl"] = _collapse(r["toplam_maliyet_tl_aralik"], 0)
    payload["daire_basi_maliyet_tl"] = _collapse(r["daire_basi_maliyet_tl_aralik"], 0)
    payload["daire_basi_tasarruf_tl_yil"] = _collapse(r["daire_basi_tasarruf_tl_aralik"], 0)
    payload["daire_basi_indirim_tl_ay"] = _collapse(r["daire_basi_indirim_tl_ay_aralik"], 0)
    payload["bina_toplam_tasarruf_tl_yil"] = _collapse(r["bina_toplam_tasarruf_tl_aralik"], 0)
    payback = r["geri_odeme_yil_aralik"]
    payload["geri_odeme_yil"] = "hesaplanamaz" if None in payback else _collapse(payback, 1)
    if r.get("maliyet_bandi_tl_kwp"):
        payload["maliyet_bandi_tl_kwp"] = [int(v) for v in r["maliyet_bandi_tl_kwp"]]
    payload["maliyet_bayraklari"] = list(r.get("maliyet_bayraklari") or [])
    if r.get("fatura_karsilama_yuzde_aralik") is not None:
        payload["fatura_karsilama_yuzde_aralik"] = _collapse(r["fatura_karsilama_yuzde_aralik"], 0)
        payload["kalan_fatura_tl_ay"] = _collapse(r["kalan_fatura_tl_ay_aralik"], 0)
        payload["tuketim_varsayimi"] = r["tuketim_varsayimi"]
        payload["mahsuplasma_varsayimi"] = r["mahsuplasma_notu"]
    payload["uyarilar"] = list(r.get("uyarilar") or [])
    payload["senaryolar"] = [
        {
            "etiket": sc["etiket"],
            "cati_yuzleri": [
                {
                    "ad": seg["ad"], "durum": _SEGMENT_STATUS_TR[seg["durum"]],
                    "kwp": round_half_up(seg["kwp"], 1), "yillik_kwh": int(round_half_up(sum(seg["aylik_kwh"]), 0)),
                }
                for seg in sc["segmentler"]
                if seg["durum"] in _SEGMENT_STATUS_TR
            ],
        }
        for sc in r["senaryolar"]
    ]


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
    terms = forbidden_terms(payload)
    financial_spans = []
    pos = 0
    for part in _SENTENCE_SPLIT_RE.split(cleaned):
        if any(term in _tr_lower(part) for term in terms):
            financial_spans.append((pos, pos + len(part)))
        pos += len(part)

    invalid = []
    for m in _NUM_RE.finditer(cleaned):
        value, decimals = _parse_number(m.group())
        in_financial_sentence = any(start <= m.start() < end for start, end in financial_spans)
        if in_financial_sentence or m.start() in forbidden_starts or not _matches(value, decimals, allowed):
            invalid.append(value)
    return not invalid, invalid


_SNAKE_RE = re.compile(r"[a-z]+_[a-z_]+")


def find_field_names(text, payload):
    """Payload keys and snake_case tokens that leaked into the text (sorted); [] when the text is clean."""
    cleaned = (text or "").replace(WARNING_LINE, "")
    leaked = set(_SNAKE_RE.findall(cleaned))
    for key in payload:
        if re.search(rf"(?<![\w]){re.escape(key)}(?![\w])", cleaned):
            leaked.add(key)
    return sorted(leaked)


# -------------------------------------------------------------------- template

def _relocatable_phrase(payload):
    parts = [f"{n} {label.lower()}" for label, n in payload.get("engel_sayilari", {}).items() if label in _RELOCATABLE_LABELS]
    return ", ".join(parts)


def _fmt(value, decimals=0):
    """Number or [lo, hi] -> Turkish text ('4,5' or '4,5–6,2')."""
    if isinstance(value, list):
        return f"{tr_num(value[0], decimals)}–{tr_num(value[1], decimals)}"
    return tr_num(value, decimals)


def _building_fallback(p):
    """Concise template: verdict, one short paragraph, at most 3 tips; only payload numbers, no notes shown on screen."""
    unknown = p["cati_yapisi"] == "Bilmiyorum"
    verdict = (
        f"Bu konumda her kWp panel yılda yaklaşık {tr_num(p['golgeli_uretim_kwh_kwp'])} kWh üretir; "
        f"çatıya yaklaşık {_fmt(p['kurulabilir_guc_kwp'], 1)} kWp sığabilir (güven: {p['guven_seviyesi']})."
    )
    if p["demo_photo"]:
        verdict += " Gösterilen fotoğraf örnektir, sizin çatınız değildir."

    terrain = (
        "arazi gölgesi yok (%100)" if p.get("arazi_golgesi") == "yok"
        else f"arazi gölgesi açısından uygunluk %{tr_num(p['uygunluk_puani_yuzde'], 1)}"
    )
    s1 = (
        f"Bina toplamı yılda yaklaşık {_fmt(p['yillik_uretim_kwh'])} kWh üretir; yıllık ışınım "
        f"{tr_num(p['yillik_isinim_kwh_m2'])} kWh/m², {terrain}."
    )
    flats = f"{p['daire_sayisi']} daire" + (" (tahmini)" if p["daire_sayisi_tahmini"] else "")
    s2 = (
        f"Üretim {flats} arasında eşit paylaşılırsa her daireye sabit indirim yıllık yaklaşık "
        f"{_fmt(p['daire_basi_tasarruf_tl_yil'])} TL (aylık ortalama {_fmt(p['daire_basi_indirim_tl_ay'])} TL) olur"
    )
    if p.get("fatura_karsilama_yuzde_aralik") is not None:
        s2 += f"; daire faturasının yaklaşık %{_fmt(p['fatura_karsilama_yuzde_aralik'])} kadarı karşılanır"
    else:
        s2 += " (geniş tahmin: kademe bilinmiyor)"
    s2 += "."
    s3 = (
        f"Kurulum maliyeti toplam yaklaşık {_fmt(p['kurulum_maliyeti_tl'])} TL, daire başına "
        f"{_fmt(p['daire_basi_maliyet_tl'])} TL"
    )
    if p["geri_odeme_yil"] != "hesaplanamaz":
        s3 += f"; geri ödeme süresi yaklaşık {_fmt(p['geri_odeme_yil'], 1)} yıl"
    s3 += "."
    if unknown:
        s4 = "En büyük belirsizlik çatı yapısı: tamamen düz ve tamamen eğimli çatı için aralık verildi."
    elif not p["fotograf_var"]:
        s4 = "En büyük belirsizlik kullanılabilir alan: fotoğraf olmadığı için ortalama orana göre varsayıldı."
    elif p.get("engel_orani") is not None:
        s4 = f"Fotoğrafa göre çatının %{tr_num(percent_from_ratio(p['engel_orani']), 1)} kadarı engellerle kaplı."
    elif p["daire_sayisi_tahmini"]:
        s4 = "En büyük belirsizlik daire sayısı: tahmindir."
    else:
        s4 = "Kesin sonuç için yerinde ölçüm gerekir."
    paragraph = " ".join([s1, s2, s3, s4])

    tips = []
    if unknown:
        tips.append("- Çatı yapısını (düz, eğimli, karma) seçin; sonuç aralığı daralır.")
    elif not p["fotograf_var"] or p.get("goruntu_turu") != "top-down":
        tips.append("- Kuşbakışı (yukarıdan çekilmiş) bir fotoğraf deneyin; alan ve engel ölçümü daha güvenilir olur.")
    elif p.get("engel_orani") is not None:
        tips.append("- Engel tablosunu kontrol edin; çatıda olmayan engellerin işaretini kaldırarak sonucu güncelleyin.")
    if p["daire_sayisi_tahmini"]:
        tips.append("- Daire sayısı tahmindir; gerçek daire sayısını girin, daire başına rakamlar buna göre güncellenir.")
    if p["tasarruf_yontemi"] == "genis_tahmin":
        tips.append("- Ortalama daire faturasını girin; tasarruf aralığı daralır.")
    relocatable = _relocatable_phrase(p)
    if relocatable:
        tips.insert(1, f"- Çatıda {relocatable} var; bunları başka bir yere taşımanın mümkün olup olmadığına bakın, kullanılabilir alan artar.")
    elif "Baca" in p.get("engel_sayilari", {}):
        tips.insert(1, f"- Çatınızdaki {p['engel_sayilari']['Baca']} bacanın çevresindeki boşluğu bir kurulum firmasına kontrol ettirin.")
    if not tips:
        tips.append("- Kat malikleri kurulu kararını netleştirin, ardından bir kurulum firmasından teklif alın.")
    return "\n".join([verdict, paragraph, *tips[:3], WARNING_LINE])


def build_fallback_explanation(result, detection=None, demo_photo=False):
    """Deterministic Turkish template with the same 4 sections; only payload numbers, no qualitative verdict."""
    return _building_fallback(build_payload(result, detection, demo_photo))


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
    config = types.GenerateContentConfig(system_instruction=system_prompt_for(payload), temperature=0.2)
    return client.models.generate_content(model=EXPLAIN_MODEL, contents=contents, config=config).text


def _retry_hint(invalid, leaked=()):
    if leaked:
        return (
            f"Önceki yanıtta girdideki alan adları geçiyordu: {', '.join(leaked)}. Alan adlarını hiç yazma; "
            "doğal Türkçe kullan."
        )
    if not invalid:
        return "Önceki yanıt boştu. Kurallara uyarak yanıtı yeniden yaz."
    numbers = ", ".join(tr_num(v, 0 if float(v).is_integer() else 2) for v in invalid)
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
            leaked = find_field_names(text, payload) if text else []
            if ok and not leaked:
                out = {"text": text, "source": "gemini", "retried": retried, "invalid_numbers": []}
                _cache[key] = out
                return dict(out)
            if attempt == 0:
                retried, hint = True, _retry_hint(invalid, leaked)
    except Exception:  # missing key, network, 429/503, SDK errors: degrade to the template
        pass
    return {
        "text": build_fallback_explanation(result, detection, demo_photo),
        "source": "template",
        "retried": retried,
        "invalid_numbers": invalid,
    }
