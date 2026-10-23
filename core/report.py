"""Traceability: step-by-step calculation breakdown ("Hesap dökümü") and the downloadable report.

Built ONLY from the result dict and config constants (no LLM, no new calculation). Every number shown here is a
result value or a configured constant, rounded with the shared half-up rule.
"""
from core.building_mode import CAPACITY_SOURCE_LABELS_TR, LEGAL_NOTE, PITCHED_TYPE_LABELS_TR, STRUCTURE_LABELS_TR
from core.config import DEFAULT_ROOF_PITCH_DEG, FACE_MIN_YIELD_RATIO, M2_PER_KWP_FLAT, M2_PER_KWP_PITCHED, PANEL_KWP
from core.formatting import round_half_up, tr_num
from core.solar import pvcalc_params, pvcalc_url

IMAGERY_NOTE = (
    "Uydu görüntüsü (Esri World Imagery) yalnızca gösterim içindir; hiçbir görüntü bir modele gönderilmez, "
    "saklanmaz veya analiz edilmez. Çizilen her şey kendi geometri ve hesaplarımızdan gelir."
)
PVGIS_CAPTION = "Avrupa Komisyonu'nun resmî PVGIS servisinin ham çıktısı"
SOURCE_DATA = "OSM/Overture"
SOURCE_PVGIS = "PVGIS"
SOURCE_EPDK = "EPDK"
SOURCE_COST = "satıcı fiyatları"
SOURCE_ASSUMPTION = "varsayım"
SOURCE_DRAWING = "kullanıcı çizimi"
_RATIO_SOURCE = {
    "fotograf": "fotoğraf (Gemini)",
    "ortalama": SOURCE_ASSUMPTION,
    "engel_yok": "kullanıcı onayı (engel yok)",
}
_RATIO_NOTE = {
    "fotograf": "fotoğraftaki engel oranı düşüldü",
    "ortalama": "ortalama oran; çizim yok",
    "engel_yok": "kullanıcı çatının boş olduğunu onayladı",
}


def ratio_panel_text(r):
    """Panel count of a ratio result (average / photo / 'Engel yok' without a layout): an estimate, not a layout."""
    return f"≈{_span(r['panel_sayisi_aralik'])} panel (orandan tahmini; yerleşim için engelleri çizin)"


def _n(value, decimals=0):
    return tr_num(value, decimals)


def _span(values, decimals=0, unit=""):
    lo, hi = (round_half_up(v, decimals) for v in values)
    text = _n(lo, decimals) if lo == hi else f"{_n(lo, decimals)}–{_n(hi, decimals)}"
    return f"{text} {unit}".strip()


def _row(step, calc, source):
    return {"Adım": step, "Hesap": calc, "Kaynak": source}


def _segment_rows(sc, geometric, prefix):
    rows = []
    cos_text = f"cos{DEFAULT_ROOF_PITCH_DEG}°"
    for seg in sc["segmentler"]:
        name = f"{prefix}{seg['ad']}"
        if seg["durum"] == "kullanilmadi":
            continue
        if seg["durum"] == "elendi":
            rows.append(_row(
                name, f"özgül verim düz bölümün %{_n(seg['verim_orani'] * 100)}'i < %{_n(FACE_MIN_YIELD_RATIO * 100)} "
                "→ verimsiz yüz, 0 kWp", SOURCE_PVGIS,
            ))
            continue
        if seg["durum"] == "alinamadi":
            rows.append(_row(name, "güneş verisi alınamadı → 0 kWp", SOURCE_PVGIS))
            continue
        if geometric:
            calc = f"{seg['panel_sayisi']} panel × {_n(PANEL_KWP, 2)} kWp = {_n(seg['kwp'], 1)} kWp ({_n(seg['plan_m2'], 1)} m² plan)"
            rows.append(_row(name, calc, SOURCE_DRAWING))
        elif seg["tur"] == "flat":
            calc = f"{_n(seg['plan_m2'], 1)} m² ÷ {_n(M2_PER_KWP_FLAT, 1)} = {_n(seg['kwp'], 1)} kWp"
            rows.append(_row(name, calc, SOURCE_ASSUMPTION))
        else:
            calc = (
                f"{_n(seg['plan_m2'], 1)} m² ÷ {cos_text} ÷ {_n(M2_PER_KWP_PITCHED, 1)} = {_n(seg['kwp'], 1)} kWp"
            )
            rows.append(_row(name, calc, SOURCE_ASSUMPTION))
    return rows


def _direction(value, positive, negative):
    return f"{tr_num(abs(value))} m {positive if value >= 0 else negative}"


def alignment_row(offset):
    """Display-only imagery alignment shift of the roof view; None at zero (or no offset)."""
    if not offset or (offset[0] == 0 and offset[1] == 0):
        return None
    return _row(
        "Görüntü hizalama kaydırması",
        f"{_direction(offset[0], 'doğu', 'batı')}, {_direction(offset[1], 'kuzey', 'güney')} (yalnızca gösterim)",
        "kullanıcı ayarı",
    )


def breakdown_rows(r, offset=None):
    """[{Adım, Hesap, Kaynak}] for the 'Hesap dökümü' table, one row per calculation step.

    offset: the roof view's imagery alignment (dx east, dy north in m); it adds a display-only row, no result changes."""
    rows = []
    geometric = r.get("guc_kaynagi") == "geometri"
    if r.get("kenar_sonrasi_alan_m2") is not None:
        rows.append(_row("Çatı alanı (bina izi)", f"{_n(r['toplam_cati_alani_m2'], 1)} m²", SOURCE_DATA))
        rows.append(_row(
            "Kenar boşluğu sonrası alan",
            f"{_n(r['toplam_cati_alani_m2'], 1)} m² → {_n(r['kenar_sonrasi_alan_m2'], 1)} m² "
            f"({_n(r['kenar_boslugu_m'], 0)} m içeri)",
            f"{SOURCE_DATA}, {SOURCE_ASSUMPTION}",
        ))
    else:
        rows.append(_row("Çatı alanı", f"{_n(r['toplam_cati_alani_m2'], 1)} m² (elle girildi)", "kullanıcı girdisi"))
    if geometric:
        rows.append(_row(
            "Panel sayısı", f"{_span(r['panel_sayisi_aralik'])} panel × {_n(PANEL_KWP, 2)} kWp = "
            f"{_span(r['kurulabilir_guc_kwp_aralik'], 1, 'kWp')} (kullanılabilir alan oranı kullanılmadı)",
            SOURCE_DRAWING,
        ))
    else:
        src = r.get("guc_kaynagi", "ortalama")
        rows.append(_row(
            "Kullanılabilir alan",
            f"{_n(r['alan_tabani_m2'], 1)} × {_n(r['kullanilabilir_oran'], 3)} ≈ {_n(r['kullanilabilir_alan_m2'], 1)} m² "
            f"({_RATIO_NOTE[src]})",
            _RATIO_SOURCE[src],
        ))
    multi = len(r["senaryolar"]) > 1
    for sc in r["senaryolar"]:
        prefix = f"{sc['etiket']} · " if multi else ""
        if not geometric and sc["duz_pay"] not in (0.0, 1.0):
            rows.append(_row(
                f"{prefix}Teras payı", f"%{_n(sc['duz_pay'] * 100)} düz, %{_n((1 - sc['duz_pay']) * 100)} eğimli",
                "kullanıcı girdisi",
            ))
        rows += _segment_rows(sc, geometric, prefix)
        rows.append(_row(
            f"{prefix}Üretim", f"Σ kWp × özgül verim = {_n(sc['kwp'], 1)} kWp → {_n(sc['kwh'])} kWh/yıl", SOURCE_PVGIS,
        ))
    n = r["daire_sayisi"]
    rows.append(_row(
        "Daire payı", f"{_span(r['yillik_uretim_kwh_aralik'])} kWh ÷ {n} ÷ 12 = {_span(r['daire_payi_kwh_ay_aralik'], 0, 'kWh/ay')}",
        f"{SOURCE_ASSUMPTION} (eşit paylaşım)",
    ))
    if r["tasarruf_yontemi"] == "fatura":
        rows.append(_row(
            "Tüketim", f"Fatura {_n(r['fatura_tl_ay'])} TL → {_n(r['tuketim_kwh_ay'])} kWh/ay (EPDK kademeleri)", SOURCE_EPDK,
        ))
        rows.append(_row(
            "İndirim", f"aylık mahsuplaşma: ortalama {_span(r['daire_basi_indirim_tl_ay_aralik'], 0, 'TL/ay')} "
            f"(yıllık {_span(r['daire_basi_tasarruf_tl_aralik'], 0, 'TL')})", SOURCE_EPDK,
        ))
    else:
        rows.append(_row(
            "İndirim", f"{_span(r['daire_payi_kwh_ay_aralik'], 0, 'kWh')} × {_n(r['tarife_dusuk_tl_kwh'], 2)}–"
            f"{_n(r['tarife_yuksek_tl_kwh'], 2)} TL = {_span(r['daire_basi_indirim_tl_ay_aralik'], 0, 'TL/ay')} "
            "(kademe bilinmiyor)", SOURCE_EPDK,
        ))
    for sc in r["senaryolar"]:
        prefix = f"{sc['etiket']} · " if multi else ""
        band = sc["maliyet_bandi"]
        calc = (
            f"{_n(sc['kwp'], 1)} kWp × {_n(band[0])}–{_n(band[1])} TL/kWp = {_span(sc['maliyet_tl'], 0, 'TL')}"
            if band else "kurulabilir panel yok → 0 TL"
        )
        rows.append(_row(f"{prefix}Maliyet", calc, SOURCE_COST))
    cost_lo, cost_hi = r["daire_basi_maliyet_tl_aralik"]
    payback = r["geri_odeme_yil_aralik"]
    pay_text = (
        f"{_span(payback, 1, 'yıl')}" if None not in payback
        else "hesaplanamaz (tasarruf çok düşük)" if payback == [None, None]
        else " – ".join(f"{_n(v, 1)} yıl" if v is not None else "hesaplanamaz" for v in payback)
    )
    rows.append(_row(
        "Geri ödeme", f"daire başına maliyet {_span([cost_lo, cost_hi], 0, 'TL')} ÷ yıllık indirim "
        f"{_span(r['daire_basi_tasarruf_tl_aralik'], 0, 'TL')} = {pay_text}", f"{SOURCE_COST}, {SOURCE_EPDK}",
    ))
    if (align := alignment_row(offset)) is not None:
        rows.append(align)
    return rows


def pvgis_requests(lat, lon, flat_solar, faces, result):
    """[{segment, url}]: the PVcalc URLs behind this result (flat ideal + terrain run, every face that has area).

    The URL stored with the (cached) PVGIS answer is used; it is rebuilt from the same parameters only when the
    answer carries none (e.g. test doubles).
    """
    def url(answer, params):
        return (answer or {}).get("request_url") or pvcalc_url(params)

    ideal, shaded = flat_solar["ideal"], flat_solar["shaded"]
    out = [
        {"segment": "Düz bölüm, en uygun açı (ufuksuz)", "url": url(ideal, pvcalc_params(lat, lon, optimal=True, usehorizon=False))},
        {"segment": "Düz bölüm, arazi ufku dahil", "url": url(
            shaded, pvcalc_params(lat, lon, angle=flat_solar["angle"], aspect=flat_solar["aspect"], usehorizon=True)
        )},
    ]
    used = {seg["ad"] for sc in result["senaryolar"] for seg in sc["segmentler"] if seg["tur"] == "face" and seg["plan_m2"] > 0}
    for face in faces:
        name = f"{face['name']} yüzü"
        if name in used:
            params = pvcalc_params(lat, lon, angle=DEFAULT_ROOF_PITCH_DEG, aspect=face["aspect"], usehorizon=True)
            out.append({"segment": name, "url": url(face.get("solar"), params)})
    return out


def pvgis_links(r):
    """[(label, url)] for 'PVGIS'te kendiniz doğrulayın': the exact PVcalc URLs used for this result."""
    return [(req["segment"], req["url"]) for req in r.get("pvgis_istekleri") or [] if req.get("url")]


def _md_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def report_markdown(r, assumptions=(), offset=None):
    """One Markdown report built in memory: key cards, calculation breakdown, assumptions, links and legal note.

    assumptions: iterable of (name, value, status, source) rows (the 'Varsayımlar ve kaynaklar' table).
    """
    structure = STRUCTURE_LABELS_TR[r["cati_yapisi"]]
    if r["cati_yapisi"] in ("pitched", "mixed", "unknown"):
        structure += f" · eğimli çatı tipi: {PITCHED_TYPE_LABELS_TR[r['egimli_tip']]}"
    lines = [
        "# Çatı GES ön değerlendirme raporu",
        "",
        f"Konum: {r['lat']:.5f}, {r['lon']:.5f} · Çatı yapısı: {structure} · Daire sayısı: {r['daire_sayisi']}",
        "",
        "## Özet",
        "",
        f"- Panel sayısı: {_span(r['panel_sayisi_aralik'])}" if r.get("guc_kaynagi") == "geometri"
        else f"- Panel: {ratio_panel_text(r)}",
        f"- Kurulabilir güç: {_span(r['kurulabilir_guc_kwp_aralik'], 1, 'kWp')}",
        f"- Yıllık üretim: {_span(r['yillik_uretim_kwh_aralik'], 0, 'kWh')}",
        f"- CO₂ azaltımı: {_span(r['co2_ton_yil_aralik'], 1, 'ton/yıl')}",
        f"- Toplam kurulum maliyeti: {_span(r['toplam_maliyet_tl_aralik'], 0, 'TL')}",
        f"- Daire başına maliyet: {_span(r['daire_basi_maliyet_tl_aralik'], 0, 'TL')}",
        f"- Daire başına sabit indirim: {_span(r['daire_basi_tasarruf_tl_aralik'], 0, 'TL/yıl')} "
        f"(ortalama {_span(r['daire_basi_indirim_tl_ay_aralik'], 0, 'TL/ay')})",
        f"- Güç kaynağı: {CAPACITY_SOURCE_LABELS_TR.get(r.get('guc_kaynagi'), '—')}",
        f"- Güven seviyesi: {r['guven_seviyesi']}",
        "",
        "## Hesap dökümü",
        "",
        "| Adım | Hesap | Kaynak |",
        "|---|---|---|",
    ]
    lines += [f"| {_md_cell(row['Adım'])} | {_md_cell(row['Hesap'])} | {_md_cell(row['Kaynak'])} |" for row in breakdown_rows(r, offset)]
    assumptions = list(assumptions)
    if assumptions:
        lines += ["", "## Varsayımlar ve kaynaklar", "", "| Değer | Sonuç | Durum | Kaynak / not |", "|---|---|---|---|"]
        lines += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in assumptions]
    links = pvgis_links(r)
    if links:
        lines += ["", f"## PVGIS'te kendiniz doğrulayın ({PVGIS_CAPTION})", ""]
        lines += [f"- {label}: <{url}>" for label, url in links]
    lines += ["", "## Notlar", ""]
    if r.get("mahsuplasma_notu"):
        lines.append(f"- {r['mahsuplasma_notu']}")
    lines += [f"- {LEGAL_NOTE}", f"- {IMAGERY_NOTE}", ""]
    if r.get("guven_nedenleri"):
        lines += ["## Güven seviyesinin nedenleri", ""] + [f"- {x}" for x in r["guven_nedenleri"]] + [""]
    return "\n".join(lines)

