"""Usable area, two-tier mesken tariff (bill, savings) and building-mode confidence."""
import math

from core.config import (
    EDGE_FACTOR,
    TARIFF_HIGH_TL_PER_KWH_EXCL_TAX,
    TARIFF_LOW_TL_PER_KWH_EXCL_TAX,
    TAX_FACTOR,
    TIER_LIMIT_KWH_PER_DAY,
    USABLE_RATIO_NO_PHOTO,
)

SAVINGS_HIGH_FACTOR = 1.0  # monthly netting (aylik mahsuplasma): the whole month's production offsets that month's consumption
DAYS_PER_MONTH = 30

# Two-tier residential tariff, tax included (TAX_FACTOR is applied exactly once, here)
LOW = TARIFF_LOW_TL_PER_KWH_EXCL_TAX * TAX_FACTOR
HIGH = TARIFF_HIGH_TL_PER_KWH_EXCL_TAX * TAX_FACTOR
LIMIT = TIER_LIMIT_KWH_PER_DAY * DAYS_PER_MONTH  # kWh/month billed at LOW; only the excess is billed at HIGH
BILL_AT_LIMIT = LIMIT * LOW


def _is_number(x):
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


# ------------------------------------------------------------------ tariff

def bill_tl(kwh):
    """Monthly bill (TL, tax included) for `kwh`: LOW tier up to LIMIT, HIGH tier above it."""
    if not _is_number(kwh) or kwh <= 0:
        return 0.0
    if kwh <= LIMIT:
        return kwh * LOW
    return BILL_AT_LIMIT + (kwh - LIMIT) * HIGH


def kwh_from_bill(bill):
    """Invert bill_tl (closed form). None, NaN or <= 0 means no bill: 0 kWh."""
    if not _is_number(bill) or bill <= 0:
        return 0.0
    if bill <= BILL_AT_LIMIT:
        return bill / LOW
    return LIMIT + (bill - BILL_AT_LIMIT) / HIGH


def monthly_savings(consumption_kwh, production_kwh_month, self_use_share):
    """Monthly TL saved: bill(C) - bill(C - production x share). PV offsets the HIGH tier first.

    production_kwh_month is the ROOF-scaled monthly production (kWp x E_m), never raw E_m.
    Surplus beyond consumption is worth 0 (conservative).
    """
    offset = max(production_kwh_month, 0) * self_use_share
    return bill_tl(consumption_kwh) - bill_tl(max(consumption_kwh - offset, 0))


def fallback_savings(annual_kwh):
    """Range when consumption is unknown: kWh x [LOW, HIGH]. Monthly netting removes the self-use uncertainty; only the
    tariff tier is unknown (tax is already in LOW/HIGH)."""
    return [annual_kwh * LOW, annual_kwh * HIGH]


# ------------------------------------------------------------ area and confidence

def usable_plan_area(roof_area_m2, obstacle_ratio, usable_base_area_m2=None):
    """Usable PLAN area (m2): footprint after setback x photo/average share, or manual area x factors."""
    if usable_base_area_m2 is not None:
        if obstacle_ratio is None:
            return usable_base_area_m2 * (USABLE_RATIO_NO_PHOTO / EDGE_FACTOR)
        return usable_base_area_m2 * (1 - obstacle_ratio)
    if obstacle_ratio is None:
        return roof_area_m2 * USABLE_RATIO_NO_PHOTO
    return roof_area_m2 * EDGE_FACTOR * (1 - obstacle_ratio)


_LEVELS = ("Düşük", "Orta", "Yüksek")


def building_confidence(
    image_view, has_polygon, roof_known, n_unavailable_faces=0, has_bill=False, flats_estimated=False,
    obstacles_marked=False,
):
    """Building-mode confidence (SPEC section 6): returns (level, reasons, improvements).

    Base: Orta with a footprint or a top-down photo, else Düşük. An unknown roof structure and an unavailable
    pitched face each drop one level (not below Düşük). A user-set Karma counts as known. Obstacles marked on the
    roof view replace the photo/average reason but never raise the level (Yüksek needs neighbour shading, P2).
    """
    top_down = image_view == "top-down"
    level = 1 if (top_down or has_polygon) else 0
    reasons, improve = [], []
    reasons.append(
        "Çatı alanı bina izinden ölçüldü (kenar boşluğu düşüldü)."
        if has_polygon
        else "Çatı alanı elle girildi; bina izi kullanılmadı."
    )
    if obstacles_marked:
        reasons.append("Engeller kullanıcı tarafından işaretlendi.")
    elif top_down:
        reasons.append("Fotoğraf kuşbakışı çekilmiş; engel oranı güvenilir biçimde ölçülebildi.")
    elif image_view == "oblique":
        reasons.append("Fotoğraf yandan/eğik açıdan çekilmiş; engel alanı yaklaşık, güveni artırmıyor.")
    else:
        reasons.append("Fotoğraf yok; kullanılabilir alan ortalama orana göre tahmin edildi (ortalama varsayım).")
    if not roof_known:
        level -= 1
        reasons.append("Çatı yapısı bilinmiyor; hem tamamen düz hem tamamen eğimli çatı için aralık hesaplandı.")
    if n_unavailable_faces:
        level -= 1
        reasons.append("Bir çatı yüzü için güneş verisi alınamadı; sonuç eksik olabilir.")
    if flats_estimated:
        improve.append("Daire sayısını kontrol edin.")
    if not has_bill:
        improve.append("Ortalama daire faturasını girin.")
    if not roof_known:
        improve.append("Çatı yapısını seçin.")
    if not top_down and not obstacles_marked:
        improve.append("Çatı fotoğrafı yükleyin.")
    reasons.append("Komşu bina gölgesi henüz hesaba katılmıyor; yalnızca arazi gölgesi hesaplandı.")
    return _LEVELS[max(level, 0)], reasons, improve


def run_scenario(base_result, change):
    """Re-run the calculation with a change (e.g. fewer obstacles); return new numbers + delta."""
    raise NotImplementedError
