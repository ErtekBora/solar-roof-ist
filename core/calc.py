"""Area, capacity (kWp), suitability score, annual kWh, CO2, savings (TL), confidence and scenarios."""
import math

from core.config import (
    CO2_FACTOR_T_PER_MWH,
    EDGE_FACTOR,
    M2_PER_KWP_FLAT,
    M2_PER_KWP_PITCHED,
    TARIFF_HIGH_TL_PER_KWH_EXCL_TAX,
    TARIFF_LOW_TL_PER_KWH_EXCL_TAX,
    TARIFF_SOURCE,
    TAX_FACTOR,
    TIER_LIMIT_KWH_PER_DAY,
    USABLE_RATIO_NO_PHOTO,
)

SAVINGS_LOW_FACTOR = 0.5  # share of production used in the household (lower bound)
SAVINGS_HIGH_FACTOR = 1.0
DAYS_PER_MONTH = 30

# Two-tier residential tariff, tax included (TAX_FACTOR is applied exactly once, here)
LOW = TARIFF_LOW_TL_PER_KWH_EXCL_TAX * TAX_FACTOR
HIGH = TARIFF_HIGH_TL_PER_KWH_EXCL_TAX * TAX_FACTOR
LIMIT = TIER_LIMIT_KWH_PER_DAY * DAYS_PER_MONTH  # kWh/month billed at LOW; only the excess is billed at HIGH
BILL_AT_LIMIT = LIMIT * LOW

_COVERAGE_FULL_EPS = 1e-6
_SELF_USED_MIN = 0.5


def _is_number(x):
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def _check_monthly(name, values):
    if values is None:
        return None
    values = list(values)
    if len(values) != 12 or not all(_is_number(v) and v >= 0 for v in values):
        raise ValueError(f"{name} must have exactly 12 non-negative numbers")
    return values


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


def savings_from_bill(monthly_bill_tl, monthly_production_kwh):
    """Bill-based savings from 12 roof-scaled monthly production values (kWh).

    Returns the annual TL range (self-use share 0.5 .. 1.0), the bill coverage range in % (capped at 100),
    the consumption, self_used_share (production used in one household at share 1.0, 0..1) and the shared-roof flag.
    """
    c = kwh_from_bill(monthly_bill_tl)
    low = sum(monthly_savings(c, p, SAVINGS_LOW_FACTOR) for p in monthly_production_kwh)
    high = sum(monthly_savings(c, p, SAVINGS_HIGH_FACTOR) for p in monthly_production_kwh)
    annual_bill = 12 * monthly_bill_tl
    coverage = [min(1.0, v / annual_bill) * 100 for v in (low, high)]
    annual_kwh = sum(monthly_production_kwh)
    used = sum(min(p * SAVINGS_HIGH_FACTOR, c) for p in monthly_production_kwh)
    self_used = used / annual_kwh if annual_kwh > 0 else 0.0
    full = coverage[1] >= 100 * (1 - _COVERAGE_FULL_EPS)
    return {
        "consumption_kwh": c,
        "range": [low, high],
        "coverage_pct": coverage,
        "self_used_share": self_used,
        "shared_roof": bool(annual_kwh > 0 and (full or self_used < _SELF_USED_MIN)),
    }


def fallback_savings(annual_kwh):
    """Wide range when consumption is unknown: kWh x [0.5 x LOW, 1.0 x HIGH] (tax is already in LOW/HIGH)."""
    return [annual_kwh * SAVINGS_LOW_FACTOR * LOW, annual_kwh * SAVINGS_HIGH_FACTOR * HIGH]


# -------------------------------------------------------------- confidence

def _confidence(roof_type, image_view, has_polygon, roof_type_known):
    """Capacity confidence from the area source, the photo and the roof type (SPEC section 6, Prototype P1).

    Returns (level, reasons, improvements). Only inputs that change a shown result count.
    TODO(P2): building height joins the confidence once neighbour shading uses it; Yüksek is unreachable until then.
    """
    top_down = image_view == "top-down"
    level = "Orta" if (top_down or has_polygon) else "Düşük"
    reasons, improve = [], []
    reasons.append(
        "Çatı alanı bina izinden ölçüldü (kenar boşluğu düşüldü)."
        if has_polygon
        else "Çatı alanı elle girildi; bina izi kullanılmadı."
    )
    if top_down:
        reasons.append("Fotoğraf kuşbakışı çekilmiş; engel oranı güvenilir biçimde ölçülebildi.")
    elif image_view == "oblique":
        reasons.append("Fotoğraf yandan/eğik açıdan çekilmiş; engel alanı yaklaşık, güveni artırmıyor.")
    else:
        reasons.append("Fotoğraf yok; kullanılabilir alan ortalama orana göre tahmin edildi (ortalama varsayım).")
    if not top_down:
        improve.append("Kendi çatınızın tepeden (kuşbakışı) fotoğrafını yükleyin.")
    if not roof_type_known:
        level = "Düşük"  # one level down, never below Düşük
        reasons.append("Çatı tipi bilinmiyor; düz çatı varsayıldı.")
        improve.append("Çatı tipini seçin.")
    elif roof_type == "pitched":
        reasons.append("Eğimli çatıda gerçek yön bilinmiyor; paneller en uygun eğim ve yönde varsayıldı.")
    if not has_polygon:
        improve.append("Haritada kapsama alanı içindeki bir binayı seçin.")
    reasons.append("Komşu bina gölgesi henüz hesaba katılmıyor; yalnızca arazi gölgesi hesaplandı.")
    return level, reasons, improve


def compute_capacity(
    roof_area_m2,
    obstacle_ratio,
    roof_type,
    yield_ideal,
    yield_shaded,
    monthly_bill_tl=None,
    image_view=None,
    monthly_ideal=None,
    monthly_shaded=None,
    usable_base_area_m2=None,
    roof_type_known=True,
):
    """Return all result numbers described in SPEC section 6 (Turkish keys, SPEC section 9).

    obstacle_ratio None means no usable photo. monthly_* are PVGIS per-kWp monthly values (E_m); they are
    scaled here by kWp to roof totals in kWh, and those roof-scaled SHADED values feed the savings.
    usable_base_area_m2 is the footprint area after the geometric setback; EDGE_FACTOR is not applied then.
    monthly_bill_tl None, NaN or <= 0 means no bill: a wide fallback savings range is returned.
    """
    if not _is_number(roof_area_m2) or roof_area_m2 <= 0:
        raise ValueError("roof_area_m2 must be > 0")
    if usable_base_area_m2 is not None and (not _is_number(usable_base_area_m2) or usable_base_area_m2 < 0):
        raise ValueError("usable_base_area_m2 must be None or >= 0")
    if obstacle_ratio is not None and (not _is_number(obstacle_ratio) or not 0 <= obstacle_ratio < 1):
        raise ValueError("obstacle_ratio must be None or 0 <= ratio < 1")
    if roof_type not in ("flat", "pitched"):
        raise ValueError("roof_type must be 'flat' or 'pitched'")
    if not _is_number(yield_ideal) or yield_ideal <= 0:
        raise ValueError("yield_ideal must be > 0")
    if not _is_number(yield_shaded) or yield_shaded < 0:
        raise ValueError("yield_shaded must be >= 0")
    if image_view not in (None, "top-down", "oblique"):
        raise ValueError("image_view must be None, 'top-down' or 'oblique'")
    monthly_ideal = _check_monthly("monthly_ideal", monthly_ideal)
    monthly_shaded = _check_monthly("monthly_shaded", monthly_shaded)

    has_polygon = usable_base_area_m2 is not None
    if has_polygon:
        if obstacle_ratio is None:
            usable = usable_base_area_m2 * (USABLE_RATIO_NO_PHOTO / EDGE_FACTOR)
        else:
            usable = usable_base_area_m2 * (1 - obstacle_ratio)
    elif obstacle_ratio is None:
        usable = roof_area_m2 * USABLE_RATIO_NO_PHOTO
    else:
        usable = roof_area_m2 * EDGE_FACTOR * (1 - obstacle_ratio)
    obstacle_area = None if obstacle_ratio is None else round(roof_area_m2 * obstacle_ratio, 2)

    kwp = usable / (M2_PER_KWP_FLAT if roof_type == "flat" else M2_PER_KWP_PITCHED)
    suitability = min(100.0, max(0.0, yield_shaded / yield_ideal * 100))
    kwh = kwp * yield_shaded
    level, reasons, improve = _confidence(roof_type, image_view, has_polygon, roof_type_known)

    # roof-scaled shaded monthly production (kWp x E_m); uniform split only when no monthly data was given
    production = [kwp * m for m in monthly_shaded] if monthly_shaded is not None else [kwh / 12] * 12
    bill = monthly_bill_tl if _is_number(monthly_bill_tl) and monthly_bill_tl > 0 else None
    if bill is not None:
        sv = savings_from_bill(bill, production)
        savings_range, method = sv["range"], "fatura"
        bill_fields = {
            "fatura_tl_ay": bill,
            "tuketim_kwh_ay": round(sv["consumption_kwh"], 1),
            "fatura_karsilama_yuzde_aralik": [round(v, 2) for v in sv["coverage_pct"]],
            "kendi_kullanim_orani_yuzde": round(sv["self_used_share"] * 100, 2),
            "tek_daire_uyarisi": sv["shared_roof"],
        }
    else:
        savings_range, method = fallback_savings(kwh), "genis_tahmin"
        bill_fields = {
            "fatura_tl_ay": None,
            "tuketim_kwh_ay": None,
            "fatura_karsilama_yuzde_aralik": None,
            "kendi_kullanim_orani_yuzde": None,
            "tek_daire_uyarisi": False,
        }

    return {
        "cati_tipi": roof_type,
        "goruntu_turu": image_view,
        "toplam_cati_alani_m2": roof_area_m2,
        "engel_orani": obstacle_ratio,
        "engel_alani_m2": obstacle_area,
        "kullanilabilir_alan_m2": round(usable, 2),
        "kurulabilir_guc_kwp": round(kwp, 3),
        "ideal_uretim_kwh_kwp": yield_ideal,
        "golgeli_uretim_kwh_kwp": yield_shaded,
        "golge_kaybi_yuzde": round(max(0.0, 100.0 - suitability), 2),
        "uygunluk_puani_yuzde": round(suitability, 2),
        "yillik_uretim_kwh": round(kwh, 1),
        "aylik_uretim_kwh": None if monthly_shaded is None else [round(v, 1) for v in production],
        "aylik_ideal_kwh": None if monthly_ideal is None else [round(kwp * m, 1) for m in monthly_ideal],
        "yillik_tasarruf_tl_aralik": [round(v, 2) for v in savings_range],
        "tasarruf_yontemi": method,
        "tarife_dusuk_tl_kwh": round(LOW, 2),
        "tarife_yuksek_tl_kwh": round(HIGH, 2),
        "tarife_kaynagi": TARIFF_SOURCE,
        **bill_fields,
        "co2_ton_yil": round(kwh * CO2_FACTOR_T_PER_MWH / 1000, 2),
        "guven_seviyesi": level,
        "guven_nedenleri": reasons,
        "guven_iyilestirme": improve,
    }


def run_scenario(base_result, change):
    """Re-run the calculation with a change (e.g. fewer obstacles); return new numbers + delta."""
    raise NotImplementedError
