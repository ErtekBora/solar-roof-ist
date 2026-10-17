"""Area, capacity (kWp), suitability score, annual kWh, CO2, savings (TL), confidence and scenarios."""
import math

from core.config import (
    CO2_FACTOR_T_PER_MWH,
    EDGE_FACTOR,
    M2_PER_KWP_FLAT,
    M2_PER_KWP_PITCHED,
    USABLE_RATIO_NO_PHOTO,
)

SAVINGS_LOW_FACTOR = 0.5
SAVINGS_HIGH_FACTOR = 1.0


def _is_number(x):
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def _check_monthly(name, values):
    if values is None:
        return None
    values = list(values)
    if len(values) != 12 or not all(_is_number(v) and v >= 0 for v in values):
        raise ValueError(f"{name} must have exactly 12 non-negative numbers")
    return values


def _confidence(roof_type, image_view):
    """Demo-stage confidence: photo only (SPEC section 6)."""
    if image_view == "top-down":
        level = "Orta"
        reasons = ["Fotoğraf kuşbakışı çekilmiş; çatı yüzeyi güvenilir biçimde ölçülebildi."]
    elif image_view == "oblique":
        level = "Düşük"
        reasons = ["Fotoğraf yandan/eğik açıdan çekilmiş; çatı ve engel alanları daha az güvenilir."]
    else:
        level = "Düşük"
        reasons = ["Fotoğraf yüklenmedi; kullanılabilir alan ortalama orana göre tahmin edildi."]
    reasons.append("Bina yüksekliği ve komşu bina verisi henüz kullanılmıyor; yalnızca arazi gölgesi hesaplandı.")
    if roof_type == "pitched":
        reasons.append("Eğimli çatıda gerçek yön bilinmiyor; paneller en uygun eğim ve yönde varsayıldı.")
    return level, reasons


def compute_capacity(
    roof_area_m2,
    obstacle_ratio,
    roof_type,
    yield_ideal,
    yield_shaded,
    tariff,
    image_view=None,
    monthly_ideal=None,
    monthly_shaded=None,
):
    """Return all result numbers described in SPEC section 6 (Turkish keys, SPEC section 9).

    obstacle_ratio None means no photo. monthly_* are PVGIS per-kWp monthly values; they are
    scaled here to roof totals in kWh.
    """
    if not _is_number(roof_area_m2) or roof_area_m2 <= 0:
        raise ValueError("roof_area_m2 must be > 0")
    if obstacle_ratio is not None and (not _is_number(obstacle_ratio) or not 0 <= obstacle_ratio < 1):
        raise ValueError("obstacle_ratio must be None or 0 <= ratio < 1")
    if roof_type not in ("flat", "pitched"):
        raise ValueError("roof_type must be 'flat' or 'pitched'")
    if not _is_number(yield_ideal) or yield_ideal <= 0:
        raise ValueError("yield_ideal must be > 0")
    if not _is_number(yield_shaded) or yield_shaded < 0:
        raise ValueError("yield_shaded must be >= 0")
    if not _is_number(tariff) or tariff <= 0:
        raise ValueError("tariff must be > 0")
    if image_view not in (None, "top-down", "oblique"):
        raise ValueError("image_view must be None, 'top-down' or 'oblique'")
    monthly_ideal = _check_monthly("monthly_ideal", monthly_ideal)
    monthly_shaded = _check_monthly("monthly_shaded", monthly_shaded)

    if obstacle_ratio is None:
        usable = roof_area_m2 * USABLE_RATIO_NO_PHOTO
        obstacle_area = None
    else:
        usable = roof_area_m2 * EDGE_FACTOR * (1 - obstacle_ratio)
        obstacle_area = round(roof_area_m2 * obstacle_ratio, 2)

    kwp = usable / (M2_PER_KWP_FLAT if roof_type == "flat" else M2_PER_KWP_PITCHED)
    suitability = min(100.0, max(0.0, yield_shaded / yield_ideal * 100))
    kwh = kwp * yield_shaded
    level, reasons = _confidence(roof_type, image_view)

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
        "aylik_uretim_kwh": None if monthly_shaded is None else [round(kwp * m, 1) for m in monthly_shaded],
        "aylik_ideal_kwh": None if monthly_ideal is None else [round(kwp * m, 1) for m in monthly_ideal],
        "yillik_tasarruf_tl_aralik": [
            round(kwh * tariff * SAVINGS_LOW_FACTOR, 2),
            round(kwh * tariff * SAVINGS_HIGH_FACTOR, 2),
        ],
        "co2_ton_yil": round(kwh * CO2_FACTOR_T_PER_MWH / 1000, 2),
        "guven_seviyesi": level,
        "guven_nedenleri": reasons,
    }


def run_scenario(base_result, change):
    """Re-run the calculation with a change (e.g. fewer obstacles); return new numbers + delta."""
    raise NotImplementedError
