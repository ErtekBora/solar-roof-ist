"""Building mode: one rooftop PV system for an apartment building, production shared EQUALLY among its flats.

Pure functions (no network, no Streamlit): flat-count estimate, roof segments (flat + two pitched faces),
equal sharing of the production, savings and payback. PVGIS results are passed in by the caller.
"""
import math

from core import calc, cost
from core.config import (
    CO2_FACTOR_T_PER_MWH,
    COST_SOURCE,
    DEFAULT_ROOF_PITCH_DEG,
    EDGE_FACTOR,
    FACE_MIN_YIELD_RATIO,
    FLAT_GROSS_AREA_M2,
    LEVEL_HEIGHT_M,
    M2_PER_KWP_FLAT,
    M2_PER_KWP_PITCHED,
    PANEL_KWP,
    PAYBACK_MAX_YEARS,
    SETBACK_M,
    TARIFF_SOURCE,
    USABLE_RATIO_NO_PHOTO,
)
from core.cost import clamp_flats
from core.formatting import round_half_up, tr_num
from core.solar import normalize_aspect

STRUCTURES = ("flat", "pitched", "mixed", "unknown")
STRUCTURE_LABELS_TR = {
    "flat": "Tamamen düz",
    "pitched": "Tamamen eğimli",
    "mixed": "Karma (teras + eğimli)",
    "unknown": "Bilmiyorum",
}
MIN_SAVINGS_TL_YR = 1.0  # a savings bound below this gives no payback bound
PITCHED_TYPES = ("hip", "gable")
PITCHED_TYPE_LABELS_TR = {"hip": "Kırma (4 yüz)", "gable": "Beşik (2 yüz)"}
FLAT_SEGMENT = "Düz teras"
# where the usable capacity comes from (priority: user drawing > photo obstacle ratio > average share)
CAPACITY_SOURCE_LABELS_TR = {
    "geometri": "kullanıcı çizimi (panel yerleşimi)",
    "fotograf": "fotoğraftaki engel oranı",
    "ortalama": "ortalama kullanılabilir alan oranı",
    "engel_yok": "kullanıcı onayı: çatıda engel yok",
}
HIGH_SHARE_NOTE = (
    "Daire payı (≈{kwh} kWh/ay) tipik bir hane tüketiminin üstünde olabilir; fazla üretim değerlendirilemeyebilir. "
    "Netleştirmek için ortalama daire faturasını girin."
)

ASSUMPTION_CONSTANT_CONSUMPTION = (
    "Girilen fatura yılın her ayı için ortalama tüketim kabul edilmiştir; mevsimsel farklar (klima, ısıtıcı) "
    "dikkate alınmamıştır. Yaz aylarındaki fazla üretim 0 TL sayıldığı için tasarruf ihtiyatlı hesaplanmıştır."
)
LEGAL_NOTE = (
    "Üretimin dairelere eşit paylaştırılması mevzuata, kat malikleri kararına ve dağıtım şirketinin onayına tabidir. "
    "En yaygın uygulama sistemin ortak alan sayacına bağlanmasıdır; bu durumda ortak alan tüketimini aşan üretim "
    "değerlendirilemeyebilir ve geri ödeme uzar. "
    "Mevzuat üretim ve tüketim tesisinin aynı kişiye ait olmasını şart koşar; bu nedenle üretim dairelerin bireysel "
    "faturalarından doğrudan düşülemez, sistem ortak alan aboneliğine bağlanır ve tasarrufun aidattan eşit düşülmesi "
    "önerilir."
)
NETTING_ASSUMPTION = "Aylık mahsuplaşma varsayımıyla hesaplanmıştır."
NETTING_NOTE = (
    "Hesap aylık mahsuplaşma varsayar. EPDK'nın 30 Nisan 2026 kararıyla lisanssız üretimde mahsuplaşma saatlik "
    "bazda yapılıyor; mesken ve apartman ortak alan uygulamasının ayrıntıları dağıtım şirketiyle netleşmelidir. "
    "Saatlik mahsuplaşmada gündüz tüketimi düşük olan sistemlerde tasarruf daha düşük çıkabilir."
)
USABLE_RATIO_NOTE = (
    "Ember brüt kullanılabilir oranı %27,6 ÷ 0,85 (kenar boşluğu sonrası) = %32,5; küçük binalarda kenar kaybı "
    "oransal olarak büyük olduğundan sonuç ihtiyatlıdır."
)
SEASONAL_NOTE = "Yıllık ortalamadır; kışın daha az, yazın daha fazla indirim olur ve indirim faturayı aşamaz."
PHYSICAL_SHARING_NOTE = (
    "Fiziksel paylaşımda (her daireye eşit kWh) tüketimi yüksek daireler kademe farkı nedeniyle biraz daha fazla "
    "kazanır. Herkes kurulumu eşit ödediği için tasarrufun yönetim tarafından aidattan eşit düşülmesi önerilir; bu, "
    "sistemin ortak sayaca bağlandığı yaygın uygulamayla da uyumludur."
)
FACE_UNAVAILABLE_MSG = "Bir çatı yüzü için güneş verisi alınamadı; sonuç eksik olabilir"

_COMPASS_TR = ("Kuzey", "Kuzeydoğu", "Doğu", "Güneydoğu", "Güney", "Güneybatı", "Batı", "Kuzeybatı")
_DATIVE_TR = {
    "Kuzey": "Kuzeye", "Kuzeydoğu": "Kuzeydoğuya", "Doğu": "Doğuya", "Güneydoğu": "Güneydoğuya",
    "Güney": "Güneye", "Güneybatı": "Güneybatıya", "Batı": "Batıya", "Kuzeybatı": "Kuzeybatıya",
}


def _is_number(x):
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def _round_half_up_int(x):
    return int(math.floor(x + 0.5))


# ------------------------------------------------------------------ flat count

def estimate_flat_count(num_floors, height_m, footprint_area_m2):
    """Estimated number of flats, always >= 1.

    floors = num_floors if known, else max(1, round(height_m / 3)); flats = max(1, round(floors x footprint / 120)).
    """
    if _is_number(num_floors) and num_floors > 0:
        floors = num_floors
    elif _is_number(height_m) and height_m > 0:
        floors = max(1, _round_half_up_int(height_m / LEVEL_HEIGHT_M))
    else:
        floors = 1
    if not _is_number(footprint_area_m2) or footprint_area_m2 <= 0:
        return 1
    return max(1, _round_half_up_int(floors * footprint_area_m2 / FLAT_GROSS_AREA_M2))


# --------------------------------------------------------------------- geometry

def to_pvgis_aspect(north_azimuth_deg):
    """North-based clockwise azimuth -> PVGIS aspect (0 = South, 90 = West, -90 = East, 180 = North)."""
    aspect = (north_azimuth_deg - 180) % 360
    return aspect - 360 if aspect > 180 else aspect


def compass_tr(azimuth_deg):
    """Turkish 8-point compass name of a North-based azimuth."""
    return _COMPASS_TR[int(((azimuth_deg % 360) + 22.5) // 45) % 8]


def face_azimuths(ridge_azimuth_deg):
    """The two pitched-face azimuths (North-based): perpendicular to the ridge, opposite to each other."""
    first = (ridge_azimuth_deg + 90) % 360
    return [first, (first + 180) % 360]


def face_specs(ridge_azimuth_deg, pitched_type="gable"):
    """[{name, azimuth_deg, aspect, weight}] for the pitched faces; `aspect` is what PVGIS PVcalc needs (1 degree).

    Gable: the two faces perpendicular to the ridge, 1/2 each. Hip: those two plus the two hip ends facing along the
    ridge, 1/4 each (the square assumption used when there is no footprint).
    """
    if pitched_type not in PITCHED_TYPES:
        raise ValueError(f"pitched_type must be one of {PITCHED_TYPES}")
    azimuths = face_azimuths(ridge_azimuth_deg)
    if pitched_type == "hip":
        azimuths += [ridge_azimuth_deg % 360, (ridge_azimuth_deg + 180) % 360]
    return [
        {"name": compass_tr(a), "azimuth_deg": a, "aspect": normalize_aspect(to_pvgis_aspect(a)),
         "weight": 1 / len(azimuths)}
        for a in azimuths
    ]


# ---------------------------------------------------------------------- segments

def roof_scenarios(structure, flat_share_pct=50):
    """[(label, flat_share 0..1)]: one scenario for a known structure, the two extremes for 'unknown'."""
    if structure == "flat":
        return [(STRUCTURE_LABELS_TR["flat"], 1.0)]
    if structure == "pitched":
        return [(STRUCTURE_LABELS_TR["pitched"], 0.0)]
    if structure == "mixed":
        pct = flat_share_pct if _is_number(flat_share_pct) else 50
        return [(STRUCTURE_LABELS_TR["mixed"], min(100.0, max(0.0, pct)) / 100)]
    return [(STRUCTURE_LABELS_TR["flat"], 1.0), (STRUCTURE_LABELS_TR["pitched"], 0.0)]


def face_segment_name(face_name):
    return f"{face_name} yüzü"


def numeric_panel_count(kwp):
    """Whole panels drawn for a numeric kWp: floor(kWp / PANEL_KWP)."""
    return int(math.floor(kwp / PANEL_KWP + 1e-9)) if _is_number(kwp) and kwp > 0 else 0


def compute_segments(usable_plan_m2, flat_share, flat_solar, faces, geometry=None):
    """Split the usable PLAN area into a flat segment and the pitched faces and compute kWp and monthly kWh.

    flat_solar: result of core.solar.get_solar_yield (the SHADED run is the flat-optimal reference).
    faces: [{name, azimuth_deg, aspect, solar, weight}] with solar = a PVcalc result or None (call failed) and
    weight = the face's share of the pitched plan area (proportional to its intersected plan area; equal if missing).
    Numeric: flat kWp = plan / 6.5; face surface = plan / cos(pitch), kWp = surface / 5.5.
    geometry = {segment name: {"plan_m2", "panels"}} from the panel layout (user-drawn obstacles): kWp = panels x
    PANEL_KWP and the plan areas are the geometric ones; the usable share is not used.
    A face below 80 % of the flat segment's shaded specific yield is dropped (0 kWp, 0 kWh, 0 panels).
    """
    share = min(1.0, max(0.0, flat_share))
    flat_ref = flat_solar["shaded"]
    empty = {"plan_m2": 0.0, "panels": 0}
    if geometry is None:
        usable = max(0.0, usable_plan_m2 or 0.0)
        flat_plan = usable * share
        flat_kwp = flat_plan / M2_PER_KWP_FLAT
        flat_panels = numeric_panel_count(flat_kwp)
    else:
        g = geometry.get(FLAT_SEGMENT, empty)
        flat_plan, flat_panels = g["plan_m2"], int(g["panels"])
        flat_kwp = flat_panels * PANEL_KWP
        usable = flat_plan + sum(geometry.get(face_segment_name(f["name"]), empty)["plan_m2"] for f in faces)
    flat_seg = {
        "ad": FLAT_SEGMENT, "tur": "flat", "durum": "kullanildi" if flat_plan > 0 else "kullanilmadi",
        "plan_m2": flat_plan, "kwp": flat_kwp, "e_y": flat_ref["e_y"], "panel_sayisi": flat_panels,
        "aylik_kwh": [flat_kwp * m for m in flat_ref["e_m"]],
    }
    face_segs, warnings = [], []
    for face in faces:
        name = face_segment_name(face["name"])
        if geometry is None:
            face_plan = usable * (1 - share) * face.get("weight", 1 / len(faces))
        else:
            face_plan = geometry.get(name, empty)["plan_m2"]
        surface = face_plan / math.cos(math.radians(DEFAULT_ROOF_PITCH_DEG))
        seg = {
            "ad": name, "tur": "face", "yon": face["name"], "azimut_derece": face["azimuth_deg"],
            "aspect": face["aspect"], "plan_m2": face_plan, "yuzey_m2": surface, "durum": "kullanildi",
            "kwp": 0.0, "e_y": None, "verim_orani": None, "aylik_kwh": [0.0] * 12, "panel_sayisi": 0,
        }
        solar = face.get("solar")
        if face_plan <= 0:
            seg["durum"] = "kullanilmadi"
        elif solar is None:
            seg["durum"] = "alinamadi"
            warnings.append(FACE_UNAVAILABLE_MSG)
        else:
            ratio = solar["e_y"] / flat_ref["e_y"] if flat_ref["e_y"] > 0 else 0.0
            seg["e_y"], seg["verim_orani"] = solar["e_y"], ratio
            if ratio < FACE_MIN_YIELD_RATIO:
                seg["durum"] = "elendi"
                warnings.append(f"{_DATIVE_TR[face['name']]} bakan çatı yüzü verimsiz olduğu için hesaba katılmadı")
            else:
                if geometry is None:
                    seg["kwp"] = surface / M2_PER_KWP_PITCHED
                    seg["panel_sayisi"] = numeric_panel_count(seg["kwp"])
                else:
                    seg["panel_sayisi"] = int(geometry.get(name, empty)["panels"])
                    seg["kwp"] = seg["panel_sayisi"] * PANEL_KWP
                seg["aylik_kwh"] = [seg["kwp"] * m for m in solar["e_m"]]
        face_segs.append(seg)
    segments = [flat_seg, *face_segs]
    monthly = [sum(s["aylik_kwh"][i] for s in segments) for i in range(12)]
    kwp = sum(s["kwp"] for s in segments)
    return {
        "duz_pay": share, "kullanilabilir_plan_m2": usable, "segmentler": segments, "kwp": kwp,
        "aylik_kwh": monthly, "kwh": sum(monthly),
        "panel_sayisi": sum(s["panel_sayisi"] for s in segments if s["durum"] == "kullanildi"),
        "uyarilar": list(dict.fromkeys(warnings)),
        "alinamayan_yuz": sum(1 for s in face_segs if s["durum"] == "alinamadi"),
        "elenen_yuz": sum(1 for s in face_segs if s["durum"] == "elendi"),
    }


def unusable_faces(flat_solar, faces):
    """Segment names of the faces that get no panels: below the 80 % rule or without PVGIS data."""
    ref = flat_solar["shaded"]["e_y"]
    return {
        face_segment_name(f["name"]) for f in faces
        if f.get("solar") is None or ref <= 0 or f["solar"]["e_y"] / ref < FACE_MIN_YIELD_RATIO
    }


# ----------------------------------------------------------------------- sharing

def allocate_equal_share(monthly_kwh, n_flats, bill_tl=None):
    """Share the production: equal kWh per flat AND an equal TL discount per flat (mesken tariff).

    kWh: each month the production is split equally among N flats (`daire_aylik_kwh`).
    TL (bill given): monthly netting, share 1.0 only: building monthly savings = N x monthly_savings(C_avg,
    production_m / N, 1.0), surplus worth 0; the fixed per-flat discount is the yearly total divided by N, because
    every flat pays the installation equally. Without a bill the wide fallback range is divided per flat. No
    individual bills are used. The range of the final result comes from roof scenarios and cost bands only.
    """
    n = clamp_flats(n_flats)
    per_flat = [max(0.0, m) / n for m in monthly_kwh]
    annual_per_flat = sum(per_flat)
    base = {"daire_sayisi": n, "daire_uretim_kwh": annual_per_flat, "daire_aylik_kwh": per_flat}
    if _is_number(bill_tl) and bill_tl > 0:
        c = calc.kwh_from_bill(bill_tl)
        annual_bill = 12 * bill_tl
        # monthly netting (aylik mahsuplasma): share 1.0, the month's production offsets that month's consumption
        building_monthly = [n * calc.monthly_savings(c, p, calc.SAVINGS_HIGH_FACTOR) for p in per_flat]
        building_total = sum(building_monthly)
        discount_monthly = [v / n for v in building_monthly]
        discount_year = min(building_total / n, annual_bill)
        coverage = min(1.0, discount_year / annual_bill) * 100
        return {
            **base, "yontem": "fatura", "tasarruf_tl": [discount_year, discount_year],
            "indirim_yillik_tl": [discount_year, discount_year], "indirim_aylik_tl": [discount_monthly, discount_monthly],
            "indirim_ay_ortalama_tl": [discount_year / 12] * 2, "bina_toplam_tasarruf_tl": [building_total] * 2,
            "karsilama_yuzde": [coverage, coverage],
            "kalan_fatura_tl_ay": [(annual_bill - discount_year) / 12] * 2,
            "tuketim_kwh_ay": c, "varsayim": ASSUMPTION_CONSTANT_CONSUMPTION,
        }
    discount_year = calc.fallback_savings(annual_per_flat)
    return {
        **base, "yontem": "genis_tahmin", "tasarruf_tl": discount_year, "indirim_yillik_tl": discount_year,
        "indirim_aylik_tl": None, "indirim_ay_ortalama_tl": [v / 12 for v in discount_year],
        "bina_toplam_tasarruf_tl": [v * n for v in discount_year], "karsilama_yuzde": None,
        "kalan_fatura_tl_ay": None, "tuketim_kwh_ay": None, "varsayim": None,
    }


def payback_range(cost_low, cost_high, sav_low, sav_high):
    """(low_years, high_years): low = cost_low / sav_high, high = cost_high / sav_low; None where not computable.

    A savings bound below 1 TL/yr (or a non-positive cost) gives None for that bound, and so does a payback above
    PAYBACK_MAX_YEARS. Never raises.
    """
    def years(cost_tl, savings_tl):
        if not (_is_number(cost_tl) and _is_number(savings_tl)) or cost_tl <= 0 or savings_tl < MIN_SAVINGS_TL_YR:
            return None
        value = cost_tl / savings_tl
        return value if value <= PAYBACK_MAX_YEARS else None

    return years(cost_low, sav_high), years(cost_high, sav_low)


# ------------------------------------------------------------------------ result

def _span(values):
    return [min(values), max(values)]


def building_specific_yield(kwh, kwp):
    """Annual kWh per installed kWp of the whole system (display and explainer only); None without installed kWp."""
    return kwh / kwp if _is_number(kwp) and kwp > 0 else None


def _yield_bounds(scenarios):
    """[lo, hi] of the scenarios' building yields; a bound without installed kWp is None (both None: no card)."""
    values = [s["ozgul_verim_bina"] for s in scenarios if s["ozgul_verim_bina"] is not None]
    return [min(values), max(values)] if values else [None, None]


def _scenario(label, seg, n, bill):
    kwp_cost = cost.estimate_install_cost(seg["kwp"])
    share = allocate_equal_share(seg["aylik_kwh"], n, bill)
    return {
        "etiket": label, "duz_pay": seg["duz_pay"], "kwp": seg["kwp"], "kwh": seg["kwh"],
        "ozgul_verim_bina": building_specific_yield(seg["kwh"], seg["kwp"]),
        "aylik_kwh": seg["aylik_kwh"], "segmentler": seg["segmentler"], "uyarilar": seg["uyarilar"],
        "alinamayan_yuz": seg["alinamayan_yuz"], "elenen_yuz": seg["elenen_yuz"],
        "panel_sayisi": seg["panel_sayisi"], "kullanilabilir_plan_m2": seg["kullanilabilir_plan_m2"],
        "maliyet_tl": [kwp_cost[0], kwp_cost[1]], "maliyet_bandi": kwp_cost[2], "maliyet_bayraklari": kwp_cost[3],
        "paylasim": share,
    }


def analyze_building(
    *, roof_area_m2, usable_base_area_m2, obstacle_ratio, image_view, structure, flat_share_pct, n_flats,
    n_flats_estimated, bill_tl, flat_solar, faces, pitched_type="gable", geometry=None, capacity_source=None,
    drawn=None,
):
    """Full building-mode result (Turkish keys): segments per scenario, ranges, cost, sharing, payback, confidence.

    geometry: None (numeric capacity) or one {segment: {"plan_m2", "panels"}} per scenario of `roof_scenarios`
    (user-drawn obstacles: kWp = panels x PANEL_KWP). capacity_source: "geometri" | "fotograf" | "ortalama" |
    "engel_yok" (derived when None). drawn: {"count", "area_m2", "terrace", "marked"} from the roof drawing.
    """
    if not _is_number(roof_area_m2) or roof_area_m2 <= 0:
        raise ValueError("roof_area_m2 must be > 0")
    if obstacle_ratio is not None and (not _is_number(obstacle_ratio) or not 0 <= obstacle_ratio < 1):
        raise ValueError("obstacle_ratio must be None or 0 <= ratio < 1")
    if structure not in STRUCTURES:
        raise ValueError(f"structure must be one of {STRUCTURES}")
    n = clamp_flats(n_flats)
    bill = bill_tl if _is_number(bill_tl) and bill_tl > 0 else None
    has_polygon = usable_base_area_m2 is not None
    usable = calc.usable_plan_area(roof_area_m2, obstacle_ratio, usable_base_area_m2)
    ideal, shaded = flat_solar["ideal"], flat_solar["shaded"]

    if pitched_type not in PITCHED_TYPES:
        raise ValueError(f"pitched_type must be one of {PITCHED_TYPES}")
    if capacity_source is None:
        capacity_source = "geometri" if geometry is not None else ("fotograf" if obstacle_ratio is not None else "ortalama")
    drawn = drawn or {}
    scenarios = []
    for i, (label, share) in enumerate(roof_scenarios(structure, flat_share_pct)):
        geo = geometry[i] if geometry is not None else None
        scenarios.append(_scenario(label, compute_segments(usable, share, flat_solar, faces, geo), n, bill))
    if geometry is not None:
        usable = scenarios[0]["kullanilabilir_plan_m2"]

    total_cost = cost.combine_cost_scenarios([s["maliyet_tl"] for s in scenarios])
    flat_cost = cost.per_flat_cost(total_cost, n)
    savings = _span([v for s in scenarios for v in s["paylasim"]["tasarruf_tl"]])
    payback = payback_range(flat_cost[0], flat_cost[1], savings[0], savings[1])
    kwp_range = _span([s["kwp"] for s in scenarios])
    kwh_range = _span([s["kwh"] for s in scenarios])
    with_bill = bill is not None
    pitched_used = any(s["duz_pay"] < 1 for s in scenarios)
    unavailable = max((s["alinamayan_yuz"] for s in scenarios), default=0) if pitched_used else 0
    level, reasons, improve = calc.building_confidence(
        image_view, has_polygon, structure != "unknown", unavailable, with_bill, n_flats_estimated,
        obstacles_marked=bool(drawn.get("marked")),
    )
    share_kwh = _span([s["kwh"] / n / 12 for s in scenarios])
    base_area = usable_base_area_m2 if has_polygon else roof_area_m2
    warnings = list(dict.fromkeys(w for s in scenarios for w in s["uyarilar"]))
    suitability = min(100.0, max(0.0, shaded["e_y"] / ideal["e_y"] * 100))
    sharing = [s["paylasim"] for s in scenarios]
    result = {
        "cati_yapisi": structure,
        "duz_pay_yuzde": round(scenarios[0]["duz_pay"] * 100) if structure == "mixed" else None,
        "goruntu_turu": image_view,
        "toplam_cati_alani_m2": roof_area_m2,
        "engel_orani": obstacle_ratio,
        "engel_alani_m2": None if obstacle_ratio is None else round(roof_area_m2 * obstacle_ratio, 2),
        "kullanilabilir_alan_m2": usable,  # unrounded: display rounds once (no 83,9 vs 83,8)
        "daire_sayisi": n,
        "daire_sayisi_tahmini": bool(n_flats_estimated),
        "senaryolar": scenarios,
        "kurulabilir_guc_kwp_aralik": kwp_range,
        "yillik_uretim_kwh_aralik": kwh_range,
        "co2_ton_yil_aralik": [v * CO2_FACTOR_T_PER_MWH / 1000 for v in kwh_range],
        "toplam_maliyet_tl_aralik": list(total_cost),
        "daire_basi_maliyet_tl_aralik": list(flat_cost),
        "daire_basi_tasarruf_tl_aralik": savings,
        "daire_basi_indirim_tl_ay_aralik": _span([v for sh in sharing for v in sh["indirim_ay_ortalama_tl"]]),
        "bina_toplam_tasarruf_tl_aralik": _span([v for sh in sharing for v in sh["bina_toplam_tasarruf_tl"]]),
        "indirim_notu": SEASONAL_NOTE,
        "mahsuplasma_notu": NETTING_NOTE if with_bill else None,
        "paylasim_notu": PHYSICAL_SHARING_NOTE,
        "geri_odeme_yil_aralik": list(payback),
        "tasarruf_yontemi": "fatura" if with_bill else "genis_tahmin",
        "fatura_tl_ay": bill,
        "fatura_karsilama_yuzde_aralik": _span([v for sh in sharing for v in sh["karsilama_yuzde"]]) if with_bill else None,
        "kalan_fatura_tl_ay_aralik": _span([v for sh in sharing for v in sh["kalan_fatura_tl_ay"]]) if with_bill else None,
        "tuketim_kwh_ay": sharing[0]["tuketim_kwh_ay"],
        "tuketim_varsayimi": ASSUMPTION_CONSTANT_CONSUMPTION if with_bill else None,
        "maliyet_bandi_tl_kwp": _cost_band_range(scenarios),
        "maliyet_bayraklari": list(dict.fromkeys(f for s in scenarios for f in s["maliyet_bayraklari"])),
        "maliyet_kaynagi": COST_SOURCE,
        "dislanan_kalemler": list(cost.EXCLUSIONS_TR),
        "uyarilar": warnings,
        "ideal_uretim_kwh_kwp": ideal["e_y"],
        "golgeli_uretim_kwh_kwp": shaded["e_y"],  # flat-section PVGIS yield: feeds the 80 % rule, not the building average
        "ozgul_verim_bina_aralik": _yield_bounds(scenarios),
        "yillik_isinim_kwh_m2": shaded["h_i_y"],
        "sicaklik_kaybi_yuzde": abs(shaded["l_tg"]),
        "uygunluk_puani_yuzde": round(suitability, 2),
        "golge_kaybi_yuzde": round(max(0.0, 100.0 - suitability), 2),
        "panel_egim_derece": flat_solar.get("angle", shaded.get("angle")),
        "panel_yon_derece": flat_solar.get("aspect", shaded.get("aspect")),
        "tarife_dusuk_tl_kwh": round(calc.LOW, 2),
        "tarife_yuksek_tl_kwh": round(calc.HIGH, 2),
        "tarife_kaynagi": TARIFF_SOURCE,
        "guven_seviyesi": level,
        "guven_nedenleri": reasons,
        "guven_iyilestirme": improve,
        "egimli_tip": pitched_type,
        "guc_kaynagi": capacity_source,
        "panel_sayisi_aralik": _span([s["panel_sayisi"] for s in scenarios]),
        "kenar_boslugu_m": SETBACK_M if has_polygon else None,
        "kenar_sonrasi_alan_m2": usable_base_area_m2,
        "kullanilabilir_oran": _usable_ratio(capacity_source, obstacle_ratio, has_polygon),
        "alan_tabani_m2": base_area,
        "cizilen_engel_sayisi": drawn.get("count") if drawn.get("marked") else None,
        "cizilen_engel_alani_m2": drawn.get("area_m2") if drawn.get("marked") else None,
        "teras_cizildi": bool(drawn.get("terrace")),
        "daire_payi_kwh_ay_aralik": share_kwh,
        "daire_payi_yuksek": not with_bill and share_kwh[1] > calc.LIMIT,
    }
    return result


def _usable_ratio(source, obstacle_ratio, has_polygon):
    """Share applied to the area base in the numeric modes (None in geometry mode)."""
    if source == "geometri":
        return None
    if obstacle_ratio is None:
        return USABLE_RATIO_NO_PHOTO / EDGE_FACTOR if has_polygon else USABLE_RATIO_NO_PHOTO
    return (1 - obstacle_ratio) if has_polygon else EDGE_FACTOR * (1 - obstacle_ratio)


def high_share_note(result):
    """Turkish note when (without a bill) a flat's monthly share exceeds the 240 kWh tier limit, else None."""
    if not result.get("daire_payi_yuksek"):
        return None
    lo, hi = result["daire_payi_kwh_ay_aralik"]
    text = tr_num(hi) if round_half_up(lo) == round_half_up(hi) else f"{tr_num(lo)}–{tr_num(hi)}"
    return HIGH_SHARE_NOTE.format(kwh=text)


def _cost_band_range(scenarios):
    bands = [s["maliyet_bandi"] for s in scenarios if s["maliyet_bandi"] is not None]
    return [min(b[0] for b in bands), max(b[1] for b in bands)] if bands else None


def chart_rows(result, months):
    """Rows {Ay, Sıra, Grup, Seri, kWh} for the monthly chart: segment split per scenario (stacked by segment).

    Only used segments appear; a scenario without any (0 kWp) adds no rows, so no misleading label is shown.
    """
    rows = []
    for sc in result["senaryolar"]:
        for seg in (s for s in sc["segmentler"] if s["durum"] == "kullanildi"):
            for i, month in enumerate(months):
                rows.append({"Ay": month, "Sıra": i, "Grup": sc["etiket"], "Seri": seg["ad"], "kWh": seg["aylik_kwh"][i]})
    return rows
