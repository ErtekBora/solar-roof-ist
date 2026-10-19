"""Installation cost tool: pure functions, every number comes from core/config.py."""
import math

from core.config import COST_BAND_MAX_SOURCED_KWP, COST_BANDS_TL_PER_KWP

FLAG_NO_PANELS = "kurulabilir panel yok"
FLAG_OUT_OF_RANGE = "kaynak aralığı dışında, tahmin daha belirsiz"
EXCLUSIONS_TR = (
    "Çatı güçlendirme (eski binalarda gerekebilir)",
    "Yaklaşık 10–12 yılda bir inverter değişimi",
    "Bakım",
    "Panel verim kaybı (yıllar içinde)",
    "Kat malikleri karar süreci",
)


def clamp_flats(n_flats):
    """Flat count clamped to >= 1 (None, NaN or garbage count as 1)."""
    try:
        n = int(n_flats)
    except (TypeError, ValueError, OverflowError):
        return 1
    return max(1, n)


def cost_band(kwp):
    """(low, high) TL/kWp of the band containing `kwp` (upper limits are exclusive)."""
    for upper, low, high in COST_BANDS_TL_PER_KWP:
        if kwp < upper:
            return low, high
    return COST_BANDS_TL_PER_KWP[-1][1:]


def estimate_install_cost(kwp):
    """Return (cost_low, cost_high, band, flags) for ONE kWp value; kwp <= 0 gives (0, 0, None, [no panels])."""
    if isinstance(kwp, bool) or not isinstance(kwp, (int, float)) or not math.isfinite(kwp) or kwp <= 0:
        return 0.0, 0.0, None, [FLAG_NO_PANELS]
    band = cost_band(kwp)
    flags = [FLAG_OUT_OF_RANGE] if kwp > COST_BAND_MAX_SOURCED_KWP else []
    return kwp * band[0], kwp * band[1], band, flags


def combine_cost_scenarios(costs):
    """Total range over several scenarios: [min of all lows, max of all highs]; never assumes which has more kWp.

    `costs` is a list of (cost_low, cost_high); an empty list gives (0, 0).
    """
    costs = list(costs)
    if not costs:
        return 0.0, 0.0
    return min(c[0] for c in costs), max(c[1] for c in costs)


def per_flat_cost(total_range, n_flats):
    """Per-flat cost range = total / N (N >= 1)."""
    n = clamp_flats(n_flats)
    return total_range[0] / n, total_range[1] / n
