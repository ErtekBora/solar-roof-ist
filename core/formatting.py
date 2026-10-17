"""Shared number rounding and Turkish formatting: one rule (round half up) for the UI, tables and AI payload."""
from decimal import ROUND_HALF_UP, Decimal


def _quantize(value, decimals):
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-decimals), ROUND_HALF_UP)


def round_half_up(value, decimals=0):
    """Round half up on the decimal repr (27.15 -> 27.2, 1738.5 -> 1739); int when decimals == 0."""
    rounded = _quantize(value, decimals)
    return int(rounded) if decimals == 0 else float(rounded)


def tr_num(value, decimals=0):
    """Round half up, then format with Turkish separators (1.234,5)."""
    text = f"{_quantize(value, decimals):,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def percent_from_ratio(ratio):
    """Percentage of a ratio rounded to 3 decimals first, as the AI payload stores it (0.2715 -> 27.2)."""
    return float(Decimal(str(round_half_up(ratio, 3))) * 100)
