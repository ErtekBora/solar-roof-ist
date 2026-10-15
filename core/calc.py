"""Area, capacity (kWp), suitability score, annual kWh, CO2, savings (TL), confidence and scenarios."""


def compute_capacity(roof_area_m2, obstacle_ratio, roof_type, yield_ideal, yield_shaded, tariff):
    """Return all result numbers described in SPEC section 6."""
    raise NotImplementedError


def run_scenario(base_result, change):
    """Re-run the calculation with a change (e.g. fewer obstacles); return new numbers + delta."""
    raise NotImplementedError
