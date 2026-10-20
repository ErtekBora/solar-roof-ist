import json
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

from core import building_mode, config, explain, report, solar
from core.formatting import tr_num
from core.solar import _parse_pvcalc

FIXTURES = Path(__file__).parent / "fixtures"


def _yield():
    load = lambda n: _parse_pvcalc(json.loads((FIXTURES / n).read_text(encoding="utf-8")))
    ideal, shaded = load("pvgis_optimal.json"), load("pvgis_terrain.json")
    return {"ideal": ideal, "shaded": shaded, "angle": ideal["angle"], "aspect": ideal["aspect"]}


def _faces(ridge=0.0, kind="hip", north=0.55):
    shaded = _yield()["shaded"]
    out = []
    for spec in building_mode.face_specs(ridge, kind):
        k = north if abs(spec["aspect"]) > 135 else 0.93
        out.append({**spec, "solar": {**shaded, "e_y": shaded["e_y"] * k, "e_m": [m * k for m in shaded["e_m"]]}})
    return out


def _result(structure="pitched", bill=None, geometry=None, ratio=None, source=None, ridge=90.0):
    faces = _faces(ridge)
    r = building_mode.analyze_building(
        roof_area_m2=593.9, usable_base_area_m2=499.5, obstacle_ratio=ratio, image_view=None, structure=structure,
        flat_share_pct=40, n_flats=25, n_flats_estimated=False, bill_tl=bill, flat_solar=_yield(), faces=faces,
        pitched_type="hip", geometry=geometry, capacity_source=source,
        drawn={"count": 2, "area_m2": 6.5, "terrace": False, "marked": geometry is not None},
    )
    r.update(lat=40.97, lon=29.07)
    r["pvgis_istekleri"] = report.pvgis_requests(40.97, 29.07, _yield(), faces, r)
    return r


def _allowed(r):
    """Every result number, ratios as % and the configured constants the formulas show."""
    values = []
    explain._collect_numbers(r, values)
    allowed = set(values)
    for sc in r["senaryolar"]:
        allowed.add(sc["duz_pay"] * 100)
        allowed.add((1 - sc["duz_pay"]) * 100)
        for seg in sc["segmentler"]:
            if seg.get("verim_orani") is not None:
                allowed.add(seg["verim_orani"] * 100)
    allowed |= {
        config.SETBACK_M, config.M2_PER_KWP_FLAT, config.M2_PER_KWP_PITCHED, config.PANEL_KWP,
        config.DEFAULT_ROOF_PITCH_DEG, config.FACE_MIN_YIELD_RATIO * 100, 12,
    }
    return allowed


def _numbers_in(text):
    cleaned = text.replace("cos30°", " ").replace("Σ", " ").replace("kWh/kWp", " ")
    return [explain._parse_number(m.group()) for m in explain._NUM_RE.finditer(cleaned)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"structure": "unknown"},
        {"structure": "mixed", "bill": 2000.0},
        {"ratio": 0.2},
        {"ratio": 0.0, "source": "engel_yok"},
        {"geometry": [{"Düz teras": {"plan_m2": 0.0, "panels": 0}, "Güney yüzü": {"plan_m2": 120.0, "panels": 40},
                       "Kuzey yüzü": {"plan_m2": 120.0, "panels": 40}, "Doğu yüzü": {"plan_m2": 60.0, "panels": 18},
                       "Batı yüzü": {"plan_m2": 60.0, "panels": 18}}]},
    ],
)
def test_every_number_in_the_breakdown_equals_a_result_value(kwargs):
    r = _result(**kwargs)
    allowed = _allowed(r)
    rows = report.breakdown_rows(r)
    assert rows and all(set(row) == {"Adım", "Hesap", "Kaynak"} for row in rows)
    for row in rows:
        for value, decimals in _numbers_in(row["Hesap"]):
            assert explain._matches(value, decimals, allowed), (row, value)


def test_breakdown_rows_follow_the_capacity_source():
    numeric = report.breakdown_rows(_result())
    steps = [row["Adım"] for row in numeric]
    assert "Kenar boşluğu sonrası alan" in steps and "Kullanılabilir alan" in steps and "Panel sayısı" not in steps
    usable = next(row for row in numeric if row["Adım"] == "Kullanılabilir alan")
    assert usable["Hesap"].startswith("499,5 × 0,325 ≈ 162,2 m²") and usable["Kaynak"] == report.SOURCE_ASSUMPTION
    geo = _result(geometry=[{"Güney yüzü": {"plan_m2": 120.0, "panels": 40}}])
    geo_steps = [row["Adım"] for row in report.breakdown_rows(geo)]
    assert "Panel sayısı" in geo_steps and "Kullanılabilir alan" not in geo_steps
    south = next(row for row in report.breakdown_rows(geo) if row["Adım"] == "Güney yüzü")
    assert south["Hesap"].startswith("40 panel × 0,40 kWp = 16,0 kWp") and south["Kaynak"] == report.SOURCE_DRAWING
    north = next(row for row in report.breakdown_rows(_result()) if row["Adım"] == "Kuzey yüzü")
    assert "verimsiz" in north["Hesap"] and north["Kaynak"] == report.SOURCE_PVGIS


def test_breakdown_shows_the_bill_inversion_and_the_flat_share():
    r = _result(bill=2000.0)
    text = {row["Adım"]: row["Hesap"] for row in report.breakdown_rows(r)}
    assert text["Tüketim"].startswith(f"Fatura 2.000 TL → {tr_num(r['tuketim_kwh_ay'])} kWh/ay")
    assert text["Daire payı"].startswith(f"{tr_num(r['yillik_uretim_kwh_aralik'][0])} kWh ÷ 25 ÷ 12 = ")


def test_pvgis_url_construction_matches_the_request_parameters(monkeypatch):
    calls = []

    class Resp:
        status_code = 200

        def json(self):
            return json.loads((FIXTURES / "pvgis_terrain.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(solar.requests, "get", lambda url, params=None, **kw: calls.append((url, params)) or Resp())
    monkeypatch.setattr(solar, "_throttle_sleep", lambda s: None)
    out = solar.fetch_pvcalc(40.97, 29.07, angle=30, aspect=-90, usehorizon=True)
    url, params = calls[0]
    parts = urlsplit(out["request_url"])
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == url == solar.PVCALC_URL
    assert dict(parse_qsl(parts.query)) == {k: str(v) for k, v in params.items()}
    assert dict(parse_qsl(parts.query))["outputformat"] == "json" and dict(parse_qsl(parts.query))["loss"] == "14"


def test_pvgis_requests_list_flat_runs_and_faces_with_area():
    r = _result()
    reqs = r["pvgis_istekleri"]
    assert [q["segment"] for q in reqs[:2]] == ["Düz bölüm, en uygun açı (ufuksuz)", "Düz bölüm, arazi ufku dahil"]
    assert "optimalangles=1" in reqs[0]["url"] and "usehorizon=0" in reqs[0]["url"]
    assert "angle=33" in reqs[1]["url"] and "aspect=8" in reqs[1]["url"] and "usehorizon=1" in reqs[1]["url"]
    faces = {q["segment"]: q["url"] for q in reqs[2:]}
    assert set(faces) == {"Güney yüzü", "Kuzey yüzü", "Doğu yüzü", "Batı yüzü"}
    assert "aspect=0" in faces["Güney yüzü"] and "aspect=180" in faces["Kuzey yüzü"]
    flat_only = _result(structure="flat")
    assert len(flat_only["pvgis_istekleri"]) == 2


def test_report_contains_the_main_numbers_and_notes():
    r = _result(bill=2000.0)
    md = report.report_markdown(r, [("Kenar boşluğu", "1,0 m", "Varsayım", "not")])
    for text in (
        "## Hesap dökümü", "## Varsayımlar ve kaynaklar", building_mode.LEGAL_NOTE, report.IMAGERY_NOTE,
        f"{tr_num(r['kurulabilir_guc_kwp_aralik'][0], 1)} kWp", f"{tr_num(r['yillik_uretim_kwh_aralik'][0])} kWh",
        f"{tr_num(r['toplam_maliyet_tl_aralik'][0])}", f"Panel sayısı: {r['panel_sayisi_aralik'][0]}",
        "Kırma (4 yüz)", r["guven_seviyesi"], "re.jrc.ec.europa.eu/api/v5_3/PVcalc",
    ):
        assert text in md, text
    assert isinstance(md, str) and md.count("| Kenar boşluğu | 1,0 m |") == 1


def test_report_notes_carry_the_hourly_netting_and_same_person_caveats():
    md = report.report_markdown(_result(bill=2000.0), [])
    assert "saatlik" in md and "aynı kişiye ait" in md
