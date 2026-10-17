import io
import math

import pytest
from google.genai import errors
from PIL import Image
from shapely.geometry import Polygon

from core import config, vision
from core.vision import VisionError

ROOT_IMG = "data/demo_images/roof_10.jpg"


def png(width=1000, height=1000, color=(120, 120, 120)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "PNG")
    return buf.getvalue()


def payload(**over):
    """A minimal valid Gemini answer: main roof [100..900]^2 (y, x), no obstacles."""
    data = {
        "roof_polygon": [[100, 100], [100, 900], [900, 900], [900, 100]],
        "roof_box": [100, 100, 900, 900],
        "obstacles": [],
        "roof_type": "flat",
        "image_view": "top-down",
        "multiple_buildings": False,
    }
    data.update(over)
    return data


def api_error(code, status):
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": "boom", "status": status}})


# ------------------------------------------------------------ coordinates / image

def test_point_and_box_conversion_non_square():
    # Gemini [y, x] on an 800x400 image: x scales with the width, y with the height
    assert vision._point_to_px([250, 500], 800, 400) == (400.0, 100.0)
    assert vision._box_to_px([100, 200, 300, 600], 800, 400) == (160.0, 40.0, 480.0, 120.0)


def test_downscale_keeps_ratio():
    img = vision._load_image(png(4000, 3000))
    assert img.size == (1280, 960)


def test_small_image_not_upscaled():
    assert vision._load_image(png(300, 200)).size == (300, 200)


def test_corrupt_bytes_raise_vision_error():
    with pytest.raises(VisionError, match="Dosya okunamadı"):
        vision._load_image(b"this is not an image")


def test_detect_corrupt_bytes_raises_vision_error(fake_gemini):
    client = fake_gemini([])
    with pytest.raises(VisionError, match="Dosya okunamadı"):
        vision.detect_roof_objects(b"garbage")
    assert client.calls == 0


# --------------------------------------------------------------- roof extraction

def test_valid_polygon_is_used(fake_gemini):
    fake_gemini([payload()])
    r = vision.detect_roof_objects(png())
    assert r["roof_source"] == "polygon"
    assert r["roof_area_px"] == pytest.approx(640000)
    assert r["width"] == 1000 and r["height"] == 1000 and r["model"] == config.VISION_MODEL


def test_bowtie_polygon_repaired_or_falls_back_to_box(fake_gemini):
    bowtie = [[100, 100], [100, 900], [900, 100], [900, 900]]
    fake_gemini([payload(roof_polygon=bowtie, roof_box=[0, 0, 1000, 1000])])
    r = vision.detect_roof_objects(png())
    assert r["roof_source"] in ("polygon", "box")
    assert Polygon(r["roof_polygon_px"]).is_valid and r["roof_area_px"] > 0


def test_tiny_polygon_falls_back_to_box_with_four_corners(fake_gemini):
    tiny = [[500, 500], [500, 520], [520, 520], [520, 500]]
    fake_gemini([payload(roof_polygon=tiny)])
    r = vision.detect_roof_objects(png(800, 400))
    assert r["roof_source"] == "box"
    # box [ymin 100, xmin 100, ymax 900, xmax 900] on 800x400 -> x1=80, y1=40, x2=720, y2=360
    assert r["roof_polygon_px"] == [(80.0, 40.0), (720.0, 40.0), (720.0, 360.0), (80.0, 360.0)]
    assert r["roof_area_px"] == pytest.approx(640 * 320)


def test_polygon_much_smaller_than_box_falls_back(fake_gemini):
    small = [[100, 100], [100, 300], [300, 300], [300, 100]]  # 4% of the image, 6% of the box
    fake_gemini([payload(roof_polygon=small, roof_box=[0, 0, 1000, 1000])])
    assert vision.detect_roof_objects(png())["roof_source"] == "box"


def test_invalid_box_with_valid_polygon_uses_polygon(fake_gemini):
    fake_gemini([payload(roof_box=[100, 100, 100, 900])])  # zero height
    r = vision.detect_roof_objects(png())
    assert r["roof_source"] == "polygon"


def test_polygon_is_not_compared_to_an_invalid_box(fake_gemini):
    # box is tiny (invalid, <2%); polygon would be < 25% of a huge box but there is no valid box
    fake_gemini([payload(roof_box=[500, 500, 510, 510])])
    assert vision.detect_roof_objects(png())["roof_source"] == "polygon"


def test_box_coordinates_are_clipped(fake_gemini):
    fake_gemini([payload(roof_polygon=[], roof_box=[-50, -50, 1200, 1200])])
    r = vision.detect_roof_objects(png(100, 100))
    assert r["roof_polygon_px"] == [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def test_zero_area_box_and_no_polygon_raises(fake_gemini):
    fake_gemini([payload(roof_polygon=[], roof_box=[300, 300, 300, 700])])
    with pytest.raises(VisionError, match="geçerli bir çatı yüzeyi"):
        vision.detect_roof_objects(png())


def test_both_invalid_raises(fake_gemini):
    fake_gemini([payload(roof_polygon=[[1, 1], [1, 2], [2, 2]], roof_box=None)])
    with pytest.raises(VisionError, match="geçerli bir çatı yüzeyi"):
        vision.detect_roof_objects(png())


def test_closing_duplicate_point_is_dropped():
    pts = vision._clean_points([[100, 100], [100, 900], [900, 900], [100, 100]], 1000, 1000)
    assert pts == [(100.0, 100.0), (900.0, 100.0), (900.0, 900.0)]


def test_duplicates_reduced_below_three_points_fall_back(fake_gemini):
    dup = [[100, 100], [100, 100], [500, 500], [500, 500], [100, 100]]
    fake_gemini([payload(roof_polygon=dup)])
    assert vision.detect_roof_objects(png())["roof_source"] == "box"


def test_polygon_construction_error_falls_back(monkeypatch, fake_gemini):
    def boom(*a, **k):
        raise ValueError("bad ring")

    monkeypatch.setattr(vision, "Polygon", boom)
    assert vision._validate_polygon([(0, 0), (1, 0), (1, 1)], None, 10, 10) is None


# ------------------------------------------------------------------- obstacles

def test_obstacle_filtering_overlap_rule(fake_gemini):
    obstacles = [
        {"label": "chimney", "box_2d": [400, 500, 450, 550], "confidence": 0.9},  # inside
        {"label": "skylight", "box_2d": [400, 860, 500, 960], "confidence": 0.9},  # 40% overlap, centre outside
        {"label": "skylight", "box_2d": [400, 890, 500, 990], "confidence": 0.9},  # 10% overlap
        {"label": "chimney", "box_2d": [0, 0, 50, 50], "confidence": 0.9},  # far away
        {"label": "chimney", "box_2d": [400, 500, 400, 550], "confidence": 0.9},  # zero height
    ]
    fake_gemini([payload(obstacles=obstacles)])
    r = vision.detect_roof_objects(png())
    assert [o["id"] for o in r["obstacles"]] == [0, 1]
    assert [o["label"] for o in r["obstacles"]] == ["chimney", "skylight"]
    assert r["obstacles"][1]["box_px"] == (860.0, 400.0, 960.0, 500.0)


def test_centre_inside_roof_is_kept_even_with_overlap_below_30_percent(fake_gemini):
    # box centred just inside the roof corner (895, 895): overlap is 29.3% < 30%, but the centre is on the roof
    obstacles = [{"label": "chimney", "box_2d": [835, 835, 955, 955], "confidence": 0.9}]
    fake_gemini([payload(obstacles=obstacles)])
    assert len(vision.detect_roof_objects(png())["obstacles"]) == 1


def test_label_normalization():
    assert vision._normalize_label("Water Tank ") == "water tank"
    assert vision._normalize_label("banana") == "other"
    assert vision._normalize_label(None) == "other"
    assert vision._normalize_label("other") == "other"
    assert all(vision._normalize_label(c) == c for c in config.OBSTACLE_CLASSES)


@pytest.mark.parametrize(
    "value, expected",
    [(None, 0.5), (85, 0.85), (-0.2, 0.0), (0.7, 0.7), (1, 1.0), (100, 1.0), (250, 1.0), (True, 0.5),
     (math.nan, 0.5), (math.inf, 0.5), ("high", 0.5)],
)
def test_confidence_normalization(value, expected):
    assert vision._normalize_confidence(value) == pytest.approx(expected)


def test_obstacle_confidence_normalized_in_result(fake_gemini):
    obstacles = [{"label": "chimney", "box_2d": [400, 500, 450, 550], "confidence": 85}]
    fake_gemini([payload(obstacles=obstacles)])
    assert vision.detect_roof_objects(png())["obstacles"][0]["confidence"] == pytest.approx(0.85)


# ------------------------------------------------------------ obstacle ratio

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_ratio_single_obstacle():
    assert vision.compute_obstacle_ratio(SQUARE, [(0, 0, 50, 50)], 100, 100) == pytest.approx(0.25)


def test_ratio_obstacle_inside_offset_roof():
    roof = [(50, 50), (150, 50), (150, 150), (50, 150)]
    assert vision.compute_obstacle_ratio(roof, [(60, 60, 110, 110)], 200, 200) == pytest.approx(0.25)


def test_ratio_overlapping_obstacles_not_double_counted():
    boxes = [(0, 0, 50, 50), (25, 25, 75, 75)]  # union = 2500 + 2500 - 625
    assert vision.compute_obstacle_ratio(SQUARE, boxes, 100, 100) == pytest.approx(0.4375)


def test_ratio_obstacle_outside_roof_ignored():
    roof = [(0, 0), (50, 0), (50, 50), (0, 50)]
    assert vision.compute_obstacle_ratio(roof, [(60, 60, 90, 90)], 100, 100) == 0.0


def test_ratio_clipped_to_095():
    assert vision.compute_obstacle_ratio(SQUARE, [(0, 0, 100, 100)], 100, 100) == 0.95


def test_ratio_empty_list_is_zero():
    assert vision.compute_obstacle_ratio(SQUARE, [], 100, 100) == 0.0


def test_ratio_zero_roof_area_raises():
    with pytest.raises(ValueError):
        vision.compute_obstacle_ratio([(0, 0), (10, 0), (20, 0)], [(0, 0, 5, 5)], 100, 100)


# -------------------------------------------------------------- Gemini calls

def test_cache_hit_makes_no_second_call(fake_gemini):
    client = fake_gemini([payload()])
    data = png()
    first = vision.detect_roof_objects(data)
    second = vision.detect_roof_objects(data)
    assert client.calls == 1 and first == second
    assert first["image_hash"] == __import__("hashlib").sha256(data).hexdigest()


def test_cache_returns_copy(fake_gemini):
    fake_gemini([payload()])
    data = png()
    vision.detect_roof_objects(data)["obstacles"].append("junk")
    assert vision.detect_roof_objects(data)["obstacles"] == []


def test_malformed_json_retried_once_then_ok(fake_gemini):
    client = fake_gemini(["{not json", payload()])
    assert vision.detect_roof_objects(png())["roof_source"] == "polygon"
    assert client.calls == 2


def test_malformed_json_twice_raises_after_one_retry(fake_gemini):
    client = fake_gemini(["{not json", "[1, 2]"])
    with pytest.raises(VisionError, match="okunamadı"):
        vision.detect_roof_objects(png())
    assert client.calls == 2


@pytest.mark.parametrize("code, status", [(429, "RESOURCE_EXHAUSTED"), (503, "UNAVAILABLE")])
def test_transient_error_retried_once_after_3s(fake_gemini, code, status):
    client = fake_gemini([api_error(code, status), payload()])
    assert vision.detect_roof_objects(png())["roof_source"] == "polygon"
    assert client.calls == 2 and client.sleeps == [3]


def test_429_twice_raises_quota_message(fake_gemini):
    client = fake_gemini([api_error(429, "RESOURCE_EXHAUSTED")] * 2)
    with pytest.raises(VisionError, match="Gemini kullanım kotası doldu; lütfen biraz sonra tekrar deneyin."):
        vision.detect_roof_objects(png())
    assert client.calls == 2


def test_503_twice_raises_busy_message(fake_gemini):
    fake_gemini([api_error(503, "UNAVAILABLE")] * 2)
    with pytest.raises(VisionError, match="yoğun"):
        vision.detect_roof_objects(png())


def test_status_detected_from_message_text(fake_gemini):
    client = fake_gemini([RuntimeError("503 UNAVAILABLE: overloaded"), payload()])
    vision.detect_roof_objects(png())
    assert client.calls == 2


def test_other_api_error_not_retried(fake_gemini):
    client = fake_gemini([api_error(400, "INVALID_ARGUMENT")])
    with pytest.raises(VisionError, match="şu an yapılamadı"):
        vision.detect_roof_objects(png())
    assert client.calls == 1 and client.sleeps == []


def test_missing_api_key_raises_vision_error(monkeypatch):
    vision.clear_cache()

    def no_key():
        raise config.MissingApiKeyError("Gemini API anahtarı bulunamadı.")

    monkeypatch.setattr(vision, "get_gemini_api_key", no_key)
    with pytest.raises(VisionError, match="anahtarı"):
        vision.detect_roof_objects(png())


# ------------------------------------------------------- real fixture (roof_10)

def test_roof10_fixture_end_to_end(fake_gemini, roof10_json):
    fake_gemini([roof10_json])
    with open(ROOT_IMG, "rb") as f:
        r = vision.detect_roof_objects(f.read())
    assert r["roof_source"] == "polygon"
    assert r["roof_type"] == "pitched" and r["image_view"] == "oblique" and r["multiple_buildings"] is False
    assert len(r["obstacles"]) == 8
    assert {o["label"] for o in r["obstacles"]} == {"skylight", "satellite dish", "air conditioner unit"}
    assert 0 < r["obstacle_ratio"] < 0.95


# ---------------------------------------------------------------------- overlay

def test_draw_overlay_returns_valid_png(fake_gemini, roof10_json):
    fake_gemini([roof10_json])
    with open(ROOT_IMG, "rb") as f:
        data = f.read()
    r = vision.detect_roof_objects(data)
    for ids in (None, {0, 1}, set()):
        out = vision.draw_overlay(data, r, ids)
        img = Image.open(io.BytesIO(out))
        assert img.format == "PNG" and img.size == (r["width"], r["height"])
        img.verify()


def test_draw_overlay_marks_included_red_and_excluded_grey(fake_gemini):
    obstacles = [
        {"label": "chimney", "box_2d": [200, 200, 300, 300], "confidence": 0.9},
        {"label": "chimney", "box_2d": [600, 600, 700, 700], "confidence": 0.9},
    ]
    fake_gemini([payload(obstacles=obstacles)])
    data = png()
    r = vision.detect_roof_objects(data)
    img = Image.open(io.BytesIO(vision.draw_overlay(data, r, {0})))
    assert img.getpixel((200, 250)) == (230, 30, 30)  # left edge of included obstacle
    assert img.getpixel((600, 650)) == (150, 150, 150)  # left edge of excluded obstacle
