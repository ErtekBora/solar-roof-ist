"""Optional, fail-safe HTTP client for the separate yolo_service."""


def detect(image_bytes):
    """POST the image to {YOLO_URL}/detect; return detections, or None on any failure."""
    raise NotImplementedError
