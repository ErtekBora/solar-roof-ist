"""Merge Gemini and YOLO detections (IoU matching) and mark confirmed ones."""


def fuse_detections(gemini_boxes, yolo_boxes, iou_threshold=0.3):
    """Return detections labelled doğrulandı (both models) or kontrol et (one model)."""
    raise NotImplementedError
