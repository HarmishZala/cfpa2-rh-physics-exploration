"""Object detector — YOLO-World open-vocabulary detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single object detection result."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center_x: int
    center_y: int
    area_fraction: float  # bbox area / frame area
    region: str  # "left", "center", or "right"


class ObjectDetector:
    """Open-vocabulary object detector using YOLO-World.

    Usage
    -----
    >>> detector = ObjectDetector(model_size="s")
    >>> detector.set_classes(["red ball", "fire extinguisher"])
    >>> detections = detector.detect(frame)
    """

    _MODEL_MAP = {
        "s": "yolov8s-worldv2.pt",
        "m": "yolov8m-worldv2.pt",
        "l": "yolov8l-worldv2.pt",
    }

    def __init__(
        self,
        model_size: str = "s",
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLOWorld

        weight = self._MODEL_MAP.get(model_size)
        if weight is None:
            raise ValueError(
                f"Unknown model_size '{model_size}'. Choose from: {list(self._MODEL_MAP)}"
            )

        self._model = YOLOWorld(weight)
        self._confidence = confidence
        self._classes: list[str] = []

        if device:
            self._device = device
        else:
            import torch
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"

    def set_classes(self, classes: list[str]) -> None:
        """Set the target object classes to detect (open-vocabulary)."""
        self._classes = classes
        self._model.set_classes(classes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame.

        Returns a list of Detection objects sorted by confidence (highest first).
        """
        if not self._classes:
            return []

        results = self._model.predict(
            frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        h, w = frame.shape[:2]
        frame_area = h * w
        detections: list[Detection] = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu())
            cls_idx = int(box.cls[0].cpu())
            label = self._classes[cls_idx] if cls_idx < len(self._classes) else "unknown"

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bbox_area = (x2 - x1) * (y2 - y1)

            # Classify screen region based on center x
            if cx < w / 3:
                region = "left"
            elif cx > 2 * w / 3:
                region = "right"
            else:
                region = "center"

            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center_x=cx,
                    center_y=cy,
                    area_fraction=bbox_area / frame_area,
                    region=region,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
