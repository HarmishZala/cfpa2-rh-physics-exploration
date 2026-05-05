"""Live display — camera feed with detection overlays and VLM direction."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection


# Colour palette for bounding boxes (BGR)
_COLOURS = [
    (0, 255, 0),    # green
    (255, 128, 0),   # blue-ish
    (0, 200, 255),   # yellow-ish
    (255, 0, 255),   # magenta
    (0, 255, 255),   # cyan
    (128, 0, 255),   # purple
]

# Direction → arrow endpoint offset (dx, dy) relative to frame center
_ARROW_OFFSETS = {
    "forward": (0, -120),
    "left": (-120, 0),
    "right": (120, 0),
    "back": (0, 120),
    "stop": (0, 0),
}


class Display:
    """OpenCV-based live display with detection boxes and VLM overlay."""

    WINDOW_NAME = "Real-World Exploration"

    def __init__(self) -> None:
        self._last_direction: str = ""
        self._last_reasoning: str = ""
        self._artifacts_found: int = 0
        self._artifacts_total: int = 0

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        direction: str = "",
        reasoning: str = "",
        artifacts_found: int = 0,
        artifacts_total: int = 0,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Draw overlays on frame and show it. Returns the annotated frame."""
        if direction:
            self._last_direction = direction
        if reasoning:
            self._last_reasoning = reasoning
        self._artifacts_found = artifacts_found
        self._artifacts_total = artifacts_total

        canvas = frame.copy()
        self._draw_detections(canvas, detections)
        self._draw_direction_arrow(canvas)
        self._draw_hud(canvas, fps)

        cv2.imshow(self.WINDOW_NAME, canvas)
        return canvas

    def poll_key(self, wait_ms: int = 1) -> int:
        """Poll for keypress. Returns key code or -1."""
        return cv2.waitKey(wait_ms) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> None:
        for i, det in enumerate(detections):
            colour = _COLOURS[i % len(_COLOURS)]
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            label_text = f"{det.label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(
                frame, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
            )

    def _draw_direction_arrow(self, frame: np.ndarray) -> None:
        direction = self._last_direction
        if not direction:
            return

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        dx, dy = _ARROW_OFFSETS.get(direction, (0, 0))

        if direction == "stop":
            cv2.circle(frame, (cx, cy), 30, (0, 0, 255), 3)
            cv2.putText(
                frame, "STOP", (cx - 30, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )
        else:
            cv2.arrowedLine(
                frame, (cx, cy), (cx + dx, cy + dy),
                (0, 255, 255), 4, tipLength=0.3,
            )
            cv2.putText(
                frame, direction.upper(), (cx + dx - 40, cy + dy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )

    def _draw_hud(self, frame: np.ndarray, fps: float) -> None:
        h, w = frame.shape[:2]
        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # FPS
        cv2.putText(
            frame, f"FPS: {fps:.0f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )
        # Artifacts
        cv2.putText(
            frame,
            f"Artifacts: {self._artifacts_found}/{self._artifacts_total}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
        )
        # Direction
        if self._last_direction:
            cv2.putText(
                frame,
                f"Nav: {self._last_direction}",
                (200, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA,
            )
        # Reasoning (truncated)
        if self._last_reasoning:
            text = self._last_reasoning[:80] + ("..." if len(self._last_reasoning) > 80 else "")
            cv2.putText(
                frame, text, (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
            )
