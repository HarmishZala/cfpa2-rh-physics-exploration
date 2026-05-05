"""Camera provider — captures frames from webcam or iPhone (Continuity Camera)."""

from __future__ import annotations

import base64
import cv2
import numpy as np


class CameraProvider:
    """Wraps cv2.VideoCapture for consistent frame capture."""

    def __init__(
        self,
        device: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {device}. "
                "Check that your webcam is connected or iPhone Continuity Camera is enabled."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._width = width
        self._height = height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Return a single BGR frame (numpy array)."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        return frame

    def capture_b64(self, fmt: str = ".jpg", quality: int = 85) -> str:
        """Capture a frame and return it as a base64-encoded string.

        Parameters
        ----------
        fmt : image format extension (default ".jpg")
        quality : JPEG quality 0-100 (ignored for PNG)
        """
        frame = self.capture()
        encode_params = []
        if fmt == ".jpg":
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ok, buf = cv2.imencode(fmt, frame, encode_params)
        if not ok:
            raise RuntimeError("Failed to encode frame.")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> CameraProvider:
        return self

    def __exit__(self, *exc) -> None:
        self.release()
