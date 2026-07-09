from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from .stabilize import GlobalStabilizer
except ImportError:  # pragma: no cover
    from src.stabilize import GlobalStabilizer


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")


@dataclass
class VideoFrame:
    frame_index: int
    timestamp_sec: float
    frame: np.ndarray
    raw_frame: np.ndarray
    motion_score: float
    stabilization_method: str


class VideoReader:
    def __init__(
        self,
        video_path: str | Path,
        *,
        stabilize_enabled: bool = False,
        stabilizer: GlobalStabilizer | None = None,
    ) -> None:
        _require_cv2()
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.stabilize_enabled = bool(stabilize_enabled)
        self.stabilizer = stabilizer or GlobalStabilizer()
        self.prev_output_frame: np.ndarray | None = None
        self.frame_index = -1
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.fps = 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> VideoFrame | None:
        ok, raw_frame = self.cap.read()
        if not ok:
            return None

        self.frame_index += 1
        timestamp_sec = self.frame_index / self.fps

        if not self.stabilize_enabled:
            output_frame = raw_frame
            motion_score = 0.0
            method = "disabled"
        elif self.prev_output_frame is None:
            output_frame = raw_frame
            motion_score = 0.0
            method = "bootstrap"
        else:
            result = self.stabilizer.stabilize(self.prev_output_frame, raw_frame)
            output_frame = result.stabilized_frame
            motion_score = float(result.motion_score)
            method = result.method

        self.prev_output_frame = output_frame.copy()
        return VideoFrame(
            frame_index=self.frame_index,
            timestamp_sec=timestamp_sec,
            frame=output_frame,
            raw_frame=raw_frame,
            motion_score=motion_score,
            stabilization_method=method,
        )

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

