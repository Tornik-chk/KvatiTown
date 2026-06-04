"""
sign_detector.py — April tag detection and traffic sign classification.

Uses pupil-apriltags for tag detection. Install with:
    pip install pupil-apriltags

Tag family: tag36h11, IDs 1-199 are traffic signs.
"""

from __future__ import annotations
import cv2
import numpy as np
from enum import Enum, auto
from typing import Optional, Tuple

try:
    from pupil_apriltags import Detector
    _APRILTAG_AVAILABLE = True
except ImportError:
    _APRILTAG_AVAILABLE = False
    print("[SignDetector] WARNING: pupil-apriltags not installed. Sign detection disabled.")


class SignAction(Enum):
    NONE        = auto()  # ignore / informational
    STOP        = auto()  # full stop, then yield to cross-traffic
    YIELD       = auto()  # yield to cross-traffic without full stop
    SLOW        = auto()  # reduce speed temporarily


# Duckietown tag36h11 ID → action mapping (IDs 1-199 are traffic signs)
# Only actionable signs are mapped; everything else → NONE
_ID_TO_ACTION: dict[int, SignAction] = {
    1:  SignAction.STOP,   # stop sign
    2:  SignAction.YIELD,  # yield
    5:  SignAction.SLOW,   # slow down
    6:  SignAction.SLOW,   # speed limit 10
    7:  SignAction.SLOW,   # speed limit 15
}

# Human-readable names for logging
_ID_TO_NAME: dict[int, str] = {
    1:  "stop",
    2:  "yield",
    3:  "no-entry",
    4:  "service vehicle",
    5:  "slow down",
    6:  "speed limit 10",
    7:  "speed limit 15",
    8:  "speed limit 20",
    9:  "speed limit 30",
    10: "speed limit 40",
    20: "right turn only",
    21: "left turn only",
    22: "oneway right",
    23: "oneway left",
    24: "junction",
    25: "traffic light ahead",
    26: "pedestrian",
    27: "t-intersection",
    28: "crossing",
    29: "keep right",
}

# Minimum tag decision margin to trust a detection (lower = noisier)
_MIN_DECISION_MARGIN = 30
# Minimum tag area in pixels to act on (filters out far-away tags)
_MIN_TAG_AREA = 800


class SignDetector:
    """
    Wraps pupil-apriltags detector and converts tag IDs to SignActions.
    Call detect(frame) each frame; it returns the most actionable sign seen.
    """

    def __init__(self):
        if _APRILTAG_AVAILABLE:
            self._detector = Detector(
                families="tag36h11",
                nthreads=2,
                quad_decimate=2.0,   # downsample before detection — faster on Duckiebot
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
            )
        else:
            self._detector = None

    def detect(self, frame_bgr: np.ndarray) -> Tuple[SignAction, Optional[str], Optional[int]]:
        """
        Detect the most actionable traffic sign in the frame.

        Args:
            frame_bgr: BGR image from camera.read()

        Returns:
            (action, sign_name, tag_id)
            action    — SignAction to take (NONE if nothing actionable)
            sign_name — human-readable name, or None
            tag_id    — raw April tag ID, or None
        """
        if self._detector is None:
            return SignAction.NONE, None, None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tags = self._detector.detect(gray)

        best_action = SignAction.NONE
        best_name   = None
        best_id     = None
        best_area   = 0

        for tag in tags:
            # Filter weak or distant detections
            if tag.decision_margin < _MIN_DECISION_MARGIN:
                continue

            corners = tag.corners
            area = _polygon_area(corners)
            if area < _MIN_TAG_AREA:
                continue

            tag_id = tag.tag_id

            # Only act on traffic sign range
            if not (1 <= tag_id <= 199):
                continue

            action = _ID_TO_ACTION.get(tag_id, SignAction.NONE)
            name   = _ID_TO_NAME.get(tag_id, f"sign_{tag_id}")

            # Prefer higher-priority actions; break ties by tag area (closer = bigger)
            if _action_priority(action) > _action_priority(best_action) or (
                action == best_action and area > best_area
            ):
                best_action = action
                best_name   = name
                best_id     = tag_id
                best_area   = area

        return best_action, best_name, best_id


def _polygon_area(corners: np.ndarray) -> float:
    """Shoelace formula for polygon area from corner array (4x2)."""
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _action_priority(action: SignAction) -> int:
    return {
        SignAction.NONE:  0,
        SignAction.SLOW:  1,
        SignAction.YIELD: 2,
        SignAction.STOP:  3,
    }[action]