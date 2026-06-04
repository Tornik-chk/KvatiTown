"""AprilTag traffic-sign detection (single-file module for the team repo).

Detects tag36h11 markers, maps tag IDs to sign types, returns distance/bearing.
Must match agent.py imports: SignType, TagObservation, SignDetector.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    from pupil_apriltags import Detector  # type: ignore
    _APRILTAG_OK = True
except Exception as _e:  # noqa: BLE001
    Detector = None  # type: ignore
    _APRILTAG_OK = False
    _IMPORT_ERROR = repr(_e)


class SignType(Enum):
    UNKNOWN = "unknown"
    STOP = "stop"
    YIELD = "yield"
    T_INTERSECTION = "t_intersection"
    FOUR_WAY = "four_way"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"
    STRAIGHT_ONLY = "straight_only"
    NO_LEFT = "no_left"
    NO_RIGHT = "no_right"
    ONE_WAY_LEFT = "one_way_left"
    ONE_WAY_RIGHT = "one_way_right"
    PARKING = "parking"
    DUCKIEBOT = "duckiebot"


@dataclass
class TagObservation:
    tag_id: int
    sign_type: SignType
    distance_m: float
    bearing_rad: float
    corners: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    center_px: Tuple[int, int] = (0, 0)


_TAG_ID_TO_SIGN: Dict[int, SignType] = {
    25: SignType.STOP, 26: SignType.STOP, 31: SignType.STOP,
    32: SignType.STOP, 33: SignType.STOP,
    27: SignType.YIELD, 28: SignType.YIELD, 29: SignType.YIELD, 30: SignType.YIELD,
    75: SignType.PARKING, 207: SignType.PARKING,
}

_RANGE_RULES = [
    ((1, 7), SignType.DUCKIEBOT),
    ((8, 30), SignType.FOUR_WAY),
    ((61, 76), SignType.T_INTERSECTION),
    ((100, 130), SignType.LEFT_ONLY),
    ((131, 160), SignType.RIGHT_ONLY),
    ((161, 190), SignType.STRAIGHT_ONLY),
    ((200, 220), SignType.PARKING),
]


def lookup_sign(tag_id: int) -> SignType:
    if tag_id in _TAG_ID_TO_SIGN:
        return _TAG_ID_TO_SIGN[tag_id]
    for (lo, hi), st in _RANGE_RULES:
        if lo <= tag_id <= hi:
            return st
    return SignType.UNKNOWN


_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CAMERA_CFG = os.path.join(_PROJECT_ROOT, "duckiebot", "camera_driver", "config", "camera_config.yaml")
_PROJECT_CFG = os.path.join(_PROJECT_ROOT, "config", "project_config.yaml")


def _load_yaml(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _intrinsics(width: int, height: int, fov_deg: float):
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


class SignDetector:
    """AprilTag detector — used by agent.py FSM."""

    def __init__(self, tag_family: Optional[str] = None, tag_size_m: Optional[float] = None):
        cam = _load_yaml(_CAMERA_CFG)
        cfg = _load_yaml(_PROJECT_CFG)
        self.width = int(cam.get("resolution", {}).get("width", 640))
        self.height = int(cam.get("resolution", {}).get("height", 480))
        self.fov_deg = float(cam.get("fov", 160))
        self.tag_family = tag_family or cfg.get("tag_family", "tag36h11")
        self.tag_size_m = float(tag_size_m if tag_size_m is not None else cfg.get("tag_size_m", 0.065))
        self._cam_params = _intrinsics(self.width, self.height, self.fov_deg)
        self.available = _APRILTAG_OK
        self.last_error: Optional[str] = None
        if _APRILTAG_OK:
            self._det = Detector(
                families=self.tag_family, nthreads=2,
                quad_decimate=1.0, refine_edges=True,
            )
        else:
            self._det = None
            self.last_error = (
                f"pupil_apriltags import failed: {_IMPORT_ERROR}. "
                "Run: pip install -r requirements.txt"
            )

    def detect(self, frame_rgb_or_bgr: np.ndarray) -> List[TagObservation]:
        if not self.available or self._det is None or frame_rgb_or_bgr is None:
            return []
        if frame_rgb_or_bgr.ndim == 3:
            gray = np.mean(frame_rgb_or_bgr, axis=2).astype(np.uint8)
        else:
            gray = frame_rgb_or_bgr.astype(np.uint8)
        try:
            results = self._det.detect(
                gray, estimate_tag_pose=True,
                camera_params=self._cam_params, tag_size=self.tag_size_m,
            )
        except Exception as e:  # noqa: BLE001
            self.last_error = repr(e)
            return []
        out: List[TagObservation] = []
        for r in results:
            x, y, z = (float(v) for v in r.pose_t.flatten())
            distance = float(np.linalg.norm([x, y, z]))
            bearing = math.atan2(x, z)
            out.append(TagObservation(
                tag_id=int(r.tag_id),
                sign_type=lookup_sign(int(r.tag_id)),
                distance_m=distance,
                bearing_rad=bearing,
                corners=tuple((float(c[0]), float(c[1])) for c in r.corners),
                center_px=(int(r.center[0]), int(r.center[1])),
            ))
        return out
