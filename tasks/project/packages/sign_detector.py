"""AprilTag traffic-sign detection for the Godot project maps.

Textures on signs are tag36h11 images named tag36_11_XXXXX.png → tag ID XXXXX.
Lookup matches tags used on project.tscn / project_signs.tscn (see config/sign_tag_map.yaml).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
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


_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CAMERA_CFG = os.path.join(_PROJECT_ROOT, "duckiebot", "camera_driver", "config", "camera_config.yaml")
_PROJECT_CFG = os.path.join(_PROJECT_ROOT, "config", "project_config.yaml")
_TAG_MAP_CFG = os.path.join(_PROJECT_ROOT, "config", "sign_tag_map.yaml")

# Duckietown apriltagsDB defaults (overridden by config/sign_tag_map.yaml)
_DEFAULT_TAG_MAP: Dict[int, str] = {
    1: "stop",
    2: "yield",
    8: "four_way",
    25: "stop",
    26: "stop",
    27: "stop",
    28: "stop",
    31: "stop",
    33: "stop",
    57: "t_intersection",
    58: "t_intersection",
    61: "t_intersection",
    75: "parking",
}

_STR_TO_SIGN = {m.value: m for m in SignType}


def _load_yaml(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_tag_map() -> Dict[int, SignType]:
    raw = _load_yaml(_TAG_MAP_CFG).get("tags", {})
    merged = dict(_DEFAULT_TAG_MAP)
    merged.update(raw)
    out: Dict[int, SignType] = {}
    for k, v in merged.items():
        try:
            tid = int(k)
            name = str(v).strip().lower()
            out[tid] = _STR_TO_SIGN.get(name, SignType.UNKNOWN)
        except Exception:
            continue
    return out


def lookup_sign(tag_id: int, table: Dict[int, SignType]) -> SignType:
    return table.get(tag_id, SignType.UNKNOWN)


def _intrinsics(width: int, height: int, fov_deg: float):
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def _distance_from_corners(corners: np.ndarray, tag_size_m: float, fx: float) -> float:
    """Fallback distance when pose estimate is noisy (wide fisheye sim camera)."""
    if corners is None or len(corners) < 4:
        return 99.0
    pts = np.asarray(corners, dtype=np.float64)
    sides = [
        float(np.linalg.norm(pts[i] - pts[(i + 1) % 4]))
        for i in range(4)
    ]
    px = max(1.0, float(np.mean(sides)))
    return float(tag_size_m * fx / px)


class SignDetector:
    """AprilTag detector — used by agent.py FSM."""

    def __init__(self, tag_family: Optional[str] = None, tag_size_m: Optional[float] = None):
        cam = _load_yaml(_CAMERA_CFG)
        cfg = _load_yaml(_PROJECT_CFG)
        self.width = int(cam.get("resolution", {}).get("width", 640))
        self.height = int(cam.get("resolution", {}).get("height", 480))
        self.fov_deg = float(cam.get("fov", 160))
        self.tag_family = tag_family or cfg.get("tag_family", "tag36h11")
        # Godot sign plane is 0.07 m (obj_stop_sign.tscn PlaneMesh)
        self.tag_size_m = float(
            tag_size_m if tag_size_m is not None else cfg.get("tag_size_m", 0.07)
        )
        self.min_margin = float(cfg.get("tag_min_decision_margin", 20.0))
        self._fx, self._fy, self._cx, self._cy = _intrinsics(self.width, self.height, self.fov_deg)
        self._cam_params = (self._fx, self._fy, self._cx, self._cy)
        self._tag_table = _load_tag_map()
        self.last_raw_ids: List[int] = []
        self.available = _APRILTAG_OK
        self.last_error: Optional[str] = None

        if _APRILTAG_OK:
            self._det = Detector(
                families=self.tag_family,
                nthreads=2,
                quad_decimate=1.5,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
            )
        else:
            self._det = None
            self.last_error = (
                f"pupil_apriltags import failed: {_IMPORT_ERROR}. "
                "Run: pip install -r requirements.txt"
            )

    def _to_gray(self, frame_rgb_or_bgr: np.ndarray) -> np.ndarray:
        if frame_rgb_or_bgr.ndim != 3:
            return frame_rgb_or_bgr.astype(np.uint8)
        # Agent passes RGB from camera.read_rgb(); OpenCV needs BGR for cvtColor
        bgr = cv2.cvtColor(frame_rgb_or_bgr, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def detect(self, frame_rgb_or_bgr: np.ndarray) -> List[TagObservation]:
        if not self.available or self._det is None or frame_rgb_or_bgr is None:
            return []

        gray = self._to_gray(frame_rgb_or_bgr)
        try:
            results = self._det.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self._cam_params,
                tag_size=self.tag_size_m,
            )
        except Exception as e:  # noqa: BLE001
            self.last_error = repr(e)
            self.last_raw_ids = []
            return []

        self.last_raw_ids = [int(r.tag_id) for r in results]
        out: List[TagObservation] = []
        for r in results:
            if float(r.decision_margin) < self.min_margin:
                continue

            tag_id = int(r.tag_id)
            sign_type = lookup_sign(tag_id, self._tag_table)

            dist_pose = 99.0
            bearing = 0.0
            try:
                x, y, z = (float(v) for v in r.pose_t.flatten())
                dist_pose = float(np.linalg.norm([x, y, z]))
                bearing = math.atan2(x, z)
            except Exception:
                pass

            dist_px = _distance_from_corners(r.corners, self.tag_size_m, self._fx)
            # Prefer sane pose; otherwise use pixel-size geometry
            if 0.05 < dist_pose < 8.0:
                distance = dist_pose
            else:
                distance = dist_px

            out.append(TagObservation(
                tag_id=tag_id,
                sign_type=sign_type,
                distance_m=distance,
                bearing_rad=bearing,
                corners=tuple((float(c[0]), float(c[1])) for c in r.corners),
                center_px=(int(r.center[0]), int(r.center[1])),
            ))

        return out
