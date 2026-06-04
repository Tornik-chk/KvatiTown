__future__ import annotations

import time
import threading
import cv2
import numpy as np
from enum import Enum, auto
from typing import Optional

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.project.packages.sign_detector import SignDetector, SignAction


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# How long to pause at a STOP sign before checking cross-traffic (seconds)
_STOP_PAUSE_S = 2.0

# How long to wait at YIELD before assuming road is clear (seconds)
_YIELD_TIMEOUT_S = 4.0

# Speed multiplier while SLOW sign is active
_SLOW_FACTOR = 0.6

# How long SLOW stays active after the sign leaves frame (seconds)
_SLOW_DURATION_S = 3.0

# Smooth deceleration / acceleration steps
_RAMP_STEPS = 10
_RAMP_STEP_INTERVAL = 0.05   # seconds between steps → 0.5s total ramp

# Cross-traffic motion detection: bounding box must shift this many pixels
# between frames to count as a moving robot
_MOTION_THRESHOLD_PX = 15

# Minimum object detection score to count as a robot in intersection
_ROBOT_SCORE_THRESHOLD = 0.55

# How many consecutive clear frames before we consider intersection empty
_CLEAR_FRAMES_NEEDED = 8

# Object detection: class ID for truck/robot
_ROBOT_CLASS_ID = 1

# Sign must be seen this many consecutive frames before acting on it
# (prevents acting on a single noisy detection)
_SIGN_CONFIRM_FRAMES = 3

# After resuming from a stop, ignore signs for this many seconds
# (prevents immediately re-triggering on the same sign)
_SIGN_COOLDOWN_S = 4.0


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class State(Enum):
    LANE_FOLLOWING = auto()
    SIGN_SEEN      = auto()
    DECELERATING   = auto()
    WAITING        = auto()
    RESUMING       = auto()


# ---------------------------------------------------------------------------
# LED helpers
# ---------------------------------------------------------------------------

def _leds_normal(leds):
    if not leds:
        return
    leds.set_rgb(0, [0.0, 1.0, 0.0])   # front: green
    leds.set_rgb(2, [0.0, 1.0, 0.0])
    leds.set_rgb(3, [0.0, 0.0, 0.0])   # back: off
    leds.set_rgb(4, [0.0, 0.0, 0.0])


def _leds_stopping(leds):
    if not leds:
        return
    for idx in (0, 2, 3, 4):
        leds.set_rgb(idx, [1.0, 0.0, 0.0])   # all red


def _leds_waiting(leds):
    if not leds:
        return
    for idx in (0, 2, 3, 4):
        leds.set_rgb(idx, [1.0, 0.6, 0.0])   # all amber


def _leds_slow(leds):
    if not leds:
        return
    leds.set_rgb(0, [1.0, 1.0, 0.0])   # front: yellow
    leds.set_rgb(2, [1.0, 1.0, 0.0])
    leds.set_rgb(3, [0.0, 0.0, 0.0])
    leds.set_rgb(4, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Cross-traffic motion detector
# ---------------------------------------------------------------------------

class MotionYieldMonitor:
    """
    Tracks whether a robot is moving in the intersection using bounding box
    displacement between frames. Yields while motion is detected.
    """

    def __init__(self):
        self._prev_bbox: Optional[tuple] = None
        self._clear_count = 0

    def reset(self):
        self._prev_bbox   = None
        self._clear_count = 0

    def is_clear(self, detections: list, img_w: int) -> bool:
        """
        Returns True when intersection has been clear for _CLEAR_FRAMES_NEEDED frames.
        detections: list of ((x1,y1,x2,y2), score, class_id)
        """
        robots = [
            d for d in detections
            if d[2] == _ROBOT_CLASS_ID and d[1] >= _ROBOT_SCORE_THRESHOLD
        ]

        if not robots:
            self._clear_count += 1
            self._prev_bbox = None
            return self._clear_count >= _CLEAR_FRAMES_NEEDED

        # Robot present — check if it's moving
        bbox = robots[0][0]   # (x1, y1, x2, y2)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        moving = False
        if self._prev_bbox is not None:
            pcx = (self._prev_bbox[0] + self._prev_bbox[2]) / 2.0
            pcy = (self._prev_bbox[1] + self._prev_bbox[3]) / 2.0
            dist = np.hypot(cx - pcx, cy - pcy)
            moving = dist > _MOTION_THRESHOLD_PX

        self._prev_bbox   = bbox
        self._clear_count = 0   # robot present, reset clear counter

        # If robot is stationary and not blocking center, tentatively clear
        # (parked robot on side — don't wait forever)
        center_x = img_w / 2.0
        near_center = abs(cx - center_x) < img_w * 0.35
        if not moving and not near_center:
            self._clear_count += 1
            return self._clear_count >= _CLEAR_FRAMES_NEEDED

        return False   # moving robot or stationary robot blocking center


# ---------------------------------------------------------------------------
# Smooth speed ramp
# ---------------------------------------------------------------------------

def _ramp(wheels, from_l, from_r, to_l, to_r):
    for i in range(1, _RAMP_STEPS + 1):
        t = i / _RAMP_STEPS
        l = from_l + t * (to_l - from_l)
        r = from_r + t * (to_r - from_r)
        wheels.set_wheels_speed(l, r)
        time.sleep(_RAMP_STEP_INTERVAL)


# ---------------------------------------------------------------------------
# Object detection thread
# ---------------------------------------------------------------------------

class DetectionThread:
    """
    Runs object detection in a background thread so it doesn't block the
    main control loop. Access latest detections via .detections property.
    """

    def __init__(self):
        self._detections = []
        self._lock       = threading.Lock()
        self._frame      = None
        self._frame_lock = threading.Lock()
        self._stop       = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True)

        # Lazy-import object detection to avoid hard dependency
        try:
            from tasks.object_detection.packages import object_detection_activity as oda
            self._oda = oda
            self._enabled = True
        except ImportError:
            print("[DetectionThread] Object detection unavailable.")
            self._enabled = False

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def push_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self._frame = frame

    @property
    def detections(self) -> list:
        with self._lock:
            return list(self._detections)

    def _run(self):
        if not self._enabled:
            return
        while not self._stop.is_set():
            frame = None
            with self._frame_lock:
                if self._frame is not None:
                    frame = self._frame.copy()
                    self._frame = None

            if frame is None:
                time.sleep(0.02)
                continue

            try:
                dets = self._oda.run_detection(frame)
                with self._lock:
                    self._detections = dets
            except Exception as e:
                print(f"[DetectionThread] error: {e}")

            time.sleep(0.05)   # ~20 Hz max



def main(camera, wheels, leds, stop_event):
    NotImplemented