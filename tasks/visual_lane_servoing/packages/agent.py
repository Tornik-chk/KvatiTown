import os
import yaml
import numpy as np
import cv2
from collections import deque
from typing import Tuple

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_config.yaml'
))

_LINE_OFFSET = 160
_ROI_START   = 0.47
_NUM_SLICES  = 3
_SLICE_TOL   = 5

# ---------------------------------------------------------------------------
# Red line detection constants
# Red wraps around in HSV — two ranges needed
# ---------------------------------------------------------------------------
_RED_LOWER1 = np.array([0,   80,  80])
_RED_UPPER1 = np.array([10,  255, 255])
_RED_LOWER2 = np.array([170, 80,  80])
_RED_UPPER2 = np.array([179, 255, 255])
# Vertical ROI: only the bottom portion of frame — red strip only appears
# this low when the bot is genuinely close to the stop line.
_RED_ROI_START    = 0.50   # start higher — detect strip while still approaching straight
_RED_ROI_END      = 0.90
_RED_COL_START    = 0.30
_RED_COL_END      = 0.70
_RED_LINE_FRAC    = 0.20
_RED_CONFIRM_N    = 3      # fewer frames needed — fire earlier while still straight
_RED_COOLDOWN_S   = 5.0


def _detect_red_line(frame_bgr: np.ndarray) -> bool:
    """Return True if a red stop line is visible close ahead in the lane center."""
    h, w = frame_bgr.shape[:2]
    # Crop to vertical ROI (close range only) and center horizontal strip
    roi = frame_bgr[
        int(h * _RED_ROI_START): int(h * _RED_ROI_END),
        int(w * _RED_COL_START): int(w * _RED_COL_END),
    ]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, _RED_LOWER1, _RED_UPPER1),
        cv2.inRange(hsv, _RED_LOWER2, _RED_UPPER2),
    )
    roi_w = mask.shape[1]
    for row in range(mask.shape[0]):
        if np.count_nonzero(mask[row]) / roi_w > _RED_LINE_FRAC:
            return True
    return False


def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white:  np.ndarray,
    h: int,
) -> Tuple[list, list]:
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y      = int(h * _ROI_START)
    yellow_xs, white_xs = [], []
    _, w = mask_yellow.shape
    mid  = w // 2

    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2

        # Yellow: take the rightmost cluster in the LEFT half of the strip.
        # This is the inner edge of the yellow dash closest to the lane centre.
        # Using the left half avoids picking up yellow that has swung into the
        # right half on a sharp turn — if it's that far right, use the right
        # half as a fallback but prefer left.
        strip_y = mask_yellow[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx_y   = np.where(strip_y > 0)[1]
        if len(idx_y) > 0:
            left_half  = idx_y[idx_y <  mid]
            right_half = idx_y[idx_y >= mid]
            if len(left_half) > 0:
                yellow_xs.append(int(np.max(left_half)))   # rightmost in left half
            else:
                yellow_xs.append(int(np.min(right_half)))  # leftmost in right half

        # White: take the leftmost cluster in the RIGHT half of the strip.
        # This is the inner edge of the white line closest to the lane centre.
        strip_w = mask_white[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx_w   = np.where(strip_w > 0)[1]
        if len(idx_w) > 0:
            right_half = idx_w[idx_w >= mid]
            left_half  = idx_w[idx_w <  mid]
            if len(right_half) > 0:
                white_xs.append(int(np.min(right_half)))   # leftmost in right half
            else:
                white_xs.append(int(np.max(left_half)))    # rightmost in left half

    return yellow_xs, white_xs


class LaneServoingAgent:

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.p_gain              = cfg.get('p_gain',              0.1)
        self.d_gain              = cfg.get('d_gain',              0.35)
        self.max_steer           = cfg.get('max_steer',           0.4)
        self.base_speed          = cfg.get('base_speed',          0.2)
        self.curve_speed         = cfg.get('curve_speed',         0.2)
        self.curve_threshold     = cfg.get('curve_threshold',     350)
        self.steering_threshold  = cfg.get('steering_threshold',  0.2)
        self.curve_boost         = cfg.get('curve_boost',         1.3)
        self.detection_threshold = cfg.get('detection_threshold', 500)

        self.frame_count        = 0
        self._prev_error        = 0.0
        self._filtered_error    = 0.0
        self._lane_half_width   = float(_LINE_OFFSET)
        self._left_history      = deque(maxlen=3)
        self._right_history     = deque(maxlen=3)
        self.last_debug_info    = self._empty_debug_info(480, 640)

        # Red line state
        self._red_confirm_count  = 0
        self._red_cooldown_until = 0.0
        self.at_red_line         = False   # readable by project agent
        self.approaching_red     = False   # True while count is building

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.mean(yellow_xs))
            w_mean = float(np.mean(white_xs))
            measured = (w_mean - y_mean) / 2.0
            if measured > 20:
                self._lane_half_width = 0.9 * self._lane_half_width + 0.1 * measured
            error = w / 2.0 - (y_mean + w_mean) / 2.0

        elif left_det and yellow_xs:
            error = w / 2.0 - (float(np.mean(yellow_xs)) + self._lane_half_width)

        elif right_det and white_xs:
            error = w / 2.0 - (float(np.mean(white_xs)) - self._lane_half_width)

        else:
            error = self._prev_error

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error: float) -> float:
        error_diff       = error - self._prev_error
        self._prev_error = error
        steering = self.p_gain * error + self.d_gain * error_diff
        return float(np.clip(steering, -self.max_steer, self.max_steer))

    def _motor_commands(self, steering: float, recovery: bool, is_curve: bool, both_visible: bool):
        if recovery:
            speed = self.base_speed * 0.5
            left  = float(np.clip(speed - steering, 0.0, 1.0))
            right = float(np.clip(speed + steering, 0.0, 1.0))
            return left, right

        speed = self.curve_speed if is_curve else self.base_speed

        if not both_visible:
            speed *= 0.85

        left  = speed - steering
        right = speed + steering

        if is_curve and abs(steering) > self.steering_threshold:
            if steering > 0:
                right *= self.curve_boost
            else:
                left  *= self.curve_boost

        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _smooth(self, left, right, both_visible):
        buf = 2 if both_visible else 1
        if self._left_history.maxlen != buf:
            self._left_history  = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)
        self._left_history.append(left)
        self._right_history.append(right)
        return (sum(self._left_history)  / len(self._left_history),
                sum(self._right_history) / len(self._right_history))

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Args:
            image: BGR frame directly from camera (Duckiebot camera_node via
                   cv_bridge / imdecode always delivers BGR — do NOT pre-convert).
        """
        import time
        self.frame_count += 1
        now = time.time()

        # FIX (Bug 2): frame is already BGR from the Duckiebot camera pipeline.
        # The original code did COLOR_RGB2BGR here which swapped red↔blue on
        # the physical bot and broke both red-line and lane-colour detection.
        bgr = image  # no conversion needed

        # ------------------------------------------------------------------
        # Red line detection
        # ------------------------------------------------------------------
        self.at_red_line = False
        if now > self._red_cooldown_until:
            if _detect_red_line(bgr):
                self._red_confirm_count += 1
            else:
                self._red_confirm_count = max(0, self._red_confirm_count - 1)

            if self._red_confirm_count >= _RED_CONFIRM_N:
                self.at_red_line = True
                self.approaching_red = False
                self._red_confirm_count = 0
                print("[LaneAgent] Red line confirmed — stopping.")
                self.last_debug_info['at_red_line'] = True
                return 0.0, 0.0
            # Seeing red but not yet confirmed — flag so project agent can
            # straighten the bot while it's still on the approach.
            self.approaching_red = self._red_confirm_count > 0
        else:
            self._red_confirm_count = 0
            self.approaching_red = False

        self.last_debug_info['at_red_line'] = False

        # ------------------------------------------------------------------
        # Normal lane following
        # ------------------------------------------------------------------
        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print(f"[Agent] detect_lane_markings error: {e}")
            return 0.0, 0.0

        mask_y = (mask_left  * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels  = int(np.count_nonzero(mask_w))
        total_pixels  = yellow_pixels + white_pixels

        combined = np.clip(mask_left + mask_right, 0, 1)
        self.last_debug_info = {
            'roi':               image,
            'lane_mask':         (combined * 255).astype(np.uint8),
            'white_mask':        mask_w,
            'yellow_mask':       mask_y,
            'total_lane_pixels': total_pixels,
            'lateral_error':     float(np.clip(self._prev_error, -1.0, 1.0)),
            'lane_detected':     total_pixels >= self.detection_threshold,
            'frame_count':       self.frame_count,
            'at_red_line':       False,
        }

        h, w      = mask_y.shape
        left_det  = yellow_pixels > 0
        right_det = white_pixels  > 0

        recovery = (not left_det) and (not right_det)

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
        both_visible        = left_det and right_det
        is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)

        raw_error            = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        self._filtered_error = 0.7 * self._filtered_error + 0.3 * raw_error
        steering             = self._calculate_steering(self._filtered_error)
        left, right          = self._motor_commands(steering, recovery, is_curve, both_visible)
        left, right          = self._smooth(left, right, both_visible)

        slice_height = int(h * 0.35 / _NUM_SLICES)
        start_y      = int(h * _ROI_START)
        self.last_debug_info.update({
            'yellow_xs': yellow_xs,
            'white_xs':  white_xs,
            'slice_ys':  [start_y + i * slice_height + slice_height // 2 for i in range(_NUM_SLICES)],
            'is_curve':  is_curve,
            'curve_dir': curve_dir,
        })

        return left, right

    def reset_red_cooldown(self):
        """Call this after completing an intersection so red line is ignored briefly."""
        import time
        self._red_cooldown_until = time.time() + _RED_COOLDOWN_S
        self._red_confirm_count  = 0
        self.at_red_line         = False

    def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def get_debug_info(self, image: np.ndarray) -> dict:
        return self.last_debug_info

    def _empty_debug_info(self, h, w):
        return {
            'roi':               np.zeros((h, w, 3), dtype=np.uint8),
            'lane_mask':         np.zeros((h, w),    dtype=np.uint8),
            'white_mask':        np.zeros((h, w),    dtype=np.uint8),
            'yellow_mask':       np.zeros((h, w),    dtype=np.uint8),
            'total_lane_pixels': 0,
            'lateral_error':     0.0,
            'lane_detected':     False,
            'frame_count':       0,
            'at_red_line':       False,
        }