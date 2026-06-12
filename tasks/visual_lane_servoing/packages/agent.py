import os
import time
import yaml
import numpy as np
import cv2
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


def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white:  np.ndarray,
    h: int,
) -> Tuple[list, list]:
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y      = int(h * _ROI_START)
    yellow_xs, white_xs = [], []

    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2

        strip_y = mask_yellow[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_y > 0)[1]
        if len(idx) > 0:
            yellow_xs.append(int(np.mean(idx)))

        strip_w = mask_white[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_w > 0)[1]
        if len(idx) > 0:
            white_xs.append(int(np.mean(idx)))

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
        self.max_steer           = cfg.get('max_steer',           0.38)
        self.base_speed          = cfg.get('base_speed',          0.21)
        self.min_cruise_speed    = cfg.get('min_cruise_speed',    0.17)
        self.min_wheel_speed     = cfg.get('min_wheel_speed',     0.08)
        self.turn_speed_ratio    = cfg.get('turn_speed_ratio',    0.12)
        self.curve_feedforward   = cfg.get('curve_feedforward',   0.12)
        self.curve_threshold     = cfg.get('curve_threshold',     350)
        self.detection_threshold = cfg.get('detection_threshold', 500)
        self.smooth_alpha        = cfg.get('smooth_alpha',        0.6)
        self.steer_smooth        = cfg.get('steer_smooth',        0.6)

        self.frame_count        = 0
        self._prev_error        = 0.0
        self._filtered_error    = 0.0
        self._filtered_steering = 0.0
        self._smooth_left       = None
        self._smooth_right      = None
        self._lane_half_width   = float(_LINE_OFFSET)
        self._white_lane = []
        self.last_debug_info    = self._empty_debug_info(480, 640)

        # Left-turn state machine: triggered when yellow disappears (intersection)
        # Phase 'straight' – drive forward for one lane width
        # Phase 'turning'  – hard left until white line reappears (or timeout)
        self._yellow_visible_frames  = 0
        self._left_turn_state        = 'none'   # 'none' | 'straight' | 'turning'
        self._left_turn_start        = 0.0
        self._left_turn_cooldown_end = 0.0
        self._left_straight_duration = cfg.get('left_straight_duration', 1.1)
        self._left_turn_max_duration = cfg.get('left_turn_max_duration',  2.5)
        self._left_straight_speed    = cfg.get('left_straight_speed',     0.23)
        self._left_turn_wheel_inner  = cfg.get('left_turn_wheel_inner',   0.07)
        self._left_turn_wheel_outer  = cfg.get('left_turn_wheel_outer',   0.26)


    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.mean(yellow_xs))
            w_mean = float(np.mean(white_xs))

            # White to the left of yellow → wrong side (intersection / oncoming lane).
            # Discard white and follow yellow only.
            if w_mean <= y_mean:
                error = w / 2.0 - (y_mean + self._lane_half_width)
                return float(np.clip(error / (w / 2.0), -1.0, 1.0))

            # Both valid: update lane half-width and track the centre.
            measured = (w_mean - y_mean) / 2.0
            if measured > 20:
                self._lane_half_width = 0.9 * self._lane_half_width + 0.1 * measured
            error = w / 2.0 - (y_mean + w_mean) / 2.0

        elif left_det and yellow_xs:
            # Yellow only: keep yellow at lane_half_width to the left of centre.
            error = w / 2.0 - (float(np.mean(yellow_xs)) + self._lane_half_width)

        elif right_det and white_xs:
            # White only: keep white at lane_half_width to the right of centre.
            error = w / 2.0 - (float(np.mean(white_xs)) - self._lane_half_width)

        else:
            error = self._prev_error

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error: float) -> float:
        error_diff       = error - self._prev_error
        self._prev_error = error
        steering = self.p_gain * error + self.d_gain * error_diff
        return float(np.clip(steering, -self.max_steer, self.max_steer))

    def _cruise_speed(self, steering: float, is_curve: bool, both_visible: bool) -> float:
        return self.base_speed

    def _apply_wheel_floor(self, left: float, right: float) -> Tuple[float, float]:
        min_val = min(left, right)
        if min_val < self.min_wheel_speed:
            shift = self.min_wheel_speed - min_val
            left += shift
            right += shift

        peak = max(left, right)
        if peak > 1.0:
            scale = 1.0 / peak
            left *= scale
            right *= scale

        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _motor_commands(
        self,
        steering: float,
        recovery: bool,
        both_visible: bool,
        is_curve: bool,
    ):
        speed = self._cruise_speed(steering, is_curve, both_visible)
        diff = float(np.clip(steering, -self.max_steer, self.max_steer))

        left  = speed - diff
        right = speed + diff

        return self._apply_wheel_floor(left, right)

    def _smooth(self, left: float, right: float) -> Tuple[float, float]:
        alpha = self.smooth_alpha
        if self._smooth_left is None:
            self._smooth_left  = left
            self._smooth_right = right
        else:
            self._smooth_left  = alpha * left  + (1.0 - alpha) * self._smooth_left
            self._smooth_right = alpha * right + (1.0 - alpha) * self._smooth_right
        return self._smooth_left, self._smooth_right

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        self.frame_count += 1
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        now = time.monotonic()

        # ── 1. Lane detection ────────────────────────────────────────────────
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

        h, w      = mask_y.shape
        left_det  = yellow_pixels > 0
        right_det = white_pixels  > 0

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)

        yellow_slice_count = len(yellow_xs)
        white_slice_count  = len(white_xs)

        # ── Always update debug visualizations ───────────────────────────────
        combined = np.clip(mask_left + mask_right, 0, 1)
        slice_height = int(h * 0.35 / _NUM_SLICES)
        start_y      = int(h * _ROI_START)
        self.last_debug_info = {
            'roi':               image,
            'lane_mask':         (combined * 255).astype(np.uint8),
            'white_mask':        mask_w,
            'yellow_mask':       mask_y,
            'red_mask':          np.zeros((h, w), dtype=np.uint8),
            'red_px':            0,
            'red_line':          False,
            'total_lane_pixels': total_pixels,
            'lateral_error':     float(np.clip(self._prev_error, -1.0, 1.0)),
            'lane_detected':     total_pixels >= self.detection_threshold,
            'frame_count':       self.frame_count,
            'yellow_xs':         yellow_xs,
            'white_xs':          white_xs,
            'slice_ys':          [start_y + i * slice_height + slice_height // 2 for i in range(_NUM_SLICES)],
            'is_curve':          False,
            'curve_dir':         0,
        }

        if yellow_xs and white_xs:
            self._white_lane = white_xs
        else:
            self._white_lane = []

        # ── Yellow-end tracker (intersection detection) ───────────────────────
        _YELLOW_MIN_FRAMES = 8
        if yellow_slice_count > 0:
            self._yellow_visible_frames = min(self._yellow_visible_frames + 1, 999)
        else:
            if (self._yellow_visible_frames >= _YELLOW_MIN_FRAMES
                    and self._left_turn_state == 'none'
                    and now >= self._left_turn_cooldown_end):
                self._left_turn_state = 'straight'
                self._left_turn_start = now
                print("[Agent] Yellow gone — left turn: driving straight")
            self._yellow_visible_frames = 0

        # ── Left-turn state machine ───────────────────────────────────────────
        if self._left_turn_state == 'straight':
            elapsed = now - self._left_turn_start
            if elapsed < self._left_straight_duration:
                s = self._left_straight_speed
                return s, s
            self._left_turn_state = 'turning'
            self._left_turn_start = now
            print("[Agent] Left turn: now turning")

        if self._left_turn_state == 'turning':
            elapsed = now - self._left_turn_start
            # Exit: white line reappears OR hard timeout
            white_reappeared = white_slice_count >= 2
            timed_out        = elapsed >= self._left_turn_max_duration
            if white_reappeared or timed_out:
                self._left_turn_state        = 'none'
                self._left_turn_cooldown_end = now + 3.0
                self._yellow_visible_frames  = 0
                reason = "white reappeared" if white_reappeared else "timeout"
                print(f"[Agent] Left turn done ({reason}) — resuming lane follow")
            else:
                return self._left_turn_wheel_inner, self._left_turn_wheel_outer
        # ─────────────────────────────────────────────────────────────────────

        recovery  = total_pixels < self.detection_threshold

        # White to the left of yellow means wrong-side detection — treat as yellow-only.
        white_on_wrong_side = (
            left_det and right_det
            and yellow_xs and white_xs
            and float(np.mean(white_xs)) <= float(np.mean(yellow_xs))
        )
        effective_right_det = right_det and not white_on_wrong_side
        both_visible        = left_det and effective_right_det and not recovery

        # Curve detection: prefer yellow when white is discarded.
        is_curve, curve_dir = detect_curve(
            yellow_xs,
            white_xs if not white_on_wrong_side else [],
            self.curve_threshold,
        )

        raw_error            = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        self._filtered_error = 0.7 * self._filtered_error + 0.3 * raw_error
        steering = self._calculate_steering(self._filtered_error)
        if is_curve:
            steering -= curve_dir * self.curve_feedforward
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))
        self._filtered_steering = (
            self.steer_smooth * steering
            + (1.0 - self.steer_smooth) * self._filtered_steering
        )
        left, right = self._motor_commands(
            self._filtered_steering, recovery, both_visible, is_curve,
        )
        left, right = self._smooth(left, right)

        self.last_debug_info.update({
            'is_curve':  is_curve,
            'curve_dir': curve_dir,
        })

        return left, right

    def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def reset(self):
        self.frame_count             = 0
        self._prev_error             = 0.0
        self._filtered_error         = 0.0
        self._filtered_steering      = 0.0
        self._smooth_left            = None
        self._smooth_right           = None
        self._lane_half_width        = float(_LINE_OFFSET)
        self._yellow_visible_frames  = 0
        self._left_turn_state        = 'none'
        self._left_turn_start        = 0.0
        self._left_turn_cooldown_end = 0.0
        self._white_lane = []
        print("[Agent] State reset")

    def get_debug_info(self, image: np.ndarray) -> dict:
        return self.last_debug_info



    def _empty_debug_info(self, h, w):
        return {
            'roi':               np.zeros((h, w, 3), dtype=np.uint8),
            'lane_mask':         np.zeros((h, w),    dtype=np.uint8),
            'white_mask':        np.zeros((h, w),    dtype=np.uint8),
            'yellow_mask':       np.zeros((h, w),    dtype=np.uint8),
            'red_mask':          np.zeros((h, w),    dtype=np.uint8),
            'red_px':            0,
            'red_line':          False,
            'total_lane_pixels': 0,
            'lateral_error':     0.0,
            'lane_detected':     False,
            'frame_count':       0,
        }