"""Traffic Signs — final with cooldown for stop/yield and correct T-intersection choices."""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

import yaml

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.object_detection.packages.agent import ObjectDetectionAgent
from tasks.object_detection.packages.stop_activity import should_stop as yolo_should_stop

from .sign_detector import SignDetector, SignType, TagObservation

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG_FILE = os.path.join(_PROJECT_ROOT, "config", "project_config.yaml")
_SIGN_TAG_MAP_PATH = os.path.join(_PROJECT_ROOT, "config", "sign_tag_map.yaml")

# ----------------------------------------------------------------------
# Load tag sets from YAML
# ----------------------------------------------------------------------
def _load_tag_sets():
    try:
        with open(_SIGN_TAG_MAP_PATH) as f:
            data = yaml.safe_load(f) or {}
            tags = data.get("tags", {})
    except Exception:
        tags = {}

    stop_ids = set()
    yield_ids = set()
    intersection_ids = set()
    left_t_ids = set()
    right_t_ids = set()

    for tag_id, stype in tags.items():
        tag_id = int(tag_id)
        stype = str(stype).lower()
        if stype == "stop":
            stop_ids.add(tag_id)
        elif stype == "yield":
            yield_ids.add(tag_id)
        elif stype == "left_t_intersect":
            left_t_ids.add(tag_id)
            intersection_ids.add(tag_id)
        elif stype == "right_t_intersect":
            right_t_ids.add(tag_id)
            intersection_ids.add(tag_id)
        elif stype in ("four_way", "t_intersection",
                       "left_only", "right_only", "straight_only",
                       "no_left", "no_right", "one_way_left", "one_way_right"):
            intersection_ids.add(tag_id)

    return stop_ids, yield_ids, intersection_ids, left_t_ids, right_t_ids

STOP_TAG_IDS, YIELD_TAG_IDS, INTERSECTION_TAG_IDS, LEFT_T_IDS, RIGHT_T_IDS = _load_tag_sets()

def choose_maneuver_for_tag(tag_id: int, sign_type: SignType) -> str:
    """Return maneuver based on sign type (and tag ID for left/right T)."""
    if sign_type == SignType.T_INTERSECTION:
        return random.choice(["left", "right"])
    if sign_type == SignType.LEFT_ONLY:
        return "left"
    if sign_type == SignType.RIGHT_ONLY:
        return "right"
    if sign_type == SignType.STRAIGHT_ONLY:
        return "straight"
    if tag_id in LEFT_T_IDS:
        return random.choice(["left", "straight"])
    if tag_id in RIGHT_T_IDS:
        return random.choice(["right", "straight"])
    # Default for four‑way and others
    return random.choice(["left", "right", "straight"])

# ----------------------------------------------------------------------
# Global status
# ----------------------------------------------------------------------
LATEST_STATUS: dict = {
    "state": "INIT",
    "last_sign": None,
    "chosen_maneuver": None,
    "obstacle": False,
    "tag_count": 0,
}
AGENT_PAUSED: bool = False
Detection = Tuple[Tuple[int, int, int, int], float, int]

# ----------------------------------------------------------------------
# FSM states and config
# ----------------------------------------------------------------------
class State(Enum):
    LANE_FOLLOW = "LANE_FOLLOW"
    APPROACH_SIGN = "APPROACH_SIGN"
    STOP_HOLD = "STOP_HOLD"
    YIELD_HOLD = "YIELD_HOLD"
    TURN_DELAY = "TURN_DELAY"          # forward motion before turning
    EXECUTE_TURN = "EXECUTE_TURN"
    WAIT_RIGHT_OF_WAY = "WAIT_RIGHT_OF_WAY"
    OBSTACLE_STOP = "OBSTACLE_STOP"

@dataclass
class DriveCommand:
    left_pwm: float = 0.0
    right_pwm: float = 0.0
    state_name: str = "INIT"

@dataclass
class FsmInputs:
    tags: List[TagObservation]
    obstacle: bool = False
    lane_cmd: DriveCommand = field(default_factory=DriveCommand)
    now: float = field(default_factory=time.monotonic)

@dataclass
class FsmConfig:
    approach_distance_m: float = 1.2      # start slowing down
    stop_distance_m: float = 0.30
    yield_distance_m: float = 0.25
    intersection_choice_m: float = 0.25   # VERY close to sign before deciding
    stop_hold_seconds: float = 2.0
    yield_hold_seconds: float = 0.5
    turn_delay_seconds: float = 1.0       # move straight for 1 s before turning
    turn_duration_seconds: float = 1.2    # arc turn duration (calibrate for 90°)
    turn_choice_cooldown: float = 3.0
    base_speed: float = 0.20
    turn_speed: float = 0.18
    right_of_way_rule: str = "first_stopped"

def _decide_right_of_way(my_stop_time: Optional[float], other, rule: str):
    return True  # not used

# ----------------------------------------------------------------------
# Behavior FSM
# ----------------------------------------------------------------------
class BehaviorFSM:
    def __init__(self, cfg: Optional[FsmConfig] = None):
        self.cfg = cfg or FsmConfig()
        self.state = State.LANE_FOLLOW
        self._state_entered = time.monotonic()
        self._pending_sign: Optional[TagObservation] = None
        self._chosen: Optional[str] = None
        self._sign_cooldown = 0.0
        self._intersection_cooldown = 0.0

    def _go(self, new_state: State, now: float, log: Optional[Callable[[str], None]] = None):
        if new_state != self.state and log:
            log(f"[FSM] {self.state.name} -> {new_state.name}")
        self.state = new_state
        self._state_entered = now

    def _time_in_state(self, now: float) -> float:
        return now - self._state_entered

    @staticmethod
    def _closest_sign(tags: List[TagObservation]) -> Optional[TagObservation]:
        relevant = [t for t in tags if t.sign_type not in (SignType.UNKNOWN, SignType.DUCKIEBOT)]
        return min(relevant, key=lambda t: t.distance_m) if relevant else None

    def step(self, inp: FsmInputs, log: Optional[Callable[[str], None]] = None) -> DriveCommand:
        s = self.state

        # Global STOP / YIELD check (except when already held or obstacle)
        if s not in (State.STOP_HOLD, State.YIELD_HOLD, State.OBSTACLE_STOP) \
                and inp.now >= self._sign_cooldown:
            for tag in inp.tags:
                if tag.tag_id in STOP_TAG_IDS and tag.distance_m <= self.cfg.stop_distance_m:
                    self._pending_sign = tag
                    self._go(State.STOP_HOLD, inp.now, log)
                    return DriveCommand(0.0, 0.0, s.name)
                if tag.tag_id in YIELD_TAG_IDS and tag.distance_m <= self.cfg.yield_distance_m:
                    self._pending_sign = tag
                    self._go(State.YIELD_HOLD, inp.now, log)
                    return DriveCommand(0.0, 0.0, s.name)

        # --- STATE MACHINE ---
        if s == State.OBSTACLE_STOP:
            if not inp.obstacle:
                self._go(State.LANE_FOLLOW, inp.now, log)
            return DriveCommand(0.0, 0.0, s.name)

        if s == State.LANE_FOLLOW:
            if inp.now >= self._intersection_cooldown:
                sign = self._closest_sign(inp.tags)
                if sign and sign.tag_id in INTERSECTION_TAG_IDS and sign.distance_m <= self.cfg.approach_distance_m:
                    self._pending_sign = sign
                    self._go(State.APPROACH_SIGN, inp.now, log)
            return DriveCommand(inp.lane_cmd.left_pwm, inp.lane_cmd.right_pwm, s.name)

        if s == State.APPROACH_SIGN:
            sign = self._closest_sign(inp.tags) if inp.tags else None
            if sign is None:
                # Sign lost – abort approach
                self._go(State.LANE_FOLLOW, inp.now, log)
                return DriveCommand(inp.lane_cmd.left_pwm, inp.lane_cmd.right_pwm, s.name)

            self._pending_sign = sign

            # Slow down while approaching
            speed_factor = 0.6
            left_cmd = inp.lane_cmd.left_pwm * speed_factor
            right_cmd = inp.lane_cmd.right_pwm * speed_factor

            # Only decide when the sign is truly close
            if sign.distance_m <= self.cfg.intersection_choice_m and self._chosen is None:
                chosen = choose_maneuver_for_tag(sign.tag_id, sign.sign_type)
                self._chosen = chosen
                if log:
                    log(f"[FSM] Chose {chosen} at tag {sign.tag_id} (dist {sign.distance_m:.3f}m)")
                self._intersection_cooldown = inp.now + self.cfg.turn_choice_cooldown
                self._pending_sign = None
                self._go(State.TURN_DELAY, inp.now, log)
                return DriveCommand(self.cfg.base_speed, self.cfg.base_speed, s.name)
            else:
                return DriveCommand(left_cmd, right_cmd, s.name)

        if s == State.TURN_DELAY:
            # Move straight forward for turn_delay_seconds
            if self._time_in_state(inp.now) < self.cfg.turn_delay_seconds:
                return DriveCommand(self.cfg.base_speed, self.cfg.base_speed, s.name)
            self._go(State.EXECUTE_TURN, inp.now, log)
            left_pwm, right_pwm = self._turn_wheels()
            return DriveCommand(left_pwm, right_pwm, s.name)

        if s == State.EXECUTE_TURN:
            left_pwm, right_pwm = self._turn_wheels()
            if self._time_in_state(inp.now) < self.cfg.turn_duration_seconds:
                return DriveCommand(left_pwm, right_pwm, s.name)
            self._chosen = None
            self._go(State.LANE_FOLLOW, inp.now, log)
            return DriveCommand(inp.lane_cmd.left_pwm, inp.lane_cmd.right_pwm, s.name)

        if s == State.STOP_HOLD:
            if self._time_in_state(inp.now) < self.cfg.stop_hold_seconds:
                return DriveCommand(0.0, 0.0, s.name)
            self._sign_cooldown = inp.now + 2.0
            self._chosen = None
            self._pending_sign = None
            self._go(State.LANE_FOLLOW, inp.now, log)
            return DriveCommand(inp.lane_cmd.left_pwm, inp.lane_cmd.right_pwm, s.name)

        if s == State.YIELD_HOLD:
            if self._time_in_state(inp.now) < self.cfg.yield_hold_seconds:
                return DriveCommand(0.0, 0.0, s.name)
            self._sign_cooldown = inp.now + 1.0
            self._pending_sign = None
            self._go(State.LANE_FOLLOW, inp.now, log)
            return DriveCommand(inp.lane_cmd.left_pwm, inp.lane_cmd.right_pwm, s.name)

        # Fallback
        return DriveCommand(0.0, 0.0, s.name)

    def _turn_wheels(self) -> Tuple[float, float]:
        """Return (left_pwm, right_pwm) for the current chosen maneuver."""
        if self._chosen == "left":
            return (self.cfg.base_speed - self.cfg.turn_speed,
                    self.cfg.base_speed + self.cfg.turn_speed)
        elif self._chosen == "right":
            return (self.cfg.base_speed + self.cfg.turn_speed,
                    self.cfg.base_speed - self.cfg.turn_speed)
        else:  # straight
            return (self.cfg.base_speed, self.cfg.base_speed)

# ----------------------------------------------------------------------
# Obstacle detector (unchanged)
# ----------------------------------------------------------------------
class ObstacleDetector:
    def __init__(self) -> None:
        self._agent = ObjectDetectionAgent()
        self._last: List[Detection] = []

    @property
    def model_loaded(self) -> bool:
        return self._agent.model_loaded

    @property
    def load_error(self):
        return self._agent.load_error

    def step(self, frame_rgb) -> List[Detection]:
        result = self._agent.detect(frame_rgb)
        if result is not None:
            self._last = result
        return self._last

    def should_stop(self):
        non_sign = [d for d in self._last if d[2] != 2]
        return yolo_should_stop(non_sign, self._agent.img_size)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def _load_cfg() -> dict:
    try:
        with open(_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _build_fsm_cfg(raw: dict) -> FsmConfig:
    return FsmConfig(
        approach_distance_m=float(raw.get("approach_distance_m", 1.2)),
        stop_distance_m=float(raw.get("stop_distance_m", 0.30)),
        yield_distance_m=float(raw.get("yield_distance_m", 0.25)),
        intersection_choice_m=float(raw.get("intersection_choice_m", 0.25)),
        stop_hold_seconds=float(raw.get("stop_hold_seconds", 2.0)),
        yield_hold_seconds=float(raw.get("yield_hold_seconds", 0.5)),
        turn_delay_seconds=float(raw.get("turn_delay_seconds", 1.0)),
        turn_duration_seconds=float(raw.get("turn_duration_seconds", 1.2)),
        turn_choice_cooldown=float(raw.get("turn_choice_cooldown", 3.0)),
        base_speed=float(raw.get("base_speed", 0.20)),
        turn_speed=float(raw.get("turn_speed", 0.18)),
    )

_STATE_COLOR = {
    State.STOP_HOLD: [1.0, 0.0, 0.0],
    State.YIELD_HOLD: [1.0, 0.7, 0.0],
}
_DEFAULT_LED = [0.0, 0.7, 0.0]

def _set_led_state_color(leds, state: State):
    color = _STATE_COLOR.get(state, _DEFAULT_LED)
    for led in (0, 2, 3, 4):
        leds.set_rgb(led, color)

def main(camera, wheels, leds, stop_event):
    cfg = _load_cfg()
    print("[Project] Initialising agents...")
    lane_agent = LaneServoingAgent()
    sign_det = SignDetector(
        tag_family=cfg.get("tag_family", "tag36h11"),
        tag_size_m=float(cfg.get("tag_size_m", 0.065)),
    )
    obstacle_det = ObstacleDetector()
    fsm = BehaviorFSM(_build_fsm_cfg(cfg))

    detection_skip = int(cfg.get("detection_skip", 1))
    frame_count = 0

    print("[Project] Entering main loop")
    try:
        while not stop_event.is_set():
            ok, frame_bgr = camera.read()
            if not ok or frame_bgr is None:
                time.sleep(0.01)
                continue
            try:
                ok_rgb, frame_rgb = camera.read_rgb()
                if not ok_rgb or frame_rgb is None:
                    frame_rgb = frame_bgr[..., ::-1]
            except Exception:
                frame_rgb = frame_bgr[..., ::-1]

            frame_count += 1

            try:
                pwm_l, pwm_r = lane_agent.compute_commands(frame_rgb)
            except Exception as e:
                pwm_l, pwm_r = 0.0, 0.0
                print(f"[Project] Lane agent error: {e}")

            tags = []
            if detection_skip <= 1 or (frame_count % (detection_skip + 1) == 0):
                tags = sign_det.detect(frame_rgb)
                if tags:
                    for t in tags:
                        print(f"[TAG] ID={t.tag_id} type={t.sign_type.value} dist={t.distance_m:.3f}m")
                obstacle_det.step(frame_rgb)
            stop_flag, _ = obstacle_det.should_stop()

            cmd = fsm.step(
                FsmInputs(
                    tags=tags,
                    obstacle=stop_flag,
                    lane_cmd=DriveCommand(pwm_l, pwm_r, "LANE"),
                    now=time.monotonic(),
                ),
                log=print,
            )

            if not AGENT_PAUSED:
                try:
                    if not wheels.is_game_over():
                        wheels.set_wheels_speed(cmd.left_pwm, cmd.right_pwm)
                except AttributeError:
                    wheels.set_wheels_speed(cmd.left_pwm, cmd.right_pwm)

            closest = min(tags, key=lambda t: t.distance_m) if tags else None
            LATEST_STATUS["state"] = cmd.state_name
            LATEST_STATUS["last_sign"] = closest.sign_type.value if closest else None
            LATEST_STATUS["chosen_maneuver"] = fsm._chosen
            LATEST_STATUS["obstacle"] = stop_flag
            LATEST_STATUS["tag_count"] = len(tags)

            if leds is not None:
                try:
                    _set_led_state_color(leds, fsm.state)
                except Exception:
                    pass

    finally:
        try:
            wheels.set_wheels_speed(0.0, 0.0)
        except Exception:
            pass
        if leds is not None:
            try:
                leds.all_off()
            except Exception:
                pass
        print("[Project] Stopped cleanly.")