"""Traffic Signs — main loop (single agent file for the team repo).

Only local import: sign_detector.py
Also uses course modules: visual_lane_servoing, object_detection (YOLO obstacles).

Maps / simulation / UI do NOT import other files from packages/ — only this module
via servers/project/virtual_server.py (needs LATEST_STATUS + AGENT_PAUSED here).
"""
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

LATEST_STATUS: dict = {
    "state": "INIT",
    "last_sign": None,
    "chosen_maneuver": None,
    "obstacle": False,
    "tag_count": 0,
}
AGENT_PAUSED: bool = False

Detection = Tuple[Tuple[int, int, int, int], float, int]


# ---------------------------------------------------------------------------
# Types + FSM (merged from sign_types / behavior_fsm / intersection / ROW)
# ---------------------------------------------------------------------------
class Maneuver(Enum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class DriveCommand:
    left_pwm: float = 0.0
    right_pwm: float = 0.0
    state_name: str = "INIT"


class State(Enum):
    LANE_FOLLOW = "LANE_FOLLOW"
    APPROACH_SIGN = "APPROACH_SIGN"
    STOP_HOLD = "STOP_HOLD"
    YIELD_HOLD = "YIELD_HOLD"
    WAIT_RIGHT_OF_WAY = "WAIT_RIGHT_OF_WAY"
    EXECUTE_TURN = "EXECUTE_TURN"
    OBSTACLE_STOP = "OBSTACLE_STOP"
    DONE = "DONE"


class Priority(Enum):
    GO = "go"
    WAIT = "wait"


@dataclass
class OtherBotState:
    present: bool = False
    stopped: bool = False
    on_my_right: bool = False
    stopped_at: Optional[float] = None


_INTERSECTION_SIGNS = {
    SignType.FOUR_WAY: [Maneuver.STRAIGHT, Maneuver.LEFT, Maneuver.RIGHT],
    SignType.T_INTERSECTION: [Maneuver.LEFT, Maneuver.RIGHT],
    SignType.LEFT_ONLY: [Maneuver.LEFT],
    SignType.RIGHT_ONLY: [Maneuver.RIGHT],
    SignType.STRAIGHT_ONLY: [Maneuver.STRAIGHT],
    SignType.NO_LEFT: [Maneuver.STRAIGHT, Maneuver.RIGHT],
    SignType.NO_RIGHT: [Maneuver.STRAIGHT, Maneuver.LEFT],
    SignType.ONE_WAY_LEFT: [Maneuver.LEFT],
    SignType.ONE_WAY_RIGHT: [Maneuver.RIGHT],
}


def _choose_maneuver(sign: SignType) -> Maneuver:
    options = _INTERSECTION_SIGNS.get(sign, [Maneuver.STRAIGHT])
    return random.choice(options)


def _is_intersection_sign(sign: SignType) -> bool:
    return sign in _INTERSECTION_SIGNS


def _decide_right_of_way(my_stop_time: Optional[float], other: OtherBotState, rule: str) -> Priority:
    if not other.present:
        return Priority.GO
    if rule == "right_priority":
        return Priority.WAIT if other.on_my_right else Priority.GO
    if my_stop_time is None:
        return Priority.WAIT
    if other.stopped_at is None or my_stop_time < other.stopped_at:
        return Priority.GO
    if my_stop_time == other.stopped_at:
        return Priority.WAIT if other.on_my_right else Priority.GO
    return Priority.WAIT


@dataclass
class FsmInputs:
    tags: List[TagObservation]
    obstacle: bool = False
    other_bot: OtherBotState = field(default_factory=OtherBotState)
    lane_cmd: DriveCommand = field(default_factory=DriveCommand)
    now: float = field(default_factory=time.monotonic)


@dataclass
class FsmConfig:
    approach_distance_m: float = 0.50
    stop_distance_m: float = 0.18
    intersection_choice_m: float = 0.30
    stop_hold_seconds: float = 2.0
    yield_hold_seconds: float = 0.5
    turn_seconds: float = 1.6
    turn_speed: float = 0.18
    right_of_way_rule: str = "first_stopped"


class BehaviorFSM:
    def __init__(self, cfg: Optional[FsmConfig] = None):
        self.cfg = cfg or FsmConfig()
        self.state = State.LANE_FOLLOW
        self._state_entered = time.monotonic()
        self._stopped_at: Optional[float] = None
        self._pending_sign: Optional[TagObservation] = None
        self._chosen: Optional[Maneuver] = None

    def _go(self, new_state: State, now: float, log: Optional[Callable[[str], None]] = None):
        if new_state != self.state and log:
            log(f"[FSM] {self.state.name} -> {new_state.name}")
        self.state = new_state
        self._state_entered = now

    def _time_in_state(self, now: float) -> float:
        return now - self._state_entered

    @staticmethod
    def _closest_sign(tags: List[TagObservation]) -> Optional[TagObservation]:
        relevant = [
            t for t in tags
            if t.sign_type not in (SignType.UNKNOWN, SignType.DUCKIEBOT)
        ]
        return min(relevant, key=lambda t: t.distance_m) if relevant else None

    def step(self, inp: FsmInputs, log: Optional[Callable[[str], None]] = None) -> DriveCommand:
        if inp.obstacle and self.state != State.WAIT_RIGHT_OF_WAY:
            self._go(State.OBSTACLE_STOP, inp.now, log)
        s = self.state

        if s == State.OBSTACLE_STOP:
            if not inp.obstacle:
                self._go(State.LANE_FOLLOW, inp.now, log)
            return DriveCommand(0.0, 0.0, s.name)

        if s == State.LANE_FOLLOW:
            sign = self._closest_sign(inp.tags)
            if sign and sign.distance_m <= self.cfg.approach_distance_m:
                self._pending_sign = sign
                self._go(State.APPROACH_SIGN, inp.now, log)
            return DriveCommand(inp.lane_cmd.left_pwm, inp.lane_cmd.right_pwm, s.name)

        if s == State.APPROACH_SIGN:
            sign = self._pending_sign
            if sign is None:
                self._go(State.LANE_FOLLOW, inp.now, log)
                return DriveCommand(0.0, 0.0, s.name)
            if sign.sign_type == SignType.STOP and sign.distance_m <= self.cfg.stop_distance_m:
                self._stopped_at = inp.now
                self._go(State.STOP_HOLD, inp.now, log)
                return DriveCommand(0.0, 0.0, s.name)
            if sign.sign_type == SignType.YIELD and sign.distance_m <= self.cfg.stop_distance_m:
                self._go(State.YIELD_HOLD, inp.now, log)
                return DriveCommand(0.0, 0.0, s.name)
            if _is_intersection_sign(sign.sign_type):
                if sign.distance_m <= self.cfg.intersection_choice_m and self._chosen is None:
                    self._chosen = _choose_maneuver(sign.sign_type)
                    if log:
                        log(f"[FSM] Chose {self._chosen.name} at {sign.sign_type.name}")
                    self._go(State.EXECUTE_TURN, inp.now, log)
                    return DriveCommand(0.0, 0.0, s.name)
            return DriveCommand(inp.lane_cmd.left_pwm * 0.6, inp.lane_cmd.right_pwm * 0.6, s.name)

        if s == State.STOP_HOLD:
            if self._time_in_state(inp.now) < self.cfg.stop_hold_seconds:
                return DriveCommand(0.0, 0.0, s.name)
            self._go(State.WAIT_RIGHT_OF_WAY, inp.now, log)
            return DriveCommand(0.0, 0.0, s.name)

        if s == State.YIELD_HOLD:
            if self._time_in_state(inp.now) < self.cfg.yield_hold_seconds:
                return DriveCommand(0.0, 0.0, s.name)
            self._go(State.WAIT_RIGHT_OF_WAY, inp.now, log)
            return DriveCommand(0.0, 0.0, s.name)

        if s == State.WAIT_RIGHT_OF_WAY:
            if _decide_right_of_way(self._stopped_at, inp.other_bot, self.cfg.right_of_way_rule) == Priority.GO:
                self._pending_sign = None
                self._stopped_at = None
                self._go(State.LANE_FOLLOW, inp.now, log)
                return DriveCommand(inp.lane_cmd.left_pwm, inp.lane_cmd.right_pwm, s.name)
            return DriveCommand(0.0, 0.0, s.name)

        if s == State.EXECUTE_TURN:
            v = self.cfg.turn_speed
            m = self._chosen
            if m == Maneuver.LEFT:
                left, right = -v * 0.4, v
            elif m == Maneuver.RIGHT:
                left, right = v, -v * 0.4
            else:
                left, right = v, v
            if self._time_in_state(inp.now) >= self.cfg.turn_seconds:
                self._chosen = None
                self._pending_sign = None
                self._go(State.LANE_FOLLOW, inp.now, log)
            return DriveCommand(left, right, s.name)

        return DriveCommand(0.0, 0.0, s.name)


# ---------------------------------------------------------------------------
# Obstacle (merged from obstacle.py — still uses object_detection task)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main entry (called by virtual_server / real_server)
# ---------------------------------------------------------------------------
def _load_cfg() -> dict:
    try:
        with open(_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _build_fsm_cfg(raw: dict) -> FsmConfig:
    return FsmConfig(
        approach_distance_m=float(raw.get("approach_distance_m", 0.50)),
        stop_distance_m=float(raw.get("stop_distance_m", 0.18)),
        intersection_choice_m=float(raw.get("intersection_choice_m", 0.30)),
        stop_hold_seconds=float(raw.get("stop_hold_seconds", 2.0)),
        yield_hold_seconds=float(raw.get("yield_hold_seconds", 0.5)),
        turn_speed=float(raw.get("turn_speed", 0.18)),
        right_of_way_rule=str(raw.get("right_of_way_rule", "first_stopped")),
    )


_STATE_COLOR = {
    State.STOP_HOLD: [1.0, 0.0, 0.0],
    State.YIELD_HOLD: [1.0, 0.7, 0.0],
    State.EXECUTE_TURN: [0.0, 0.4, 1.0],
    State.OBSTACLE_STOP: [1.0, 0.0, 0.0],
    State.WAIT_RIGHT_OF_WAY: [1.0, 0.5, 0.0],
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

    if not sign_det.available:
        print(f"[Project] Sign detector unavailable: {sign_det.last_error}")
    if not obstacle_det.model_loaded:
        print(f"[Project] Obstacle model not loaded: {obstacle_det.load_error}")

    detection_skip = int(cfg.get("detection_skip", 1))
    frame_count = 0

    if leds is not None:
        try:
            leds.all_off()
        except Exception:
            pass

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
            except Exception as e:  # noqa: BLE001
                pwm_l, pwm_r = 0.0, 0.0
                print(f"[Project] Lane agent error: {e}")

            tags = []
            if detection_skip <= 1 or (frame_count % (detection_skip + 1) == 0):
                tags = sign_det.detect(frame_rgb)
                obstacle_det.step(frame_rgb)
            stop_flag, _ = obstacle_det.should_stop()

            cmd = fsm.step(
                FsmInputs(
                    tags=tags,
                    obstacle=stop_flag,
                    other_bot=OtherBotState(),
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
            LATEST_STATUS["chosen_maneuver"] = fsm._chosen.value if fsm._chosen else None
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
