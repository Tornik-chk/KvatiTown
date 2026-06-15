"""
sign_behavior_config.py
"""

from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple


class TagID(IntEnum):
    TURN_RIGHT_FWD = 9
    TURN_LEFT_FWD = 10
    TURN_LEFT_RIGHT = 11
    STOP = 26
    YIELD = 39


_TAG_ID_MAP: Dict[int, TagID] = {
    1: TagID.STOP,

    # 1: TagID.TURN_RIGHT_FWD,
    9: TagID.TURN_RIGHT_FWD,
    10: TagID.TURN_LEFT_FWD,
    11: TagID.TURN_LEFT_RIGHT,

    20: TagID.STOP,
    24: TagID.STOP,
    25: TagID.STOP,
    26: TagID.STOP,

    39: TagID.YIELD,
}


def resolve_tag(raw_id: int) -> Optional[TagID]:
    return _TAG_ID_MAP.get(raw_id)


_TAG_TURNS: Dict[TagID, List[str]] = {

    # 9 = Forward + Right
    TagID.TURN_RIGHT_FWD: ["forward", "right"],

    # 10 = Left + Right
    TagID.TURN_LEFT_FWD: ["left", "right"],

    # 11 = Forward + Left
    TagID.TURN_LEFT_RIGHT: ["forward", "left"],
}


class State(IntEnum):
    MOVING = auto()
    APPROACHING = auto()
    SLOWING = auto()
    STOPPED = auto()
    CHECKPATH = auto()
    POST_STOP = auto()
    INTERSECT = auto()
    PRE_TURN = auto()
    TURNING = auto()
    EXITING = auto()


class SignBehaviorConfig:
    """
    Config for sign behavior.

    Accepts **kwargs so old code does not crash if it passes old values.
    """

    def __init__(self, **kwargs):
        # Red-line detection
        self.red_strip_frac: float = kwargs.pop("red_strip_frac", 0.32)

        self.red_roi_left: float = kwargs.pop("red_roi_left", 0.12)
        self.red_roi_right: float = kwargs.pop("red_roi_right", 0.9)

        # Bigger = stop later / closer to red line.
        self.red_line_close_y2_ratio: float = kwargs.pop("red_line_close_y2_ratio", 0.3)

        self.red_pixel_frac: float = kwargs.pop("red_pixel_frac", 0.012)
        self.red_min_area: float = kwargs.pop("red_min_area", 60.0)
        self.red_min_width_frac: float = kwargs.pop("red_min_width_frac", 0.12)

        self.red_hsv_low1: Tuple[int, int, int] = kwargs.pop("red_hsv_low1", (0, 100, 70))
        self.red_hsv_high1: Tuple[int, int, int] = kwargs.pop("red_hsv_high1", (12, 255, 255))
        self.red_hsv_low2: Tuple[int, int, int] = kwargs.pop("red_hsv_low2", (168, 100, 70))
        self.red_hsv_high2: Tuple[int, int, int] = kwargs.pop("red_hsv_high2", (179, 255, 255))

        # Important:
        # Do NOT start default intersection behavior when red line is seen without a saved sign.
        self.default_forward_on_red_without_tag: bool = kwargs.pop(
            "default_forward_on_red_without_tag",
            False,
        )

        # After finishing sign behavior, ignore the same red line for a while.
        self.red_ignore_after_frames: int = kwargs.pop("red_ignore_after_frames", 100)

        self.approach_duration = kwargs.pop(
            "approach_duration",
            3,
        )

        self.approach_speed = kwargs.pop(
            "approach_speed",
            0.1,
        )

        # Slow approach after seeing a sign but before red line.
        self.approach_speed_factor: float = kwargs.pop("approach_speed_factor", 0.75)
        self.approach_ramp_factor: float = kwargs.pop("approach_ramp_factor", 0.985)

        # Full stop behavior
        self.stop_hold_frames: int = kwargs.pop("stop_hold_frames", 50)
        self.slow_ramp_factor: float = kwargs.pop("slow_ramp_factor", 0.8)
        self.stopped_speed_threshold: float = kwargs.pop("stopped_speed_threshold", 0.055)

        # CHECKPATH sweep
        self.check_left_frames: int = kwargs.pop("check_left_frames", 20)
        self.check_right_frames: int = kwargs.pop("check_right_frames", 10)
        self.check_turn_speed: float = kwargs.pop("check_turn_speed", 0.25)
        self.check_settle_frames: int = kwargs.pop("check_settle_frames", 30)

        # POST_STOP
        self.post_stop_frames: int = kwargs.pop("post_stop_frames", 40)
        self.post_stop_speed: float = kwargs.pop("post_stop_speed", 0.4)

        # Pre-turn forward creep
        self.preturn_right_frames: int = kwargs.pop("preturn_right_frames", 6)
        self.preturn_left_frames: int = kwargs.pop("preturn_left_frames", 3)
        self.preturn_speed: float = kwargs.pop("preturn_speed", 0.2)

        # Intersection manoeuvre
        self.intersect_forward_frames: int = kwargs.pop("intersect_forward_frames", 40)
        self.intersect_left_frames: int = kwargs.pop("intersect_left_frames", 55)
        self.intersect_right_frames: int = kwargs.pop("intersect_right_frames", 40)

        # for new bot right needs more speed
        self.intersect_forward_speed: Tuple[float, float] = kwargs.pop(
            "intersect_forward_speed",
            (0.4, 0.5),
        )
        self.intersect_left_speed: Tuple[float, float] = kwargs.pop(
            "intersect_left_speed",
            (0.2, 0.5),
        )
        self.intersect_right_speed: Tuple[float, float] = kwargs.pop(
            "intersect_right_speed",
            (0.60, 0.05),
        )

        # Exiting after intersection
        self.exit_speed: float = kwargs.pop("exit_speed", 0.40)
        self.exit_timeout_frames: int = kwargs.pop("exit_timeout_frames", 3)

        for key, value in kwargs.items():
            setattr(self, key, value)