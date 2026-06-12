"""
agent_with_signs.py
===================

Lane-following agent with sign/intersection behavior added on top.
"""

from typing import List, Optional, Tuple

import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.sign_detection.packages.sign_behaviour import SignBehaviorFSM
from tasks.sign_detection.packages.sign_behaviour import SignBehaviorConfig


class LaneServoingAgentWithSigns(LaneServoingAgent):
    """
    Extends LaneServoingAgent with AprilTag + red-line behavior.

    compute_commands(image, detections=None)
    """

    def __init__(self, config_path=None, sign_config=None, **kwargs):
        # Do not pass old sign kwargs into LaneServoingAgent.
        # Some old server/config code may pass stop_hold_frames, min_margin, etc.
        super().__init__(config_path=config_path)

        if sign_config is None:
            cfg = SignBehaviorConfig(**kwargs)
        elif isinstance(sign_config, dict):
            cfg = SignBehaviorConfig(**sign_config)
        else:
            cfg = sign_config

        self._sign_fsm = SignBehaviorFSM(config=cfg)

    def reset_behavior(self):
        if hasattr(self, "_sign_fsm") and self._sign_fsm is not None:
            self._sign_fsm.reset()

        if hasattr(self, "_prev_error"):
            self._prev_error = 0.0

        if hasattr(self, "_prev_diff"):
            self._prev_diff = 0.0

        if hasattr(self, "_filtered_error"):
            self._filtered_error = 0.0

        if hasattr(self, "_filtered_steering"):
            self._filtered_steering = 0.0

        if hasattr(self, "_lost_lane_frames"):
            self._lost_lane_frames = 0

        if hasattr(self, "_left_history"):
            self._left_history.clear()

        if hasattr(self, "_right_history"):
            self._right_history.clear()

        print("[LaneAgentWithSigns] behavior reset")

    def compute_commands(self, image, detections=None):
        # type: (np.ndarray, Optional[List]) -> Tuple[float, float]

        base_left, base_right = super().compute_commands(image)

        left, right, state = self._sign_fsm.step(
            image,
            base_left,
            base_right,
            detections or [],
        )

        return left, right, state

    @property
    def sign_state(self):
        return self._sign_fsm.state_name

    @property
    def sign_debug(self):
        return self._sign_fsm.debug

    def step(self, image, wheels_driver, detections=None):
        left, right, _ = self.compute_commands(image, detections)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def get_debug_info(self, image):
        info = super().get_debug_info(image)

        info["sign_state"] = self.sign_state
        info["sign_debug"] = self.sign_debug

        return info