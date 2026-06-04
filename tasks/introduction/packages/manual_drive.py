from typing import Dict, Tuple
import logging
logger = logging.getLogger(__name__)
import numpy as np

SPEED = 1
TURN = 0.5


def get_motor_speeds(keys_pressed: Dict[str, bool]) -> Tuple[float, float]:
    left_speed = 0.0
    right_speed = 0.0
    kp = keys_pressed

    if kp.get("up"): 
        left_speed += 0.5
        right_speed += 0.5
    if kp.get("down"): 
        left_speed -= 0.5
        right_speed -= 0.5
    if kp.get("right"): 
        left_speed += 0.3
        right_speed -= 0.3
    if kp.get("left"): 
        left_speed -= 0.3
        right_speed += 0.3
    
    return (left_speed, right_speed)