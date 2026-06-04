from typing import List, Tuple
import numpy as np


def detect_curve(
    yellow_xs: List[int],
    white_xs: List[int],
    curve_threshold: int = 350,
) -> Tuple[bool, int]:
    """
    Detect whether the road is curving based on lane marking x-positions.

    Args:
        yellow_xs: x-positions of yellow (left) lane across image rows,
                   xs[0] nearest to robot, xs[-1] farthest ahead.
        white_xs:  x-positions of white (right) lane, same convention.
        curve_threshold: pixel shift between near and far to classify as a curve.

    Returns:
        (is_curve, direction) where direction > 0 means curving right,
        < 0 means curving left, 0 means straight.
    """
    shifts = []

    if len(yellow_xs) >= 2:
        shifts.append(yellow_xs[-1] - yellow_xs[0])
    if len(white_xs) >= 2:
        shifts.append(white_xs[-1] - white_xs[0])

    if not shifts:
        return False, 0

    shift = int(np.mean(shifts))

    if abs(shift) > curve_threshold:
        return True, int(np.sign(shift))

    return False, 0