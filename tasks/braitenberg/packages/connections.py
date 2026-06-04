from typing import Tuple
import numpy as np

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Left motor weight matrix: highest at bottom-left, decreasing toward top-right."""
    h, w = shape
    x = np.linspace(0, 1, w).reshape(1, -1)
    y = np.linspace(0, 1, h).reshape(-1, 1)

    M = y * (1 - x)
    return M


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Right motor weight matrix: highest at bottom-right, decreasing toward top-left."""
    h, w = shape
    x = np.linspace(0, 1, w).reshape(1, -1)
    y = np.linspace(0, 1, h).reshape(-1, 1)

    M = y * x
    return M