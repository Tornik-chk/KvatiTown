from typing import Tuple
import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    delta_ticks = ticks - prev_ticks
    rotation_per_tick = 2 * np.pi / resolution
    wheel_rotation = delta_ticks * rotation_per_tick
    return wheel_rotation, ticks


def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:
    
    d_left = R * delta_phi_left
    d_right = R * delta_phi_right
    
    d_A = (d_left + d_right) / 2.0
    
    delta_theta = (d_right - d_left) / (2.0 * baseline)
    
    delta_x = d_A * np.cos(theta_prev)
    delta_y = d_A * np.sin(theta_prev)
    
    x = x_prev + delta_x
    y = y_prev + delta_y
    theta = theta_prev + delta_theta
    
    return x, y, theta
