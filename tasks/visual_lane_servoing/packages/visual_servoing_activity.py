from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml')
try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _h = {}

_yellow_lower = np.array([_h.get('yellow_lower_h', 0),  _h.get('yellow_lower_s', 0),  _h.get('yellow_lower_v', 0)])
_yellow_upper = np.array([_h.get('yellow_upper_h', 0),  _h.get('yellow_upper_s', 0), _h.get('yellow_upper_v', 0)])

_white_lower = np.array([_h.get('white_lower_h', 0),   _h.get('white_lower_s', 0), _h.get('white_lower_v', 0)])
_white_upper = np.array([_h.get('white_upper_h', 0), _h.get('white_upper_s', 0), _h.get('white_upper_v', 0)])

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect lane markings (yellow dashed and white solid lines) from camera image.

    Args:
        image: BGR image from camera

    Returns:
        Tuple of (left_lane, right_lane) where:
        - left_lane: filtered gradient magnitudes for yellow dashed line
        - right_lane: filtered gradient magnitudes for white solid line
    """
    # Convert BGR to grayscale and HSV
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Gaussian blur — ksize=(0,0) lets OpenCV derive kernel from sigma
    sigma = 1.5
    img_gaussian = cv2.GaussianBlur(img_gray, (0, 0), sigma)

    # Sobel derivatives
    sobelx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1)

    # Gradient magnitude
    Gmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Magnitude threshold — normalized to image max to be lighting-invariant
    gmag_max = Gmag.max()
    if gmag_max == 0:
        return np.zeros_like(Gmag), np.zeros_like(Gmag)
    threshold = 0.15 * gmag_max
    mask_mag = Gmag > threshold

    # Spatial half-masks
    height, width = img_gray.shape
    mid = width // 2
    mask_left  = np.zeros((height, width), dtype=bool)
    mask_right = np.zeros((height, width), dtype=bool)
    mask_left[:,  :mid] = True
    mask_right[:, mid:] = True

    # Sobel sign masks — no sobely constraint, only sobelx matters
    # Left/yellow: left edge of bright marking → sobelx > 0
    # Right/white: right edge of bright marking → sobelx < 0
    mask_sobelx_pos = sobelx > 0
    mask_sobelx_neg = sobelx < 0

    # HSV color masks — bool for clean arithmetic
    mask_yellow = cv2.inRange(img_hsv, _yellow_lower, _yellow_upper).astype(bool)
    mask_white  = cv2.inRange(img_hsv, _white_lower,  _white_upper ).astype(bool)

    # Combine masks — all bool, no dtype mixing
    mask_left_final  = mask_left  & mask_mag & mask_sobelx_pos & mask_yellow
    mask_right_final = mask_right & mask_mag & mask_sobelx_neg & mask_white

    left_lane  = np.where(mask_left_final,  Gmag, 0.0)
    right_lane = np.where(mask_right_final, Gmag, 0.0)

    return left_lane, right_lane




def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower    = np.array(yellow_lower)
    _yellow_upper    = np.array(yellow_upper)
    _white_lower = np.array(white_lower)
    _white_upper = np.array(white_upper)

def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]),    'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]),    'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]),    'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]), 'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]), 'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]), 'white_upper_v':  int(_white_upper[2]),
    }