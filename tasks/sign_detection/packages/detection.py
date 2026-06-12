import numpy as np
import cv2
from typing import List, Tuple, Optional

from tasks.sign_detection.packages.sign_behaviour_config import State


# todo - at intersection not many signs are seen. area will be enough
# todo - when moving between lines - magram

# todo - ubralod area sakmarsia swor gzaze
# intersection-ze marto logika ro mqondes eg gonia kvati da cherdeba
# lanes shoris logika imushaves yvelgan intersections garda (marto erti lane aris mand visible)
# magram samagierod false positives are eqneba


# todo lessen number of frames stopped for duck

# =========================================================
# GLOBAL CONFIG
# =========================================================
IMG_WIDTH = 640
IMG_HEIGHT = 480

CLASS_NAMES = {
    0: "duckie",
    1: "truck",
    2: "sign",
}

CLASS_COLORS = {
    0: (0, 215, 255),
    1: (180, 100, 220),
    2: (50, 205, 50),
}

Detection = Tuple[Tuple[int, int, int, int], float, int]





# =========================================================
# DUCK STOP CONFIG
# =========================================================
# Bigger = stop later / closer.
# Smaller = stop earlier / farther.
DUCK_STOP_Y2_RATIO = 0.70

DUCK_CONFIRM_FRAMES = 2
TRUCK_CONFIRM_FRAMES = 1
STOP_LATCH_FRAMES = 5

_duck_counter = 0
_truck_counter = 0
_stop_latch = 0


# =========================================================
# MASK HELPERS
# =========================================================
def _clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    return cleaned


def _mask_to_bboxes(mask):
    """
    This is kept for the truck path because your current truck detection works well.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h, area))

    return boxes


def _mask_to_candidates(mask: np.ndarray, min_area: float = 80.0):
    """
    More detailed candidate extraction for ducks.
    We need this because duck and yellow road line have similar color.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        contour_area = float(cv2.contourArea(cnt))

        if contour_area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w <= 2 or h <= 2:
            continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        bbox_area = float(w * h)
        extent = contour_area / (bbox_area + 1e-6)

        rect = cv2.minAreaRect(cnt)
        rect_w, rect_h = rect[1]

        if rect_w <= 1 or rect_h <= 1:
            rotated_aspect = max(w, h) / (min(w, h) + 1e-6)
        else:
            rotated_aspect = max(rect_w, rect_h) / (min(rect_w, rect_h) + 1e-6)

        candidates.append({
            "bbox": (x1, y1, x2, y2),
            "area": contour_area,
            "bbox_area": bbox_area,
            "extent": extent,
            "rotated_aspect": rotated_aspect,
        })

    return candidates


# =========================================================
# DUCK FILTERING
# =========================================================
def _is_duck_candidate(candidate: dict, frame_w: int, frame_h: int) -> bool:
    x1, y1, x2, y2 = candidate["bbox"]

    box_w = x2 - x1
    box_h = y2 - y1

    bbox_area = candidate["bbox_area"]
    rotated_aspect = candidate["rotated_aspect"]
    extent = candidate["extent"]

    cx = (x1 + x2) / 2.0
    y2_ratio = y2 / float(frame_h)
    aspect = box_w / float(box_h + 1e-6)

    # Ignore very far/top yellow noise.
    if y2_ratio < 0.28:
        return False

    # Duck should be somewhere in front / near driving area.
    # Keep this wide because at intersections there may be no clear lane lines.
    if cx < frame_w * 0.10 or cx > frame_w * 0.90:
        return False

    # Real duck can be small, so do not make this too strict.
    if box_h < frame_h * 0.025:
        return False

    if bbox_area < frame_w * frame_h * 0.00045:
        return False

    # Main yellow-line rejection:
    # road yellow line/dashes are usually long and thin.
    # duck is more compact.
    if aspect > 2.25:
        return False

    if aspect < 0.30:
        return False

    if rotated_aspect > 2.70:
        return False

    # Very filled long rectangles are often road markings, not ducks.
    if extent > 0.90 and rotated_aspect > 1.80 and aspect > 1.60:
        return False

    return True


def _duck_close_enough(
    bbox: Tuple[int, int, int, int],
    score: float,
    state,
    white_x = [],
) -> bool:
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    area = box_w * box_h
    cx = (x1 + x2) / 2.0
    aspect = box_w / float(box_h + 1e-6)

    if score < 0.20:
        return False

    if cx < IMG_WIDTH * 0.16 or cx > IMG_WIDTH * 0.84:
        return False

    if aspect > 2.25:
        return False

    if aspect < 0.30:
        return False

    if box_h < IMG_HEIGHT * 0.025:
        return False

    if area < IMG_WIDTH * IMG_HEIGHT * 0.00045:
        return False

    # -----------------------------
    # Intersection:
    # area-only rule
    # -----------------------------
    if state != State.MOVING:
        print("INTERSECTIONNNNNNNNNNNNNNNNNN")
        if area < 10000:
            return False

    # -----------------------------
    # Normal driving:
    # use white-lane distance
    # -----------------------------
    else:
        if area < 3000:
            return False

        if white_x is not None:
            white_x = float(np.median(white_x))
            dist = abs(cx - white_x)

            print(
                f"duck-white distance={dist:.1f}px "
                f"area={area}"
            )

            if dist > 250:
                return False

    return y2 >= IMG_HEIGHT * DUCK_STOP_Y2_RATIO


# =========================================================
# DETECTION
# =========================================================
def detect_obstacles(frame_rgb: np.ndarray) -> List[Detection]:
    """
    Returns:
        [((x1, y1, x2, y2), score, cls_id), ...]

    cls_id:
        0 = duckie
        1 = truck

    Truck detection is kept close to your current working version.
    Duck detection is improved so yellow road line is rejected by shape.
    """
    global IMG_WIDTH, IMG_HEIGHT

    frame_h, frame_w = frame_rgb.shape[:2]
    IMG_WIDTH = frame_w
    IMG_HEIGHT = frame_h

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # -----------------------------
    # DUCK / YELLOW DETECTION
    # -----------------------------
    # Wider than before because real duck can be darker/smaller.
    yellow_lower = np.array([18, 80, 80], dtype=np.uint8)
    yellow_upper = np.array([38, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Ignore very top area.
    yellow_mask[:int(frame_h * 0.10), : ] = 0

    yellow_mask = _clean_mask(yellow_mask, open_size=3, close_size=5)

    detections: List[Detection] = []

    for candidate in _mask_to_candidates(yellow_mask, min_area=70.0):
        if not _is_duck_candidate(candidate, frame_w, frame_h):
            continue

        bbox = candidate["bbox"]
        score = min(1.0, candidate["bbox_area"] / 6500.0)
        detections.append((bbox, score, 0))

    # -----------------------------
    # TRUCK / BLUE DETECTION
    # -----------------------------
    # Kept from your working uncommented version.
    blue_lower = np.array([90, 80, 80], dtype=np.uint8)
    blue_upper = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    for x1, y1, x2, y2, area in _mask_to_bboxes(blue_mask):
        detections.append(((x1, y1, x2, y2), min(1.0, area / 5000.0), 1))

    return detections



vehicle_min_bbox_area = 800
# intersection crossing check
def vehicle_detected(detections):
    # Used by sign_behavior.py during CHECKPATH.
    if detections is None:
        detections = []

    for (x1, y1, x2, y2), score, cls_id in detections:
        if cls_id != 1:
            continue

        area = (x2 - x1) * (y2 - y1)

        if area < vehicle_min_bbox_area:
            continue

        centre_x = (x1 + x2) / 2.0
        offset = abs(centre_x - IMG_WIDTH / 2) / IMG_WIDTH

        print(f"[SignBehavior] vehicle detected (area={area:.0f}, offset={offset:.3f})")
        return True, offset

    return False, 0.0


# =========================================================
# STOP LOGIC
# =========================================================
def should_stop(
    detections: List[Detection],
    state: Optional[State] = None,
    white_x = []
) -> Tuple[bool, str]:
    """
    Stops for:
        - close duck
        - close truck

    Does not stop during CHECKPATH because sign behavior is already checking traffic.
    """

    global _duck_counter, _truck_counter, _stop_latch

    if detections is None:
        detections = []

    # Allow both enum State.CHECKPATH and string "CHECKPATH".
    if state == State.CHECKPATH or state == "CHECKPATH":
        return False, ""

    duck_threat = False
    truck_threat = False
    reason = ""

    for bbox, score, cls_id in detections:
        x1, y1, x2, y2 = bbox

        area = (x2 - x1) * (y2 - y1)
        height = y2 - y1
        width = x2 - x1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # -----------------------------
        # DUCK STOP
        # -----------------------------
        # if cls_id == 0:
        #     if _duck_close_enough(bbox, score, state, white_x):
        #         duck_threat = True
        #         reason = f"duckie close enough score={score:.2f} bbox={bbox}"
        #         print(f"[ObstacleStop] DUCKIE candidate close (area={area:.0f})")

        # -----------------------------
        # TRUCK STOP
        # -----------------------------
        # Kept from your working uncommented version.
        if cls_id == 1:
            if area < 50000:
                continue
            if height < 15:
                continue
            if cy < IMG_HEIGHT * 0.25:
                continue
            if cx < 0.08 * IMG_WIDTH or cx > 0.92 * IMG_WIDTH:
                continue

            truck_threat = True
            reason = f"truck too close area={area:.0f} h={height:.0f}"
            print(f"[ObstacleStop] TRUCK STOP (area={area:.0f}, h={height:.0f})")

    if duck_threat:
        _duck_counter += 1
    else:
        _duck_counter = max(0, _duck_counter - 1)

    if truck_threat:
        _truck_counter += 1
    else:
        _truck_counter = max(0, _truck_counter - 1)

    confirmed_duck = _duck_counter >= DUCK_CONFIRM_FRAMES
    confirmed_truck = _truck_counter >= TRUCK_CONFIRM_FRAMES

    if confirmed_duck or confirmed_truck:
        _stop_latch = STOP_LATCH_FRAMES
    else:
        _stop_latch = max(0, _stop_latch - 1)

    if _stop_latch > 0:
        return True, reason or "stop latch active"

    return False, ""


def reset_detection_state():
    global _duck_counter, _truck_counter, _stop_latch

    _duck_counter = 0
    _truck_counter = 0
    _stop_latch = 0

    print("[Detection] state reset")