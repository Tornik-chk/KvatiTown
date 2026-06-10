from typing import List, Tuple
Detection = Tuple[Tuple[int, int, int, int], float, int]
class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}
_OBSTACLE_CLASSES = {0}  # 0 = 'duckie'
_MIN_HEIGHT_FRACTION = 0.15  # box must be ≥ 15 % of frame height
# Minimum confidence for a detection to trigger a stop, as a second guard.
_MIN_STOP_SCORE = 0.50
_MAX_LATERAL_OFFSET = 0.60  # centre must be within 60 % of img width from centre
def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    if not detections:
        return False, "no detections"

    for bbox, score, class_id in detections:
        if class_id not in _OBSTACLE_CLASSES:
            continue
        if score < _MIN_STOP_SCORE:
            continue

        xmin, ymin, xmax, ymax = bbox
        box_h = ymax - ymin
        height_fraction = box_h / img_size
        if height_fraction < _MIN_HEIGHT_FRACTION:
            continue
        cx_frac = ((xmin + xmax) / 2.0) / img_size
        lateral_offset = abs(cx_frac - 0.5) * 2.0
        # Close duckies (big box) get more lateral tolerance
        max_offset = 0.80 if height_fraction > 0.30 else _MAX_LATERAL_OFFSET

        if lateral_offset > max_offset:
            continue
        label = class_names.get(class_id, f"class_{class_id}")
        reason = (
            f"{label} detected: score={score:.2f}, "
            f"height={height_fraction:.2f}, lateral_offset={lateral_offset:.2f}"
        )
        return True, reason
    return False, "no qualifying obstacle in path"