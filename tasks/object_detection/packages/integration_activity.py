from typing import Tuple

MODEL_PATH = "tasks/object_detection/models/best.onnx"

_FRAME_SKIP = 1

_MIN_SCORE = 0.45
_MIN_BBOX_AREA = 600
_MAX_BBOX_AREA = 640 * 480 * 0.65

_MIN_ASPECT = 0.25
_MAX_ASPECT = 3.20


def NUMBER_FRAMES_SKIPPED() -> int:
    return _FRAME_SKIP


def filter_by_classes(pred_class: int) -> bool:
    return int(pred_class) == 0


def filter_by_scores(score: float) -> bool:
    return float(score) >= _MIN_SCORE


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    xmin, ymin, xmax, ymax = bbox

    width = max(0, xmax - xmin)
    height = max(0, ymax - ymin)

    area = width * height

    if width <= 2 or height <= 2:
        return False

    if area < _MIN_BBOX_AREA:
        return False

    if area > _MAX_BBOX_AREA:
        return False

    aspect = width / float(height + 1e-6)

    if aspect < _MIN_ASPECT:
        return False

    if aspect > _MAX_ASPECT:
        return False

    return True