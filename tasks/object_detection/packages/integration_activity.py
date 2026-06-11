from typing import Tuple

# Path to the trained model weights (.onnx file).
# Relative paths resolve from the project root.
MODEL_PATH = "tasks/object_detection/models/best.onnx"

# ── Tuning knobs ──────────────────────────────────────────────────────────────

# Run inference on every (N+1)-th frame.
# 0 = every frame (slowest, most accurate)
# 1 = every 2nd frame  (good balance on most hardware)
# 2 = every 3rd frame  (lighter, still reliable at low speeds)
_FRAME_SKIP = 1

# Minimum bounding-box area (pixels²) to consider a detection real.
# Tiny boxes are usually false positives far in the distance.
_MIN_BBOX_AREA = 800  # ≈ 28×28 px – ignore specks

# Maximum fraction of the image width a box may occupy.
# A box covering >80 % of the width is almost certainly a misfire.
_MAX_WIDTH_FRACTION = 0.95

# Only these class IDs trigger a stop.  trucks and signs are present in the
# dataset but should NOT stop the robot.
_STOP_CLASSES = {0}  # 0 = 'duckie'

# Minimum confidence to keep a detection at all.
_MIN_SCORE = 0.45

# ── Student-facing functions ──────────────────────────────────────────────────

def NUMBER_FRAMES_SKIPPED() -> int:
    """
    Return the number of frames to skip between inference calls.

    The object-detection agent calls this every frame.  Returning N means
    inference runs once every N+1 frames, so returning 0 means every frame.
    Experiment with 1–3 to trade latency against CPU load.
    """
    return _FRAME_SKIP


def filter_by_classes(pred_class: int) -> bool:
    return pred_class == 0  # only duckies


def filter_by_scores(score: float) -> bool:
    """
    Return False to discard low-confidence predictions.

    Anything below _MIN_SCORE is considered noise and thrown away early
    before NMS and the bbox filter run.
    """
    return score >= _MIN_SCORE


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Return False to discard geometrically implausible bounding boxes.

    bbox = (xmin, ymin, xmax, ymax) in original image pixels.

    Two heuristics:
    1. Drop boxes that are too small (likely far-away or noisy detections).
    2. Drop boxes that are unrealistically wide (likely a bad detection).
    """
    xmin, ymin, xmax, ymax = bbox

    width  = xmax - xmin
    height = ymax - ymin
    area   = width * height

    if area < _MIN_BBOX_AREA:
        return False

    # We don't know img_w here, but xmax is clamped to img_w-1 by the agent,
    # so we use xmax as a proxy for image width.
    img_w_approx = xmax if xmax > 0 else 1
    if width / img_w_approx > _MAX_WIDTH_FRACTION:
        return False

    return True