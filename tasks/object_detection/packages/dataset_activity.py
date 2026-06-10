import json
from typing import List

# Classes the model is trained to detect.
# The index here is the class ID written into YOLO label files.
CLASSES = ['duckie', 'truck', 'sign']

# Images are resized to this square size before training.
IMAGE_SIZE = 416


def convert_labelme_json(json_path: str, img_w: int, img_h: int) -> List[str]:
    """
    Convert a LabelMe annotation JSON to YOLO-format label lines.

    Steps (exactly as specified in the assignment notebook):
    1. Load the JSON file.
    2. Iterate over shapes, skip labels not in CLASSES.
    3. Get class_id via CLASSES.index(label).
    4. Extract xmin, ymin, xmax, ymax from the two corner points.
    5. Scale corners from original image space → IMAGE_SIZE space.
    6. Compute normalised cx, cy, w, h relative to IMAGE_SIZE.
    7. Return list of YOLO label strings.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    lines: List[str] = []

    for shape in data.get('shapes', []):
        label = shape.get('label', '').lower().strip()
        if label not in CLASSES:
            continue

        cls_id = CLASSES.index(label)
        points = shape.get('points', [])

        if len(points) < 2:
            continue

        # points = [[x1, y1], [x2, y2]] (top-left, bottom-right)
        xmin = min(points[0][0], points[1][0])
        ymin = min(points[0][1], points[1][1])
        xmax = max(points[0][0], points[1][0])
        ymax = max(points[0][1], points[1][1])

        # Step 5: scale from original image space to IMAGE_SIZE space
        xmin = xmin * IMAGE_SIZE / img_w
        xmax = xmax * IMAGE_SIZE / img_w
        ymin = ymin * IMAGE_SIZE / img_h
        ymax = ymax * IMAGE_SIZE / img_h

        # Step 6: normalise to [0, 1] relative to IMAGE_SIZE
        cx = (xmin + xmax) / 2 / IMAGE_SIZE
        cy = (ymin + ymax) / 2 / IMAGE_SIZE
        w  = (xmax - xmin) / IMAGE_SIZE
        h  = (ymax - ymin) / IMAGE_SIZE

        # Skip degenerate boxes
        if w <= 0 or h <= 0:
            continue

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines