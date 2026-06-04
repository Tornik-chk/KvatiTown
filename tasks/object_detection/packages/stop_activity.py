from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    for (x1, y1, x2, y2), score, class_id in detections:
        # Signs are informational, not obstacles — ignore them
        if class_id == 2:
            continue

        box_h = y2 - y1
        box_w = x2 - x1
        area_ratio = (box_h * box_w) / (img_size * img_size)

        # Object is close if its bottom edge is low in the frame
        # and it occupies a meaningful fraction of the image
        if y2 > img_size * 0.60 and area_ratio > 0.04:
            name = class_names[class_id]
            return True, f"{name} detected close ahead (y2={y2}, area={area_ratio:.2f})"

    return False, ""