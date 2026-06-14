from tasks.object_detection.packages.agent import ObjectDetectionAgent


class DuckDetector:

    def __init__(self):
        self.detector = ObjectDetectionAgent()
        self._cached_ducks = []

    def detect(self, frame_rgb):
        detections = self.detector.detect(frame_rgb)

        # skipped frame -> reuse previous result
        if detections is None:
            return self._cached_ducks

        ducks = []

        for bbox, score, cls_id in detections:
            if cls_id == 0:  # duckie
                ducks.append((bbox, score, cls_id))

        self._cached_ducks = ducks
        return ducks