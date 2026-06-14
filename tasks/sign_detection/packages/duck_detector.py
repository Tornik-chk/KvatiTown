from tasks.object_detection.packages.agent import ObjectDetectionAgent


class DuckDetector:
    """
    Thin wrapper around ObjectDetectionAgent that keeps only duckie detections.

    Important: the ONNX model is shared at class level, so even if DuckDetector()
    is constructed more than once, the heavy model loads only once per process.
    """

    _shared_detector = None

    def __init__(self):
        if DuckDetector._shared_detector is None:
            DuckDetector._shared_detector = ObjectDetectionAgent()

        self.detector = DuckDetector._shared_detector
        self._cached_ducks = []

    def detect(self, frame_rgb):
        detections = self.detector.detect(frame_rgb)

        # ObjectDetectionAgent returns None on skipped frames.
        if detections is None:
            return list(self._cached_ducks)

        ducks = []

        for bbox, score, cls_id in detections:
            if int(cls_id) == 0:
                ducks.append((bbox, float(score), int(cls_id)))

        self._cached_ducks = ducks
        return list(ducks)