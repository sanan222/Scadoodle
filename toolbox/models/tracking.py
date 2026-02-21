from .detection import YOLODetector


class ByteTracker:
    def __init__(self, detector: YOLODetector):
        self.detector = detector
    
    def track(self, video_path, tracker='bytetrack.yaml', **kwargs):
        return self.detector.track(video_path, tracker=tracker, **kwargs)
