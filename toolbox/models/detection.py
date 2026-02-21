from ultralytics import YOLO
from .base import BaseModel


class YOLODetector(BaseModel):
    def load(self):
        self.model = YOLO(str(self.checkpoint_path))
    
    def inference(self, frame, classes=None, conf=0.25):
        if classes:
            results = self.model(frame, classes=classes, conf=conf)
        else:
            results = self.model(frame, conf=conf)
        return results[0]
    
    def track(self, video_path, tracker='bytetrack.yaml', **kwargs):
        return self.model.track(source=video_path, tracker=tracker, **kwargs)
