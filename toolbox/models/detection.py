from ultralytics import YOLO
from .base import BaseModel


class YOLODetector(BaseModel):
    def load(self):
        self.model = YOLO(str(self.checkpoint_path))

    def inference(self, frame, classes=None, conf=0.25):
        if classes:
            results = self.model(frame, classes=classes, conf=conf, verbose=False)
        else:
            results = self.model(frame, conf=conf, verbose=False)

        detections = []
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": result.names[int(box.cls[0])],
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
            })
        return detections

    def track(self, video_path, tracker='bytetrack.yaml', **kwargs):
        return self.model.track(source=video_path, tracker=tracker, **kwargs)
