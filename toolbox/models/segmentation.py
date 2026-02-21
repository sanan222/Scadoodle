from ultralytics import FastSAM, SAM
from .base import BaseModel


class FastSAMSegmenter(BaseModel):
    def load(self):
        self.model = FastSAM(str(self.checkpoint_path))
    
    def inference(self, frame, prompts=None, imgsz=640):
        results = self.model(frame, imgsz=imgsz)
        if prompts:
            return results[0].masks
        return results[0]


class MobileSAMSegmenter(BaseModel):
    def load(self):
        self.model = SAM(str(self.checkpoint_path))
    
    def inference(self, frame, bboxes=None, points=None):
        if bboxes:
            results = self.model(frame, bboxes=bboxes)
        elif points:
            results = self.model(frame, points=points)
        else:
            results = self.model(frame)
        return results[0]
