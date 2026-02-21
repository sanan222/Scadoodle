import numpy as np
from ultralytics import FastSAM, SAM
from .base import BaseModel


class FastSAMSegmenter(BaseModel):
    def load(self):
        self.model = FastSAM(str(self.checkpoint_path))

    def inference(self, frame, point_prompt=None, bbox_prompt=None, imgsz=640):
        results = self.model(frame, imgsz=imgsz, verbose=False)
        if results[0].masks is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        masks = results[0].masks.data.cpu().numpy()
        if masks.shape[0] == 0:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        return (masks[0] * 255).astype(np.uint8)


class MobileSAMSegmenter(BaseModel):
    def load(self):
        self.model = SAM(str(self.checkpoint_path))

    def inference(self, frame, bbox_prompt=None, point_prompt=None):
        if bbox_prompt is not None:
            results = self.model(frame, bboxes=[bbox_prompt], verbose=False)
        elif point_prompt is not None:
            results = self.model(frame, points=[point_prompt], verbose=False)
        else:
            results = self.model(frame, verbose=False)

        if results[0].masks is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        masks = results[0].masks.data.cpu().numpy()
        if masks.shape[0] == 0:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        h, w = frame.shape[:2]
        mask = masks[0]
        if mask.shape != (h, w):
            import cv2
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask * 255).astype(np.uint8)
