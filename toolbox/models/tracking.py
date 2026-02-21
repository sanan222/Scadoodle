from ultralytics import YOLO
import numpy as np


class ByteTracker:
    def __init__(self):
        self._tracker = None
        self._next_id = 1
        self._tracks = {}

    def inference(self, detections):
        if not detections:
            return []

        tracked = []
        for det in detections:
            bbox = det.get("bbox", det.get("bbox", [0, 0, 0, 0]))
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            matched_id = None
            best_dist = 80
            for tid, prev in self._tracks.items():
                dx = cx - prev[0]
                dy = cy - prev[1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    matched_id = tid

            if matched_id is None:
                matched_id = self._next_id
                self._next_id += 1

            self._tracks[matched_id] = (cx, cy)

            tracked.append({
                "track_id": matched_id,
                "bbox": bbox,
                "frame_id": det.get("frame_id", -1),
            })
        return tracked
