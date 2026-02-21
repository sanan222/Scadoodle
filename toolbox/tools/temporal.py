import cv2
import numpy as np
from typing import List, Dict, Callable


class TemporalTools:
    @staticmethod
    def temporal_event_localizer(signal: List[float], condition: Callable) -> int:
        left, right = 0, len(signal) - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if condition(signal[mid]):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        return result
    
    @staticmethod
    def frame_sampler(video_path: str, every_n: int) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % every_n == 0:
                frames.append(frame)
            frame_id += 1
        cap.release()
        return frames
    
    @staticmethod
    def frame_annotator(frame: np.ndarray, annotations: Dict) -> np.ndarray:
        annotated = frame.copy()
        if 'bboxes' in annotations:
            for bbox in annotations['bboxes']:
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        if 'labels' in annotations:
            for i, label in enumerate(annotations['labels']):
                cv2.putText(annotated, label, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated
    
    @staticmethod
    def threshold_trigger(signal: list, threshold=0.5) -> int:
        """Return the index of the first value that meets or exceeds threshold.

        Works with both numeric signals (val >= threshold) and boolean signals
        (first True entry).
        """
        for i, val in enumerate(signal):
            if isinstance(val, bool):
                if val:
                    return i
            elif val >= threshold:
                return i
        return None

    @staticmethod
    def rising_edge_detector(signal: list, min_gap: int = 3) -> int:
        """Find the first False→True transition after at least `min_gap` consecutive Falses.

        Useful for detecting *pickup* events: the object was visible but untouched
        for several frames, then contact begins.  Returns the index of the first
        True after the gap, or None.
        """
        consecutive_false = 0
        for i, val in enumerate(signal):
            if isinstance(val, bool):
                is_true = val
            else:
                is_true = val >= 0.5
            if not is_true:
                consecutive_false += 1
            else:
                if consecutive_false >= min_gap:
                    return i
                consecutive_false = 0
        return None
