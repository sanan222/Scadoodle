import cv2
import numpy as np
from typing import Tuple, List, Dict


class SpatialTools:
    @staticmethod
    def mask_extractor(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(frame, frame, mask=mask)
    
    @staticmethod
    def bbox_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]
    
    @staticmethod
    def iou_calculator(bbox1: Tuple, bbox2: Tuple) -> float:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        return inter_area / (bbox1_area + bbox2_area - inter_area)
    
    @staticmethod
    def contact_proximity_checker(mask1: np.ndarray, mask2: np.ndarray, threshold: int) -> bool:
        dist_transform = cv2.distanceTransform(mask1, cv2.DIST_L2, 3)
        overlap = cv2.bitwise_and(mask2, mask2, mask=mask1)
        if overlap.sum() > 0:
            return True
        return dist_transform[mask2 > 0].min() <= threshold if mask2.sum() > 0 else False
    
    @staticmethod
    def centroid_tracker(detections: List[Dict]) -> List[Tuple[float, float, int]]:
        centroids = []
        for det in detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centroids.append((cx, cy, det['frame_id']))
        return centroids
