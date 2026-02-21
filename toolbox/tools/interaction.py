import numpy as np
from typing import List


class InteractionTools:
    @staticmethod
    def hand_object_contact_detector(hand_landmarks: List, object_mask: np.ndarray) -> bool:
        for landmark in hand_landmarks:
            x, y = int(landmark['x']), int(landmark['y'])
            if 0 <= y < object_mask.shape[0] and 0 <= x < object_mask.shape[1]:
                if object_mask[y, x] > 0:
                    return True
        return False
    
    @staticmethod
    def object_appearance_detector(frames: List[np.ndarray], reference_features: np.ndarray) -> int:
        for i, frame in enumerate(frames):
            if np.any(frame):
                return i
        return -1
