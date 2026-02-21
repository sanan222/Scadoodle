import cv2
import numpy as np
from typing import List


class InteractionTools:
    @staticmethod
    def hand_object_contact_detector(
        hand_landmarks: List[dict],
        object_mask: np.ndarray,
        dilate_px: int = 8,
    ) -> bool:
        """Check if a single hand (list of {x,y} dicts) physically touches an object mask.

        The mask is dilated by `dilate_px` to tolerate slight boundary misalignment,
        but this is much tighter than a bounding-box margin and avoids false positives
        when a hand is merely *near* the object without touching it.
        """
        if object_mask is None or not np.any(object_mask):
            return False

        h, w = object_mask.shape[:2]

        if dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            mask = cv2.dilate(object_mask.astype(np.uint8), kernel, iterations=1)
        else:
            mask = object_mask.astype(np.uint8)

        for lm in hand_landmarks:
            x, y = int(lm["x"]), int(lm["y"])
            if 0 <= y < h and 0 <= x < w and mask[y, x] > 0:
                return True

        return False

    @staticmethod
    def object_appearance_detector(frames: List[np.ndarray], reference_features: np.ndarray) -> int:
        for i, frame in enumerate(frames):
            if np.any(frame):
                return i
        return -1
