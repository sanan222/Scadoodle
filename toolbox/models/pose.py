import cv2
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


class MediaPipeHands:
    def __init__(self):
        self.detector = None

    def load(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = str(MODELS_DIR / "hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=2,
            min_hand_detection_confidence=0.4,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def inference(self, frame):
        """Returns list of hands. Each hand is a list of {"x": int, "y": int} pixel-coordinate dicts."""
        if self.detector is None:
            return []

        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return []

        h, w = frame.shape[:2]
        hands_out = []
        for hand in result.hand_landmarks:
            landmarks = [{"x": int(lm.x * w), "y": int(lm.y * h)} for lm in hand]
            hands_out.append(landmarks)
        return hands_out


class MediaPipePose:
    def __init__(self):
        self.detector = None

    def load(self):
        pass

    def inference(self, frame):
        return []
