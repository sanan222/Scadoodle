class MediaPipeHands:
    def __init__(self):
        self.hands = None
    
    def load(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.hands = vision.HandLandmarker
    
    def inference(self, frame):
        if self.hands:
            return self.hands.detect(frame)
        return None


class MediaPipePose:
    def __init__(self):
        self.pose = None
    
    def load(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.pose = vision.PoseLandmarker
    
    def inference(self, frame):
        if self.pose:
            return self.pose.detect(frame)
        return None
