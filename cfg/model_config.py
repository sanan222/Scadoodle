from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    'yolov10n': MODELS_DIR / 'yolov10n.pt',
    'yolov9c': MODELS_DIR / 'yolov9c.pt',
    'fastsam': MODELS_DIR / 'fastsam-x.pt',
    'mobilesam': MODELS_DIR / 'mobile_sam.pt',
    'raft': MODELS_DIR / 'raft_small.pth',
}

MODEL_CONFIGS = {
    'yolov10n': {'conf': 0.25, 'iou': 0.45},
    'fastsam': {'imgsz': 640, 'conf': 0.4},
    'raft': {'device': 'cpu'},
}
