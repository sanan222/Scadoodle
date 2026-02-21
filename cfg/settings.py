from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).parent.parent
    WORKSPACE_DIR = BASE_DIR / "workspace"
    PROMPTS_DIR = BASE_DIR / "prompts"
    
    GEMINI_MODEL = "gemini-2.0-flash"
    
    VIDEO_SAMPLE_RATE = 5
    TRACKING_CONFIDENCE = 0.5
    PROXIMITY_THRESHOLD = 10
    
    @classmethod
    def ensure_directories(cls):
        cls.WORKSPACE_DIR.mkdir(exist_ok=True)
        cls.PROMPTS_DIR.mkdir(exist_ok=True)
