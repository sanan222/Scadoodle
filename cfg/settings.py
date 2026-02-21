from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).parent.parent
    WORKSPACE_DIR = BASE_DIR / "workspace"
    PROMPTS_DIR = BASE_DIR / "prompts"
    VIDEOS_DIR = BASE_DIR / "videos"

    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_TEMPERATURE = 0.2
    GEMINI_MAX_OUTPUT_TOKENS = 16384
    GEMINI_TOP_P = 0.95
    GEMINI_TOP_K = 40

    DEFAULT_FRAMES_DIR = "videos/sumka_512x512_2fps"
    VIDEO_SAMPLE_RATE = 5
    TRACKING_CONFIDENCE = 0.5
    PROXIMITY_THRESHOLD = 10

    @classmethod
    def ensure_directories(cls):
        cls.WORKSPACE_DIR.mkdir(exist_ok=True)
        cls.PROMPTS_DIR.mkdir(exist_ok=True)
