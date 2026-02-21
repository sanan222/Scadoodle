import sys
import json
from pathlib import Path
import google.generativeai as genai
from cfg import APIKeys, Settings


class GeminiPlanner:
    def __init__(self):
        api_key = APIKeys.get_gemini_key()
        genai.configure(api_key=api_key)
        self.model_name = Settings.GEMINI_MODEL
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        prompt_path = Settings.PROMPTS_DIR / "system_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        return "You are a video analysis task planner. Output valid JSON only."
    
    def plan_task(self, user_prompt: str, video_path: str) -> dict:
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=self.system_prompt
        )
        response = model.generate_content(
            f"VIDEO PATH: {video_path}\nUSER TASK: {user_prompt}",
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)


class CursorBridge:
    def __init__(self):
        self.workspace = Settings.WORKSPACE_DIR
        Settings.ensure_directories()
    
    def send_task(self, cursor_prompt: str, video_path: str):
        task_file = self.workspace / "current_task.json"
        task_file.write_text(json.dumps({
            "prompt": cursor_prompt,
            "video_path": video_path
        }, indent=2))


class VideoAnalyticsOrchestrator:
    def __init__(self):
        self.planner = GeminiPlanner()
        self.bridge = CursorBridge()
    
    def execute(self, video_path: str, user_prompt: str):
        plan = self.planner.plan_task(user_prompt, video_path)
        self.bridge.send_task(plan['cursor_prompt'], video_path)
        return plan


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <video_path> <user_prompt>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    user_prompt = sys.argv[2]
    
    orchestrator = VideoAnalyticsOrchestrator()
    orchestrator.execute(video_path, user_prompt)


if __name__ == "__main__":
    main()
