import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from google.genai import types

from cfg import APIKeys, Settings


class ToolboxRegistry:
    @staticmethod
    def get_available_models() -> List[Dict]:
        return [
            {
                "name": "yolov10",
                "type": "detection",
                "output": "[{label, bbox:[x1,y1,x2,y2], confidence, frame_id}]",
                "import": "from toolbox.models.detection import YOLODetector",
                "usage": "detector = YOLODetector(MODEL_PATHS['yolov10n']); detector.load(); results = detector.inference(frame)",
            },
            {
                "name": "fastsam",
                "type": "segmentation",
                "output": "binary mask",
                "import": "from toolbox.models.segmentation import FastSAMSegmenter",
                "usage": "seg = FastSAMSegmenter(MODEL_PATHS['fastsam']); seg.load(); mask = seg.inference(frame, point_prompt=(x,y))",
                "constraint": "NO text prompts. Only point_prompt or bbox_prompt.",
            },
            {
                "name": "mobilesam",
                "type": "segmentation",
                "output": "binary mask",
                "import": "from toolbox.models.segmentation import MobileSAMSegmenter",
                "usage": "seg = MobileSAMSegmenter(MODEL_PATHS['mobilesam']); seg.load(); mask = seg.inference(frame, bbox_prompt=[x1,y1,x2,y2])",
                "constraint": "NO text prompts. Only point_prompt or bbox_prompt.",
            },
            {
                "name": "bytetrack",
                "type": "tracking",
                "output": "[{track_id, bbox, frame_id}]",
                "import": "from toolbox.models.tracking import ByteTracker",
                "usage": "tracker = ByteTracker(); tracks = tracker.inference(detections)",
            },
            {
                "name": "raft_small",
                "type": "optical_flow",
                "output": "flow field (H,W,2)",
                "import": "from toolbox.models.flow import RAFTFlow",
                "usage": "flow = RAFTFlow(MODEL_PATHS['raft']); flow.load(); field = flow.inference(frame1, frame2)",
            },
            {
                "name": "mediapipe_hands",
                "type": "pose",
                "output": "[{landmark:{x,y}, frame_id}]",
                "import": "from toolbox.models.pose import MediaPipeHands",
                "usage": "hands = MediaPipeHands(); hands.load(); landmarks = hands.inference(frame)",
            },
            {
                "name": "mediapipe_pose",
                "type": "pose",
                "output": "[{keypoints, frame_id}]",
                "import": "from toolbox.models.pose import MediaPipePose",
                "usage": "pose = MediaPipePose(); pose.load(); keypoints = pose.inference(frame)",
            },
        ]

    @staticmethod
    def get_available_tools() -> List[Dict]:
        return [
            {"name": "mask_extractor", "module": "SpatialTools", "signature": "SpatialTools.mask_extractor(frame, mask) -> cropped_region"},
            {"name": "bbox_crop", "module": "SpatialTools", "signature": "SpatialTools.bbox_crop(frame, bbox) -> cropped_region"},
            {"name": "iou_calculator", "module": "SpatialTools", "signature": "SpatialTools.iou_calculator(bbox_a, bbox_b) -> float"},
            {"name": "contact_proximity_checker", "module": "SpatialTools", "signature": "SpatialTools.contact_proximity_checker(mask_a, mask_b, threshold_px) -> bool"},
            {"name": "centroid_tracker", "module": "SpatialTools", "signature": "SpatialTools.centroid_tracker(detections) -> [(x,y,frame)]"},
            {"name": "flow_magnitude_map", "module": "MotionTools", "signature": "MotionTools.flow_magnitude_map(flow_field) -> magnitude_map"},
            {"name": "motion_peak_detector", "module": "MotionTools", "signature": "MotionTools.motion_peak_detector(flow_magnitudes, threshold) -> [frame_ids]"},
            {"name": "trajectory_builder", "module": "MotionTools", "signature": "MotionTools.trajectory_builder(track_data, track_id) -> [(x,y,frame)]"},
            {"name": "velocity_calculator", "module": "MotionTools", "signature": "MotionTools.velocity_calculator(trajectory) -> [velocity_per_frame]"},
            {"name": "takeoff_landing_detector", "module": "MotionTools", "signature": "MotionTools.takeoff_landing_detector(trajectory) -> frame_id"},
            {"name": "temporal_event_localizer", "module": "TemporalTools", "signature": "TemporalTools.temporal_event_localizer(signal, condition) -> frame_id"},
            {"name": "frame_sampler", "module": "TemporalTools", "signature": "TemporalTools.frame_sampler(video_path, every_n) -> [frames]"},
            {"name": "frame_annotator", "module": "TemporalTools", "signature": "TemporalTools.frame_annotator(frame, annotations) -> annotated_frame"},
            {"name": "threshold_trigger", "module": "TemporalTools", "signature": "TemporalTools.threshold_trigger(signal_array, threshold) -> frame_id"},
            {"name": "hand_object_contact_detector", "module": "InteractionTools", "signature": "InteractionTools.hand_object_contact_detector(hand_landmarks, object_mask) -> bool"},
            {"name": "object_appearance_detector", "module": "InteractionTools", "signature": "InteractionTools.object_appearance_detector(frames, reference_embedding) -> frame_id"},
        ]

    @classmethod
    def build_toolbox_description(cls) -> str:
        models = cls.get_available_models()
        tools = cls.get_available_tools()

        models_block = "\n".join([
            f"- {m['name']} ({m['type']}): output={m['output']}\n"
            f"  import: {m['import']}\n"
            f"  usage: {m['usage']}"
            + (f"\n  CONSTRAINT: {m['constraint']}" if 'constraint' in m else "")
            for m in models
        ])

        tools_block = "\n".join([
            f"- {t['signature']}" for t in tools
        ])

        return f"AVAILABLE MODELS:\n{models_block}\n\nAVAILABLE TOOLS:\n{tools_block}"


class FrameLoader:
    def __init__(self, frames_dir: str):
        self.frames_dir = Path(frames_dir)
        self.frame_paths = self._discover_frames()

    def _discover_frames(self) -> List[Path]:
        patterns = ["frame_*.jpg", "frame_*.png", "*.jpg", "*.png"]
        for pattern in patterns:
            found = sorted(self.frames_dir.glob(pattern))
            if found:
                return found
        return []

    @property
    def count(self) -> int:
        return len(self.frame_paths)

    def get_sample_paths(self, max_samples: int = 5) -> List[Path]:
        if self.count <= max_samples:
            return list(self.frame_paths)
        step = self.count // max_samples
        return [self.frame_paths[i * step] for i in range(max_samples)]

    def summary(self) -> str:
        if not self.frame_paths:
            return "No frames found."
        return (
            f"Directory: {self.frames_dir}\n"
            f"Total frames: {self.count}\n"
            f"First frame: {self.frame_paths[0].name}\n"
            f"Last frame: {self.frame_paths[-1].name}\n"
            f"Format: {self.frame_paths[0].suffix}"
        )


class GeminiAgent:
    def __init__(self):
        self.client = self._create_client()
        self.system_prompt = self._load_system_prompt()

    def _create_client(self) -> genai.Client:
        return genai.Client(
            vertexai=True,
            project=APIKeys.get_project_id(),
            location=APIKeys.get_location(),
            credentials=APIKeys.get_credentials(),
        )

    def _load_system_prompt(self) -> str:
        prompt_file = Settings.PROMPTS_DIR / "system_prompt.txt"
        if prompt_file.exists():
            return prompt_file.read_text()
        return self._build_fallback_prompt()

    def _build_fallback_prompt(self) -> str:
        toolbox_desc = ToolboxRegistry.build_toolbox_description()
        return (
            "You are a video analysis task planner and code architect.\n"
            "Given a user request about video analysis, produce a JSON execution plan "
            "and a complete Python script using ONLY the toolbox below.\n\n"
            f"{toolbox_desc}\n\n"
            "CONSTRAINTS:\n"
            "- NO text-based semantic segmentation (SAM models only accept point/bbox prompts)\n"
            "- NO VLMs available\n"
            "- CPU-efficient processing only\n"
            "- Frames are pre-extracted as frame_NNNNN.jpg\n\n"
            "OUTPUT: strict JSON with keys: task_summary, reasoning, execution_plan, python_script"
        )

    def generate_plan(self, user_prompt: str, frame_loader: FrameLoader,
                      sample_frames: bool = True) -> Dict:
        contents = self._build_contents(user_prompt, frame_loader, sample_frames)

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=Settings.GEMINI_TEMPERATURE,
            max_output_tokens=Settings.GEMINI_MAX_OUTPUT_TOKENS,
            top_p=Settings.GEMINI_TOP_P,
            top_k=Settings.GEMINI_TOP_K,
            response_mime_type="application/json",
        )

        response = self.client.models.generate_content(
            model=Settings.GEMINI_MODEL,
            contents=contents,
            config=config,
        )

        return self._parse_response(response.text)

    def _build_text_context(self, user_prompt: str, frame_loader: FrameLoader) -> str:
        return (
            f"USER TASK: {user_prompt}\n\n"
            f"VIDEO FRAMES INFO:\n{frame_loader.summary()}\n\n"
            f"FRAMES DIRECTORY PATH: {frame_loader.frames_dir}\n\n"
            f"TOOLBOX REFERENCE:\n{ToolboxRegistry.build_toolbox_description()}\n\n"
            "Analyze the task. Reason about which models and tools to use. "
            "Produce the execution plan and a COMPLETE runnable Python script. "
            "The script must accept the frames directory as sys.argv[1]."
        )

    def _build_contents(self, user_prompt: str, frame_loader: FrameLoader,
                        sample_frames: bool) -> List:
        text_context = self._build_text_context(user_prompt, frame_loader)
        parts = []

        if sample_frames:
            sample_paths = frame_loader.get_sample_paths(max_samples=5)
            if sample_paths:
                text_context += (
                    f"\n\nI'm providing {len(sample_paths)} sample frames "
                    "from the video so you can understand the visual content:"
                )
                parts.append(types.Part.from_text(text=text_context))
                for fp in sample_paths:
                    image_bytes = fp.read_bytes()
                    parts.append(types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ))
                    parts.append(types.Part.from_text(text=f"[Frame: {fp.name}]"))
                return parts

        parts.append(types.Part.from_text(text=text_context))
        return parts

    def _parse_response(self, response_text: str) -> Dict:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)


class PlanExecutor:
    def __init__(self):
        self.workspace = Settings.WORKSPACE_DIR
        Settings.ensure_directories()

    def save_plan(self, plan: Dict, frames_dir: str) -> Path:
        plan_file = self.workspace / "execution_plan.json"
        plan_file.write_text(json.dumps(plan, indent=2))

        if "python_script" in plan:
            script_file = self.workspace / "generated_script.py"
            script_file.write_text(plan["python_script"])
            return script_file

        if "final_prompt" in plan:
            prompt_file = self.workspace / "generated_prompt.txt"
            prompt_file.write_text(plan["final_prompt"])
            return prompt_file

        return plan_file

    def display_plan(self, plan: Dict):
        print("\n" + "=" * 80)
        print("  GEMINI 2.5 — TASK ANALYSIS & EXECUTION PLAN")
        print("=" * 80)

        print(f"\n  Task: {plan.get('task_summary', 'N/A')}")

        if "reasoning" in plan:
            print(f"\n  Reasoning:\n  {plan['reasoning']}")

        steps = plan.get("execution_plan", [])
        print(f"\n  Pipeline ({len(steps)} steps):")
        for step in steps:
            action_tag = step.get("action", "?").upper()
            print(f"    [{step.get('step', '?')}] {action_tag}: {step.get('name', '?')}")
            if "params" in step:
                print(f"        params: {step['params']}")
            print(f"        -> {step.get('output_variable', '?')}  ({step.get('reason', '')})")

        if "python_script" in plan:
            print(f"\n{'=' * 80}")
            print("  GENERATED PYTHON SCRIPT")
            print("=" * 80)
            script_lines = plan["python_script"].split("\n")
            for i, line in enumerate(script_lines, 1):
                print(f"  {i:4d} | {line}")

        print("\n" + "=" * 80)

    def display_final_prompt(self, plan: Dict):
        prompt = plan.get("python_script") or plan.get("final_prompt", "")
        if not prompt:
            print("\n  [No generated output found in plan]")
            return
        print(f"\n{'=' * 80}")
        print("  FINAL OUTPUT (generated by Gemini 2.5 agent)")
        print("=" * 80)
        print(f"\n{prompt}")
        print(f"\n{'=' * 80}")


class CursorMCPBridge:
    def __init__(self):
        self.workspace = Settings.WORKSPACE_DIR
        self.task_file = self.workspace / "current_task.json"
        Settings.ensure_directories()

    def publish_task(self, plan: Dict, frames_dir: str) -> Path:
        cursor_prompt = plan.get("python_script") or plan.get("final_prompt", "")

        task_payload = {
            "task_summary": plan.get("task_summary", ""),
            "reasoning": plan.get("reasoning", ""),
            "execution_plan": plan.get("execution_plan", []),
            "cursor_prompt": cursor_prompt,
            "frames_dir": frames_dir,
            "instruction": (
                "You are a Python video analysis engineer. "
                "Using the cursor_prompt above, write a complete runnable Python script "
                "in workspace/solution.py. The script must:\n"
                "1. Accept frames_dir as sys.argv[1]\n"
                "2. Import models from toolbox.models.* and tools from toolbox.tools.*\n"
                "3. Load model checkpoints via cfg.model_config.MODEL_PATHS\n"
                "4. Execute the full pipeline described in the execution_plan\n"
                "5. Print results to stdout (frame indices, metrics)\n"
                "6. Save result visualization to workspace/result.jpg\n"
                "Then execute the script and return the output."
            ),
        }

        self.task_file.write_text(json.dumps(task_payload, indent=2))
        return self.task_file

    def get_task_status(self) -> str:
        if not self.task_file.exists():
            return "no_task"
        result_file = self.workspace / "result.jpg"
        solution_file = self.workspace / "solution.py"
        if result_file.exists():
            return "completed"
        if solution_file.exists():
            return "code_written"
        return "pending"


class ScriptRunner:
    def __init__(self):
        self.workspace = Settings.WORKSPACE_DIR
        self.project_root = Settings.BASE_DIR
        Settings.ensure_directories()

    def write_solution(self, script_code: str) -> Path:
        solution_path = self.workspace / "solution.py"
        header = (
            "import sys, os\n"
            f"sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))\n\n"
        )
        if "sys.path.insert" not in script_code:
            script_code = header + script_code
        solution_path.write_text(script_code)
        return solution_path

    def execute(self, frames_dir: str, timeout: int = 300) -> Optional[str]:
        solution_path = self.workspace / "solution.py"
        if not solution_path.exists():
            return None

        result = subprocess.run(
            [sys.executable, str(solution_path), frames_dir],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.project_root),
        )

        output = result.stdout
        if result.stderr:
            for line in result.stderr.splitlines():
                if "error" in line.lower() or "traceback" in line.lower():
                    output += "\n" + result.stderr
                    break

        return output


class VideoAnalyticsPipeline:
    WORKSPACE_ARTIFACTS = [
        "solution.py", "generated_script.py",
        "execution_plan.json", "current_task.json", "result.jpg",
    ]

    def __init__(self):
        self.agent = GeminiAgent()
        self.executor = PlanExecutor()
        self.mcp_bridge = CursorMCPBridge()
        self.runner = ScriptRunner()

    def _clean_workspace(self):
        for name in self.WORKSPACE_ARTIFACTS:
            p = Settings.WORKSPACE_DIR / name
            if p.exists():
                p.unlink()

    def run(self, frames_dir: str, user_prompt: str,
            send_frames: bool = True) -> Dict:
        self._clean_workspace()
        frame_loader = FrameLoader(frames_dir)

        if frame_loader.count == 0:
            raise FileNotFoundError(
                f"No frames found in {frames_dir}. "
                "Expected frame_NNNNN.jpg files."
            )

        print(f"\n  Frames: {frame_loader.count} found in {frames_dir}")
        print(f"  Prompt: \"{user_prompt}\"")
        print(f"  Model:  {Settings.GEMINI_MODEL}")
        print(f"  Project: {APIKeys.get_project_id()}")
        print(f"  Sending sample frames: {send_frames}")

        print(f"\n  [1/4] Querying Gemini 2.5...\n")
        plan = self.agent.generate_plan(user_prompt, frame_loader, send_frames)

        print("  [2/4] Plan received. Saving artifacts...\n")
        self.executor.display_plan(plan)
        output_file = self.executor.save_plan(plan, frames_dir)
        print(f"\n  Plan saved to: {output_file}")

        print("\n  [3/4] Publishing task to Cursor MCP bridge...\n")
        task_file = self.mcp_bridge.publish_task(plan, frames_dir)
        print(f"  Task published to: {task_file}")

        exec_output = ""
        print("\n  [4/4] Executing generated script...\n")
        script_code = plan.get("python_script", "")
        if script_code:
            self.runner.write_solution(script_code)
            exec_output = self.runner.execute(frames_dir) or ""

            print("=" * 80)
            print("  EXECUTION OUTPUT")
            print("=" * 80)
            if exec_output:
                for line in exec_output.strip().splitlines():
                    print(f"  {line}")
            else:
                print("  [No output captured]")
            print("=" * 80)

            result_path = Settings.WORKSPACE_DIR / "result.jpg"
            if result_path.exists():
                print(f"\n  Result visualization: {result_path}")
        else:
            print("  [No python_script in plan, skipping execution]")

        print()
        return {"plan": plan, "output": exec_output}


def main():
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python main.py <user_prompt> [frames_dir] [--no-frames]\n\n"
            "Examples:\n"
            "  python main.py \"Find when the bag is picked up\"\n"
            "  python main.py \"Track all people\" videos/sumka_512x512_2fps\n"
            "  python main.py \"Detect hand-object contact\" --no-frames"
        )
        sys.exit(1)

    user_prompt = sys.argv[1]

    frames_dir = Settings.DEFAULT_FRAMES_DIR
    send_frames = True

    for arg in sys.argv[2:]:
        if arg == "--no-frames":
            send_frames = False
        elif not arg.startswith("--"):
            frames_dir = arg

    frames_path = Path(frames_dir)
    if not frames_path.is_absolute():
        frames_path = Settings.BASE_DIR / frames_dir

    if not frames_path.exists():
        print(f"Error: Frames directory not found: {frames_path}")
        sys.exit(1)

    try:
        pipeline = VideoAnalyticsPipeline()
        result = pipeline.run(str(frames_path), user_prompt, send_frames)
    except json.JSONDecodeError as e:
        print(f"\n  Error parsing Gemini response as JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
