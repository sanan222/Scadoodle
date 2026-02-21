# Full Workflow: Gemini Orchestrator → Cursor MCP Pipeline

---

## SYSTEM ARCHITECTURE OVERVIEW

```
User Natural Language Prompt
        │
        ▼
┌──────────────────┐
│  Gemini 2.0 Flash │  ← Structured JSON Orchestrator
│  (Task Planner)   │    (Never processes video itself)
└────────┬─────────┘
         │ Structured JSON Plan
         ▼
┌──────────────────┐
│  Cursor MCP      │  ← Receives plan + writes Python
│  (Code Agent)    │    Runs script, returns results
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Python Runtime  │  ← Executes toolbox functions
│  (Local)         │    on the actual video
└──────────────────┘
```

---

## STEP 1: GEMINI SYSTEM PROMPT (The Orchestrator Brain)

This is the fixed system prompt you give Gemini. It defines all available tools/models and forces structured output.

```
You are a video analysis task planner. You have access to a toolbox of lightweight CPU-efficient computer vision models and tools. Your job is to decompose a user's video analysis request into an ordered execution plan.

AVAILABLE MODELS:
- yolov10: object detection → output: [{label, bbox, confidence, frame_id}]
- fastsam / mobilesam: segmentation with point/bbox/text prompt → output: binary mask
- bytetrack: multi-object tracking → output: [{track_id, bbox, frame_id}]
- raft_small: optical flow → output: flow field (H,W,2)
- midas_small: depth estimation → output: depth map (H,W)
- clip_vit: visual embedding of image region → output: 768-dim vector
- dinov2_small: patch-level visual embedding → output: 384-dim vector
- mediapipe_hands: hand keypoint detection → output: [{landmark, frame_id}]
- mediapipe_pose: full body pose → output: [{keypoints, frame_id}]
- depth_anything_v2: relative depth map → output: depth map (H,W)

AVAILABLE TOOLS:
- cosine_similarity_calculator(vec_a, vec_b) → float
- spatial_relation_checker(bbox_a, bbox_b, depth_map) → {relation: "in_front"|"behind"|"left"|...}
- contact_proximity_checker(mask_a, mask_b, threshold_px) → bool
- motion_peak_detector(flow_magnitudes, threshold) → [frame_ids]
- temporal_event_localizer(signal, condition) → frame_id
- frame_sampler(video, every_n) → [frames]
- keyframe_extractor(video) → [frames]
- mask_extractor(frame, mask) → cropped_region
- hand_object_contact_detector(hand_landmarks, object_mask) → bool
- occlusion_detector(bbox_a, bbox_b, depth_map) → bool
- takeoff_landing_detector(trajectory) → frame_id
- object_appearance_detector(frames, reference_embedding) → frame_id
- threshold_trigger(signal_array, threshold) → frame_id
- trajectory_builder(track_data, track_id) → [(x,y,frame)]
- velocity_calculator(trajectory) → [velocity_per_frame]
- depth_order_comparator(depth_map, region_a, region_b) → {"a_closer": bool}
- frame_differencer(frame_t, frame_t1) → diff_map
- signal_smoother(signal, method) → smoothed_signal

RULES:
1. Minimize model calls. Prefer tools over models where possible.
2. Always prefer detection → tracking → analysis over per-frame inference.
3. Use binary search (temporal_event_localizer) to avoid processing all frames.
4. Output ONLY valid JSON. No explanation text.
5. Specify the exact prompt to pass into Cursor MCP to generate Python code.

OUTPUT FORMAT:
{
  "task_summary": "...",
  "input_requirements": {
    "video_path": "...",
    "reference_image": "optional",
    "reference_mask": "optional"
  },
  "execution_plan": [
    {
      "step": 1,
      "action": "model|tool",
      "name": "yolov10",
      "params": {"classes": ["person", "bag"], "every_n_frames": 5},
      "output_variable": "detections",
      "reason": "Detect bag in sampled frames first"
    }
  ],
  "cursor_prompt": "Full precise coding prompt to give Cursor MCP"
}
```

---

## STEP 2: EXAMPLE GEMINI OUTPUTS

### Prompt: *"Find the takeoff moment in the given video"*

```json
{
  "task_summary": "Detect the frame where an airborne object or person transitions from grounded to airborne state.",
  "input_requirements": {
    "video_path": "required"
  },
  "execution_plan": [
    {
      "step": 1, "action": "model", "name": "yolov10",
      "params": {"classes": ["person", "bird", "plane"], "every_n_frames": 3},
      "output_variable": "detections",
      "reason": "Find target object class first"
    },
    {
      "step": 2, "action": "model", "name": "bytetrack",
      "params": {"detections": "detections"},
      "output_variable": "tracks",
      "reason": "Build consistent track IDs"
    },
    {
      "step": 3, "action": "tool", "name": "trajectory_builder",
      "params": {"track_data": "tracks", "track_id": "auto_primary"},
      "output_variable": "trajectory",
      "reason": "Build y-coordinate trajectory over time"
    },
    {
      "step": 4, "action": "tool", "name": "velocity_calculator",
      "params": {"trajectory": "trajectory"},
      "output_variable": "velocity_signal"
    },
    {
      "step": 5, "action": "tool", "name": "takeoff_landing_detector",
      "params": {"trajectory": "trajectory", "velocity": "velocity_signal"},
      "output_variable": "takeoff_frame",
      "reason": "Detect the frame where vertical velocity changes sign from 0 to negative (upward in image coords)"
    }
  ],
  "cursor_prompt": "Write a Python script using ultralytics YOLOv10 and ByteTrack to detect and track persons/birds in a video (sample every 3rd frame). Build a vertical trajectory (y centroid per frame) for the primary track. Compute per-frame vertical velocity. Find and return the frame index where vertical velocity first crosses below a threshold indicating upward motion (takeoff). Output the frame index and save that frame as takeoff_frame.jpg. Video path is passed as sys.argv[1]."
}
```

---

### Prompt: *"Find first frame where one object is put behind another"*

```json
{
  "execution_plan": [
    {"step": 1, "name": "yolov10", "params": {"every_n_frames": 2}, "output_variable": "detections"},
    {"step": 2, "name": "midas_small", "params": {"frames": "keyframes_only"}, "output_variable": "depth_maps"},
    {"step": 3, "name": "depth_order_comparator",
     "params": {"depth_map": "depth_maps", "region_a": "object_a_bbox", "region_b": "object_b_bbox"},
     "output_variable": "depth_order_per_frame"},
    {"step": 4, "name": "temporal_event_localizer",
     "params": {"signal": "depth_order_per_frame", "condition": "order_changed"},
     "output_variable": "result_frame"}
  ],
  "cursor_prompt": "Write a Python script that: 1) Runs YOLOv8 detection on every 2nd frame. 2) For each frame with 2+ objects detected, runs MiDaS-small depth estimation. 3) Compares average depth of two target object bounding box regions. 4) Tracks the depth ordering across frames. 5) Returns the first frame_id where the depth order between object A and object B flips (A goes behind B). Save that frame as result.jpg."
}
```

---

## STEP 3: CURSOR MCP SETUP

### What is Cursor MCP?
Cursor's **Model Context Protocol** allows external programs (like your orchestrator) to programmatically send prompts to Cursor's AI agent, which then writes and executes code inside a project workspace.

### Installation & Configuration

**1. Install Cursor** (if not already): https://cursor.sh

**2. Enable MCP in Cursor settings:**
```json
// ~/.cursor/mcp.json
{
  "mcpServers": {
    "video-analytics": {
      "command": "python",
      "args": ["/your/project/mcp_server.py"],
      "env": {
        "WORKSPACE": "/your/project/workspace"
      }
    }
  }
}
```

**3. Create your MCP Server (`mcp_server.py`):**
```python
# Minimal MCP server that bridges Gemini output → Cursor
from mcp.server.fastmcp import FastMCP
import subprocess, json, sys

mcp = FastMCP("video-analytics")

@mcp.tool()
def run_video_task(cursor_prompt: str, video_path: str) -> str:
    """Receive a structured coding prompt, write script, run it."""
    # Cursor's agent reads this prompt and writes the code
    # The actual code generation happens inside Cursor's composer
    return json.dumps({
        "prompt": cursor_prompt,
        "video_path": video_path,
        "instruction": "Write and execute this script in the workspace."
    })

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**4. Install MCP SDK:**
```bash
pip install mcp fastmcp
```

---

## STEP 4: FULL ORCHESTRATION SCRIPT

This is the glue that connects everything end-to-end.

```python
# orchestrate.py
import google.generativeai as genai
import json, subprocess, sys
from pathlib import Path

# --- Config ---
GEMINI_MODEL = "gemini-2.0-flash"
WORKSPACE = Path("./workspace")
WORKSPACE.mkdir(exist_ok=True)

genai.configure(api_key="YOUR_GEMINI_API_KEY")

SYSTEM_PROMPT = open("gemini_system_prompt.txt").read()  # The prompt from Step 1

def plan_task(user_prompt: str, video_path: str) -> dict:
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(
        f"VIDEO PATH: {video_path}\nUSER TASK: {user_prompt}",
        generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

def send_to_cursor(cursor_prompt: str, video_path: str):
    """Write prompt file that Cursor MCP picks up and executes."""
    task_file = WORKSPACE / "current_task.json"
    task_file.write_text(json.dumps({
        "prompt": cursor_prompt,
        "video_path": video_path
    }, indent=2))
    print(f"[✓] Task written to {task_file}")
    print("[→] Cursor MCP will now generate and run the script.")
    # Optionally trigger Cursor via CLI or watch mechanism here

def main():
    video_path = sys.argv[1]
    user_prompt = sys.argv[2]

    print(f"[1] Planning task with Gemini: '{user_prompt}'")
    plan = plan_task(user_prompt, video_path)

    print(f"[2] Task Summary: {plan['task_summary']}")
    print(f"[3] Execution Steps: {len(plan['execution_plan'])}")
    for step in plan['execution_plan']:
        print(f"    Step {step['step']}: {step['name']} → {step.get('output_variable','')}")

    print(f"[4] Sending to Cursor MCP...")
    send_to_cursor(plan['cursor_prompt'], video_path)

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python orchestrate.py ./video.mp4 "find the takeoff moment in the given video"
```

---

## STEP 5: CURSOR AGENT BEHAVIOR

Inside Cursor, configure the **Agent / Composer** with this standing instruction:

```
You are a Python video analysis engineer. When you receive a task from current_task.json:
1. Read the prompt and video_path fields.
2. Write a complete, runnable Python script in workspace/solution.py using only:
   - ultralytics, opencv-python, torch, torchvision, transformers,
     fastsam, mediapipe, numpy, scipy
3. Use the exact pipeline described in the prompt.
4. Add a __main__ block that reads video_path from the JSON and prints the result frame index.
5. Run the script immediately using the terminal tool.
6. Report the output frame index and save the result frame as workspace/result.jpg.
```

---

## EFFICIENCY PRINCIPLES GEMINI MUST FOLLOW

| Principle | Implementation |
|---|---|
| **Sample first, densify later** | Always run detection on every N frames, then zoom in with binary search |
| **Track, don't detect per frame** | Run detector once, hand off to tracker for the rest |
| **Depth only where needed** | Only run depth model on candidate frames, not all frames |
| **Avoid VLMs** | Never use GPT-4V/Gemini Vision on video frames — only embeddings (CLIP/DINOv2) |
| **Binary search for first-frame tasks** | Instead of scanning all frames, use temporal_event_localizer to bisect |
| **Embeddings over VLMs for similarity** | Use CLIP cosine similarity instead of asking a VLM "does this look like X?" |