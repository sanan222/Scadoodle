import os
import re
import sys
import uuid
import shutil
import traceback
from pathlib import Path

import cv2
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

from cfg import Settings
from main import VideoAnalyticsPipeline

app = Flask(__name__, static_folder=None)
CORS(app)

UPLOAD_DIR = Settings.BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_sessions = {}


def extract_frames(video_path: Path, output_dir: Path, fps: int = 2) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(video_fps / fps))
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            saved += 1
            fname = f"frame_{saved:05d}.jpg"
            h, w = frame.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            cv2.imwrite(str(output_dir / fname), frame)
        frame_idx += 1

    cap.release()
    return saved


# ── Static file serving ─────────────────────────────────────

FRONTEND_DIR = Settings.BASE_DIR / "frontend" / "frontend"


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    if (FRONTEND_DIR / filename).is_file():
        return send_from_directory(str(FRONTEND_DIR), filename)
    return "Not found", 404


# ── API routes ───────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    video_id = str(uuid.uuid4())[:8]
    session_dir = UPLOAD_DIR / video_id
    session_dir.mkdir(parents=True, exist_ok=True)

    video_path = session_dir / video_file.filename
    video_file.save(str(video_path))

    frames_dir = session_dir / "frames"
    try:
        frame_count = extract_frames(video_path, frames_dir, fps=2)
    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        return jsonify({"error": f"Frame extraction failed: {e}"}), 500

    frame_names = sorted(f.name for f in frames_dir.iterdir() if f.suffix == ".jpg")

    _sessions[video_id] = {
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "frame_count": frame_count,
        "video_name": video_file.filename,
    }

    return jsonify({
        "video_id": video_id,
        "frame_count": frame_count,
        "video_name": video_file.filename,
        "frames": frame_names,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    video_id = data.get("video_id")
    prompt = data.get("prompt", "").strip()

    if not video_id or video_id not in _sessions:
        return jsonify({"error": "Invalid or missing video_id"}), 400
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    session = _sessions[video_id]
    frames_dir = session["frames_dir"]

    try:
        pipeline = VideoAnalyticsPipeline()
        result = pipeline.run(frames_dir, prompt, send_frames=True)

        plan = result["plan"]
        exec_output = result["output"]

        result_path = Settings.WORKSPACE_DIR / "result.jpg"
        has_result_image = result_path.exists()

        frame_id = None
        frame_name = None
        for line in exec_output.splitlines():
            lower = line.lower()
            if "frame" in lower and any(c.isdigit() for c in line):
                numbers = re.findall(r"\d+", line)
                if numbers:
                    frame_id = int(numbers[-1])
                    frame_name = f"frame_{frame_id:05d}.jpg"
                    break

        ts = int(os.path.getmtime(str(result_path))) if has_result_image else 0

        return jsonify({
            "task_summary": plan.get("task_summary", ""),
            "reasoning": plan.get("reasoning", ""),
            "frame_id": frame_id,
            "frame_name": frame_name,
            "has_result_image": has_result_image,
            "result_image_url": f"/api/result/{video_id}?t={ts}" if has_result_image else None,
            "output": exec_output.strip(),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/result/<video_id>")
def get_result_image(video_id):
    result_path = Settings.WORKSPACE_DIR / "result.jpg"
    if result_path.exists():
        return send_file(str(result_path), mimetype="image/jpeg")
    return "No result image", 404


@app.route("/api/frames/<video_id>/<frame_name>")
def get_frame(video_id, frame_name):
    if video_id not in _sessions:
        return "Invalid session", 404
    frames_dir = Path(_sessions[video_id]["frames_dir"])
    frame_path = frames_dir / frame_name
    if frame_path.exists():
        return send_file(str(frame_path), mimetype="image/jpeg")
    return "Frame not found", 404


if __name__ == "__main__":
    print("\n  Scadoodle Backend Server")
    print("  ========================")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
