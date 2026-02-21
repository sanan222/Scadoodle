# Project Structure Overview

## Directory Layout

```
Cursor_hack/                            # Root directory
│
├── .cursor/                            # Cursor IDE configuration
│   ├── .cursorrules                    # AI coding rules (305 lines)
│   └── STRUCTURE.md                    # This file
│
├── cfg/                                # Configuration management
│   ├── __init__.py
│   ├── api_keys.py                     # API key handling
│   ├── model_config.py                 # Model paths & configs
│   └── settings.py                     # General settings
│
├── toolbox/                            # Core CV library
│   ├── __init__.py
│   │
│   ├── models/                         # Computer vision models
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseModel abstract class
│   │   ├── detection.py                # YOLODetector
│   │   ├── segmentation.py             # FastSAM, MobileSAM
│   │   ├── tracking.py                 # ByteTracker
│   │   ├── flow.py                     # RAFTFlow
│   │   └── pose.py                     # MediaPipe Hands/Pose
│   │
│   ├── tools/                          # Analysis tools
│   │   ├── __init__.py
│   │   ├── spatial.py                  # Spatial analysis
│   │   ├── motion.py                   # Motion analysis
│   │   ├── temporal.py                 # Temporal analysis
│   │   └── interaction.py              # Interaction detection
│   │
│   └── toolbox.md                      # Toolbox documentation
│
├── scripts/                            # Utility scripts
│   └── download_models.py              # Model downloader (142 lines)
│
├── models/                             # Model checkpoints (236 MB)
│   ├── yolov10n.pt                     # 5.6 MB
│   ├── yolov9c.pt                      # 50 MB
│   ├── fastsam-x.pt                    # 139 MB
│   ├── mobile_sam.pt                   # 39 MB
│   ├── raft_small.pth                  # 3.9 MB
│   └── *.txt, *.md                     # Documentation
│
├── prompts/                            # LLM system prompts
│   └── system_prompt.txt               # Gemini orchestrator prompt
│
├── workspace/                          # Code execution (gitignored)
│
├── main.py                             # Entry point (66 lines)
├── requirements.txt                    # Python dependencies
├── project_overview.md                 # Architecture reference
├── README.md                           # Project documentation
└── .gitignore                          # Git ignore rules
```

## Code Statistics

- **Total Lines**: 670 (excluding documentation)
- **Main Entry**: 66 lines
- **Configuration**: 4 files
- **Models**: 7 classes across 6 files
- **Tools**: 4 classes across 4 files
- **Rules**: 305 lines in .cursorrules

## Key Principles

1. **No automatic README generation**
2. **Strict OOP architecture** (abstract base classes, dependency injection)
3. **Configuration isolation** (all config in cfg/)
4. **Clean code** (no comments, self-documenting)
5. **Modular structure** (models and tools separately organized)

## Three-Layer Architecture

```
User Prompt
    ↓
[Gemini Planner] → JSON execution plan
    ↓
[Cursor MCP] → Generate Python code
    ↓
[Python Runtime] → Execute on video
    ↓
Result (frame index, saved image)
```

## Import Examples

```python
# Configuration
from cfg import APIKeys, MODEL_PATHS, Settings

# Models
from toolbox.models import YOLODetector, FastSAMSegmenter
from toolbox.models import ByteTracker, RAFTFlow
from toolbox.models import MediaPipeHands

# Tools
from toolbox.tools import SpatialTools, MotionTools
from toolbox.tools import TemporalTools, InteractionTools
```

## Development Workflow

1. User runs: `python main.py video.mp4 "find takeoff moment"`
2. GeminiPlanner reads system_prompt.txt and creates JSON plan
3. CursorBridge writes task to workspace/current_task.json
4. Cursor agent generates Python script using toolbox
5. Script executes, outputs result frame and index

---

Last updated: 2026-02-20
