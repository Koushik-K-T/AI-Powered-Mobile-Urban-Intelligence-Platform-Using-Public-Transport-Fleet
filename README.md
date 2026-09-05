# RoadSense — AI-Powered Pothole Detection System

> Real-time pothole detection using **YOLO26m** (mfranzon/pothole-yolo26) with a FastAPI backend, SQLite database, and Preact+Leaflet frontend.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [File-by-File Documentation](#file-by-file-documentation)
- [Technology Stack](#technology-stack)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Detection Pipeline](#detection-pipeline)
- [Database Schema](#database-schema)
- [Frontend Tabs](#frontend-tabs)
- [Model Details](#model-details)
- [Optional: Multi-Class Training (RDD2022)](#optional-multi-class-training-rdd2022)
- [Known Issues & Gotchas](#known-issues--gotchas)
- [Troubleshooting](#troubleshooting)
- [Development Notes for AI Assistants](#development-notes-for-ai-assistants)

---

## Overview

RoadSense is a web application that detects **potholes** from dashcam video uploads and simulated live bus GPS feeds. It uses a YOLO26m model trained on real pothole data to perform object detection, pins detections on an interactive Leaflet map, and clusters nearby detections into confirmed road issues.

### Key Features

1. **Video Upload & Detect** — Upload an MP4/AVI/MOV video, YOLO26 processes every 10th frame, streams results via SSE (Server-Sent Events)
2. **Live Bus Feed** — Simulated bus GPS pings with fake defect detections (for demo purposes)
3. **Road Issues Dashboard** — Clustered/fused detections using 20m haversine radius; 2+ hits = "Confirmed"
4. **Connectivity Settings** — UI demo of AIS-140, Nirbhaya 4G, and M2M IoT depot WiFi modes

### What Changed (YOLO26 Upgrade)

The app was originally using a generic COCO-trained YOLOv8m model (`yolov8m.pt`) with a `COCO_REMAP` dict that faked road defect detections by relabeling COCO classes (car, bus, person) as potholes/cracks. **The app was never detecting real road damage.**

This has been replaced with:
- **Model**: `mfranzon/pothole-yolo26` (YOLO26m architecture, 44MB, single class: Pothole)
- **No COCO fallback** — if the model file is missing, the app errors with a clear message
- **No COCO_REMAP** — completely removed from detect.py
- **Single class**: `{0: "Pothole"}` — the model only detects potholes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (Frontend)                    │
│  Preact + htm (no build step) │ Leaflet.js maps              │
│  4 tabs: Upload | Live Feed | Issues | Connectivity          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend (main.py)                  │
│  POST /upload      → save video, return job_id               │
│  GET  /stream/:id  → SSE stream of YOLO detections           │
│  POST /bus-ping    → receive simulated bus detections         │
│  GET  /live-feed   → latest 50 bus pings                     │
│  GET  /issues      → clustered road issues                   │
│  GET  /health      → health check                            │
└──────┬────────────────────────┬─────────────────────────────┘
       │                        │
┌──────▼──────┐    ┌────────────▼────────────────────────────┐
│  detect.py  │    │           database.py                    │
│  YOLO26m    │    │  SQLite: detections, issues, bus_pings   │
│  GPU infer  │    │  Haversine clustering (20m radius)       │
└─────────────┘    └─────────────────────────────────────────┘
       │
┌──────▼──────────────┐
│  models/             │
│  road_damage.pt      │
│  (YOLO26m, 44MB)     │
└──────────────────────┘
```

---

## Directory Structure

```
web/
├── main.py                 # FastAPI application — all HTTP routes
├── detect.py               # YOLO26 inference engine — SSE async generator
├── database.py             # SQLite schema, CRUD, haversine clustering
├── download_model.py       # Model verification script (no longer downloads)
├── simulate_bus.py         # Fake GPS bus pings → POST /bus-ping
├── requirements.txt        # Python dependencies
├── setup.bat               # Windows setup script
├── setup.sh                # Linux/macOS setup script
├── convert_rdd2022.py      # [OPTIONAL] RDD2022 VOC→YOLO dataset converter
├── train_rdd.py            # [OPTIONAL] Fine-tune script for multi-class
├── road_defects.db         # SQLite database (auto-created)
├── frontend/
│   └── index.html          # Single-file Preact+htm frontend (no build step)
├── models/
│   └── road_damage.pt      # YOLO26m pothole model (44MB)
└── uploads/
    └── frames/             # Saved detection frame JPEGs for lightbox
```

---

## File-by-File Documentation

### `main.py` — FastAPI Application (231 lines)

The main HTTP server. All routes are defined here.

**Key routes:**
| Route | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "timestamp": "..."}` |
| `/upload` | POST | Accepts `multipart/form-data` with a video file. Returns `{"job_id": "uuid", "filename": "..."}` |
| `/stream/{job_id}` | GET | SSE endpoint — streams YOLO processing events for a job. Event types: `progress`, `detection`, `done`, `error` |
| `/detections/{job_id}` | GET | Returns all detections for a completed job |
| `/bus-ping` | POST | Accepts JSON `{bus_id, lat, lng, defect_type?, confidence?}` from bus simulator |
| `/live-feed` | GET | Returns latest 50 bus pings/detections |
| `/issues` | GET | Returns all fused/clustered road issues |
| `/issues/{id}/status` | PATCH | Update issue status (`Pending`/`Assigned`/`Fixed`) |
| `/thumbnail/{det_id}` | GET | Returns detection thumbnail as JPEG |
| `/frame/{det_id}` | GET | Returns full detection frame as JPEG (with bbox drawn) |
| `/` | GET | Serves `frontend/index.html` via `StaticFiles` mount |

**Important behaviors:**
- Only **one video** can be processed at a time (global `current_processing_job` lock)
- The frontend is served as static files from `frontend/` directory
- Database is initialized on startup via `lifespan` context manager
- CORS is fully open (`allow_origins=["*"]`) for development

**Dependencies:** `fastapi`, `uvicorn`, `aiosqlite`, `python-multipart`

---

### `detect.py` — YOLO26 Inference Engine (282 lines)

The core detection pipeline. Called by `main.py`'s `/stream/{job_id}` endpoint.

**Key constants:**
```python
MODEL_PATH = Path("models/road_damage.pt")
SAMPLE_EVERY = 10           # Process every 10th frame
CONFIDENCE_THRESHOLD = 0.25 # Minimum detection confidence
BENGALURU_LAT = 12.9716     # Default GPS if video has no metadata
BENGALURU_LNG = 77.5946

CLASS_MAP = {0: "Pothole"}  # Single class from pothole-yolo26
```

**Key functions:**
| Function | Description |
|---|---|
| `load_model()` | Loads YOLO model from `models/road_damage.pt`. Validates it has "Pothole" class. Raises `FileNotFoundError` if missing. |
| `get_model()` | Singleton pattern — caches the model after first load |
| `extract_gps_from_video(path)` | Tries `ffprobe` to extract GPS from video metadata (ISO 6709 format) |
| `frame_to_base64(frame, bbox)` | Crops to bounding box, resizes to 120px max, encodes as base64 JPEG thumbnail |
| `full_frame_to_base64(frame, bbox)` | Full frame with bbox rectangle drawn, max 1280px wide, base64 JPEG |
| `process_video(video_path, job_id)` | **Async generator** — the main pipeline. Yields SSE events. |

**`process_video()` flow:**
1. Load model (in thread pool — blocking call)
2. Open video with `cv2.VideoCapture`
3. Extract GPS from video metadata (or use Bengaluru defaults)
4. For every `SAMPLE_EVERY`th frame:
   - Run YOLO prediction (in thread pool)
   - For each detection with `cls_id in CLASS_MAP` and `conf >= 0.25`:
     - Generate thumbnail + full frame base64
     - Insert detection into database
     - Run `cluster_and_fuse()` for issue tracking
     - Yield SSE `detection` event
   - Yield SSE `progress` event (always)
5. Yield SSE `done` event with total detection count

**GPS simulation:** If no GPS is embedded in the video, positions are simulated by incrementally offsetting from Bengaluru center coordinates.

---

### `database.py` — SQLite Schema & Clustering (169 lines)

Async SQLite database using `aiosqlite`. Auto-creates tables on startup.

**Tables:**

#### `detections`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `job_id` | TEXT | UUID of the upload job, or `"bus_feed"` for bus pings |
| `source` | TEXT | `"upload"` or `"bus"` |
| `bus_id` | TEXT | Null for uploads, bus name for bus pings |
| `frame_num` | INTEGER | Video frame number (null for bus pings) |
| `defect_type` | TEXT | `"Pothole"` |
| `confidence` | REAL | 0.0–1.0 |
| `lat` | REAL | GPS latitude |
| `lng` | REAL | GPS longitude |
| `thumbnail` | TEXT | Base64-encoded JPEG thumbnail |
| `created_at` | TEXT | ISO 8601 UTC timestamp |

#### `issues`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `cluster_key` | TEXT UNIQUE | `"{lat_4dp}_{lng_4dp}_{defect_type}"` |
| `defect_type` | TEXT | `"Pothole"` |
| `lat` / `lng` | REAL | Cluster center coordinates |
| `detection_count` | INTEGER | Number of times detected at this location |
| `priority` | TEXT | `"High"` (Pothole) or `"Low"` (other) |
| `status` | TEXT | `"Pending"` / `"Assigned"` / `"Fixed"` |
| `confirmed` | INTEGER | `1` if `detection_count >= 2`, else `0` |
| `first_seen` / `last_seen` | TEXT | ISO 8601 timestamps |

#### `bus_pings`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `bus_id` | TEXT | Bus identifier (e.g., `"Bus 12"`) |
| `lat` / `lng` | REAL | GPS coordinates |
| `defect_type` | TEXT | Null if no detection |
| `confidence` | REAL | Null if no detection |
| `created_at` | TEXT | ISO 8601 UTC timestamp |

**Key functions:**
| Function | Description |
|---|---|
| `init_db()` | Creates all tables if they don't exist |
| `insert_detection(...)` | Inserts a detection row, returns `lastrowid` |
| `insert_bus_ping(...)` | Inserts a bus ping row |
| `cluster_and_fuse(lat, lng, defect_type)` | **Core clustering logic** — finds existing issues within 20m (haversine), increments count or creates new issue |
| `get_detections_by_job(job_id)` | Returns all detections for a job, ordered by frame |
| `get_live_feed(limit)` | Returns latest bus pings + bus detections (UNION ALL) |
| `get_issues()` | Returns all issues, ordered by detection count DESC |
| `defect_priority(defect_type)` | Returns `"High"` for Pothole, `"Low"` for everything else |
| `haversine(lat1, lon1, lat2, lon2)` | Returns distance in metres between two GPS coords |

**Clustering algorithm:**
1. For each new detection at `(lat, lng)` with type `defect_type`:
2. Query all existing issues with matching `defect_type`
3. For each existing issue, compute haversine distance
4. If any issue is within **20 metres**: increment its `detection_count`, set `confirmed=1` if count ≥ 2
5. Else: create a new issue with `cluster_key = "{lat:.4f}_{lng:.4f}_{defect_type}"`

---

### `download_model.py` — Model Verification (56 lines)

**This file NO LONGER downloads anything.** It verifies that `models/road_damage.pt` exists and has the correct classes.

```bash
python download_model.py
# Output: ✅ Custom road-damage model loaded.
#         Path:    C:\...\models\road_damage.pt
#         Size:    42.0 MB
#         Classes: {0: 'Pothole'}
```

If the model is missing, it prints setup instructions. **There is NO silent fallback to a COCO model.**

---

### `simulate_bus.py` — Bus GPS Simulator (109 lines)

Standalone script that simulates 3 buses on Bengaluru routes, POSTing fake detections to `/bus-ping` every 10 seconds.

**Buses:**
| Bus ID | Starting Route |
|---|---|
| Bus 12 | Silk Board → Koramangala |
| Bus 7 | Electronic City → HSR Layout |
| Bus 23 | Yeshwantpur → Rajajinagar |

**Behavior:**
- Each tick, each bus randomly walks its GPS position (±0.003° per tick)
- 65% chance of generating a fake defect detection per tick
- Defect types sent: `"Pothole"`, `"Longitudinal Crack"`, `"Transverse Crack"`, `"Alligator Crack"`

> **Note:** The bus simulator still sends all 4 defect types for demo/testing purposes. These are accepted by the backend and stored in the database — they are NOT model detections, just simulated data. Only the video upload pipeline uses the actual YOLO26 model.

---

### `frontend/index.html` — Single-File Frontend (968 lines)

A complete single-file web application using:
- **Preact + htm** (loaded from CDN, no build step needed)
- **Leaflet.js** for interactive maps
- **Google Fonts**: Inter (body) + Outfit (headings)
- **Vanilla CSS** with CSS variables for theming

**No build step required.** The file is served directly as a static file by FastAPI.

**4 Tabs:**

| Tab | Description |
|---|---|
| **Upload & Detect** | Video drag-and-drop upload zone, progress bar with SSE streaming, detection results table with thumbnails, Leaflet map with detection pins |
| **Live Bus Feed** | Bus status cards, recent detections table, bus position map. Auto-polls `/live-feed` every 5 seconds |
| **Road Issues** | Stats row (total/confirmed/unverified/fixed), filterable issues table (type/status/priority), issues map with clickable pins |
| **Connectivity** | Radio card selector for AIS-140 SIM, Nirbhaya 4G, M2M IoT modes with simulated status panels |

**Key JavaScript globals:**
- `API = 'http://localhost:8000'` — backend URL (hardcoded)
- `uploadMap`, `liveMap`, `issuesMap` — Leaflet map instances
- `BLAT = 12.9716, BLNG = 77.5946` — Bengaluru center coordinates

**Lightbox:** Clicking a thumbnail opens a full-frame lightbox overlay with the detection bbox drawn. Press Escape to close.

**SSE Flow (Upload & Detect):**
1. User drops video file → `POST /upload` → gets `job_id`
2. Opens `EventSource` to `GET /stream/{job_id}`
3. Receives `progress` events → updates progress bar
4. Receives `detection` events → appends row to table, adds map pin
5. Receives `done` event → closes EventSource, shows completion message

---

### `convert_rdd2022.py` — [OPTIONAL] Dataset Converter (210 lines)

Converts RDD2022 PASCAL VOC dataset to YOLO format for multi-class training. **Not needed for pothole-only detection.**

```bash
python convert_rdd2022.py --input_dir path/to/RDD2022 --output_dir path/to/output
```

Maps: `D00→0 (Longitudinal Crack)`, `D10→1 (Transverse Crack)`, `D20→2 (Alligator Crack)`, `D40→3 (Pothole)`

---

### `train_rdd.py` — [OPTIONAL] Fine-Tune Script (44 lines)

Fine-tunes the pothole-yolo26 checkpoint on RDD2022 4-class data. **Not needed for pothole-only detection.**

```bash
python train_rdd.py
```

Uses: `epochs=60`, `imgsz=512`, `batch=-1` (AutoBatch), `patience=15`, `device=0`

---

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| **Backend** | Python + FastAPI | Python 3.14, FastAPI ≥0.110 |
| **ASGI Server** | Uvicorn | ≥0.29 |
| **AI Model** | YOLO26m (Ultralytics) | ultralytics ≥8.1 |
| **Deep Learning** | PyTorch + CUDA | torch 2.14.0+cu126 |
| **GPU** | NVIDIA RTX 3050 Laptop | 4GB VRAM |
| **Database** | SQLite via aiosqlite | aiosqlite ≥0.20 |
| **Image Processing** | OpenCV + Pillow | opencv-python-headless ≥4.9, Pillow ≥10 |
| **HTTP Client** | httpx | ≥0.27 (for bus simulator) |
| **Model Hub** | HuggingFace Hub | ≥0.22 |
| **Frontend** | Vanilla HTML/CSS/JS | Single file, no build |
| **Maps** | Leaflet.js | 1.9.4 (CDN) |
| **Fonts** | Google Fonts | Inter + Outfit |

### `requirements.txt`
```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
ultralytics>=8.1.0
aiosqlite>=0.20.0
python-multipart>=0.0.9
httpx>=0.27.0
Pillow>=10.0.0
opencv-python-headless>=4.9.0
huggingface-hub>=0.22.0
```

> **Note:** `torch` is NOT in requirements.txt — it is installed automatically as a dependency of `ultralytics`. If you need GPU support, install the CUDA version of PyTorch BEFORE installing ultralytics: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

---

## Setup & Installation

### Prerequisites
- Python 3.10+ (tested on 3.14)
- NVIDIA GPU with CUDA support (optional, CPU also works but slower)
- `pip` package manager

### Step-by-Step

```bash
# 1. Clone the repository
git clone <repo-url>
cd web

# 2. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/macOS

# 3. Install PyTorch with CUDA (if you have an NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Download the YOLO26 pothole model
python -c "from huggingface_hub import hf_hub_download; import shutil; p = hf_hub_download(repo_id='mfranzon/pothole-yolo26', filename='best.pt'); shutil.copy(p, 'models/road_damage.pt')"

# 6. Verify model loaded correctly
python download_model.py
# Should print: ✅ Custom road-damage model loaded.

# 7. Initialize database
python -c "import asyncio; import database; asyncio.run(database.init_db()); print('Database OK')"
```

Or use the setup scripts:
```bash
setup.bat   # Windows
./setup.sh  # Linux/macOS
```

---

## Running the Application

You need **two terminals**:

### Terminal 1 — Backend Server
```bash
uvicorn main:app --reload
# or
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2 — Bus Simulator (optional, for demo)
```bash
python simulate_bus.py
```

### Browser
Open **http://localhost:8000**

---

## API Reference

### `POST /upload`
Upload a video file for processing.

**Request:** `multipart/form-data` with field `file` (video file)

**Response:**
```json
{"job_id": "550e8400-e29b-41d4-a716-446655440000", "filename": "dashcam.mp4"}
```

**Error 409:** Another video is already being processed.

---

### `GET /stream/{job_id}`
SSE endpoint — streams YOLO processing events.

**Response:** `text/event-stream`

**Event types:**

```json
// Progress update (emitted every SAMPLE_EVERY frames)
{"type": "progress", "frame": 100, "total": 5000, "pct": 20.0, "sample_index": 10, "sample_total": 500}

// Detection found
{"type": "detection", "frame": 100, "total": 5000, "pct": 20.0, "defect_type": "Pothole", "confidence": 0.87, "lat": 12.9716, "lng": 77.5946, "id": 42, "thumbnail": "base64...", "full_frame": "base64..."}

// Processing complete
{"type": "done", "total_detections": 15, "job_id": "..."}

// Error
{"type": "error", "message": "Cannot open video file"}
```

---

### `POST /bus-ping`
Accept a bus detection ping.

**Request body:**
```json
{"bus_id": "Bus 12", "lat": 12.9716, "lng": 77.5946, "defect_type": "Pothole", "confidence": 0.85}
```

**Response:** `{"status": "ok"}`

---

### `GET /live-feed`
Returns latest 50 bus pings/detections.

**Response:** Array of `{bus_id, defect_type, confidence, lat, lng, created_at, kind}` where `kind` is `"detection"` or `"ping"`.

---

### `GET /issues`
Returns all fused/clustered road issues, ordered by detection count DESC.

---

### `PATCH /issues/{id}/status`
Update issue status.

**Request body:** `{"status": "Pending"}` or `"Assigned"` or `"Fixed"`

---

### `GET /thumbnail/{det_id}`
Returns detection thumbnail as `image/jpeg`.

### `GET /frame/{det_id}`
Returns full detection frame (with bbox drawn) as `image/jpeg`.

---

## Detection Pipeline

```
Video File
    │
    ▼
cv2.VideoCapture ──► Read every 10th frame
    │
    ▼
YOLO26m.predict(frame, conf=0.25)
    │
    ▼
For each box where cls_id ∈ {0: "Pothole"}:
    ├──► Generate thumbnail (120px max, base64 JPEG)
    ├──► Generate full frame with bbox (1280px max, base64 JPEG)
    ├──► INSERT INTO detections (...)
    ├──► cluster_and_fuse(lat, lng, "Pothole")
    └──► Yield SSE "detection" event
    │
    ▼
Yield SSE "progress" event (always, every sampled frame)
    │
    ▼
Yield SSE "done" event (when video ends)
```

---

## Model Details

### Current Model: `mfranzon/pothole-yolo26`

| Property | Value |
|---|---|
| Architecture | YOLO26m |
| File | `models/road_damage.pt` (44MB) |
| Source | [HuggingFace: mfranzon/pothole-yolo26](https://huggingface.co/mfranzon/pothole-yolo26) |
| Classes | `{0: 'Pothole'}` (single class) |
| Layers | 280 |
| Parameters | 21,774,430 |
| GFLOPs | 75.0 |
| Inference speed | ~19ms per frame on RTX 3050 at 512×384 |

### How to swap to a different model

1. Place your `.pt` file at `models/road_damage.pt`
2. Update `CLASS_MAP` in `detect.py` to match your model's classes
3. Update `defect_priority()` in `database.py` if adding new defect types
4. Update the filter `<select>` in `frontend/index.html` (search for `id="filterType"`)
5. Update `defectColor()` in `frontend/index.html` to add colors for new classes
6. Run `python download_model.py` to verify

---

## Optional: Multi-Class Training (RDD2022)

If you want to detect 4 types of road damage (not just potholes), you can fine-tune the model on the RDD2022 dataset.

### Target classes (after training):
```
0: D00 — Longitudinal Crack
1: D10 — Transverse Crack
2: D20 — Alligator Crack
3: D40 — Pothole
```

### Steps:

1. **Download RDD2022** dataset (PASCAL VOC format, ~3GB)

2. **Convert to YOLO format:**
   ```bash
   python convert_rdd2022.py --input_dir path/to/RDD2022 --output_dir rdd2022_yolo
   ```

3. **Fine-tune:**
   ```bash
   # Edit train_rdd.py to set correct DATA_YAML path
   python train_rdd.py
   ```

4. **Deploy trained model:**
   ```bash
   copy runs\pothole_train\yolo26_rdd_v1\weights\best.pt models\road_damage.pt
   ```

5. **Update code for 4 classes** — modify `CLASS_MAP` in `detect.py`:
   ```python
   CLASS_MAP = {
       0: "Longitudinal Crack",  # D00
       1: "Transverse Crack",    # D10
       2: "Alligator Crack",     # D20
       3: "Pothole",             # D40
   }
   ```

6. **Update `defect_priority()` in `database.py`:**
   ```python
   def defect_priority(defect_type: str) -> str:
       if defect_type in ("Pothole", "Alligator Crack"):
           return "High"
       if defect_type == "Transverse Crack":
           return "Medium"
       return "Low"
   ```

7. **Update frontend filter dropdown** in `frontend/index.html`:
   ```html
   <option value="Pothole">Pothole</option>
   <option value="Alligator Crack">Alligator Crack</option>
   <option value="Transverse Crack">Transverse Crack</option>
   <option value="Longitudinal Crack">Longitudinal Crack</option>
   ```

8. **Update `defectColor()`** in `frontend/index.html`:
   ```javascript
   function defectColor(type) {
     const map = {
       'Pothole':'#EF4444',
       'Alligator Crack':'#F59E0B',
       'Transverse Crack':'#8B5CF6',
       'Longitudinal Crack':'#3B82F6'
     };
     return map[type] || '#64748b';
   }
   ```

---

## Known Issues & Gotchas

### 1. `COCO_REMAP` is gone — intentionally
The old code had a `COCO_REMAP` dict that relabeled COCO classes as road defects. This was a demo hack and produced completely fake results. It has been **removed entirely**. If someone re-adds it, they are undoing the fix.

### 2. Single-class model
The current model only detects `Pothole`. The bus simulator (`simulate_bus.py`) still sends all 4 defect types — this is fine because those are simulated, not real model detections.

### 3. No yolov8m.pt fallback
The old code silently fell back to `yolov8m.pt` (a COCO model) if the road damage model was missing. This fallback has been **removed**. If `models/road_damage.pt` is missing, the app will error with a clear message.

### 4. GPU memory
The RTX 3050 has only 4GB VRAM. YOLO26m at 512×384 uses ~1.5GB. If you run other GPU applications simultaneously, you may get CUDA OOM errors.

### 5. One video at a time
The backend enforces a global lock — only one video can be processed at a time. Attempting to upload while another is processing returns HTTP 409.

### 6. Frontend API URL is hardcoded
`const API = 'http://localhost:8000'` is hardcoded in `frontend/index.html` (line 513). If you deploy to a different host/port, you must change this.

### 7. Database file location
`road_defects.db` is created in the working directory (wherever you run `uvicorn` from). Always run from the project root.

### 8. `bus.jpg` artifact
Running YOLO predict with a URL (e.g., `https://ultralytics.com/images/bus.jpg`) downloads the image to CWD. This is an Ultralytics behavior, not a bug. The file has been cleaned up.

---

## Troubleshooting

### Model not found error
```
FileNotFoundError: Model not found at models/road_damage.pt
```
**Fix:** Run the download command:
```bash
python -c "from huggingface_hub import hf_hub_download; import shutil; p = hf_hub_download(repo_id='mfranzon/pothole-yolo26', filename='best.pt'); shutil.copy(p, 'models/road_damage.pt')"
```

### CUDA out of memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Fix:** Close other GPU applications, or reduce `SAMPLE_EVERY` in `detect.py` (higher = fewer frames processed = less memory pressure).

### Port already in use
```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)
```
**Fix:** Kill the existing process or use a different port: `uvicorn main:app --port 8001`

### No detections in video
The model may not detect potholes in every video. Try a video that clearly shows road damage. Also check `CONFIDENCE_THRESHOLD` in `detect.py` — lowering from `0.25` to `0.15` may help at the cost of more false positives.

### Frontend map not loading
Leaflet.js is loaded from CDN. Ensure you have internet access. The tiles use OpenStreetMap (`https://{s}.tile.openstreetmap.org`).

---

## Development Notes for AI Assistants

### Critical rules — DO NOT violate these:

1. **Do NOT re-add `COCO_REMAP`** — the entire point of the YOLO26 upgrade was to remove fake COCO-based detections.

2. **Do NOT add a fallback to `yolov8m.pt`** or any COCO model — if the model is missing, error loudly.

3. **`CLASS_MAP` must match the model's actual classes** — currently `{0: "Pothole"}` for the single-class model.

4. **The frontend has no build step** — `frontend/index.html` is a single file with inline CSS and JS. Do NOT introduce React, Vue, Vite, or any build tooling unless explicitly asked.

5. **The database schema should not change** — the `detections`, `issues`, and `bus_pings` tables are stable. Adding columns is fine, removing or renaming is not (existing data would break).

6. **SSE event format must stay stable** — the frontend parses `type`, `detection`, `progress`, and `done` events with specific field names. Do not rename fields.

### When modifying the detection pipeline:

- `detect.py` handles ALL inference logic. `main.py` only calls `detect.process_video()`.
- The model is loaded once (singleton via `get_model()`). Do not reload it per-request.
- YOLO inference is run in `loop.run_in_executor()` because it blocks. Do not call it directly in async context.
- Frame thumbnails are base64-encoded and stored in the `thumbnail` column of `detections` table. Full frames are saved to `uploads/frames/{det_id}.jpg`.

### When modifying the frontend:

- Maps are initialized lazily (only when the tab is first opened).
- The live feed tab auto-polls every 5 seconds via `setInterval`.
- `defectColor()` must return a valid CSS color for every defect type the model can produce.
- The lightbox uses a simple overlay — Escape key and click-outside both close it.

### Environment verified on:

| Property | Value |
|---|---|
| OS | Windows |
| Python | 3.14 |
| PyTorch | 2.14.0+cu126 |
| CUDA | Available |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) |
| Ultralytics | ≥8.1 |
| Model | YOLO26m, 280 layers, 21.7M params, 75.0 GFLOPs |
