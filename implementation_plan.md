# Road Defect Detection System — Full Implementation Plan
> **Resume-safe**: Every section is self-contained. If work stops, hand this document to any model/developer and they can pick up from any checkpoint. Each phase has a ✅ verification gate before proceeding.

---

## 0. Project Context (Read First)

| Item | Decision |
|---|---|
| Backend | Python 3.10+ + FastAPI + Uvicorn |
| Frontend | Single `index.html` — Preact + htm via CDN (no build step, opens in browser) |
| YOLO | `ultralytics` library, model: `keremberke/yolov8m-road-damage-detection` from HuggingFace |
| Database | SQLite (file: `road_defects.db`) via `aiosqlite` |
| Streaming | Server-Sent Events (SSE) for real-time YOLO progress |
| Map | Leaflet.js via CDN |
| GPS simulation | Bengaluru, India center: `lat=12.9716, lng=77.5946` |
| Defect classes | `D00` Longitudinal Crack, `D10` Transverse Crack, `D20` Alligator Crack, `D40` Pothole |
| GPS clustering | Haversine distance < 20 metres = same cluster; 2+ detections = "Confirmed" |
| Theme | CSS `prefers-color-scheme` (auto dark/light) |
| Font | Inter from Google Fonts |
| Accent colors | Orange `#FF6B35` (alerts), Blue `#3B82F6` (actions), Green `#10B981` (ok), Red `#EF4444` (danger) |
| Port | Backend runs on `http://localhost:8000` |

---

## 1. Final File Tree

```
project/
├── main.py              # FastAPI app — all routes
├── detect.py            # YOLO video processing — SSE generator
├── simulate_bus.py      # Standalone bus simulator (run separately)
├── database.py          # SQLite schema + all query functions
├── download_model.py    # Downloads YOLO road-damage weights
├── requirements.txt     # All Python deps with pinned versions
├── setup.bat            # Windows: installs deps + downloads model
├── setup.sh             # Linux/Mac: installs deps + downloads model
├── frontend/
│   └── index.html       # Complete Preact SPA — all 4 pages
└── uploads/             # Auto-created on first run; stores temp videos
```

---

## 2. Database Schema (database.py)

### Tables

#### `detections`
```sql
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,           -- UUID of upload job
    source      TEXT NOT NULL,           -- 'upload' | 'bus'
    bus_id      TEXT,                    -- e.g. 'Bus 12', null for uploads
    frame_num   INTEGER,                 -- frame number in video (null for bus)
    defect_type TEXT NOT NULL,           -- 'Pothole' | 'Longitudinal Crack' | etc.
    confidence  REAL NOT NULL,           -- 0.0 to 1.0
    lat         REAL NOT NULL,
    lng         REAL NOT NULL,
    thumbnail   TEXT,                    -- base64 JPEG of the cropped detection (optional)
    created_at  TEXT NOT NULL            -- ISO8601 UTC timestamp
);
```

#### `issues`
```sql
CREATE TABLE IF NOT EXISTS issues (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_key     TEXT UNIQUE NOT NULL,  -- e.g. "12.9716_77.5946_Pothole"
    defect_type     TEXT NOT NULL,
    lat             REAL NOT NULL,
    lng             REAL NOT NULL,
    detection_count INTEGER DEFAULT 1,
    priority        TEXT NOT NULL,         -- 'High' | 'Medium' | 'Low'
    status          TEXT DEFAULT 'Pending', -- 'Pending' | 'Assigned' | 'Fixed'
    confirmed       INTEGER DEFAULT 0,      -- 0 or 1 (1 = 2+ detections)
    first_seen      TEXT NOT NULL,          -- ISO8601
    last_seen       TEXT NOT NULL           -- ISO8601
);
```

#### `bus_pings`
```sql
CREATE TABLE IF NOT EXISTS bus_pings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    bus_id      TEXT NOT NULL,
    lat         REAL NOT NULL,
    lng         REAL NOT NULL,
    defect_type TEXT,           -- null if no defect at this ping
    confidence  REAL,
    created_at  TEXT NOT NULL
);
```

### Python Functions in `database.py`

```python
async def init_db()                          # Creates all tables if not exist
async def insert_detection(job_id, source, bus_id, frame_num, defect_type, confidence, lat, lng, thumbnail) -> int
async def get_detections_by_job(job_id) -> list[dict]
async def get_live_feed(limit=50) -> list[dict]   # joins detections + bus_pings, ordered by created_at desc
async def insert_bus_ping(bus_id, lat, lng, defect_type, confidence)
async def get_issues() -> list[dict]
async def cluster_and_fuse(lat, lng, defect_type)  # haversine check, upsert issues table
```

**Haversine logic** (inside `cluster_and_fuse`):
```python
# For each existing issue of same defect_type:
#   d = haversine(lat, lng, issue.lat, issue.lng)  in metres
#   if d < 20: update that issue (increment count, update last_seen, set confirmed=1 if count>=2)
#   if no match: insert new issue row
# Priority rules:
#   Pothole or Alligator Crack → 'High'
#   Transverse Crack → 'Medium'
#   Longitudinal Crack → 'Low'
```

---

## 3. Backend Routes (main.py)

### App Setup
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# On startup: await database.init_db()
# CORS: allow all origins (for local dev, frontend opens from file://)
# Mount frontend/: GET / serves frontend/index.html
```

### Routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Save video to `uploads/`, generate `job_id` (UUID), store job in memory dict `{job_id: {status, path, total_frames}}`, return `{job_id}` |
| `GET` | `/stream/{job_id}` | SSE endpoint — calls `detect.process_video()` generator, streams events |
| `GET` | `/detections/{job_id}` | Returns all DB detections for a job as JSON array |
| `GET` | `/live-feed` | Returns latest 50 from `bus_pings` + `detections` where `source='bus'` |
| `GET` | `/issues` | Returns all issues with count/status |
| `POST` | `/bus-ping` | Accepts `{bus_id, lat, lng, defect_type, confidence}`, inserts to DB, calls `cluster_and_fuse` |
| `GET` | `/health` | Returns `{"status": "ok", "timestamp": "..."}` |

### SSE Event Format (stream endpoint)
Each event is a `data: <JSON>\n\n` line. Three event types:

```jsonc
// Progress update (no detection this frame)
{"type": "progress", "frame": 45, "total": 890, "pct": 5}

// Detection found
{"type": "detection", "frame": 120, "defect_type": "Pothole", "confidence": 0.91,
 "lat": 12.9716, "lng": 77.5946, "id": 7}

// Job complete
{"type": "done", "total_detections": 14, "job_id": "abc-123"}
```

---

## 4. YOLO Processing (detect.py)

### `process_video(video_path: str, job_id: str)` — async generator

```
1. Load model:
   model = YOLO("models/road_damage.pt")  # downloaded by download_model.py

2. Open video:
   cap = cv2.VideoCapture(video_path)
   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

3. Extract GPS from video metadata:
   Run: ffprobe -v quiet -print_format json -show_streams <video_path>
   Parse GPS from 'location' or 'com.apple.quicktime.location.ISO6709' tag
   If found: parse lat/lng
   If not found: set base_lat=12.9716, base_lng=77.5946 (Bengaluru center)
   Each frame: slightly jitter GPS by ±0.0001 per frame to simulate movement

4. Frame sampling:
   Process every 10th frame (configurable: SAMPLE_EVERY = 10)
   total_sample_frames = total_frames // SAMPLE_EVERY

5. Per sampled frame loop:
   a. Read frame with cap.read()
   b. results = model.predict(frame, conf=0.4, verbose=False)
   c. If detections found:
      - For each box: get class_id → map to defect_type, get confidence, get bbox
      - Crop thumbnail: frame[y1:y2, x1:x2], encode to base64 JPEG
      - GPS: base_lat + (frame_num * 0.00001), base_lng + (frame_num * 0.00001)
      - await database.insert_detection(...)
      - await database.cluster_and_fuse(lat, lng, defect_type)
      - yield SSE detection event
   d. yield SSE progress event every frame

6. cap.release()
7. yield SSE done event
```

### Class ID Mapping
```python
CLASS_MAP = {
    0: "Longitudinal Crack",   # D00
    1: "Transverse Crack",     # D10
    2: "Alligator Crack",      # D20
    3: "Pothole"               # D40
}
```

---

## 5. Bus Simulator (simulate_bus.py)

### Run independently: `python simulate_bus.py`

```
Config:
  BACKEND_URL = "http://localhost:8000"
  BUSES = ["Bus 12", "Bus 7", "Bus 23"]
  INTERVAL_SECONDS = 10
  BENGALURU_CENTER = (12.9716, 77.5946)
  ROUTE_RADIUS = 0.05  # ~5.5km radius in degrees

Loop every 10s for each bus:
  1. Update bus position: lat += random.uniform(-0.001, 0.001)
                          lng += random.uniform(-0.001, 0.001)
     (clamp within BENGALURU_CENTER ± ROUTE_RADIUS)
  
  2. 60% chance of a detection (else ping with defect_type=null)
  
  3. If detection:
     defect_type = random.choice(["Pothole", "Longitudinal Crack", 
                                  "Transverse Crack", "Alligator Crack"])
     confidence = round(random.uniform(0.55, 0.97), 2)
  
  4. POST to BACKEND_URL/bus-ping:
     {bus_id, lat, lng, defect_type, confidence}
  
  5. Print: "[Bus 12] Pothole @ 12.9716, 77.5946 — 91% — OK 200"
```

---

## 6. Model Download (download_model.py)

```python
# Downloads keremberke/yolov8m-road-damage-detection from HuggingFace Hub
# Saves to: models/road_damage.pt
# Uses: huggingface_hub.hf_hub_download()
# Falls back to: ultralytics yolov8m.pt if HF download fails (will still work, 
#                just uses generic classes — warn user)

import os
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "road_damage.pt"
HF_REPO = "keremberke/yolov8m-road-damage-detection"
HF_FILE = "best.pt"
```

---

## 7. Requirements (requirements.txt)

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

---

## 8. Setup Scripts

### setup.bat (Windows)
```bat
@echo off
echo Installing Python dependencies...
pip install -r requirements.txt

echo Downloading YOLO road damage model...
python download_model.py

echo Creating uploads directory...
mkdir uploads 2>nul
mkdir models 2>nul

echo Done! Run: uvicorn main:app --reload
```

### setup.sh (Linux/Mac)
```bash
#!/bin/bash
pip install -r requirements.txt
python download_model.py
mkdir -p uploads models
echo "Done! Run: uvicorn main:app --reload"
```

---

## 9. Frontend Architecture (frontend/index.html)

### CDN Imports (in `<head>`)
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script type="module">
  import { h, render, useState, useEffect, useRef, useCallback } from 'https://esm.sh/preact@10.22.0/compat';
  import htm from 'https://esm.sh/htm@3.1.1';
  const html = htm.bind(h);
  // ... all app code here
</script>
```

### Component Tree
```
<App>
  ├── <NavBar>         — 4 tabs, active indicator, sticky top
  ├── <UploadPage>     — Page 1
  │   ├── <DropZone>   — drag-drop + file picker
  │   ├── <ProgressBar> — SSE-driven, fills as frames process
  │   ├── <StatsRow>   — Frame X/Y | Detections found: N
  │   ├── <DetectionTable> — live-updating rows
  │   └── <LeafletMap> — pins added in real time
  ├── <LiveFeedPage>   — Page 2
  │   ├── <BusCards>   — one card per unique bus_id
  │   ├── <FeedTable>  — auto-polls /live-feed every 5s
  │   └── <LeafletMap> — bus position markers
  ├── <IssuesPage>     — Page 3
  │   ├── <StatsRow>   — totals
  │   ├── <FilterBar>  — filter by type/status
  │   ├── <IssuesTable> — clickable rows
  │   └── <LeafletMap> — click row → pan map
  └── <ConnectivityPage> — Page 4
      ├── <RadioCard x3> — AIS-140 | Nirbhaya 4G | M2M IoT
      └── <StatusPanel>  — animated status based on selection
```

### CSS Design Tokens
```css
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-card: #ffffff;
  --text-primary: #0f172a;
  --text-secondary: #64748b;
  --border: #e2e8f0;
  --accent-blue: #3B82F6;
  --accent-orange: #FF6B35;
  --accent-green: #10B981;
  --accent-red: #EF4444;
  --accent-yellow: #F59E0B;
  --shadow: 0 1px 3px rgba(0,0,0,0.1);
  --shadow-lg: 0 10px 40px rgba(0,0,0,0.1);
  --radius: 12px;
  --radius-sm: 8px;
  --font: 'Inter', sans-serif;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #334155;
    --shadow: 0 1px 3px rgba(0,0,0,0.4);
    --shadow-lg: 0 10px 40px rgba(0,0,0,0.5);
  }
}
```

### Page 1 — Upload & Detect (detailed)
```
State: { file, jobId, status, progress, detections, mapPins }

1. DropZone:
   - Dashed border, icon, "Drag video here or click to browse"
   - On drop/pick: set file state, show filename + size

2. Upload button → POST /upload (multipart) → receive {job_id}
   - Set status = 'uploading' → show spinner

3. SSE connection: new EventSource(`/stream/${jobId}`)
   - On 'progress' event: update progress bar (frame/total*100)
   - On 'detection' event: prepend row to detections array, add pin to map
   - On 'done' event: status = 'done', close EventSource

4. DetectionTable columns:
   Frame # | Type | Confidence | Lat | Lng | (thumbnail if available)
   - New rows slide in with CSS animation (translateY + opacity)
   - Confidence shown as colored badge: ≥80% green, 60-79% yellow, <60% red

5. LeafletMap:
   - Center: Bengaluru (12.9716, 77.5946), zoom 13
   - Custom icon per defect type (colored circles)
   - Popup on click: "Pothole — 91% confidence — Frame 120"
```

### Page 2 — Live Bus Feed (detailed)
```
State: { feeds, lastPoll, blinking }

1. useEffect: poll GET /live-feed every 5000ms
   - Compare with prev feeds — if new items, trigger blink animation

2. BusCard per unique bus_id:
   - Bus name | Last seen timestamp (relative: "2 mins ago")
   - Latest defect badge | confidence | lat/lng

3. FeedTable:
   Bus ID | Defect Type | Confidence | Lat | Lng | Time Ago
   - Newest row highlighted briefly in green on arrival

4. Map shows latest position of each bus as colored marker
   - Different color per bus
```

### Page 3 — Road Issues Dashboard (detailed)
```
State: { issues, filter }

1. StatsRow: [Total] [Confirmed] [Unverified] [Fixed] — animated count-up

2. FilterBar: dropdowns for Type (all/pothole/crack), Status (all/pending/assigned/fixed)

3. IssuesTable:
   ID | Location (lat,lng) | Type | Count | Priority | Status | Last Seen
   - Priority badge: High=red, Medium=yellow, Low=blue
   - Status badge: Pending=gray, Assigned=orange, Fixed=green
   - Confirmed issues have a ✅ "Confirmed" tag; single = ⚠️ "Unverified"

4. Click row → map pans to that lat/lng, popup opens
```

### Page 4 — Connectivity Settings (detailed)
```
State: { selected: 'ais140' | 'nirbhaya4g' | 'm2m' }

Three RadioCards:

Card 1: AIS-140 SIM
  Icon: 📡
  Description: "Uses AIS-140 compliant vehicle tracking SIM. Data transmitted 
  via cellular in real-time. Mandatory for commercial vehicles per MoRTH."
  Data flow: Bus → SIM → Cellular Tower → Backend Server
  Simulated status: 🟢 Connected — Live sync

Card 2: Nirbhaya 4G
  Icon: 🔒
  Description: "Nirbhaya fund-backed 4G module. Encrypted channel for 
  women-safety + road monitoring. Priority bandwidth allocation."
  Data flow: Bus → 4G Module → Encrypted VPN → Backend Server
  Simulated status: 🟡 Store & Forward — 3 packets queued

Card 3: M2M IoT Depot WiFi
  Icon: 📶
  Description: "Machine-to-Machine IoT protocol. Data synced when bus returns 
  to depot over secured WiFi. Suitable for low-connectivity routes."
  Data flow: Bus → Local Storage → Depot WiFi → Backend Server
  Simulated status: 🔵 Syncing — Last sync 4 mins ago

Visual: Each card has animated CSS data-flow line (dots moving along path)
Selected card: accent border + glow
```

---

## 10. Build Checklist (Ordered — Resume from any ✅)

> Mark each phase complete before starting the next.

### Phase A — Project scaffold
- [ ] Create project directory structure (all folders)
- [ ] Create `requirements.txt` (exact content in §7)
- [ ] Create `setup.bat` and `setup.sh` (exact content in §8)
- [ ] Create empty `uploads/` and `models/` directories
- ✅ **Gate**: Directory tree matches §1 exactly

### Phase B — Database layer
- [ ] Create `database.py` with all 3 table schemas (exact DDL in §2)
- [ ] Implement all async functions listed in §2
- [ ] Implement `haversine()` helper and `cluster_and_fuse()` logic
- ✅ **Gate**: `python -c "import asyncio; import database; asyncio.run(database.init_db()); print('OK')"` prints OK

### Phase C — Model download
- [ ] Create `download_model.py` with HuggingFace Hub download (see §6)
- [ ] Add fallback to generic `yolov8m.pt` with warning
- [ ] Create `models/` directory
- ✅ **Gate**: `python download_model.py` creates `models/road_damage.pt`

### Phase D — YOLO detection
- [ ] Create `detect.py` with `process_video()` async generator (see §4)
- [ ] Implement CLASS_MAP (see §4)
- [ ] Implement GPS extraction via ffprobe + Bengaluru fallback
- [ ] Implement frame sampling (every 10th frame)
- [ ] Yield SSE events in exact format from §3
- ✅ **Gate**: Run against a test video, confirm SSE events print to stdout

### Phase E — FastAPI backend
- [ ] Create `main.py` with all 7 routes (see §3)
- [ ] Implement lifespan handler calling `database.init_db()`
- [ ] Implement CORS middleware (allow all origins)
- [ ] Implement in-memory job store `{job_id: status}`
- [ ] Implement `/stream/{job_id}` SSE endpoint using `StreamingResponse`
- [ ] Mount `frontend/` directory at `/`
- ✅ **Gate**: `uvicorn main:app --reload` starts, `GET /health` returns 200

### Phase F — Bus simulator
- [ ] Create `simulate_bus.py` (see §5)
- [ ] Implement 3-bus random walk within Bengaluru bounds
- [ ] 60% detection probability per ping
- ✅ **Gate**: Run `python simulate_bus.py`, confirm POSTs succeed (200 OK logged)

### Phase G — Frontend HTML
- [ ] Create `frontend/index.html`
- [ ] Add all CDN imports (see §9)
- [ ] Implement CSS design tokens (exact variables in §9)
- [ ] Implement `<NavBar>` with 4 tabs
- [ ] Implement `<UploadPage>` with DropZone + SSE + live table + map
- [ ] Implement `<LiveFeedPage>` with 5s polling + bus cards + map
- [ ] Implement `<IssuesPage>` with stats + filter + table + map
- [ ] Implement `<ConnectivityPage>` with 3 radio cards + status panel
- ✅ **Gate**: Open `http://localhost:8000` in browser — all 4 pages render

### Phase H — End-to-end test
- [ ] Start backend: `uvicorn main:app --reload`
- [ ] Start bus sim: `python simulate_bus.py` (separate terminal)
- [ ] Upload a road video → confirm SSE streams → detections appear
- [ ] Wait 30s → Live Feed page shows bus detections
- [ ] Issues page shows ≥1 confirmed issue
- [ ] Connectivity page — click all 3 cards, confirm status changes
- ✅ **Gate**: All 4 pages functional with real data

---

## 11. Key Implementation Notes for Any Model

1. **SSE in FastAPI** — Use `StreamingResponse(generator(), media_type="text/event-stream")`. Each event must be `f"data: {json}\n\n"`. Add `Cache-Control: no-cache` header.

2. **CORS** — Frontend at `http://localhost:8000` needs CORS. Use `allow_origins=["*"]` for dev.

3. **Async + YOLO** — YOLO's `model.predict()` is sync. Run in thread pool: `await asyncio.get_event_loop().run_in_executor(None, model.predict, frame)`.

4. **EventSource in browser** — Standard: `new EventSource(url)`. Parse: `source.onmessage = e => JSON.parse(e.data)`. Close: `source.close()`.

5. **Leaflet in Preact** — Initialize map in `useEffect` with empty deps `[]`. Store map ref with `useRef`. Add markers imperatively: `L.marker([lat,lng]).addTo(mapRef.current)`.

6. **Preact + htm** — Import from `esm.sh`. Use `html\`...\`` template literals instead of JSX. Props work identically to React.

7. **Haversine formula**:
   ```python
   import math
   def haversine(lat1, lon1, lat2, lon2) -> float:  # returns metres
       R = 6371000
       phi1, phi2 = math.radians(lat1), math.radians(lat2)
       dphi = math.radians(lat2 - lat1)
       dlambda = math.radians(lon2 - lon1)
       a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
       return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
   ```

8. **HuggingFace model download**:
   ```python
   from huggingface_hub import hf_hub_download
   path = hf_hub_download(repo_id="keremberke/yolov8m-road-damage-detection", filename="best.pt")
   ```

9. **Video upload** — Use `UploadFile` from FastAPI. Save with `shutil.copyfileobj`. Generate `job_id = str(uuid.uuid4())`.

10. **Relative timestamps** — In frontend: `const ago = (iso) => { const d = (Date.now() - new Date(iso))/1000; return d < 60 ? Math.floor(d)+'s ago' : Math.floor(d/60)+'m ago' }`.
