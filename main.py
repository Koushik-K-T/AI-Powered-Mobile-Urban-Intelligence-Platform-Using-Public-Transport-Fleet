"""
main.py
FastAPI application — all routes for the Road Defect Detection System.
Run: uvicorn main:app --reload
"""
import asyncio
import base64
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
import detect

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory job store: {job_id: {status, path, total_frames}}
jobs: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.init_db()
    yield


app = FastAPI(title="Road Defect Detection System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    from datetime import datetime, timezone
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Upload & Detect ────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save uploaded video, return job_id for SSE streaming."""
    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".mp4"
    dest = UPLOAD_DIR / f"{job_id}{ext}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "queued", "path": str(dest)}
    return {"job_id": job_id, "filename": file.filename}


@app.get("/stream/{job_id}")
async def stream_detections(job_id: str):
    """SSE endpoint — streams YOLO processing events for a job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] == "done":
        raise HTTPException(400, "Job already completed")

    job["status"] = "processing"

    async def event_generator():
        try:
            async for event in detect.process_video(job["path"], job_id):
                yield event
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        finally:
            job["status"] = "done"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/detections/{job_id}")
async def get_detections(job_id: str):
    """Return all detections for a completed job."""
    rows = await database.get_detections_by_job(job_id)
    return rows


# ── Live Bus Feed ──────────────────────────────────────────────────────────────

class BusPingRequest(BaseModel):
    bus_id: str
    lat: float
    lng: float
    defect_type: Optional[str] = None
    confidence: Optional[float] = None


@app.post("/bus-ping")
async def bus_ping(ping: BusPingRequest):
    """Accept a bus detection ping from simulate_bus.py."""
    await database.insert_bus_ping(
        bus_id=ping.bus_id,
        lat=ping.lat,
        lng=ping.lng,
        defect_type=ping.defect_type,
        confidence=ping.confidence,
    )
    if ping.defect_type:
        await database.insert_detection(
            job_id="bus_feed",
            source="bus",
            bus_id=ping.bus_id,
            frame_num=None,
            defect_type=ping.defect_type,
            confidence=ping.confidence or 0.0,
            lat=ping.lat,
            lng=ping.lng,
        )
        await database.cluster_and_fuse(ping.lat, ping.lng, ping.defect_type)
    return {"status": "ok"}


@app.get("/live-feed")
async def live_feed():
    """Return latest 50 bus pings/detections."""
    rows = await database.get_live_feed(limit=50)
    return rows


# ── Issues Dashboard ───────────────────────────────────────────────────────────

@app.get("/issues")
async def get_issues():
    """Return all fused/clustered road issues."""
    rows = await database.get_issues()
    return rows


@app.patch("/issues/{issue_id}/status")
async def update_issue_status(issue_id: int, body: dict):
    """Update status of an issue (Pending/Assigned/Fixed)."""
    status = body.get("status", "Pending")
    if status not in ("Pending", "Assigned", "Fixed"):
        raise HTTPException(400, "Invalid status")
    import aiosqlite
    async with aiosqlite.connect(database.DB_PATH) as db:
        await db.execute("UPDATE issues SET status=? WHERE id=?", (status, issue_id))
        await db.commit()
    return {"status": "updated"}


# ── Thumbnail Endpoint ─────────────────────────────────────────────────────────

@app.get("/thumbnail/{det_id}")
async def get_thumbnail(det_id: int):
    """Return detection thumbnail as a JPEG image."""
    async with aiosqlite.connect(database.DB_PATH) as db:
        cur = await db.execute(
            "SELECT thumbnail FROM detections WHERE id = ?", (det_id,)
        )
        row = await cur.fetchone()
        if not row or not row[0]:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        image_bytes = base64.b64decode(row[0])
        return Response(content=image_bytes, media_type="image/jpeg")


# ── Frontend (serve index.html) ────────────────────────────────────────────────

frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
