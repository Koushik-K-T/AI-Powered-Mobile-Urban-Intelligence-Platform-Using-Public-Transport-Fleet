"""
detect.py
YOLO26 pothole detection — async SSE generator.
Called by main.py /stream/{job_id} endpoint.

Model: mfranzon/pothole-yolo26 (YOLO26m, single class: Pothole)
"""
import asyncio
import base64
import json
import math
import os
import subprocess
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import io

import database

MODEL_PATH = Path("models/road_damage.pt")
FRAMES_DIR = Path("uploads/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_EVERY = 10          # process every Nth frame
BENGALURU_LAT = 12.9716
BENGALURU_LNG = 77.5946

# Pothole-yolo26 model class mapping (single class)
CLASS_MAP = {
    0: "Pothole",
}

CONFIDENCE_THRESHOLD = 0.25


def load_model():
    """Load YOLO model once (cached at module level)."""
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run 'python download_model.py' for setup instructions."
        )

    model = YOLO(str(MODEL_PATH))

    # Verify it's a valid pothole model
    names = model.names or {}
    name_values = {str(v).lower() for v in names.values()}
    if "pothole" not in name_values and "0" not in name_values:
        raise ValueError(
            f"Model at {MODEL_PATH} has unexpected classes: {names}. "
            "Expected a pothole detection model."
        )

    print(f"INFO: Loaded pothole-yolo26 model from {MODEL_PATH}")
    print(f"      Classes: {names}")
    return model


_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def extract_gps_from_video(video_path: str):
    """Try to extract GPS from video metadata using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})

        # Try various GPS tag names used by cameras/dashcams
        for key in ("location", "com.apple.quicktime.location.ISO6709",
                    "GPS", "Location"):
            val = tags.get(key, "")
            if val:
                # ISO 6709 format: +12.9716+077.5946/
                match = re.search(r'([+-]\d+\.\d+)([+-]\d+\.\d+)', val)
                if match:
                    return float(match.group(1)), float(match.group(2))
    except Exception:
        pass
    return None, None


def frame_to_base64(frame_bgr, bbox=None, max_size=120) -> str:
    """Crop to bbox, resize, encode as base64 JPEG (thumbnail)."""
    try:
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = frame_bgr[y1:y2, x1:x2]
        else:
            crop = frame_bgr

        if crop.size == 0:
            return ""

        h, w = crop.shape[:2]
        scale = max_size / max(h, w, 1)
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def full_frame_to_base64(frame_bgr, bbox=None, max_width=1280) -> str:
    """Encode full frame with bbox drawn, resized to max_width, as base64 JPEG."""
    try:
        vis = frame_bgr.copy()
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 80, 255), 3)
        h, w = vis.shape[:2]
        if w > max_width:
            scale = max_width / w
            vis = cv2.resize(vis, (max_width, int(h * scale)))
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def process_video(video_path: str, job_id: str):
    """
    Async generator — yields SSE-formatted strings.
    Events: progress | detection | done | error
    """
    loop = asyncio.get_event_loop()

    # Load model in thread pool (blocking)
    model = await loop.run_in_executor(None, get_model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield sse_event({"type": "error", "message": "Cannot open video file"})
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    sampled_total = max(1, total_frames // SAMPLE_EVERY)

    # GPS setup
    base_lat, base_lng = extract_gps_from_video(video_path)
    has_real_gps = base_lat is not None
    if not has_real_gps:
        base_lat, base_lng = BENGALURU_LAT, BENGALURU_LNG

    detection_count = 0
    sample_index = 0

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        if frame_num % SAMPLE_EVERY != 0:
            continue

        sample_index += 1
        pct = round((sample_index / sampled_total) * 100, 1)

        # GPS: simulate movement along a road if no embedded GPS
        if not has_real_gps:
            lat = base_lat + (sample_index * 0.00005)
            lng = base_lng + (sample_index * 0.00003)
        else:
            lat = base_lat + (sample_index * 0.000005)
            lng = base_lng + (sample_index * 0.000003)

        # Run YOLO in thread pool (blocking call)
        frame_copy = frame.copy()
        results = await loop.run_in_executor(
            None, lambda f=frame_copy: model.predict(
                f, conf=CONFIDENCE_THRESHOLD, verbose=False
            )
        )

        found_any = False
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                # Only process classes in our map
                if cls_id not in CLASS_MAP:
                    continue
                defect_type = CLASS_MAP[cls_id]
                thumbnail = frame_to_base64(frame, xyxy)
                full_frame = full_frame_to_base64(frame, xyxy)

                det_id = await database.insert_detection(
                    job_id=job_id,
                    source="upload",
                    bus_id=None,
                    frame_num=frame_num,
                    defect_type=defect_type,
                    confidence=round(conf, 3),
                    lat=round(lat, 6),
                    lng=round(lng, 6),
                    thumbnail=thumbnail,
                )

                # Save full frame to disk for lightbox display
                if full_frame:
                    try:
                        frame_path = FRAMES_DIR / f"{det_id}.jpg"
                        frame_path.write_bytes(base64.b64decode(full_frame))
                    except Exception:
                        pass
                await database.cluster_and_fuse(lat, lng, defect_type)

                detection_count += 1
                found_any = True

                yield sse_event({
                    "type": "detection",
                    "frame": frame_num,
                    "total": total_frames,
                    "pct": pct,
                    "defect_type": defect_type,
                    "confidence": round(conf, 3),
                    "lat": round(lat, 6),
                    "lng": round(lng, 6),
                    "id": det_id,
                    "thumbnail": thumbnail,
                    "full_frame": full_frame,
                })

        # Always emit progress
        yield sse_event({
            "type": "progress",
            "frame": frame_num,
            "total": total_frames,
            "pct": pct,
            "sample_index": sample_index,
            "sample_total": sampled_total,
        })

        # Small yield to keep event loop alive
        await asyncio.sleep(0)

    cap.release()

    yield sse_event({
        "type": "done",
        "total_detections": detection_count,
        "job_id": job_id,
    })
