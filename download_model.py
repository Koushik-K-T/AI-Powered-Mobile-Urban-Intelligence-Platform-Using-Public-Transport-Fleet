"""
download_model.py - Downloads road damage YOLO weights.
Tries HuggingFace first, then direct GitHub release URL, then generic fallback.
Run: python download_model.py
"""
import os
import sys
import shutil
import urllib.request
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "road_damage.pt"

# Direct download URLs for road damage model
DIRECT_URLS = [
    "https://github.com/keremberke/awesome-yolov8-models/releases/download/v1.0.0/best.pt",
]

HF_REPO = "keremberke/yolov8m-road-damage-detection"
HF_FILE = "best.pt"


def download_hf():
    from huggingface_hub import hf_hub_download
    print(f"[1/3] Trying HuggingFace Hub: {HF_REPO}...")
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, local_dir=str(MODEL_DIR))
    final = MODEL_DIR / HF_FILE
    if final != MODEL_PATH:
        shutil.copy(str(final), str(MODEL_PATH))
    print(f"Road damage model saved to {MODEL_PATH}")
    return True


def download_direct():
    for url in DIRECT_URLS:
        try:
            print(f"[2/3] Trying direct URL: {url}")
            urllib.request.urlretrieve(url, MODEL_PATH)
            print(f"Road damage model saved to {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"  Failed: {e}")
    return False


def download_fallback():
    from ultralytics import YOLO
    print("[3/3] Falling back to generic yolov8m.pt (COCO classes).")
    print("WARNING: Defect labels will be remapped COCO classes, not real road damage classes.")
    model = YOLO("yolov8m.pt")
    # yolov8m.pt is saved to CWD by ultralytics
    src = Path("yolov8m.pt")
    if src.exists():
        shutil.copy(str(src), str(MODEL_PATH))
    else:
        shutil.copy(model.ckpt_path, str(MODEL_PATH))
    print(f"Fallback model saved to {MODEL_PATH}")
    return True


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        print(f"Model already exists at {MODEL_PATH} ({MODEL_PATH.stat().st_size//1024//1024}MB) — skipping.")
        sys.exit(0)

    success = False
    try:
        success = download_hf()
    except Exception as e:
        print(f"  HuggingFace failed: {e}")

    if not success:
        try:
            success = download_direct()
        except Exception as e:
            print(f"  Direct download failed: {e}")

    if not success:
        try:
            success = download_fallback()
        except Exception as e:
            print(f"  Fallback failed: {e}")
            print("ERROR: Could not download any model. Place a YOLOv8 .pt file at models/road_damage.pt manually.")
            sys.exit(1)

    print("Done!")
