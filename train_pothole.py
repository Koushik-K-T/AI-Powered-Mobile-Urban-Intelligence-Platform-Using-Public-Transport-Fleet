#!/usr/bin/env python3
"""
train_pothole.py — Fine-tune pothole-yolo26 on the Pothole Detection dataset.

Dataset: Ryukijano/Pothole-detection-Yolov8 (HuggingFace, CC BY 4.0)
         ~300 real pothole images, pre-split train/valid/test
Starting weights: mfranzon/pothole-yolo26 (models/road_damage.pt)
                  Same class (Pothole→Pothole), so the detection head is KEPT.
                  This means faster convergence vs training from scratch.

Usage:
    python train_pothole.py

After training, best weights land at:
    runs/detect/runs/pothole_finetune/v1/weights/best.pt

Copy to app:
    copy runs\detect\runs\pothole_finetune\v1\weights\best.pt models\road_damage.pt
"""

from pathlib import Path
from ultralytics import YOLO

# ── Paths ───────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/road_damage.pt")
DATA_YAML  = Path("datasets/pothole_hf/data_fixed.yaml")

# ── Sanity checks ───────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Starting checkpoint not found: {MODEL_PATH}\n"
        "Run 'python download_model.py' for setup instructions."
    )
if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"data.yaml not found: {DATA_YAML}\n"
        "Re-run the dataset download step."
    )

# ── Train ───────────────────────────────────────────────────────────────────────
model = YOLO(str(MODEL_PATH))

results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=512,        # Matches pre-training resolution of mfranzon checkpoint
    batch=8,          # Fixed batch — safe for 4GB VRAM at 512px
    workers=0,        # CRITICAL on Windows: disables DataLoader multiprocessing
    device=0,         # GPU 0
    patience=10,      # Early stop if val mAP50 doesn't improve for 10 epochs
    lr0=0.001,        # Lower LR for fine-tuning (default 0.01 is for scratch)
    lrf=0.01,         # Final LR = lr0 * lrf
    warmup_epochs=3,
    augment=True,
    hsv_h=0.015,      # Hue augment
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,      # Slight rotation — potholes are viewed at odd angles
    flipud=0.1,
    fliplr=0.5,
    mosaic=0.8,
    project="runs/pothole_finetune",
    name="v1",
    exist_ok=True,
)

# ── Post-training summary ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING COMPLETE")
print("=" * 55)
best = Path("runs/detect/runs/pothole_finetune/v1/weights/best.pt")
if best.exists():
    size_mb = best.stat().st_size / (1024 * 1024)
    print(f"  Best weights : {best.resolve()} ({size_mb:.1f} MB)")
    print()
    print("  To deploy:")
    print(f"    copy {best}  models\\road_damage.pt")
    print()
    print("  Then restart uvicorn — the app will use the new model automatically.")
else:
    print("  WARNING: best.pt not found. Check training logs.")
print("=" * 55)
