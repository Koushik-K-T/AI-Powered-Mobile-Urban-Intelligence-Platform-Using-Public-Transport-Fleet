#!/usr/bin/env python3
"""
train_rdd.py — Fine-tune YOLO26m on RDD2022 4-class road damage dataset.

Prerequisites:
  1. Run convert_rdd2022.py first to produce the YOLO-format dataset + data.yaml
  2. Ensure CUDA is available (RTX 3050 4GB VRAM)

Usage:
    python train_rdd.py
"""

from ultralytics import YOLO

# ── Starting checkpoint ──
# mfranzon/pothole-yolo26 (best.pt) — single-class {0: Pothole}
# The backbone/neck carry real pothole visual features from pre-training.
# The single-class detection head will be discarded and replaced with a
# 4-class head on first training step.
MODEL_PATH = r"C:\Users\koush\.cache\huggingface\hub\models--mfranzon--pothole-yolo26\snapshots\f07efaee2eeb63b24e1ce8c8ac635da985722961\best.pt"

# ── Dataset ──
# Update this path to wherever convert_rdd2022.py wrote data.yaml
DATA_YAML = r"C:\Users\koush\Desktop\web\web\rdd2022_yolo\data.yaml"

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,
        epochs=60,
        imgsz=512,          # Matches mfranzon checkpoint's original training resolution
        batch=-1,           # AutoBatch — auto-fits ~60% of 4GB VRAM (safe for RTX 3050)
        device=0,           # GPU 0
        patience=15,        # Early stopping: stop if no improvement for 15 epochs
        project="runs/pothole_train",
        name="yolo26_rdd_v1",
    )

    # Notes:
    # ─────────────────────────────────────────────────────────────────────
    # • batch=-1 uses Ultralytics AutoBatch. It profiles your GPU and picks
    #   a batch size that uses ~60% of VRAM. Safe for 4GB cards.
    #
    # • imgsz=512 matches the mfranzon checkpoint's original training
    #   resolution. Going higher risks VRAM OOM on 4GB.
    #
    # • "Transferred X/Y items from pretrained weights" in the console
    #   output is NORMAL. It means the single-class detection head was
    #   discarded and only the backbone/neck weights were transferred.
    #
    # • If you hit a CUDA OOM error, set imgsz=416 as a fallback:
    #       model.train(..., imgsz=416, ...)
    # ─────────────────────────────────────────────────────────────────────

    print("\n✅ Training complete!")
    print("Best weights saved to: runs/pothole_train/yolo26_rdd_v1/weights/best.pt")
    print("\nNext step: copy best.pt → models/road_damage.pt")
