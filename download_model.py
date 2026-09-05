"""
download_model.py — Verify that the custom road-damage model exists.

The pothole-yolo26 model (mfranzon/pothole-yolo26) should already be
placed at models/road_damage.pt. This script verifies the model exists
and has the correct classes.

Run: python download_model.py
"""
import sys
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "road_damage.pt"

EXPECTED_CLASSES = {"Pothole"}


def verify_model() -> bool:
    """Check that models/road_damage.pt exists and has the expected classes."""
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print()
        print("To set up the model, run:")
        print('  python -c "from huggingface_hub import hf_hub_download; '
              "import shutil; "
              "p = hf_hub_download(repo_id='mfranzon/pothole-yolo26', filename='best.pt'); "
              f"shutil.copy(p, '{MODEL_PATH}')\"")
        print()
        print("Or manually place the YOLO26 pothole model at:")
        print(f"  {MODEL_PATH.resolve()}")
        return False

    # Verify the model has the correct classes
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        names = set(model.names.values()) if model.names else set()

        if not names.intersection(EXPECTED_CLASSES):
            print(f"ERROR: Model at {MODEL_PATH} has unexpected classes: {model.names}")
            print(f"Expected at least one of: {EXPECTED_CLASSES}")
            print("Please replace the model with the pothole-yolo26 model.")
            return False

        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"✅ Custom road-damage model loaded.")
        print(f"   Path:    {MODEL_PATH.resolve()}")
        print(f"   Size:    {size_mb:.1f} MB")
        print(f"   Classes: {model.names}")
        return True

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    success = verify_model()
    if not success:
        sys.exit(1)
    print("Done!")
