"""
merge_datasets.py — Merge HuggingFace pothole dataset + Kaggle Indian Roads dataset
into a single combined YOLO dataset for training.

Kaggle dataset has 3 classes: 0=SpeedBreaker, 1=Pothole, 2=UnpavedRoad
We ONLY keep class 1 (Pothole) labels, remapped to class 0 for our single-class model.

Output: datasets/pothole_combined/
    train/images/  train/labels/
    valid/images/  valid/labels/
    test/images/   test/labels/
    data.yaml
"""
import os
import shutil
import random
import yaml
from pathlib import Path

# ── Source paths ─────────────────────────────────────────────────────────────────
HF_ROOT = Path("datasets/pothole_hf")
KAGGLE_ROOT = Path("C:/Users/koush/.cache/kagglehub/datasets/mitangshu11/indian-roads-dataset/versions/1/Dataset3Class")
OUT_ROOT = Path("datasets/pothole_combined")

# ── Clean output ─────────────────────────────────────────────────────────────────
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
for split in ["train", "valid", "test"]:
    (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

# ── Step 1: Copy HuggingFace dataset as-is ───────────────────────────────────────
print("=== Step 1: Copying HuggingFace pothole dataset ===")
hf_count = 0
for split in ["train", "valid", "test"]:
    src_img = HF_ROOT / split / "images"
    src_lbl = HF_ROOT / split / "labels"
    if not src_img.exists():
        continue
    for img_file in src_img.iterdir():
        shutil.copy2(img_file, OUT_ROOT / split / "images" / img_file.name)
        lbl_file = src_lbl / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.copy2(lbl_file, OUT_ROOT / split / "labels" / lbl_file.name)
        hf_count += 1
print(f"  Copied {hf_count} images from HuggingFace dataset")

# ── Step 2: Extract pothole-only labels from Kaggle dataset ──────────────────────
print("\n=== Step 2: Extracting pothole labels from Kaggle Indian Roads ===")
kaggle_pairs = []  # (img_path, filtered_label_lines)

all_imgs = sorted([f for f in KAGGLE_ROOT.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
skipped = 0
kept = 0

for img_path in all_imgs:
    lbl_path = KAGGLE_ROOT / (img_path.stem + ".txt")
    if not lbl_path.exists():
        skipped += 1
        continue

    # Filter: keep only class 1 (Pothole), remap to class 0
    pothole_lines = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == "1":
                # Remap class 1 → 0
                parts[0] = "0"
                pothole_lines.append(" ".join(parts))

    if pothole_lines:
        kaggle_pairs.append((img_path, pothole_lines))
        kept += 1
    else:
        skipped += 1

print(f"  Total Kaggle images: {len(all_imgs)}")
print(f"  Images with pothole annotations: {kept}")
print(f"  Skipped (no pothole): {skipped}")

# ── Step 3: Split Kaggle pothole images 70/20/10 ─────────────────────────────────
print("\n=== Step 3: Splitting Kaggle data (70% train / 20% valid / 10% test) ===")
random.seed(42)
random.shuffle(kaggle_pairs)

n = len(kaggle_pairs)
n_train = int(n * 0.7)
n_valid = int(n * 0.2)
# rest goes to test

splits = {
    "train": kaggle_pairs[:n_train],
    "valid": kaggle_pairs[n_train:n_train + n_valid],
    "test": kaggle_pairs[n_train + n_valid:],
}

kaggle_count = 0
for split, pairs in splits.items():
    for img_path, label_lines in pairs:
        # Prefix with "kaggle_" to avoid filename collisions
        new_name = f"kaggle_{img_path.name}"
        shutil.copy2(img_path, OUT_ROOT / split / "images" / new_name)
        lbl_out = OUT_ROOT / split / "labels" / (f"kaggle_{img_path.stem}.txt")
        lbl_out.write_text("\n".join(label_lines) + "\n")
        kaggle_count += 1

print(f"  Added {kaggle_count} Kaggle pothole images:")
for split, pairs in splits.items():
    print(f"    {split}: {len(pairs)}")

# ── Step 4: Write data.yaml ──────────────────────────────────────────────────────
print("\n=== Step 4: Writing data.yaml ===")
data_yaml = {
    "path": str(OUT_ROOT.resolve()).replace("\\", "/"),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": {0: "Pothole"},
}
yaml_path = OUT_ROOT / "data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
print(f"  Written: {yaml_path.resolve()}")

# ── Summary ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  COMBINED DATASET READY")
print("=" * 55)
for split in ["train", "valid", "test"]:
    img_count = len(list((OUT_ROOT / split / "images").iterdir()))
    lbl_count = len(list((OUT_ROOT / split / "labels").iterdir()))
    print(f"  {split:6s}: {img_count} images, {lbl_count} labels")
total = sum(len(list((OUT_ROOT / s / "images").iterdir())) for s in ["train", "valid", "test"])
print(f"  TOTAL : {total} images")
print(f"\n  HuggingFace: {hf_count} images")
print(f"  Kaggle:      {kaggle_count} images (pothole-only, class 1→0)")
print(f"\n  data.yaml: {yaml_path.resolve()}")
print("=" * 55)
