#!/usr/bin/env python3
"""
convert_rdd2022.py — Convert RDD2022 (PASCAL VOC) to YOLO format.

Usage:
    python convert_rdd2022.py --input_dir path/to/RDD2022 --output_dir path/to/output

Input layout (one or more country folders):
    <input_dir>/<country>/train/images/*.jpg
    <input_dir>/<country>/train/annotations/xmls/*.xml

Output layout:
    <output_dir>/images/train/*.jpg
    <output_dir>/images/val/*.jpg
    <output_dir>/labels/train/*.txt
    <output_dir>/labels/val/*.txt
    <output_dir>/data.yaml
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# Only these RDD2022 classes are kept; everything else is silently skipped.
CLASS_MAP = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3,  # Pothole
}


def get_image_size(image_path: str) -> tuple[int, int]:
    """Return (width, height) using cv2 — does NOT decode the full image."""
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cv2 could not read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def parse_voc_xml(xml_path: str) -> list[dict]:
    """
    Parse a PASCAL VOC XML annotation file.
    Returns a list of dicts: {'name': str, 'bbox': (xmin, ymin, xmax, ymax)}.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        name = name_el.text.strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            continue
        annotations.append({"name": name, "bbox": (xmin, ymin, xmax, ymax)})
    return annotations


def voc_to_yolo(bbox: tuple, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """
    Convert VOC (xmin, ymin, xmax, ymax) to YOLO normalized
    (x_center, y_center, width, height), all values in [0, 1].
    """
    xmin, ymin, xmax, ymax = bbox

    # Clamp to image bounds
    xmin = max(0.0, min(xmin, img_w))
    ymin = max(0.0, min(ymin, img_h))
    xmax = max(0.0, min(xmax, img_w))
    ymax = max(0.0, min(ymax, img_h))

    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    return x_center, y_center, width, height


def discover_samples(input_dir: str) -> list[dict]:
    """
    Walk the input directory, find all image+XML pairs across country subfolders.
    Returns a list of dicts with 'image_path', 'xml_path', 'stem' keys.
    """
    input_path = Path(input_dir)
    samples = []
    seen_stems = set()

    # Look for country subfolders
    for country_dir in sorted(input_path.iterdir()):
        if not country_dir.is_dir():
            continue

        images_dir = country_dir / "train" / "images"
        xmls_dir = country_dir / "train" / "annotations" / "xmls"

        if not images_dir.is_dir() or not xmls_dir.is_dir():
            continue

        print(f"  Found country folder: {country_dir.name}")

        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            stem = img_file.stem
            xml_file = xmls_dir / f"{stem}.xml"

            if not xml_file.exists():
                continue

            # Avoid duplicate stems across countries by prefixing
            unique_stem = f"{country_dir.name}_{stem}"
            if unique_stem in seen_stems:
                continue
            seen_stems.add(unique_stem)

            samples.append({
                "image_path": str(img_file),
                "xml_path": str(xml_file),
                "stem": unique_stem,
                "suffix": img_file.suffix,
            })

    return samples


def convert(input_dir: str, output_dir: str) -> None:
    """Main conversion pipeline."""
    output_path = Path(output_dir)

    # Create output directories
    for split in ("train", "val"):
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"Scanning input directory: {input_dir}")
    samples = discover_samples(input_dir)
    print(f"  Total image+XML pairs found: {len(samples)}")

    if not samples:
        print("ERROR: No valid image+XML pairs found. Check your --input_dir layout.")
        return

    # ── Process all samples, filter to those with valid annotations ──
    valid_samples = []  # list of (sample, yolo_lines)
    class_counts = defaultdict(int)
    skipped = 0

    for sample in samples:
        annotations = parse_voc_xml(sample["xml_path"])

        # Filter to only our target classes
        yolo_lines = []
        for ann in annotations:
            cls_name = ann["name"]
            if cls_name not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[cls_name]

            # Read image dimensions
            try:
                img_w, img_h = get_image_size(sample["image_path"])
            except ValueError as e:
                print(f"  WARNING: {e} — skipping")
                break
            
            x_c, y_c, w, h = voc_to_yolo(ann["bbox"], img_w, img_h)

            # Skip degenerate boxes
            if w <= 0 or h <= 0:
                continue

            yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            class_counts[cls_name] += 1

        if not yolo_lines:
            skipped += 1
            continue

        valid_samples.append((sample, yolo_lines))

    print(f"  Valid images (≥1 target annotation): {len(valid_samples)}")
    print(f"  Skipped images (0 target annotations): {skipped}")

    # ── Split 85% train / 15% val ──
    random.seed(42)
    indices = list(range(len(valid_samples)))
    random.shuffle(indices)
    split_idx = int(len(valid_samples) * 0.85)
    train_indices = set(indices[:split_idx])

    train_count = 0
    val_count = 0

    for i, (sample, yolo_lines) in enumerate(valid_samples):
        split = "train" if i in train_indices else "val"

        # Copy image
        dst_img = output_path / "images" / split / f"{sample['stem']}{sample['suffix']}"
        shutil.copy2(sample["image_path"], str(dst_img))

        # Write label
        dst_lbl = output_path / "labels" / split / f"{sample['stem']}.txt"
        with open(dst_lbl, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    # ── Write data.yaml ──
    abs_output = str(output_path.resolve()).replace("\\", "/")
    yaml_content = (
        f"path: {abs_output}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: D00\n"
        f"  1: D10\n"
        f"  2: D20\n"
        f"  3: D40\n"
    )
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # ── Summary ──
    print("\n" + "=" * 55)
    print("  CONVERSION COMPLETE")
    print("=" * 55)
    print(f"  Total image+XML pairs scanned : {len(samples)}")
    print(f"  Images with valid annotations  : {len(valid_samples)}")
    print(f"  Images skipped (no targets)    : {skipped}")
    print(f"  Train split                    : {train_count}")
    print(f"  Val split                      : {val_count}")
    print()
    print("  Per-class annotation counts:")
    for cls_name in ("D00", "D10", "D20", "D40"):
        count = class_counts.get(cls_name, 0)
        print(f"    {cls_name} (class {CLASS_MAP[cls_name]}): {count:>6}")
    print()
    print(f"  data.yaml written to: {yaml_path.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RDD2022 PASCAL VOC annotations to YOLO format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing country subfolders with RDD2022 data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for YOLO-format dataset.",
    )
    args = parser.parse_args()

    convert(args.input_dir, args.output_dir)
