"""
prepare_dataset.py

Extract frames from a video and split into YOLO train/val image folders.
After extraction, annotate these images (LabelImg/CVAT/Roboflow) and place
matching .txt labels in labels/train and labels/val.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/val image dataset from video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", default="data/violation_dataset", help="Output dataset root")
    parser.add_argument("--every", type=int, default=10, help="Save every Nth frame")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_root = Path(args.out)
    train_dir = out_root / "images" / "train"
    val_dir = out_root / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frame_idx = 0
    saved = 0
    val_step = max(2, int(round(1.0 / max(1e-6, args.val_ratio))))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.every == 0:
            target_dir = val_dir if (saved % val_step == 0) else train_dir
            out_name = f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(target_dir / out_name), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"[prepare] Extracted {saved} frames from {args.video}")
    print(f"[prepare] Train images: {len(list(train_dir.glob('*.jpg')))}")
    print(f"[prepare] Val images: {len(list(val_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
