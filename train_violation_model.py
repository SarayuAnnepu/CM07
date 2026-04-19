"""
train_violation_model.py

Train a custom YOLO model for traffic-violation classes such as:
- person
- helmet
- seatbelt
- motorbike

Usage examples:
    python train_violation_model.py --data data/violation_dataset.yaml
    python train_violation_model.py --data data/violation_dataset.yaml --model yolo26m.pt --epochs 120 --imgsz 960
    python train_violation_model.py --data data/violation_dataset.yaml --classes person car truck motorbike helmet seatbelt traffic_light_red traffic_light_green traffic_light_yellow
"""

from __future__ import annotations

import argparse
import json
import os
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for traffic violation detection")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--project", type=str, default="runs/train", help="Output project directory")
    parser.add_argument("--name", type=str, default="violation_yolo26", help="Run name")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id, 'cpu', or '0,1'")
    parser.add_argument("--classes", nargs="*", default=[], help="Expected dataset class names")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        cache=True,
        workers=4,
        patience=30,
        close_mosaic=10,
        cos_lr=True,
        amp=True,
    )

    # Validate best checkpoint after training.
    best_path = f"{args.project}/{args.name}/weights/best.pt"
    best = YOLO(best_path)
    best.val(data=args.data, imgsz=args.imgsz, device=args.device)

    # Persist training metadata to make deployment reproducible.
    meta = {
        "best_weights": best_path,
        "base_model": args.model,
        "data_yaml": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "classes": args.classes,
    }
    out_meta = os.path.join(args.project, args.name, "training_meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
