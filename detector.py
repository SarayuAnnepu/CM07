"""
detector.py — YOLO detection engine (optimised for real-time).

Wraps ``ultralytics.YOLO`` and returns a clean list of ``Detection``
dataclass instances for downstream consumption.

Optimisations:
  • GPU acceleration with automatic CUDA detection.
  • Half-precision (FP16) inference when on a supported GPU.
  • Class filtering to skip irrelevant COCO categories.
  • Batch tensor conversion to avoid per-box .cpu().numpy() overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    MODEL_FALLBACKS,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DETECTION_IMGSZ,
    COCO_CLASS_MAP,
    CLASS_ALIASES,
    USE_GPU,
    USE_HALF_PRECISION,
    DETECT_CLASSES,
)


@dataclass
class Detection:
    """Single object detection result."""
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int = -1

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class ObjectDetector:
    """Loads a YOLO model and runs inference on individual frames."""

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        # Determine device: prefer CUDA when available & requested.
        self.device = self._pick_device()
        self.use_half = USE_HALF_PRECISION and self.device != "cpu"

        self.model = self._load_with_fallbacks(model_path)
        self.model.to(self.device)
        if self.use_half:
            self.model.half()

        self.supported_labels = self._extract_model_labels(self.model)
        self._runtime_classes = self._resolve_runtime_classes()
        print(f"[Detector] {model_path} on {self.device}" + (" (FP16)" if self.use_half else ""))

    # ── public API ──────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO inference and return a list of ``Detection`` objects."""
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=DETECTION_IMGSZ,
            classes=self._runtime_classes,
            half=self.use_half,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            names = getattr(result, "names", None)

            # Batch-convert tensors once instead of per-box .cpu().numpy()
            cls_ids = boxes.cls.int().cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxys = boxes.xyxy.int().cpu().numpy()

            for i in range(len(cls_ids)):
                cls_id = int(cls_ids[i])
                label = self._resolve_label(cls_id, names)
                conf = float(confs[i])
                x1, y1, x2, y2 = xyxys[i]
                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        class_id=cls_id,
                    )
                )
        return detections

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _pick_device() -> str:
        """Select the best available device."""
        if USE_GPU and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @staticmethod
    def _load_with_fallbacks(primary_model: str) -> YOLO:
        """Try primary model first, then configured fallbacks."""
        candidates: List[str] = [primary_model] + [m for m in MODEL_FALLBACKS if m != primary_model]
        last_error: Exception | None = None

        for candidate in candidates:
            try:
                model = YOLO(candidate)
                return model
            except Exception as exc:
                last_error = exc

        raise RuntimeError("Could not load any YOLO model candidate.") from last_error

    @staticmethod
    def _resolve_label(cls_id: int, names: dict | list | None = None) -> str:
        """Map a YOLO class id to a human-readable label."""
        raw_label: str
        if isinstance(names, dict) and cls_id in names:
            raw_label = str(names[cls_id])
        elif isinstance(names, list) and 0 <= cls_id < len(names):
            raw_label = str(names[cls_id])
        else:
            raw_label = COCO_CLASS_MAP.get(cls_id, f"class_{cls_id}")

        return ObjectDetector._normalize_label(raw_label)

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        normalized = raw_label.strip().lower().replace("_", " ")
        return CLASS_ALIASES.get(normalized, normalized)

    @staticmethod
    def _extract_model_labels(model: YOLO) -> set[str]:
        names = getattr(model, "names", None)
        labels: set[str] = set()
        if isinstance(names, dict):
            for value in names.values():
                labels.add(ObjectDetector._normalize_label(str(value)))
        elif isinstance(names, list):
            for value in names:
                labels.add(ObjectDetector._normalize_label(str(value)))
        return labels

    def _resolve_runtime_classes(self) -> list[int] | None:
        """
        Keep class filtering for stock COCO models, but disable it for
        custom models that contain helmet/seatbelt labels so those classes
        are not accidentally dropped by COCO id filtering.
        """
        if not DETECT_CLASSES:
            return None

        supports_custom_safety = bool({"helmet", "seatbelt"} & self.supported_labels)
        if supports_custom_safety:
            return None
        return DETECT_CLASSES
