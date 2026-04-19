"""
anpr.py — Automatic Number Plate Recognition.

Dual-engine OCR for maximum plate-read accuracy:
  - RapidOCR  (PaddleOCR models via ONNX Runtime) — best on colour images.
  - EasyOCR   — fallback, excellent on CLAHE-enhanced greyscale.

Pipeline per vehicle:
  1. Contour-based plate-region candidates.
  2. Lower-half crop (plates sit low on most vehicles).
  3. Full vehicle crop.
  Each crop is tried with both engines; best result wins.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ── optional OCR backends ───────────────────────────────────────────────
try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPID_OK = True
except ImportError:
    _RAPID_OK = False

try:
    import easyocr as _easyocr
    _EASY_OK = True
except ImportError:
    _EASY_OK = False

from config import OCR_LANGUAGES
from config import PLATE_MODEL_PATH, PLATE_DET_CLASSES

_PLATE_RE = re.compile(r"[^A-Za-z0-9 ]")


class ANPRReader:
    """Dual-engine license-plate reader."""

    def __init__(self, languages: list[str] | None = None) -> None:
        self._plate_model = None
        try:
            self._plate_model = YOLO(PLATE_MODEL_PATH)
        except Exception:
            self._plate_model = None

        self._rapid = RapidOCR() if _RAPID_OK else None

        self._easy = None
        if _EASY_OK:
            langs = languages or OCR_LANGUAGES
            self._easy = _easyocr.Reader(langs, gpu=True)

    # ── public API ──────────────────────────────────────────────────────
    def read_plate(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        min_confidence: float = 0.15,
    ) -> str:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        vehicle = frame[y1:y2, x1:x2]
        if vehicle.size == 0:
            return ""

        # Gather crops: plate candidates → lower-half → full
        crops: List[np.ndarray] = []
        crops.extend(self._find_plate_regions_yolo(vehicle))
        crops.extend(self._find_plate_regions(vehicle))

        ch, cw = vehicle.shape[:2]
        lower = vehicle[int(ch * 0.5):ch, :]
        if lower.size > 0:
            crops.append(lower)
        crops.append(vehicle)

        # Try every crop with both engines, collect all candidates
        candidates: List[Tuple[str, float]] = []

        for crop in crops:
            colour = self._upscale(crop)
            grey = self._enhance_grey(crop)

            # RapidOCR works best on colour
            candidates.extend(self._run_rapid(colour))
            # EasyOCR works best on enhanced greyscale
            candidates.extend(self._run_easy(grey))
            # Also try EasyOCR with adaptive threshold
            thresh = self._adaptive_thresh(grey)
            candidates.extend(self._run_easy(thresh))

        if not candidates:
            return ""

        # Pick the candidate with the highest confidence
        candidates.sort(key=lambda t: t[1], reverse=True)
        best_text, best_conf = candidates[0]
        return best_text if best_conf >= min_confidence else ""

    # ── YOLO plate localization ────────────────────────────────────────
    def _find_plate_regions_yolo(self, vehicle: np.ndarray, max_n: int = 3) -> List[np.ndarray]:
        if self._plate_model is None or vehicle.size == 0:
            return []

        try:
            results = self._plate_model(vehicle, conf=0.2, iou=0.5, verbose=False)
        except Exception:
            return []

        out: List[np.ndarray] = []
        h, w = vehicle.shape[:2]
        for result in results:
            boxes = result.boxes
            names = getattr(result, "names", {})
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls[0])
                label = str(names.get(cls_id, "")).strip().lower() if isinstance(names, dict) else ""
                if label and label not in PLATE_DET_CLASSES:
                    continue

                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = vehicle[y1:y2, x1:x2]
                if crop.size > 0:
                    out.append(crop)
                if len(out) >= max_n:
                    return out
        return out

    # ── RapidOCR ────────────────────────────────────────────────────────
    def _run_rapid(self, img: np.ndarray) -> List[Tuple[str, float]]:
        if not self._rapid:
            return []
        try:
            result, _ = self._rapid(img)
            if not result:
                return []
            out: List[Tuple[str, float]] = []
            for line in result:
                text = _PLATE_RE.sub("", str(line[1])).strip()
                conf = float(line[2])
                if len(text) >= 2:
                    out.append((text, conf))
            return out
        except Exception:
            return []

    # ── EasyOCR ─────────────────────────────────────────────────────────
    def _run_easy(self, img: np.ndarray) -> List[Tuple[str, float]]:
        if not self._easy:
            return []
        try:
            results = self._easy.readtext(img, detail=1)
            if not results:
                return []
            out: List[Tuple[str, float]] = []
            for _bbox, text, conf in results:
                cleaned = _PLATE_RE.sub("", text).strip()
                if len(cleaned) >= 2:
                    out.append((cleaned, conf))
            return out
        except Exception:
            return []

    # ── image preprocessing ─────────────────────────────────────────────
    @staticmethod
    def _upscale(crop: np.ndarray) -> np.ndarray:
        """Upscale small crops so OCR has enough pixels to work with."""
        h, w = crop.shape[:2]
        if w < 250:
            scale = 250 / w
            return cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def _enhance_grey(crop: np.ndarray) -> np.ndarray:
        """Convert to greyscale + CLAHE."""
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(grey)
        h, w = enhanced.shape[:2]
        if w < 250:
            scale = 250 / w
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
        return enhanced

    @staticmethod
    def _adaptive_thresh(grey: np.ndarray) -> np.ndarray:
        """Adaptive threshold — good for low-contrast plates."""
        if len(grey.shape) == 3:
            grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )

    # ── contour-based plate finder ──────────────────────────────────────
    @staticmethod
    def _find_plate_regions(crop: np.ndarray, max_n: int = 5) -> List[np.ndarray]:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[int, np.ndarray]] = []
        ch, cw = crop.shape[:2]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / max(h, 1)
            if (1.5 <= aspect <= 7.0
                    and area > ch * cw * 0.004
                    and area < ch * cw * 0.6
                    and w > 25 and h > 8):
                pad_x, pad_y = int(w * 0.08), int(h * 0.2)
                sub = crop[max(0, y - pad_y):min(ch, y + h + pad_y),
                           max(0, x - pad_x):min(cw, x + w + pad_x)]
                if sub.size > 0:
                    candidates.append((area, sub))

        candidates.sort(key=lambda t: t[0], reverse=True)
        return [c[1] for c in candidates[:max_n]]
