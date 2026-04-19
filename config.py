"""
config.py — Global configuration for the Traffic Violation Detection System.

All tuneable constants (model path, class lists, ROI coordinates, thresholds)
live here so every other module can simply ``from config import *``.
"""

import numpy as np

# ──────────────────────────── Model Settings ────────────────────────────
# Preferred-first model strategy:
# 1) Try YOLOv26 weights first (user requirement), then
# 2) Fall back to YOLO11/YOLOv8 weights for compatibility.
# For best helmet/seatbelt accuracy, replace with your custom-trained weights
# that include those classes.
MODEL_PATH: str = "yolov8n.pt"
MODEL_FALLBACKS: list[str] = ["yolo11x.pt", "yolo11m.pt", "yolov8x.pt", "yolov8m.pt"]
CONFIDENCE_THRESHOLD: float = 0.20                # slightly lower to recover occluded riders
IOU_THRESHOLD: float = 0.45                       # NMS IoU threshold
DETECTION_IMGSZ: int = 640                        # lower = faster, higher = more accurate

# ──────────────────────────── Performance Tuning ────────────────────────
FRAME_SKIP: int = 3                               # process every Nth frame (grab & discard in between)
DETECT_EVERY_N: int = 5                           # full YOLO detection every N *processed* frames
PROCESS_WIDTH: int = 640                          # resize input width for processing
PROCESS_HEIGHT: int = 480                         # resize input height for processing
USE_GPU: bool = True                              # enable CUDA acceleration if available
USE_HALF_PRECISION: bool = True                   # FP16 inference (2× on supported GPUs)
MINIMAL_LABELS: bool = True                       # lighter annotation drawing
DETECT_CLASSES: list[int] = [0, 1, 2, 3, 5, 7, 9]  # COCO IDs incl. traffic light

# ──────────────────────────── Target Classes ────────────────────────────
TARGET_CLASSES: list[str] = [
    "car", "motorbike", "bus", "truck",
    "person", "helmet", "seatbelt",
    "traffic light",
]

# COCO class-name mapping used by stock COCO models (e.g. YOLOv8/YOLO11).
# The generic model does NOT have "helmet" or "seatbelt" — those require
# a custom-trained model.
COCO_CLASS_MAP: dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorbike",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",     # <-- Added: traffic light detection
}

VEHICLE_CLASSES: set[str] = {"car", "motorbike", "bus", "truck"}

# If True, violations like NO_HELMET/NO_SEATBELT are only checked if the model 
# has proven it can detect those classes (i.e. >=1 detection in the frame).
# If False, the system will always check for them (useful for demo/testing).
STRICT_HELMET_CHECK: bool = True
STRICT_SEATBELT_CHECK: bool = False
STRICT_WRONG_ROUTE_CHECK: bool = True

# If the loaded model does not provide a "helmet" class, optionally use
# image-based head-region heuristics to infer likely helmet/no-helmet.
ENABLE_HELMET_FALLBACK_HEURISTIC: bool = True
HELMET_HEAD_REGION_RATIO: float = 0.30
HELMET_COLOR_RATIO_THRESHOLD: float = 0.08
NO_HELMET_EXPOSED_HEAD_THRESHOLD: float = 0.20
HELMET_HEAD_CENTER_WIDTH_RATIO: float = 0.55
HELMET_MAX_SKIN_RATIO_FOR_HELMET: float = 0.08
HELMET_HEAD_MODEL_PATH: str = "helmet_head_yolov8n.pt"
HELMET_HEAD_MODEL_FALLBACKS: list[str] = []
HELMET_HEAD_MODEL_CONFIDENCE: float = 0.35
HELMET_HEAD_MODEL_IOU: float = 0.45
HELMET_HEAD_MODEL_IMGSZ: int = 320

# Normalize varying model class names to project-standard labels.
CLASS_ALIASES: dict[str, str] = {
    "motorcycle": "motorbike",
    "bike": "motorbike",
    "motor cycle": "motorbike",
    "track": "truck",
    "lorry": "truck",
    "traffic-light": "traffic light",
    "traffic_light": "traffic light",
}

# ──────────────────────────── Detector Config ───────────────────────────
DEFAULT_MODEL: str = MODEL_PATH
# IoU = overlap between bounding boxes.
# "containment" = fraction of smaller box inside larger box.
RIDER_CONTAINMENT_THRESHOLD: float = 0.18
MIN_RIDER_CONFIDENCE: float = 0.30
HELMET_IOU_THRESHOLD: float = 0.05     # helmet overlap inside rider box
HELMET_PERSON_MIN_CONFIDENCE: float = 0.50
HELMET_BIKE_MIN_CONFIDENCE: float = 0.50
HELMET_MIN_CONFIDENCE: float = 0.50
HELMET_MIN_BIKE_WIDTH: int = 55
HELMET_MIN_BIKE_HEIGHT: int = 40
HELMET_MIN_PERSON_WIDTH: int = 20
HELMET_MIN_PERSON_HEIGHT: int = 30
HELMET_MIN_HEAD_HEIGHT: int = 10
HELMET_HEAD_MATCH_IOU: float = 0.06
HELMET_BLUR_LAPLACIAN_MIN: float = 18.0
HELMET_MIN_HEAD_WIDTH: int = 12
HELMET_RIDER_PROXIMITY_MAX: float = 180.0
HELMET_SKIP_OCCLUDED_HEAD: bool = True
SEATBELT_CONTAINMENT_THRESHOLD: float = 0.50 # person-in-car overlap (high = strict)
MIN_SEATBELT_PERSON_CONFIDENCE: float = 0.45
MIN_SEATBELT_CONFIDENCE: float = 0.40
SEATBELT_MATCH_THRESHOLD: float = 0.08
SEATBELT_TRACK_MATCH_IOU: float = 0.30
SEATBELT_VIOLATION_CONFIRM_FRAMES: int = 3
SEATBELT_VEHICLE_MIN_CONFIDENCE: float = 0.50
SEATBELT_MIN_VEHICLE_WIDTH: int = 110
SEATBELT_MIN_VEHICLE_HEIGHT: int = 90
SEATBELT_MIN_DRIVER_WIDTH: int = 26
SEATBELT_MIN_DRIVER_HEIGHT: int = 36
SEATBELT_DRIVER_REGION_MODE: str = "auto"  # auto, left, right
SEATBELT_DRIVER_LEFT_X_RANGE: tuple[float, float] = (0.05, 0.60)
SEATBELT_DRIVER_RIGHT_X_RANGE: tuple[float, float] = (0.40, 0.95)
SEATBELT_DRIVER_Y_RANGE: tuple[float, float] = (0.10, 0.72)
SEATBELT_CHEST_REGION_TOP_RATIO: float = 0.18
SEATBELT_CHEST_REGION_BOTTOM_RATIO: float = 0.68
SEATBELT_BLUR_LAPLACIAN_MIN: float = 30.0
SEATBELT_MAX_DRIVER_TOUCH_BORDER_RATIO: float = 0.35
SEATBELT_DIAGONAL_MIN_ANGLE: float = 22.0
SEATBELT_DIAGONAL_MAX_ANGLE: float = 78.0
SEATBELT_MIN_DIAGONAL_LENGTH_RATIO: float = 0.35
TRIPLE_RIDE_IOU_THRESHOLD: float = 0.0 # any overlap counts
TRIPLE_RIDE_MIN_PERSONS: int = 3

# ──────────────────────────── Red-Light ROI ─────────────────────────────
# Polygon representing the critical intersection zone. Any vehicle inside 
# this zone while the signal is RED triggers a violation.
RED_LIGHT_ROI: np.ndarray = np.array([
    [250,  200],
    [950,  200],
    [950,  720],
    [250,  720],
], dtype=np.int32)

# ──────────────────────────── Lane Configuration ────────────────────────
# Maps screen regions to expected traffic flow vectors.
#   "expected_dy": -1 (up), +1 (down), 0 (ignore)
#   "expected_dx": -1 (left), +1 (right), 0 (ignore)
LANE_CONFIG: list[dict] = [
    {
        "name": "Near Side (Westbound)",
        "polygon": np.array([[0, 400], [1280, 400], [1280, 720], [0, 720]], dtype=np.int32),
        "expected_dy": 0,    
        "expected_dx": -1,     # Vehicles in near lane must travel leftwards
    },
    {
        "name": "Far Side (Eastbound)",
        "polygon": np.array([[0, 100], [1280, 100], [1280, 400], [0, 400]], dtype=np.int32),
        "expected_dy": 0,    
        "expected_dx": +1,     # Vehicles in far lane must travel rightwards
    },
]

# Minimum pixel displacement before we judge direction (prevents jitter causing false positives)
DIRECTION_MIN_DISPLACEMENT: int = 5
WRONG_ROUTE_AXIS_MIN_COMPONENT: float = 0.8
WRONG_ROUTE_LANE_ASSIGN_TOLERANCE: float = 80.0
WRONG_ROUTE_MIN_DISPLACEMENT: float = 6.0
WRONG_ROUTE_DOT_THRESHOLD: float = 0.05
WRONG_ROUTE_TRAJECTORY_WINDOW: int = 6
WRONG_ROUTE_MIN_HISTORY_FRAMES: int = 4
WRONG_ROUTE_CONFIRM_FRAMES: int = 2
WRONG_ROUTE_LOG_COOLDOWN_SECONDS: float = 10.0
WRONG_ROUTE_MIN_CONFIDENCE: float = 0.35

# ──────────────────────────── Tracker Settings ──────────────────────────
TRACKER_BACKEND: str = "deepsort"  # options: deepsort, centroid
MAX_DISAPPEARED_FRAMES: int = 30
HISTORY_LENGTH: int = 10
DEEPSORT_MAX_AGE: int = 30
DEEPSORT_N_INIT: int = 2
DEEPSORT_MAX_IOU_DISTANCE: float = 0.7

# ByteTrack stabilization targets (used by ByteTrack YAML config).
BYTETRACK_MATCH_THRESH: float = 0.80
BYTETRACK_TRACK_BUFFER: int = 30

# Stable vehicle grouping across ID switches.
STABLE_GROUP_IOU_THRESHOLD: float = 0.40
STABLE_GROUP_CENTROID_DISTANCE: float = 90.0
STABLE_GROUP_MAX_GAP_FRAMES: int = 30
VIOLATION_DEDUP_SECONDS: float = 15.0
VIOLATION_MIN_CONFIDENCE: float = 0.50

# ──────────────────────────── OCR Settings ──────────────────────────────
OCR_LANGUAGES: list[str] = ["en"]
PLATE_MODEL_PATH: str = "yolov8n.pt"  # replace with your plate detector weights if available
PLATE_DET_CLASSES: tuple[str, ...] = ("license plate", "number plate", "plate")

# ──────────────────────────── Violation Logging ────────────────────────
ENABLE_VIOLATION_LOGGING: bool = True
VIOLATION_LOG_DIR: str = "evidence_logs"
SQLITE_DB_PATH: str = "evidence_logs/violations.db"

# ──────────────────────────── Visualisation ─────────────────────────────
COLOR_NORMAL: tuple[int, int, int] = (0, 255, 0)       # green
COLOR_VIOLATION: tuple[int, int, int] = (0, 0, 255)     # red
COLOR_ROI: tuple[int, int, int] = (255, 255, 0)         # cyan
COLOR_LANE: tuple[int, int, int] = (255, 200, 0)        # light-blue
FONT_SCALE: float = 0.55
FONT_THICKNESS: int = 2
