"""
Real-time wrong-route violation detection with YOLOv8 + ByteTrack.

Features:
- Vehicle detection (YOLOv8)
- Multi-object tracking (ByteTrack)
- Per-track centroid history
- Smoothed motion-direction estimation
- Dot-product wrong-way logic against allowed lane direction
- Noise suppression via minimum displacement and confirmation streak
- Visualization with ID, direction vector, and violation label

Example:
    python wrong_route_bytetrack.py --source traffic.mp4 --weights yolov8n.pt --allowed-dir 1,0

Notes:
- ByteTrack is used through Ultralytics tracking API with tracker="bytetrack.yaml".
- Allowed direction is given in image coordinates (x to the right, y downward).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None


# COCO classes used for faster traffic-violation analysis.
# person=0, bicycle=1, car=2, motorcycle=3
TARGET_CLASS_IDS = {0, 1, 2, 3}
TRACK_HISTORY_FRAMES = 10
MIN_STABLE_HISTORY = 8


@dataclass
class TrackState:
    history: Deque[np.ndarray]
    wrong_streak: int = 0
    last_motion: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_cosine: float = 1.0
    is_wrong_way: bool = False
    last_seen_frame: int = 0
    lane_name: str = "GLOBAL"
    frames_seen: int = 0


@dataclass
class LaneRule:
    name: str
    polygon: np.ndarray  # Nx2 int32
    allowed_dir: np.ndarray  # normalized [dx, dy]


@dataclass
class CanonicalTrack:
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    last_seen_frame: int


@dataclass
class RuntimeTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    motion: np.ndarray
    lane_name: str
    is_wrong_way: bool


class IDStabilizer:
    """Keep a stable canonical ID even when ByteTrack briefly switches raw IDs."""

    def __init__(self, max_gap: int = 20, iou_match: float = 0.35, dist_match: float = 80.0) -> None:
        self.max_gap = max_gap
        self.iou_match = iou_match
        self.dist_match = dist_match
        self.raw_to_canonical: Dict[int, int] = {}
        self.canon_tracks: Dict[int, CanonicalTrack] = {}

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        xa = max(a[0], b[0])
        ya = max(a[1], b[1])
        xb = min(a[2], b[2])
        yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter <= 0:
            return 0.0
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return float(inter) / float(area_a + area_b - inter)

    @staticmethod
    def _dist(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        return float(((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5)

    def resolve(
        self,
        raw_id: int,
        bbox: Tuple[int, int, int, int],
        centroid: Tuple[float, float],
        frame_index: int,
        used_canon_ids: Set[int],
    ) -> int:
        if raw_id in self.raw_to_canonical:
            cid = self.raw_to_canonical[raw_id]
            self.canon_tracks[cid] = CanonicalTrack(bbox=bbox, centroid=centroid, last_seen_frame=frame_index)
            used_canon_ids.add(cid)
            return cid

        best_id = None
        best_score = -1e9
        for cid, tr in self.canon_tracks.items():
            if cid in used_canon_ids:
                continue
            if frame_index - tr.last_seen_frame > self.max_gap:
                continue

            iou = self._iou(bbox, tr.bbox)
            dist = self._dist(centroid, tr.centroid)
            if iou < self.iou_match and dist > self.dist_match:
                continue

            score = iou * 2.0 - (dist / max(self.dist_match, 1.0))
            if score > best_score:
                best_score = score
                best_id = cid

        if best_id is None:
            best_id = raw_id

        self.raw_to_canonical[raw_id] = best_id
        self.canon_tracks[best_id] = CanonicalTrack(bbox=bbox, centroid=centroid, last_seen_frame=frame_index)
        used_canon_ids.add(best_id)
        return best_id

    def cleanup(self, frame_index: int) -> None:
        stale = [cid for cid, tr in self.canon_tracks.items() if frame_index - tr.last_seen_frame > self.max_gap]
        for cid in stale:
            del self.canon_tracks[cid]


class WrongWayDetector:
    def __init__(
        self,
        allowed_direction: Tuple[float, float],
        history_size: int = TRACK_HISTORY_FRAMES,
        smooth_window: int = TRACK_HISTORY_FRAMES,
        min_displacement: float = 3.0,
        opposite_cos_threshold: float = -0.15,
        violation_confirm_frames: int = 3,
        min_track_frames: int = MIN_STABLE_HISTORY,
        ema_alpha: float = 0.45,
        max_missing_frames: int = 30,
    ) -> None:
        self.allowed_dir = self._normalize(np.array(allowed_direction, dtype=np.float32))
        if np.linalg.norm(self.allowed_dir) < 1e-6:
            raise ValueError("Allowed direction vector must be non-zero.")

        self.history_size = TRACK_HISTORY_FRAMES
        self.smooth_window = TRACK_HISTORY_FRAMES
        self.min_displacement = min_displacement
        self.opposite_cos_threshold = opposite_cos_threshold
        self.violation_confirm_frames = violation_confirm_frames
        self.min_track_frames = min_track_frames
        self.ema_alpha = ema_alpha
        self.max_missing_frames = max_missing_frames

        self.tracks: Dict[int, TrackState] = {}

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return np.zeros_like(v)
        return v / n

    def update(self, track_id: int, centroid: Tuple[float, float], frame_index: int) -> TrackState:
        return self.update_with_direction(track_id, centroid, frame_index, None, None)

    def update_with_direction(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        frame_index: int,
        allowed_direction: Optional[np.ndarray],
        lane_name: Optional[str],
    ) -> TrackState:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(history=deque(maxlen=TRACK_HISTORY_FRAMES))

        st = self.tracks[track_id]
        st.history.append(np.array(centroid, dtype=np.float32))
        st.last_seen_frame = frame_index
        st.frames_seen += 1
        if lane_name:
            st.lane_name = lane_name

        motion = self._smoothed_motion(st.history)
        st.last_motion = self.ema_alpha * motion + (1.0 - self.ema_alpha) * st.last_motion

        if len(st.history) < self.min_track_frames:
            st.wrong_streak = 0
            st.is_wrong_way = False
            return st

        disp = float(np.linalg.norm(st.last_motion))
        if disp < self.min_displacement:
            st.wrong_streak = max(0, st.wrong_streak - 1)
            st.is_wrong_way = st.wrong_streak >= self.violation_confirm_frames
            return st

        motion_unit = self._normalize(st.last_motion)
        dir_vec = self.allowed_dir if allowed_direction is None else self._normalize(allowed_direction)
        if np.linalg.norm(dir_vec) < 1e-6:
            dir_vec = self.allowed_dir
        cosine = float(np.dot(motion_unit, dir_vec))
        st.last_cosine = cosine

        moving_opposite = cosine <= self.opposite_cos_threshold
        if moving_opposite:
            st.wrong_streak += 1
        else:
            st.wrong_streak = max(0, st.wrong_streak - 1)

        st.is_wrong_way = st.wrong_streak >= self.violation_confirm_frames
        return st

    def cleanup(self, frame_index: int) -> None:
        stale_ids = [
            tid
            for tid, st in self.tracks.items()
            if frame_index - st.last_seen_frame > self.max_missing_frames
        ]
        for tid in stale_ids:
            del self.tracks[tid]

    def _smoothed_motion(self, history: Deque[np.ndarray]) -> np.ndarray:
        if len(history) < 3:
            return np.zeros(2, dtype=np.float32)

        window = min(TRACK_HISTORY_FRAMES, len(history))
        pts = np.array(list(history)[-window:], dtype=np.float32)
        if len(pts) < 3:
            return np.zeros(2, dtype=np.float32)

        # Direction from first-to-last centroid across the short history window.
        start = np.mean(pts[:2], axis=0)
        end = np.mean(pts[-2:], axis=0)
        return (end - start).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wrong-route violation detection (YOLOv8 + ByteTrack)")
    p.add_argument("--source", type=str, required=True, help="Video path, RTSP URL, or webcam index (e.g., 0)")
    p.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLOv8 model weights")
    p.add_argument("--allowed-dir", type=str, default="1,0", help="Allowed direction vector as 'dx,dy'")
    p.add_argument(
        "--lane-json",
        type=str,
        default="",
        help="Optional lane config JSON path. Format: [{\"name\":...,\"polygon\":[[x,y],...],\"allowed_dir\":[dx,dy]}]",
    )
    p.add_argument(
        "--lane-tolerance",
        type=float,
        default=50.0,
        help="Distance tolerance in pixels to assign track to nearest lane polygon",
    )
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--classes", type=str, default="0,1,2,3", help="Class IDs to keep, e.g. '0,2,3'")
    p.add_argument("--device", type=str, default="auto", help="Inference device: auto, cpu, cuda, or cuda:0")
    p.add_argument("--history-size", type=int, default=10, help="Centroid history size per track")
    p.add_argument("--smooth-window", type=int, default=10, help="Window for first-last direction estimation")
    p.add_argument("--min-disp", type=float, default=3.0, help="Min displacement to evaluate direction")
    p.add_argument("--opp-cos", type=float, default=-0.15, help="Cosine threshold below which movement is opposite")
    p.add_argument("--confirm-frames", type=int, default=3, help="Consecutive opposite frames to mark violation")
    p.add_argument("--min-track-frames", type=int, default=8, help="Minimum tracked history before direction check")
    p.add_argument("--ema-alpha", type=float, default=0.45, help="EMA smoothing factor for motion vectors")
    p.add_argument("--frame-skip", type=int, default=3, help="Process every Nth input frame for speed (>=1)")
    p.add_argument("--detect-every", type=int, default=2, help="Run YOLO every N processed frames and track in-between")
    p.add_argument("--width", type=int, default=640, help="Resize width for processing/display")
    p.add_argument("--height", type=int, default=480, help="Resize height for processing/display")
    p.add_argument("--save", type=str, default="", help="Optional output video path")
    p.add_argument("--show", action="store_true", help="Show live window (requires OpenCV HighGUI support)")
    p.add_argument("--left-allowed-dir", type=str, default="-1,0", help="Allowed direction for left lane split")
    p.add_argument("--right-allowed-dir", type=str, default="1,0", help="Allowed direction for right lane split")
    p.add_argument("--debug", action="store_true", help="Print per-vehicle direction debug logs")
    p.add_argument("--tracker-cfg", type=str, default="d:/project_code/bytetrack_stable.yaml", help="ByteTrack YAML config path")
    return p.parse_args()


def parse_vector(s: str) -> Tuple[float, float]:
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError("--allowed-dir must be in 'dx,dy' format")
    return float(parts[0]), float(parts[1])


def parse_class_ids(s: str) -> List[int]:
    ids: List[int] = []
    for p in s.split(","):
        t = p.strip()
        if not t:
            continue
        ids.append(int(t))
    return ids


def open_source(src: str) -> cv2.VideoCapture:
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)


def load_lane_rules(path: str) -> List[LaneRule]:
    if not path:
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lanes: List[LaneRule] = []
    for item in data:
        name = str(item.get("name", f"lane_{len(lanes)}"))
        poly = np.array(item["polygon"], dtype=np.int32)
        vec = np.array(item["allowed_dir"], dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n < 1e-6:
            continue
        lanes.append(LaneRule(name=name, polygon=poly, allowed_dir=(vec / n)))
    return lanes


def assign_lane(
    cx: float,
    cy: float,
    lanes: List[LaneRule],
    tolerance: float,
) -> Optional[LaneRule]:
    if not lanes:
        return None

    best: Optional[LaneRule] = None
    best_dist = -1e9
    for lane in lanes:
        dist = cv2.pointPolygonTest(lane.polygon, (float(cx), float(cy)), True)
        if dist > best_dist:
            best_dist = dist
            best = lane

    if best is None or best_dist < -float(tolerance):
        return None
    return best


def draw_lanes(frame: np.ndarray, lanes: List[LaneRule]) -> None:
    for lane in lanes:
        pts = lane.polygon.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
        cx = int(lane.polygon[:, 0].mean())
        cy = int(lane.polygon[:, 1].mean())
        dx = int(lane.allowed_dir[0] * 40)
        dy = int(lane.allowed_dir[1] * 40)
        cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (255, 200, 0), 2, tipLength=0.35)
        cv2.putText(frame, lane.name, (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)


def split_lane_direction(
    cx: float,
    frame_w: int,
    left_dir: np.ndarray,
    right_dir: np.ndarray,
) -> Tuple[np.ndarray, str]:
    if cx < frame_w * 0.5:
        return left_dir, "LEFT"
    return right_dir, "RIGHT"


def main() -> None:
    args = parse_args()
    args.frame_skip = max(1, int(args.frame_skip))
    args.detect_every = max(1, int(args.detect_every))
    allowed_dir = parse_vector(args.allowed_dir)
    left_allowed_dir = np.array(parse_vector(args.left_allowed_dir), dtype=np.float32)
    right_allowed_dir = np.array(parse_vector(args.right_allowed_dir), dtype=np.float32)
    target_classes = parse_class_ids(args.classes)
    if not target_classes:
        target_classes = sorted(TARGET_CLASS_IDS)

    if args.device == "auto":
        use_cuda = bool(torch is not None and torch.cuda.is_available())
        device = "cuda:0" if use_cuda else "cpu"
    else:
        device = args.device

    lane_rules = load_lane_rules(args.lane_json)

    model = YOLO(args.weights)
    model.to(device)
    detector = WrongWayDetector(
        allowed_direction=allowed_dir,
        history_size=args.history_size,
        smooth_window=args.smooth_window,
        min_displacement=args.min_disp,
        opposite_cos_threshold=args.opp_cos,
        violation_confirm_frames=args.confirm_frames,
        min_track_frames=args.min_track_frames,
        ema_alpha=args.ema_alpha,
    )
    id_stabilizer = IDStabilizer(max_gap=20, iou_match=0.35, dist_match=90.0)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 25.0, (args.width, args.height))

    frame_index = 0
    processed_index = 0
    fps_t0 = time.time()
    fps_count = 0
    fps_value = 0.0
    active_tracks: Dict[int, RuntimeTrack] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if args.frame_skip > 1 and (frame_index % args.frame_skip != 0):
                continue
            processed_index += 1

            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            if lane_rules:
                draw_lanes(frame, lane_rules)

            run_detection = (processed_index % args.detect_every) == 1
            frame_tracks: Dict[int, RuntimeTrack] = {}

            if run_detection:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=args.tracker_cfg,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    classes=target_classes,
                    device=device,
                    verbose=False,
                )

                r = results[0]
                boxes = r.boxes

                if boxes is not None and boxes.xyxy is not None:
                    xyxy = boxes.xyxy.cpu().numpy().astype(int)
                    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(xyxy), -1)
                    used_canon_ids: Set[int] = set()

                    for i, box in enumerate(xyxy):
                        track_id = int(ids[i])
                        if track_id < 0:
                            continue

                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) * 0.5
                        cy = (y1 + y2) * 0.5

                        canonical_id = id_stabilizer.resolve(
                            raw_id=track_id,
                            bbox=(x1, y1, x2, y2),
                            centroid=(cx, cy),
                            frame_index=frame_index,
                            used_canon_ids=used_canon_ids,
                        )

                        if lane_rules:
                            lane = assign_lane(cx, cy, lane_rules, args.lane_tolerance)
                            lane_dir = lane.allowed_dir if lane is not None else detector.allowed_dir
                            lane_name = lane.name if lane is not None else "GLOBAL"
                        else:
                            lane_dir, lane_name = split_lane_direction(cx, args.width, left_allowed_dir, right_allowed_dir)

                        st = detector.update_with_direction(
                            canonical_id,
                            (cx, cy),
                            frame_index,
                            lane_dir,
                            lane_name,
                        )

                        if args.debug:
                            decision = "WRONG_ROUTE" if st.is_wrong_way else "NORMAL"
                            print(
                                f"vehicle_id={canonical_id} "
                                f"movement=({float(st.last_motion[0]):.2f},{float(st.last_motion[1]):.2f}) "
                                f"dot={st.last_cosine:.3f} "
                                f"decision={decision}"
                            )

                        frame_tracks[canonical_id] = RuntimeTrack(
                            track_id=canonical_id,
                            bbox=(x1, y1, x2, y2),
                            centroid=(cx, cy),
                            motion=st.last_motion.copy(),
                            lane_name=st.lane_name,
                            is_wrong_way=st.is_wrong_way,
                        )
            else:
                # Lightweight in-between tracking: shift bboxes using last smoothed motion.
                for canonical_id, prev in active_tracks.items():
                    st = detector.tracks.get(canonical_id)
                    motion = st.last_motion.copy() if st is not None else prev.motion
                    dx = int(round(float(motion[0])))
                    dy = int(round(float(motion[1])))

                    x1, y1, x2, y2 = prev.bbox
                    x1 = int(np.clip(x1 + dx, 0, args.width - 1))
                    y1 = int(np.clip(y1 + dy, 0, args.height - 1))
                    x2 = int(np.clip(x2 + dx, 0, args.width - 1))
                    y2 = int(np.clip(y2 + dy, 0, args.height - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5

                    if lane_rules:
                        lane = assign_lane(cx, cy, lane_rules, args.lane_tolerance)
                        lane_dir = lane.allowed_dir if lane is not None else detector.allowed_dir
                        lane_name = lane.name if lane is not None else "GLOBAL"
                    else:
                        lane_dir, lane_name = split_lane_direction(cx, args.width, left_allowed_dir, right_allowed_dir)

                    st = detector.update_with_direction(
                        canonical_id,
                        (cx, cy),
                        frame_index,
                        lane_dir,
                        lane_name,
                    )

                    frame_tracks[canonical_id] = RuntimeTrack(
                        track_id=canonical_id,
                        bbox=(x1, y1, x2, y2),
                        centroid=(cx, cy),
                        motion=st.last_motion.copy(),
                        lane_name=st.lane_name,
                        is_wrong_way=st.is_wrong_way,
                    )

            active_tracks = frame_tracks

            for tr in active_tracks.values():
                x1, y1, x2, y2 = tr.bbox
                color = (0, 0, 255) if tr.is_wrong_way else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                status = "WRONG" if tr.is_wrong_way else "OK"
                label = f"ID {tr.track_id} {status}"
                cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

            detector.cleanup(frame_index)
            id_stabilizer.cleanup(frame_index)

            fps_count += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 0.5:
                fps_value = fps_count / max(elapsed, 1e-6)
                fps_t0 = time.time()
                fps_count = 0

            if lane_rules:
                allowed_text = f"Lane mode: {len(lane_rules)} lanes"
            else:
                allowed_text = (
                    f"Lane split mode | L:({left_allowed_dir[0]:.1f},{left_allowed_dir[1]:.1f}) "
                    f"R:({right_allowed_dir[0]:.1f},{right_allowed_dir[1]:.1f})"
                )
            cv2.putText(frame, allowed_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"FPS: {fps_value:.1f} | dev:{device} | skip:{args.frame_skip} detN:{args.detect_every}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            if args.show:
                cv2.imshow("Wrong Route Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
