"""
standalone_multi_violation.py
Full robust real-time traffic violation detection system pipeline.
Supports multi-violations per vehicle, deduplication, and ByteTrack integration.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Set, Tuple

# You may need to pip install ultralytics and set up tracking dependencies
from ultralytics import YOLO


# ────────────────────────────────────────────────────────────────────────
#  1. DATA STRUCTURES & SETUP
# ────────────────────────────────────────────────────────────────────────

class VehicleData:
    def __init__(self, track_id: int, vtype: str, bbox: Tuple[int, int, int, int]):
        self.track_id = track_id
        self.type = vtype     # "car" or "motorcycle"
        self.bbox = bbox      # [x1, y1, x2, y2]
        self.riders: List[Tuple[int, int, int, int]] = []  # List of person bboxes
        self.violations: Set[str] = set()

# A mock tracker interface simulating ByteTrack / DeepSORT
class MockByteTrack:
    def __init__(self):
        self.next_id = 1
    def update(self, bboxes):
        # In a real setup, this runs IoU matching / Kalman filtering.
        # Returning mock fixed IDs for demonstration.
        tracked = []
        for b in bboxes:
            tracked.append((self.next_id, b))
            self.next_id += 1
        return tracked


class TrafficMonitor:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.tracker = MockByteTrack()
        
        # Deduplication state: (track_id, violation_type) -> last_logged_timestamp
        self.last_logged: Dict[Tuple[int, str], float] = {}
        self.log_cooldown = 10.0  # seconds
        
        # History for route tracking
        # track_id -> list of centroids over time
        self.trajectory_history: Dict[int, List[Tuple[float, float]]] = {}
        
    # ────────────────────────────────────────────────────────────────────────
    #  2. DETECTION & TRACKING
    # ────────────────────────────────────────────────────────────────────────

    def detect_objects(self, frame: np.ndarray):
        """Runs YOLO and parses common objects."""
        results = self.model(frame, conf=0.5, verbose=False)
        
        detections = {"vehicle": [], "person": [], "helmet": [], "traffic_light": []}
        
        if not results or not results[0].boxes:
            return detections
            
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            label = self.model.names.get(cls_id, "")
            
            bbox_t = (x1, y1, x2, y2)
            if label in ["car", "motorcycle", "truck", "bus"]:
                detections["vehicle"].append((label, bbox_t))
            elif label == "person":
                detections["person"].append(bbox_t)
            elif label == "helmet":
                detections["helmet"].append(bbox_t)
            elif label == "traffic light":
                detections["traffic_light"].append(bbox_t)
                
        return detections

    def track_objects(self, detections: dict) -> Dict[int, VehicleData]:
        """Track vehicles and initialize unification objects."""
        vehicle_bboxes = [d[1] for d in detections["vehicle"]]
        
        # Update tracker
        tracked_items = self.tracker.update(vehicle_bboxes)
        
        vehicles = {}
        for (tid, bbox) in tracked_items:
            # Map back to original label
            vtype = "car"
            for label, orig_box in detections["vehicle"]:
                if orig_box == bbox: # simple mapping for mock tracker
                    vtype = label
                    if vtype in ["motorcycle", "bike"]: vtype = "motorcycle"
                    break
                    
            vehicle = VehicleData(track_id=tid, vtype=vtype, bbox=bbox)
            vehicles[tid] = vehicle
            
            # Update trajectory
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            if tid not in self.trajectory_history:
                self.trajectory_history[tid] = []
            self.trajectory_history[tid].append((cx, cy))
            # Limit history
            if len(self.trajectory_history[tid]) > 30:
                self.trajectory_history[tid].pop(0)
                
        return vehicles

    def associate_riders(self, vehicles: Dict[int, VehicleData], persons: List[Tuple[int, int, int, int]]):
        """Map persons to motorcycles via box overlap / containment."""
        for p_box in persons:
            px1, py1, px2, py2 = p_box
            p_area = max(0, px2-px1) * max(0, py2-py1)
            
            best_bike = None
            best_overlap = 0.0
            
            for tid, v in vehicles.items():
                if v.type != "motorcycle":
                    continue
                bx1, by1, bx2, by2 = v.bbox
                
                ix1 = max(px1, bx1)
                iy1 = max(py1, by1)
                ix2 = min(px2, bx2)
                iy2 = min(py2, by2)
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                overlap_area = iw * ih
                
                if overlap_area > best_overlap:
                    best_overlap = overlap_area
                    best_bike = v

            # If person is largely inside the bike region
            if best_bike and (best_overlap / p_area) > 0.3:
                best_bike.riders.append(p_box)

    # ────────────────────────────────────────────────────────────────────────
    #  3. VIOLATION DETECTION LOGIC
    # ────────────────────────────────────────────────────────────────────────

    def detect_helmet_violation(self, vehicle: VehicleData, helmets: List[Tuple[int, int, int, int]]):
        """Check top 30% of each rider for proper helmet overlap."""
        if vehicle.type != "motorcycle" or not vehicle.riders:
            return

        any_missing = False
        
        for rider in vehicle.riders:
            rx1, ry1, rx2, ry2 = rider
            h = max(1, ry2 - ry1)
            head_h = max(1, int(h * 0.30))
            head_region = (rx1, ry1, rx2, ry1 + head_h)
            
            hx1, hy1, hx2, hy2 = head_region
            head_area = float(max(1, hx2 - hx1) * max(1, hy2 - hy1))

            has_helmet = False
            for h_box in helmets:
                bx1, by1, bx2, by2 = h_box
                
                ix1 = max(hx1, bx1)
                iy1 = max(hy1, by1)
                ix2 = min(hx2, bx2)
                iy2 = min(hy2, by2)
                
                inter = float(max(0, ix2 - ix1) * max(0, iy2 - iy1))
                h_bx_area = float(max(1, bx2 - bx1) * max(1, by2 - by1))
                
                # Must overlap meaningfully
                if inter > 0 and ((inter / h_bx_area > 0.2) or (inter / head_area > 0.2)):
                    has_helmet = True
                    break
            
            if not has_helmet:
                any_missing = True
                
        if any_missing:
            vehicle.violations.add("helmet_missing")

    def detect_triple_riding(self, vehicle: VehicleData):
        """More than 2 riders on a motorcycle -> Triple Riding."""
        if vehicle.type == "motorcycle" and len(vehicle.riders) > 2:
            vehicle.violations.add("triple_riding")

    def detect_seatbelt_violation(self, vehicle: VehicleData):
        """Simplistic placeholder check for seatbelts in cars."""
        if vehicle.type != "car":
            return
            
        # Example constraints: front-view check relying on box aspect ratios.
        vx1, vy1, vx2, vy2 = vehicle.bbox
        vw, vh = vx2 - vx1, vy2 - vy1
        if vw == 0 or vh == 0 or (vw / vh) > 2.5: # Side view likely
            return
            
        # In a real pipeline, you crop the window region and run a secondary 
        # classification model specifically for seatbelts on the driver.
        # If seatbelt is missing, un-comment below:
        # vehicle.violations.add("seatbelt_missing")

    def detect_wrong_route(self, vehicle: VehicleData):
        """Detect backward movement against allowed flow."""
        trajectory = self.trajectory_history.get(vehicle.track_id, [])
        if len(trajectory) < 10:
            return

        first_pos = trajectory[0]
        last_pos = trajectory[-1]
        
        dx = last_pos[0] - first_pos[0]
        dy = last_pos[1] - first_pos[1]
        
        # Suppose allowed direction is downwards (increasing Y).
        # Negative Y means going upwards (Wrong way).
        # Require significant movement to prevent noise triggering.
        movement_dist = (dx**2 + dy**2)**0.5
        if movement_dist > 50 and dy < -30: 
            vehicle.violations.add("wrong_route")

    def detect_red_light(self, vehicle: VehicleData, signal_color: str, stop_line_y: int):
        """Detect crossing standard stop line when signal is red."""
        if signal_color != "RED":
            return
            
        vx1, vy1, vx2, vy2 = vehicle.bbox
        # Example check if the bottom base of the car has crossed stop line
        if vy2 > stop_line_y:
            vehicle.violations.add("red_light_jump")

    # ────────────────────────────────────────────────────────────────────────
    #  4. LOGGING & OUTPUT
    # ────────────────────────────────────────────────────────────────────────

    def should_log_violation(self, track_id: int, vtype: str) -> bool:
        """Deduplication: Only log once every 10 seconds per unique violation per car."""
        now = time.time()
        key = (track_id, vtype)
        if key in self.last_logged:
            diff = now - self.last_logged[key]
            if diff < self.log_cooldown:
                return False
        return True

    def log_violation(self, vehicle: VehicleData, vtype: str):
        now = time.time()
        self.last_logged[(vehicle.track_id, vtype)] = now
        print(f"[ALERT] Track ID {vehicle.track_id} logged for {vtype.upper()}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # 1. Detection
        detections = self.detect_objects(frame)
        
        # 2. Tracking
        vehicles = self.track_objects(detections)
        
        # 3. Association
        self.associate_riders(vehicles, detections["person"])
        
        # Mock signal state for Red Light
        signal_color = "GREEN" # Change for testing
        stop_line_y = 600

        # Run multi-violation pipeline on every vehicle
        for tid, v in vehicles.items():
            self.detect_helmet_violation(v, detections["helmet"])
            self.detect_triple_riding(v)
            self.detect_seatbelt_violation(v)
            self.detect_wrong_route(v)
            self.detect_red_light(v, signal_color, stop_line_y)

            # Log deduplicated violations
            for vtype in v.violations:
                if self.should_log_violation(tid, vtype):
                    self.log_violation(v, vtype)

        # 4. Rendering logic
        annotated_frame = frame.copy()
        
        cv2.line(annotated_frame, (0, stop_line_y), (annotated_frame.shape[1], stop_line_y), (0, 255, 255), 2)

        for tid, v in vehicles.items():
            color = (0, 0, 255) if v.violations else (0, 255, 0)
            
            x1, y1, x2, y2 = v.bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID {tid}"
            if v.violations:
                label += " | " + " + ".join([v.replace("_", " ").title() for v in v.violations])
                
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for rider in v.riders:
                rx1, ry1, rx2, ry2 = rider
                cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
                
        return annotated_frame


# ────────────────────────────────────────────────────────────────────────
#  5. EXECUTION BOILERPLATE
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    
    video_source = "test_video.mp4" # Replace with actual path or 0 for webcam
    
    if not os.path.exists(video_source):
        print("Using random noisy frames for demonstration. Replace with real video.")
        print("Note: Skipping cv2.imshow because opencv-python-headless may be installed.")
        monitor = TrafficMonitor()
        for i in range(30):
            # Simulated dummy layout
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            res = monitor.process_frame(frame)
            print(f"Processed dummy frame {i+1}/30")
            # cv2.imshow("Multi-Violation Pipeline", res)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        # cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(video_source)
        monitor = TrafficMonitor()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Resize for optimal detection
            frame = cv2.resize(frame, (1280, 720))
            out_frame = monitor.process_frame(frame)
            
            cv2.imshow("Multi-Violation Pipeline", out_frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
