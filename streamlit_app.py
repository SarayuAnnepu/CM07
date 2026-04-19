"""
streamlit_app.py — Professional Dashboard for the AI Traffic Violation Detection System.

Run with:
    streamlit run streamlit_app.py

Layout
------
├── Sidebar: Navigation (Live Feed / Analytics / Database), Connection Status, Upload
├── Header: College name, Project title, Team
├── Main: Live Feed + Status overlay  |  Alert sidebar (last 5 violations)
├── Evidence Gallery: Snapshot cards (vehicle crop + plate crop + OCR text + confidence)
└── Footer: FPS, GPU Usage, Total Violations
"""

from __future__ import annotations

import io
import os
import random
import sqlite3
import string
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Redirect temp files to D: drive (C: is full)
os.makedirs(r"D:\temp", exist_ok=True)
tempfile.tempdir = r"D:\temp"

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from traffic_manager import TrafficManager, FrameResult
from violations import Violation
from config import VEHICLE_CLASSES, CONFIDENCE_THRESHOLD, DEFAULT_MODEL, SQLITE_DB_PATH, FRAME_SKIP, PROCESS_WIDTH, PROCESS_HEIGHT

# ═══════════════════════════════════════════════════════════════════════
#  Page Config
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Traffic Violation Detection — BEC",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  Dark Theme CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Global ─────────────────────────────────────────────────── */
.stApp {
    background: #0a0e1a;
    font-family: 'Inter', sans-serif;
    color: #e0e0f0;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(99,110,150,0.15);
}

/* ── Header Banner ──────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(99,110,150,0.2);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 70%);
}
.college-name {
    font-size: 0.85rem;
    color: #7c8db5;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.project-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.team-names {
    font-size: 0.8rem;
    color: #6b7db3;
    letter-spacing: 1px;
}

/* ── Status Chips ───────────────────────────────────────────── */
.status-bar {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin: 0.8rem 0 0.3rem;
    flex-wrap: wrap;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.chip-green  { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.chip-red    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.chip-blue   { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.chip-yellow { background: rgba(234,179,8,0.15);  color: #facc15; border: 1px solid rgba(234,179,8,0.3); }

/* ── Metric Cards ───────────────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,110,150,0.12);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(99,110,150,0.3); }
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #7b68ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: #7c8db5;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 0.25rem;
}

/* ── Violation Log ──────────────────────────────────────────── */
.vlog-entry {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #f87171;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
}
.vlog-time { color: #6b7db3; font-family: monospace; }
.vlog-type { color: #f87171; font-weight: 600; }
.vlog-plate { color: #60a5fa; }

/* ── Evidence Cards ─────────────────────────────────────────── */
.evidence-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,110,150,0.12);
    border-radius: 14px;
    padding: 0.8rem;
    transition: border-color 0.2s;
}
.evidence-card:hover { border-color: rgba(99,110,150,0.3); }
.evidence-plate-text {
    font-size: 1.3rem;
    font-weight: 700;
    color: #facc15;
    font-family: monospace;
    text-align: center;
    margin-top: 0.4rem;
}
.evidence-conf {
    font-size: 0.75rem;
    color: #6b7db3;
    text-align: center;
}

/* ── Table ──────────────────────────────────────────────────── */
.v-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    font-size: 0.82rem;
}
.v-table th {
    background: rgba(59,130,246,0.12);
    color: #93a3c0;
    padding: 0.65rem 0.8rem;
    text-align: left;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
}
.v-table td {
    background: rgba(255,255,255,0.02);
    color: #c8d0e0;
    padding: 0.55rem 0.8rem;
    border-top: 1px solid rgba(99,110,150,0.08);
}
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.75rem;
}
.b-helmet   { background: rgba(239,68,68,0.2); color: #f87171; }
.b-seatbelt { background: rgba(236,72,153,0.2); color: #ec4899; } /* Pink */
.b-triple   { background: rgba(251,146,60,0.2); color: #fb923c; }
.b-redlight { background: rgba(168,85,247,0.2); color: #a855f7; }
.b-wrong    { background: rgba(34,211,238,0.2); color: #22d3ee; }

/* ── Footer ─────────────────────────────────────────────────── */
.footer-bar {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(99,110,150,0.1);
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1.5rem;
    font-size: 0.78rem;
    color: #6b7db3;
}
.footer-stat { display: flex; align-items: center; gap: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_manager(model_path: str = DEFAULT_MODEL) -> TrafficManager:
    return TrafficManager(model_path=model_path)


def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def load_db_rows(limit: int = 500) -> list[dict]:
    if not os.path.exists(SQLITE_DB_PATH):
        return []
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, timestamp, violation_type, license_plate, detection_confidence
            FROM violations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


BADGE = {
    "helmet_missing": ("🪖 Helmet Missing", "b-helmet"),
    "seatbelt_missing": ("💺 Seatbelt Missing", "b-seatbelt"),
    "triple_riding": ("👥 Triple Riding", "b-triple"),
    "red_light_jump": ("🔴 Red Light", "b-redlight"),
    "wrong_route": ("↩️ Wrong Route", "b-wrong"),
    "NO_HELMET":     ("🪖 Helmet Missing", "b-helmet"),
    "NO_SEATBELT":   ("💺 Seatbelt Missing", "b-seatbelt"),
    "TRIPLE_RIDING": ("👥 Triple Riding", "b-triple"),
    "RED_LIGHT":     ("🔴 Red Light",      "b-redlight"),
    "WRONG_ROUTE":   ("↩️ Wrong Route",    "b-wrong"),
}

VIOLATION_TYPES = ["NO_HELMET", "NO_SEATBELT", "TRIPLE_RIDING", "RED_LIGHT", "WRONG_ROUTE"]


def normalize_violation_type(vtype: str) -> str:
    mapping = {
        "helmet_missing": "NO_HELMET",
        "seatbelt_missing": "NO_SEATBELT",
        "triple_riding": "TRIPLE_RIDING",
        "red_light_jump": "RED_LIGHT",
        "wrong_route": "WRONG_ROUTE",
    }
    key = str(vtype or "").strip()
    if key in mapping:
        return mapping[key]
    upper = key.upper()
    return mapping.get(key.lower(), upper)


def _mock_plate() -> str:
    state = random.choice(["AP", "TS", "KA", "TN", "MH"])
    num1 = f"{random.randint(1,99):02d}"
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    num2 = f"{random.randint(1000,9999)}"
    return f"{state} {num1} {letters} {num2}"


def generate_mock_violation() -> dict:
    """Generate one random mock violation row."""
    vtype = random.choice(VIOLATION_TYPES)
    return {
        "id": random.randint(1000, 9999),
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": vtype,
        "plate": _mock_plate(),
        "confidence": round(random.uniform(0.65, 0.98), 2),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Session State
# ═══════════════════════════════════════════════════════════════════════
if "violation_log" not in st.session_state:
    st.session_state.violation_log = []
if "total_violations" not in st.session_state:
    st.session_state.total_violations = 0
if "last_mock_time" not in st.session_state:
    st.session_state.last_mock_time = time.time()
if "evidence_cards" not in st.session_state:
    st.session_state.evidence_cards = []
if "signal_state" not in st.session_state:
    st.session_state.signal_state = "Green"
if "processing_fps" not in st.session_state:
    st.session_state.processing_fps = None
if "video_recent_violation_keys" not in st.session_state:
    st.session_state.video_recent_violation_keys = {}


# ═══════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Select View",
        ["🎥 Live Feed", "📊 Analytics", "🗄️ Database"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Connection status
    st.markdown("### 📡 Connection Status")
    st.markdown(
        '<div class="status-chip chip-green">● CCTV Connected</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    model_path = st.text_input("YOLOv26 Model", value=DEFAULT_MODEL)
    conf_thresh = st.slider("Confidence", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)

    st.markdown("---")
    st.markdown("### 📤 Upload Evidence")
    upload_type = st.radio("Input type", ["🖼️ Image", "🎬 Video"], horizontal=True)
    uploaded_file = st.file_uploader(
        "Drop file here",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mkv", "mov"],
    )

    st.markdown("---")
    demo_mode = st.toggle("🎭 Demo Mode (Mock Data)", value=False)
    st.markdown(
        "<p style='color:#4a5578;font-size:0.7rem;text-align:center;'>v2.0 · YOLOv26 + ALPR</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Header Banner
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
    <div class="college-name">Bapatla Engineering College</div>
    <div class="project-title">🚦 AI-Powered Traffic Violation Detection System</div>
    <div class="team-names">Team: Abhiram &bull; Sarayu &bull; Maneesha &bull; Pawan</div>
    <div class="status-bar">
        <span class="status-chip chip-green">● System Online</span>
        <span class="status-chip chip-blue">◉ YOLOv26 Active</span>
        <span class="status-chip chip-yellow">⚡ EasyOCR Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  Mock Data Generator (when Demo Mode is ON)
# ═══════════════════════════════════════════════════════════════════════
if demo_mode:
    if time.time() - st.session_state.last_mock_time > 10:
        mock = generate_mock_violation()
        st.session_state.violation_log.insert(0, mock)
        st.session_state.violation_log = st.session_state.violation_log[:20]
        st.session_state.total_violations += 1
        st.session_state.last_mock_time = time.time()

    # Auto-refresh every 10 seconds
    st.markdown(
        '<meta http-equiv="refresh" content="10">',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: LIVE FEED
# ═══════════════════════════════════════════════════════════════════════
if page == "🎥 Live Feed":

    # ── Top Row: Live Feed + Alert Sidebar ──────────────────────────────
    feed_col, alert_col = st.columns([3, 1])

    with feed_col:
        st.markdown("#### 📡 Live Surveillance Feed")

        if uploaded_file is None:
            # Placeholder grey rectangle
            placeholder = np.full((480, 854, 3), 30, dtype=np.uint8)
            cv2.putText(placeholder, "NO VIDEO FEED", (250, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 70, 100), 2)
            cv2.putText(placeholder, "Upload an image/video or connect CCTV", (170, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 60, 90), 1)

            # Status overlay
            cv2.putText(placeholder, f"Signal: {st.session_state.signal_state}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 100), 2)
            cv2.putText(placeholder, f"Violations: {st.session_state.total_violations}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Stop line
            cv2.line(placeholder, (100, 380), (754, 380), (0, 200, 220), 2)
            cv2.putText(placeholder, "STOP LINE", (350, 375),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 220), 1)

            # Direction arrows
            for x in [250, 450, 650]:
                cv2.arrowedLine(placeholder, (x, 450), (x, 350), (100, 140, 255), 2, tipLength=0.3)

            st.image(cv2_to_pil(placeholder), use_container_width=True)

        elif upload_type == "🖼️ Image":
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                st.error("❌ Could not read the uploaded image.")
            else:
                original = frame.copy()
                manager = load_manager(model_path)

                with st.spinner("🔍 Processing…"):
                    t0 = time.time()
                    try:
                        result: FrameResult = manager.process_frame(
                            frame,
                            use_temporal_confirmation=False,
                            force_detect=True,
                            image_mode=True,
                        )
                    except TypeError:
                        # Compatibility fallback for cached manager instances
                        # created before the updated method signature.
                        # Clear cache and recreate manager to get updated signature.
                        st.cache_resource.clear()
                        manager = load_manager(model_path)
                        result = manager.process_frame(
                            frame,
                            use_temporal_confirmation=False,
                            force_detect=True,
                            image_mode=True,
                        )
                    elapsed = time.time() - t0

                annotated = result.annotated_frame
                detections = result.detections
                violations = result.violations
                plate_map = result.plate_map

                st.session_state.processing_fps = None  # don't display FPS for images
                st.session_state.signal_state = manager._signal_state.capitalize()
                st.session_state.total_violations += len(result.new_violations)

                # Add to violation log
                for v in result.new_violations:
                    plate = plate_map.get(v.vehicle_bbox, "")
                    vtype_norm = normalize_violation_type(v.vtype)
                    badge_text, _ = BADGE.get(vtype_norm, (vtype_norm, ""))
                    st.session_state.violation_log.insert(0, {
                        "id": random.randint(1000, 9999),
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": vtype_norm,
                        "plate": plate or "—",
                        "confidence": round(max(
                            (d.confidence for d in detections if d.bbox == v.vehicle_bbox),
                            default=0.0
                        ), 2),
                    })

                # Build evidence cards
                seen = set()
                for v in result.new_violations:
                    if v.vehicle_bbox in seen:
                        continue
                    seen.add(v.vehicle_bbox)
                    x1, y1, x2, y2 = v.vehicle_bbox
                    h, w = original.shape[:2]
                    crop = original[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size == 0:
                        continue
                    plate = plate_map.get(v.vehicle_bbox, "")
                    conf = max(
                        (d.confidence for d in detections if d.bbox == v.vehicle_bbox),
                        default=0.0,
                    )
                    st.session_state.evidence_cards.insert(0, {
                        "crop": crop.copy(),
                        "type": normalize_violation_type(v.vtype),
                        "desc": v.description,
                        "plate": plate,
                        "confidence": round(conf, 2),
                    })

                st.session_state.violation_log = st.session_state.violation_log[:20]
                st.session_state.evidence_cards = st.session_state.evidence_cards[:12]

                st.image(cv2_to_pil(annotated), use_container_width=True)

                # ── Metrics strip below feed ────────────────────────────
                m1, m2, m3, m4, m5 = st.columns(5)
                for col, val, label in [
                    (m1, len(detections), "Detections"),
                    (m2, sum(1 for d in detections if d.label in VEHICLE_CLASSES), "Vehicles"),
                    (m3, sum(1 for d in detections if d.label == "person"), "Persons"),
                    (m4, len(violations), "Violations"),
                    (m5, f"{elapsed:.2f}s", "Processing"),
                ]:
                    with col:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-value">{val}</div>'
                            f'<div class="metric-label">{label}</div></div>',
                            unsafe_allow_html=True,
                        )

        elif upload_type == "🎬 Video":
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            tfile_path = tfile.name
            tfile.close()

            cap = cv2.VideoCapture(tfile_path)
            if not cap.isOpened():
                st.error("❌ Could not open the uploaded video.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if not np.isfinite(frame_count_raw) or frame_count_raw <= 0:
                    total_frames = 300
                else:
                    total_frames = int(frame_count_raw)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                slider_max = max(10, min(total_frames, 300))
                slider_default = min(slider_max, 60)
                max_frames = st.slider("Max frames", 10, slider_max, slider_default, 10)

                if st.button("▶️ Start Processing", type="primary", use_container_width=True):
                    manager = load_manager(model_path)
                    progress = st.progress(0)
                    frame_ph = st.empty()
                    processed = 0
                    t0 = time.time()
                    video_frame_skip = 1
                    recent_video_keys = dict(st.session_state.video_recent_violation_keys)
                    repeat_cooldown_sec = 1.5

                    while True:
                        # ── Frame skipping: grab without decoding N-1 frames ────────
                        skip = False
                        for _ in range(video_frame_skip - 1):
                            if not cap.grab():
                                skip = True
                                break
                            
                        ret, frame = cap.read()
                        if not ret or skip or processed >= max_frames:
                            break
                        
                        original = frame.copy()
                        # ── Resize for faster processing ───────────────────────────
                        proc_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
                        # Video mode prioritizes violation recall over speed.
                        # Force detection on each processed frame and relax temporal
                        # confirmation so sparse sampling does not suppress events.
                        result = manager.process_frame(
                            proc_frame,
                            use_temporal_confirmation=False,
                            force_detect=True,
                        )
                        processed += 1

                        active_violations = result.violations if result.violations else result.new_violations
                        frame_seen = set()
                        now_ts = time.time()

                        for v in active_violations:
                            vtype_norm = normalize_violation_type(v.vtype)
                            frame_key = (v.vehicle_bbox, vtype_norm)
                            if frame_key in frame_seen:
                                continue
                            frame_seen.add(frame_key)

                            last_logged = recent_video_keys.get(frame_key, 0.0)
                            if now_ts - float(last_logged) < repeat_cooldown_sec:
                                continue

                            plate = result.plate_map.get(v.vehicle_bbox, "")
                            conf = max(
                                (
                                    float(d.confidence)
                                    for d in result.detections
                                    if d.bbox == v.vehicle_bbox
                                ),
                                default=0.85,
                            )
                            st.session_state.violation_log.insert(0, {
                                "id": random.randint(1000, 9999),
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": vtype_norm,
                                "plate": plate or "—",
                                "confidence": round(conf, 2),
                            })
                            st.session_state.total_violations += 1
                            recent_video_keys[frame_key] = now_ts

                        # Keep cooldown cache small and fresh.
                        recent_video_keys = {
                            k: ts for k, ts in recent_video_keys.items()
                            if now_ts - float(ts) <= 20.0
                        }

                        if processed % 3 == 0:
                            progress.progress(processed / max_frames, f"Frame {processed}/{max_frames}")
                            frame_ph.image(cv2_to_pil(result.annotated_frame), use_container_width=True)

                    cap.release()
                    elapsed = time.time() - t0
                    progress.progress(1.0, "✅ Done!")
                    st.session_state.processing_fps = processed / elapsed if elapsed > 0 else 0
                    st.session_state.video_recent_violation_keys = recent_video_keys

                    try:
                        os.unlink(tfile_path)
                    except OSError:
                        pass

    # ── Alert Sidebar ───────────────────────────────────────────────────
    with alert_col:
        st.markdown("#### 🚨 Recent Alerts")
        log = st.session_state.violation_log[:5]
        if log:
            for entry in log:
                badge_text, _ = BADGE.get(entry["type"], (entry["type"], ""))
                st.markdown(
                    f'<div class="vlog-entry">'
                    f'<span class="vlog-time">[{entry["time"]}]</span> '
                    f'<span class="vlog-type">{badge_text}</span><br>'
                    f'<span class="vlog-plate">🔢 {entry["plate"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="vlog-entry" style="border-left-color:#4a5578;">'
                '<span style="color:#4a5578;">No violations yet</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 🚥 Signal State")
        signal = st.session_state.signal_state
        chip = "chip-red" if signal.lower() == "red" else "chip-green"
        st.markdown(
            f'<div class="status-chip {chip}" style="font-size:1rem;">● {signal}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### 📈 Session Stats")
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{st.session_state.total_violations}</div>'
            f'<div class="metric-label">Total Violations</div></div>',
            unsafe_allow_html=True,
        )

    # ── Violation Log Table ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Violation Log")
    log = st.session_state.violation_log[:10]
    if log:
        rows = ""
        for entry in log:
            badge_text, badge_cls = BADGE.get(entry["type"], (entry["type"], "b-helmet"))
            rows += (
                f"<tr>"
                f"<td><code>{entry['id']}</code></td>"
                f"<td><code>{entry['time']}</code></td>"
                f'<td><span class="badge {badge_cls}">{badge_text}</span></td>'
                f"<td style='color:#facc15;font-family:monospace;font-weight:600;'>{entry['plate']}</td>"
                f"<td>{entry['confidence']}</td>"
                f"</tr>"
            )
        st.markdown(
            f'<table class="v-table">'
            f"<tr><th>ID</th><th>Time</th><th>Type</th><th>Plate Number</th><th>Confidence</th></tr>"
            f"{rows}</table>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No violations recorded yet. Upload an image/video or enable Demo Mode.")

    # ── Evidence Gallery ────────────────────────────────────────────────
    evidence = st.session_state.evidence_cards
    if evidence:
        st.markdown("---")
        st.markdown("### 🔎 Evidence Gallery")
        cols_per_row = 3
        for row_start in range(0, min(len(evidence), 6), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, card in enumerate(evidence[row_start:row_start + cols_per_row]):
                with cols[idx]:
                    st.markdown('<div class="evidence-card">', unsafe_allow_html=True)

                    # Vehicle crop
                    st.image(cv2_to_pil(card["crop"]), use_container_width=True, caption="Vehicle Snapshot")

                    # Violation type badge
                    badge_text, badge_cls = BADGE.get(card["type"], (card["type"], "b-helmet"))
                    st.markdown(
                        f'<span class="badge {badge_cls}">{badge_text}</span>',
                        unsafe_allow_html=True,
                    )

                    # Plate text
                    plate = card["plate"] or "—"
                    st.markdown(
                        f'<div class="evidence-plate-text">🔢 {plate}</div>',
                        unsafe_allow_html=True,
                    )

                    # Confidence
                    st.markdown(
                        f'<div class="evidence-conf">AI Confidence: {card["confidence"]:.0%}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("### 📊 Analytics Dashboard")

    log = st.session_state.violation_log
    db_rows = load_db_rows(limit=1000)
    if db_rows:
        log = [
            {
                "id": r.get("id", 0),
                "time": str(r.get("timestamp", "")),
                "type": normalize_violation_type(r.get("violation_type", "")),
                "plate": r.get("license_plate", "") or "—",
                "confidence": r.get("detection_confidence", 0.0),
            }
            for r in db_rows
        ]

    if not log:
        st.info("No data yet. Process some images/videos or enable Demo Mode to populate analytics.")
    else:
        # Violation type breakdown
        type_counts = {}
        for entry in log:
            t = entry["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        c1, c2, c3, c4 = st.columns(4)
        for col, vtype, emoji in [
            (c1, "NO_HELMET", "🪖"),
            (c2, "TRIPLE_RIDING", "👥"),
            (c3, "RED_LIGHT", "🔴"),
            (c4, "WRONG_ROUTE", "↩️"),
        ]:
            with col:
                count = type_counts.get(vtype, 0)
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{emoji} {count}</div>'
                    f'<div class="metric-label">{vtype.replace("_"," ")}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        import pandas as pd
        df = pd.DataFrame(log)
        st.markdown("#### 📄 Full Violation Log")
        st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: DATABASE
# ═══════════════════════════════════════════════════════════════════════
elif page == "🗄️ Database":
    st.markdown("### 🗄️ Violation Database")
    db_rows = load_db_rows(limit=5000)
    if not db_rows:
        st.info("No records in SQLite yet. Process some data first.")
    else:
        import pandas as pd
        df = pd.DataFrame(db_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button("⬇️ Export as CSV", csv, "violations.csv", "text/csv")

        if st.button("🗑️ Clear Database", type="secondary"):
            st.session_state.violation_log = []
            st.session_state.evidence_cards = []
            st.session_state.total_violations = 0
            if os.path.exists(SQLITE_DB_PATH):
                conn = sqlite3.connect(SQLITE_DB_PATH)
                try:
                    conn.execute("DELETE FROM violations")
                    conn.commit()
                finally:
                    conn.close()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════════════
fps_display = f"{st.session_state.processing_fps:.1f}" if st.session_state.processing_fps is not None else "—"
st.markdown(
    f'<div class="footer-bar">'
    f'<div class="footer-stat">⚡ FPS: <strong>{fps_display}</strong></div>'
    f'<div class="footer-stat">🎯 GPU: <strong>CUDA</strong></div>'
    f'<div class="footer-stat">📊 Total Violations Today: <strong>{st.session_state.total_violations}</strong></div>'
    f'<div class="footer-stat">🕐 {datetime.now().strftime("%H:%M:%S")}</div>'
    f'</div>',
    unsafe_allow_html=True,
)
