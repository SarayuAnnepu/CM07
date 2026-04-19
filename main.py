"""
main.py — Entry point for the AI Traffic Violation Detection System.

Interactively asks the user for a video or image file path, processes it
through the TrafficManager pipeline, and saves annotated output.

Usage
-----
    python main.py                          # interactive prompt
    python main.py --source video.mp4       # skip prompt, use file directly
    python main.py --model custom.pt        # custom YOLO weights
    python main.py --output out.avi         # custom output path
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from traffic_manager import TrafficManager
from config import FRAME_SKIP, PROCESS_WIDTH, PROCESS_HEIGHT, DETECT_EVERY_N

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Traffic Violation Detection System (YOLO + EasyOCR)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video / image file path, webcam index (0), or stream URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO weights (default: model path from config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the annotated output (video or image).",
    )
    return parser.parse_args()


def ask_source() -> str:
    """Interactively ask the user for an input source."""
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   AI Traffic Violation Detection System              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Enter an input source:                              ║")
    print("║    • Path to a video file  (e.g. traffic.mp4)        ║")
    print("║    • Path to an image file (e.g. scene.jpg)          ║")
    print("║    • Webcam index          (e.g. 0)                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    source = input("\n>> Source path: ").strip().strip('"').strip("'")
    return source


def _has_gui() -> bool:
    """Check whether cv2.imshow is likely to work."""
    try:
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        return True
    except cv2.error:
        return False


# ═══════════════════════════════════════════════════════════════════════
#  Image processing
# ═══════════════════════════════════════════════════════════════════════

def process_image(
    source: str,
    manager: TrafficManager,
    output_path: str | None,
    gui: bool,
) -> None:
    """Load a single image, process it, and save / display the result."""
    frame = cv2.imread(source)
    if frame is None:
        print(f"[ERROR] Cannot read image: {source}")
        return

    h, w = frame.shape[:2]
    print(f"[main] Image loaded — {w}x{h}")

    result = manager.process_frame(
        frame,
        use_temporal_confirmation=False,
        force_detect=True,
        image_mode=True,
    )
    annotated = result.annotated_frame

    # Determine output path
    if not output_path:
        base, ext = os.path.splitext(source)
        output_path = f"{base}_output{ext}"

    cv2.imwrite(output_path, annotated)
    print(f"[main] ✅ Annotated image saved to: {output_path}")

    if gui:
        cv2.imshow("Traffic Violation Detection", annotated)
        print("[main] Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════
#  Video / webcam processing
# ═══════════════════════════════════════════════════════════════════════

def process_video(
    source,
    manager: TrafficManager,
    output_path: str | None,
    gui: bool,
) -> None:
    """Read frames from a video / webcam, process, and save / display."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[main] Video opened — {width}x{height} @ {fps:.1f} FPS")

    # Auto-generate output path for video files
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    elif isinstance(source, str):
        base, _ = os.path.splitext(source)
        auto_out = f"{base}_output.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(auto_out, fourcc, fps, (width, height))

    print(f"[main] Processing… skip={FRAME_SKIP}, N={DETECT_EVERY_N}, resize={PROCESS_WIDTH}x{PROCESS_HEIGHT}")
    print("[main] Press 'q' to quit (if display is on).")
    frame_count = 0
    raw_count = 0
    t_start = time.time()

    try:
        while True:
            # ── Frame skipping: grab without decoding N-1 frames ────────
            skip = False
            for _ in range(FRAME_SKIP - 1):
                if not cap.grab():
                    skip = True
                    break
                raw_count += 1

            ret, frame = cap.read()
            raw_count += 1
            if not ret or skip:
                print("[main] End of stream / video.")
                break

            # ── Resize for faster processing ───────────────────────────
            proc_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

            result = manager.process_frame(proc_frame)
            annotated = result.annotated_frame
            frame_count += 1

            # Scale annotated frame back to original resolution for output
            if writer:
                out_frame = cv2.resize(annotated, (width, height))
                writer.write(out_frame)

            if gui:
                cv2.imshow("Traffic Violation Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[main] 'q' pressed — exiting.")
                    break

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")

    finally:
        cap.release()
        if writer:
            writer.release()
        if gui:
            cv2.destroyAllWindows()
        elapsed = time.time() - t_start
        print(f"[main] ✅ Processed {frame_count} frames ({raw_count} total read) in {elapsed:.1f}s.")
        if elapsed > 0:
            print(f"[main]    Effective FPS: {frame_count / elapsed:.1f}  |  Raw FPS: {raw_count / elapsed:.1f}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Get source from CLI or interactive prompt ───────────────────────
    source_str = args.source if args.source else ask_source()
    if not source_str:
        print("[ERROR] No source provided.")
        sys.exit(1)

    # ── Initialise the traffic manager ──────────────────────────────────
    manager = TrafficManager(model_path=args.model)
    gui = _has_gui()

    if not gui:
        print("[main] GUI not available — output will be saved to file only.")

    # ── Determine source type ───────────────────────────────────────────
    ext = os.path.splitext(source_str)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        if not os.path.isfile(source_str):
            print(f"[ERROR] File not found: {source_str}")
            sys.exit(1)
        process_image(source_str, manager, args.output, gui)

    elif ext in VIDEO_EXTENSIONS or source_str.isdigit() or source_str.startswith(("rtsp://", "http://", "https://")):
        video_source = int(source_str) if source_str.isdigit() else source_str
        if isinstance(video_source, str) and not source_str.startswith(("rtsp://", "http://", "https://")) and not os.path.isfile(video_source):
            print(f"[ERROR] File not found: {source_str}")
            sys.exit(1)
        process_video(video_source, manager, args.output, gui)

    else:
        print(f"[ERROR] Unsupported file type: '{ext}'")
        print(f"  Supported images : {', '.join(sorted(IMAGE_EXTENSIONS))}")
        print(f"  Supported videos : {', '.join(sorted(VIDEO_EXTENSIONS))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
