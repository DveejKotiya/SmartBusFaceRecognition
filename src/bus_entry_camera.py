"""
Real-Time Bus Entry Camera Application
Smart Bus Face Recognition and Pass Verification System

This application runs at the bus door:
1. Captures live webcam feed.
2. Detects human faces with MTCNN.
3. Identifies students using FaceNet embeddings.
4. Verifies bus pass validity, route matching, and 5-minute cooldown with PassVerifier.
5. Renders clear real-time visual feedback:
   - 🟢 GREEN  : ENTRY ALLOWED (Student Name, Roll Number, Similarity)
   - 🔴 RED    : ENTRY DENIED (Exact rejection reason)
   - 🟡 YELLOW : UNKNOWN PERSON
6. Logs boarding attempts to the SQLite database.
7. Press 'q' to safely exit.
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.bus_entry_service import BusEntryService, BusEntryResult
from src.core.pass_verifier import ReasonCode, DEFAULT_RECOGNITION_THRESHOLD

# =====================================================================
# CONFIGURATION SETTINGS (EASY TO CHANGE)
# =====================================================================
# Default bus and route for this terminal
CURRENT_BUS_ID = "BUS-12"
CURRENT_ROUTE = "R-101"

# Performance: Process full AI recognition every N frames (1 = every frame)
# Running every 2-3 frames keeps the video smooth on standard laptop CPUs
RECOGNITION_INTERVAL = 2

# Face recognition similarity acceptance threshold
RECOGNITION_THRESHOLD = DEFAULT_RECOGNITION_THRESHOLD  # 0.60

# How long to display status banners after face leaves frame (seconds)
DISPLAY_RESULT_SECONDS = 2.5

# Debug mode: displays FPS, inference times, and raw reason codes
DEBUG_MODE = False


# =====================================================================
# COLOR PALETTE (BGR FORMAT FOR OPENCV)
# =====================================================================
COLOR_ALLOWED = (46, 204, 113)    # Bright Green
COLOR_DENIED = (50, 50, 220)      # Crimson Red
COLOR_UNKNOWN = (0, 165, 255)     # Amber / Orange
COLOR_BANNER_BG = (25, 25, 25)    # Dark Charcoal
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def draw_bounding_box(
    frame: np.ndarray,
    res: BusEntryResult,
    debug: bool = False
) -> None:
    """
    Renders bounding box and student identity card around a detected face.
    """
    x1, y1, x2, y2 = res.bounding_box
    h, w = frame.shape[:2]

    # Select color scheme based on entry decision and liveness
    if res.allowed:
        box_color = COLOR_ALLOWED
        status_text = "ENTRY ALLOWED"
    elif res.reason_code == ReasonCode.LIVENESS_INCONCLUSIVE:
        box_color = (0, 215, 255)  # Gold/Yellow
        status_text = "LIVE CHECK: ACTION REQUIRED"
    elif res.reason_code == ReasonCode.LIVENESS_FAILED:
        box_color = COLOR_DENIED
        status_text = "LIVENESS FAILED (SPOOF)"
    elif res.reason_code == ReasonCode.UNKNOWN_PERSON:
        box_color = COLOR_UNKNOWN
        status_text = "UNKNOWN PERSON"
    else:
        box_color = COLOR_DENIED
        status_text = "ENTRY DENIED"

    # 1. Draw Face Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # 2. Draw Top Tag (Student Name or UNKNOWN)
    if res.student_name:
        tag_text = f"{res.student_name} ({res.student_id})"
    else:
        tag_text = "UNKNOWN PERSON"

    tag_y1 = max(0, y1 - 26)
    tag_w = min(w - x1, max(180, len(tag_text) * 9 + 10))
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tag_w, y1), box_color, cv2.FILLED)
    cv2.putText(
        frame,
        tag_text,
        (x1 + 6, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_BLACK,
        1,
        cv2.LINE_AA
    )

    # 3. Draw Bottom Status Banner
    bot_y1 = min(h - 35, y2)
    bot_y2 = min(h, bot_y1 + 32)
    cv2.rectangle(frame, (x1, bot_y1), (x2, bot_y2), box_color, cv2.FILLED)
    cv2.putText(
        frame,
        status_text,
        (x1 + 6, bot_y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_BLACK,
        2,
        cv2.LINE_AA
    )

    # 4. If Denied, show reason beneath status
    if not res.allowed and res.reason_code != ReasonCode.UNKNOWN_PERSON:
        reason_msg = res.reason_message
        # Trim message if too long for screen
        if len(reason_msg) > 42:
            reason_msg = reason_msg[:39] + "..."

        sub_y1 = min(h - 15, bot_y2 + 2)
        cv2.putText(
            frame,
            f"Reason: {reason_msg}",
            (x1, sub_y1 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA
        )

    # 5. Debug Telemetry Overlay
    if debug:
        debug_str = f"Sim: {res.similarity:.2f} | Conf: {res.detection_confidence:.2f} | {res.reason_code.value}"
        cv2.putText(
            frame,
            debug_str,
            (x1, max(15, tag_y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            COLOR_WHITE,
            1,
            cv2.LINE_AA
        )


def draw_hud_header(
    frame: np.ndarray,
    bus_id: str,
    route: str,
    fps: float,
    debug: bool
) -> None:
    """Renders the top application header bar."""
    w = frame.shape[1]
    # Header bar
    cv2.rectangle(frame, (0, 0), (w, 42), COLOR_BANNER_BG, cv2.FILLED)
    cv2.line(frame, (0, 42), (w, 42), (80, 80, 80), 1)

    title = f"SMART BUS TERMINAL | BUS: {bus_id} | ROUTE: {route}"
    cv2.putText(
        frame,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_WHITE,
        2,
        cv2.LINE_AA
    )

    # Exit instruction
    exit_hint = "Press 'q' to Quit"
    if debug:
        exit_hint += f" | {fps:.1f} FPS (DEBUG)"
    cv2.putText(
        frame,
        exit_hint,
        (w - 220, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )


def run_camera(
    bus_id: str = CURRENT_BUS_ID,
    route: str = CURRENT_ROUTE,
    camera_index: int = 0,
    recognition_interval: int = RECOGNITION_INTERVAL,
    threshold: float = RECOGNITION_THRESHOLD,
    debug: bool = DEBUG_MODE,
    max_frames: int = 0,
    headless: bool = False
) -> None:
    """
    Main real-time camera loop.
    """
    print("\n" + "=" * 68)
    print("  STAGE 7: REAL-TIME BUS ENTRY RECOGNITION CAMERA")
    print(f"  Operating on BUS: {bus_id} | ROUTE: {route}")
    print("=" * 68)

    print("\n[1/3] Initializing BusEntryService...")
    service = BusEntryService(recognition_threshold=threshold)
    print(f"      Recognition threshold: {threshold:.2f}")
    print(f"      Enrolled students in store: {len(service.store)}")

    print(f"\n[2/3] Opening Webcam (Camera Index {camera_index})...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        print("        Check if webcam is in use by another program.")
        return

    print("\n[3/3] Live camera feed running!")
    if not headless:
        print("      Focus the window and press 'q' to exit.")
    print("-" * 68)

    frame_counter = 0
    fps = 0.0
    t_start = time.time()
    t_last_fps = time.time()
    fps_frame_count = 0

    cached_results: List[BusEntryResult] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Could not read frame from camera.")
                break

            frame_counter += 1
            fps_frame_count += 1

            # Update FPS every 0.5s
            t_now = time.time()
            if t_now - t_last_fps >= 0.5:
                fps = fps_frame_count / (t_now - t_last_fps)
                fps_frame_count = 0
                t_last_fps = t_now

            # Run full neural recognition every N frames
            is_recognition_frame = (frame_counter % max(1, recognition_interval) == 0)

            if is_recognition_frame:
                t_proc_start = time.time()
                current_utc = datetime.now(timezone.utc)
                # Process all faces in frame
                cached_results = service.process_frame(
                    frame=frame,
                    bus_id=bus_id,
                    route=route,
                    current_timestamp=current_utc,
                    log_to_db=True
                )
                proc_time_ms = (time.time() - t_proc_start) * 1000

                # Print terminal summary for every detected face
                for idx, r in enumerate(cached_results, start=1):
                    status_lbl = "ALLOWED" if r.allowed else f"DENIED ({r.reason_code.value})"
                    print(
                        f"[Frame {frame_counter:04d}] Face {idx}: {r.student_name or 'UNKNOWN'} "
                        f"({r.student_id or 'N/A'}) | Sim: {r.similarity:.2f} | {status_lbl}"
                    )

            # Draw top HUD bar
            draw_hud_header(frame, bus_id, route, fps, debug)

            # Draw visual feedback for all cached faces
            for r in cached_results:
                draw_bounding_box(frame, r, debug=debug)

            if not headless:
                cv2.imshow("Smart Bus Face Recognition Entry Terminal", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nUser pressed 'q'. Exiting...")
                    break

            if max_frames > 0 and frame_counter >= max_frames:
                print(f"\nReached target limit of {max_frames} frames. Exiting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        elapsed = time.time() - t_start
        avg_fps = frame_counter / elapsed if elapsed > 0 else 0
        print("-" * 68)
        print(f"Session complete: Processed {frame_counter} frames in {elapsed:.2f}s ({avg_fps:.1f} FPS).")
        print("Camera hardware released cleanly.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Bus Real-Time Entry Terminal")
    parser.add_argument("--bus", type=str, default=CURRENT_BUS_ID, help="Bus Number (e.g. BUS-12)")
    parser.add_argument("--route", type=str, default=CURRENT_ROUTE, help="Route Code (e.g. R-101)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--interval", type=int, default=RECOGNITION_INTERVAL, help="Process AI every N frames")
    parser.add_argument("--threshold", type=float, default=RECOGNITION_THRESHOLD, help="Cosine threshold (0.60)")
    parser.add_argument("--debug", action="store_true", help="Enable debug overlay")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to process (0 = infinite)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_camera(
        bus_id=args.bus,
        route=args.route,
        camera_index=args.camera,
        recognition_interval=args.interval,
        threshold=args.threshold,
        debug=args.debug or DEBUG_MODE,
        max_frames=args.frames,
        headless=args.headless
    )
