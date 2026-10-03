"""
Vision Pipeline Demo Script
Smart Bus Face Recognition and Pass Verification System

This demo verifies the Stage 3 vision pipeline:
1. Captures live frames from the default webcam.
2. Detects human faces using MTCNN (FaceDetector).
3. Draws bounding boxes on each detected face.
4. Generates a 512-dimensional embedding using FaceNet (FaceRecognizer).
5. Displays face number, detection confidence, and embedding dimensionality.
6. Press 'q' in the video window to quit.

Note: In accordance with Stage 3 guidelines, this demo does NOT identify students yet.
"""

import sys
from pathlib import Path

# Ensure project root is in Python sys.path so 'src' can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import argparse
import cv2
import numpy as np

from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer


def run_demo(max_frames: int = 0, camera_index: int = 0, headless: bool = False) -> None:
    """
    Runs the vision demo on webcam or synthetic frames.

    Args:
        max_frames: Maximum number of frames to process (0 = infinite until 'q').
        camera_index: Index of camera device (default 0).
        headless: If True, skips cv2.imshow (useful for automated testing/environments).
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: VISION PIPELINE LIVE DEMO")
    print("  MTCNN Face Detection + FaceNet 512-D Embedding Extraction")
    print("=" * 60)

    print("\n[1/3] Initializing FaceDetector (MTCNN)...")
    detector = FaceDetector(confidence_threshold=0.85)
    print(f"      Running on device: {detector.device}")

    print("\n[2/3] Initializing FaceRecognizer (InceptionResnetV1)...")
    recognizer = FaceRecognizer()
    print(f"      Running on device: {recognizer.device}")

    print(f"\n[3/3] Opening Webcam (Camera Index: {camera_index})...")
    # Use DirectShow backend on Windows for fast webcam initialization
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        print("        Check if another application is using your webcam,")
        print("        or verify Windows camera privacy permissions.")
        return

    print("\n>>> Webcam is active!")
    if not headless:
        print(">>> Focus the OpenCV window and press 'q' to quit.")
    print("-" * 60)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame from webcam. Exiting...")
                break

            frame_count += 1

            # Detect all faces in the frame
            detections = detector.detect_faces(frame, is_bgr=True)

            # Process each detected face
            for idx, det in enumerate(detections, start=1):
                x1, y1, x2, y2 = det.box
                confidence = det.confidence

                # Generate 512-D embedding from the cropped 160x160 face
                embedding = recognizer.generate_embedding(det.face_crop)
                emb_dim = len(embedding)

                # Terminal log
                print(f"[Frame {frame_count:04d}] Face {idx} | Conf: {confidence:.2f} | Embedding: {emb_dim}-D | Finite: {np.all(np.isfinite(embedding))}")

                # Draw green bounding box around detected face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Format overlay text
                text_top = f"Face {idx}: Conf {confidence:.2f}"
                text_bot = f"Embedding: {emb_dim}-D"

                # Draw label background for better readability
                label_y1 = max(10, y1 - 28)
                cv2.rectangle(frame, (x1, label_y1), (x1 + 180, y1), (0, 255, 0), cv2.FILLED)
                cv2.putText(
                    frame,
                    text_top,
                    (x1 + 4, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

                # Display embedding text beneath box
                cv2.putText(
                    frame,
                    text_bot,
                    (x1, min(frame.shape[0] - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

            # Display instruction banner on frame
            cv2.putText(
                frame,
                "Smart Bus AI: FaceNet 512-D | Press 'q' to Exit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if not headless:
                cv2.imshow("Smart Bus Face Recognition - Stage 3 Demo", frame)
                # Press 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nUser pressed 'q'. Exiting demo...")
                    break

            if max_frames > 0 and frame_count >= max_frames:
                print(f"\nReached target limit of {max_frames} frames. Exiting demo...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print("-" * 60)
        print(f"Demo complete. Processed {frame_count} frames in {elapsed:.2f}s ({fps:.1f} FPS).")
        print("Camera released and windows destroyed successfully.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Bus Face Recognition Stage 3 Demo")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to process (0 = infinite)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(max_frames=args.frames, camera_index=args.camera, headless=args.headless)
