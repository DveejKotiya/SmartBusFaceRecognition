"""
test_env.py - Environment & Dependency Diagnostic Script
Smart Bus Face Recognition and Pass Verification System

This script verifies that all required libraries (PyTorch, facenet-pytorch,
OpenCV, SQLite, Streamlit, etc.) are installed and can run on your system.
"""

import sys

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_python():
    print_header("1. Checking Python Version")
    print(f"Python Version: {sys.version}")
    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor >= 9:
        print("  [SUCCESS] Python version is compatible (>= 3.9).")
        return True
    else:
        print("  [WARNING] Python 3.9+ is recommended.")
        return False

def test_pytorch():
    print_header("2. Checking PyTorch & Hardware Acceleration")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA (GPU) Available: {cuda_available}")
        if cuda_available:
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("  [INFO] Running on CPU (sufficient for facenet-pytorch inference).")
        
        # Test tensor calculation
        x = torch.rand(2, 3)
        y = x * 2
        print("  [SUCCESS] PyTorch tensor operations work correctly.")
        return True
    except Exception as e:
        print(f"  [ERROR] PyTorch failed: {e}")
        return False

def test_facenet():
    print_header("3. Checking facenet-pytorch (MTCNN & InceptionResnetV1)")
    try:
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        from PIL import Image
        import numpy as np

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for models: {device}")

        print("Initializing MTCNN Face Detector...")
        mtcnn = MTCNN(image_size=160, margin=0, device=device)
        print("  [SUCCESS] MTCNN initialized.")

        print("Initializing InceptionResnetV1 (Pretrained VGGFace2)...")
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print("  [SUCCESS] InceptionResnetV1 loaded weights successfully.")

        # Test dummy embedding inference
        dummy_tensor = torch.randn(1, 3, 160, 160).to(device)
        with torch.no_grad():
            embedding = resnet(dummy_tensor)
        
        print(f"  [SUCCESS] Model inference passed. Generated embedding shape: {embedding.shape}")
        if embedding.shape == (1, 512):
            print("  [SUCCESS] Embedding shape is exactly (1, 512).")
        return True
    except Exception as e:
        print(f"  [ERROR] facenet-pytorch test failed: {e}")
        return False

def test_opencv():
    print_header("4. Checking OpenCV & Webcam")
    try:
        import cv2
        print(f"OpenCV Version: {cv2.__version__}")

        # Try opening default camera (index 0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print(f"  [SUCCESS] Webcam detected and captured test frame. Frame size: {frame.shape}")
            else:
                print("  [NOTICE] Webcam opened but failed to capture a frame (might be in use by another app).")
        else:
            print("  [NOTICE] No webcam detected at index 0 (or camera access is blocked in Windows Privacy Settings).")
            print("           The system can still run using mock images or file uploads.")
        return True
    except Exception as e:
        print(f"  [ERROR] OpenCV test failed: {e}")
        return False

def test_sqlite():
    print_header("5. Checking SQLite3 Database Engine")
    try:
        import sqlite3
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, msg TEXT);")
        cursor.execute("INSERT INTO test (msg) VALUES ('SmartBus');")
        conn.commit()
        cursor.execute("SELECT msg FROM test WHERE id = 1;")
        res = cursor.fetchone()
        conn.close()
        if res and res[0] == 'SmartBus':
            print(f"SQLite Version: {sqlite3.sqlite_version}")
            print("  [SUCCESS] In-memory SQLite operations passed.")
            return True
        else:
            print("  [ERROR] SQLite query returned unexpected result.")
            return False
    except Exception as e:
        print(f"  [ERROR] SQLite failed: {e}")
        return False

def test_analytics_and_ui():
    print_header("6. Checking Data Science & UI Libraries")
    try:
        import numpy as np
        print(f"NumPy Version: {np.__version__}")

        import pandas as pd
        print(f"Pandas Version: {pd.__version__}")

        import sklearn
        print(f"Scikit-Learn Version: {sklearn.__version__}")

        import streamlit as st
        print(f"Streamlit Version: {st.__version__}")

        print("  [SUCCESS] All data science and UI libraries imported successfully.")
        return True
    except Exception as e:
        print(f"  [ERROR] Library check failed: {e}")
        return False

def main():
    print("\n" + "#" * 60)
    print("  SMART BUS FACE RECOGNITION - ENVIRONMENT CHECK")
    print("#" * 60)

    results = {
        "Python": test_python(),
        "PyTorch": test_pytorch(),
        "facenet-pytorch": test_facenet(),
        "OpenCV": test_opencv(),
        "SQLite": test_sqlite(),
        "UI & Analytics": test_analytics_and_ui()
    }

    print_header("SUMMARY")
    all_passed = True
    for component, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  - {component:20s}: [{status}]")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL CHECKS PASSED! Ready for Stage 2.")
    else:
        print("  SOME CHECKS FAILED. Please review the errors above.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
