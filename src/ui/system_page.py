"""
System Health & Diagnostics Page
Smart Bus Face Recognition and Pass Verification System
"""

import sys
import cv2
import torch
import sqlite3
import streamlit as st
from pathlib import Path
from typing import Tuple

from src.database.db_connection import DB_PATH
from src.vision.embedding_store import EmbeddingStore
from src.database.dashboard_queries import get_dashboard_summary


def test_camera_hardware(index: int = 0) -> bool:
    """Quick hardware ping to check if webcam is accessible."""
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return bool(ret)
        return False
    except Exception:
        return False


def test_sqlite_db() -> Tuple[bool, str]:
    """Tests SQLite connection and verifies core tables."""
    if not DB_PATH.exists():
        return False, "Database file missing"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
        tables_count = cursor.fetchone()[0]
        conn.close()
        return True, f"Online ({tables_count} tables initialized)"
    except Exception as e:
        return False, f"Error: {e}"


def render_system_page():
    st.title("⚙️ System Health & Diagnostics")
    st.caption("Hardware, deep-learning runtimes, and local database operational status.")

    # 1. Environment Diagnostics
    st.subheader("1. Core Runtime & Libraries")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Python Runtime:** `{sys.version.split()[0]}`")
        st.write(f"**Platform:** `{sys.platform.capitalize()}`")

    with col2:
        st.write(f"**PyTorch Version:** `{torch.__version__}`")
        cuda_ok = torch.cuda.is_available()
        st.write(f"**Hardware Acceleration (CUDA):** {'🟢 Active' if cuda_ok else '⚪ CPU Mode'}")

    with col3:
        st.write(f"**OpenCV Version:** `{cv2.__version__}`")
        st.write(f"**Streamlit Version:** `{st.__version__}`")

    st.markdown("---")

    # 2. Database & Biometrics Status
    st.subheader("2. Storage & Biometric Integrity")

    db_ok, db_msg = test_sqlite_db()
    store = EmbeddingStore()
    store_ok = store.storage_path.exists()
    student_stats = get_dashboard_summary()

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric(
            label="SQLite Database",
            value="PASS" if db_ok else "FAIL",
            delta=db_msg if db_ok else "Disconnected",
            delta_color="normal" if db_ok else "inverse"
        )
    with sc2:
        st.metric(
            label="Embedding Store",
            value="PASS" if store_ok else "UNINITIALIZED",
            delta=f"{len(store)} Enrolled Biometric(s)"
        )
    with sc3:
        st.metric(
            label="Enrolled Students",
            value=student_stats["total_students"],
            delta=f"{student_stats['active_passes']} Active Passes"
        )

    st.markdown("---")

    # 3. Hardware Test Action
    st.subheader("3. Hardware Peripheral Diagnostics")
    st.write("Click below to ping the default camera hardware:")

    if st.button("🔌 Test Webcam Access"):
        with st.spinner("Testing camera index 0..."):
            cam_ok = test_camera_hardware(0)
            if cam_ok:
                st.success("🟢 **Camera Test PASSED**: Webcam detected and frame read successfully.")
            else:
                st.warning("⚠️ **Camera Test WARNING**: Could not access webcam at index 0. Check Windows camera privacy settings or ensure no other app is currently using the camera.")
