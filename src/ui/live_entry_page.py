"""
Live Bus Entrance Verification Terminal Page
Smart Bus Face Recognition and Pass Verification System
"""

import time
import cv2
import numpy as np
import streamlit as st
from datetime import datetime, timezone

from src.core.bus_entry_service import BusEntryService, BusEntryResult
from src.core.pass_verifier import ReasonCode
from src.vision.liveness import LivenessState
from src.core.config import LIVENESS_ENABLED
from src.bus_entry_camera import draw_bounding_box, draw_hud_header
from src.database.dashboard_queries import get_distinct_buses, get_distinct_routes


@st.cache_resource
def get_cached_service():
    """Caches the heavy AI model service to prevent reload on UI rerun."""
    return BusEntryService()


def render_live_entry_page():
    st.title("📹 Live Bus Entrance Verification")
    st.caption("AI-powered boarding verification terminal running live at the bus entrance.")

    if not LIVENESS_ENABLED:
        st.warning("⚠️ **DEVELOPMENT MODE ACTIVE**: Liveness detection (anti-spoofing) is currently DISABLED in configuration.")

    # 1. Bus & Route Configuration Bar
    col1, col2, col3 = st.columns([2, 2, 2])
    distinct_buses = get_distinct_buses() or ["BUS-12", "BUS-08"]
    distinct_routes = get_distinct_routes() or ["R-101", "R-102"]

    with col1:
        current_bus = st.selectbox("Current Bus ID:", options=distinct_buses, index=0)
    with col2:
        current_route = st.selectbox("Current Route:", options=distinct_routes, index=0)
    with col3:
        threshold = st.slider("Recognition Threshold:", min_value=0.40, max_value=0.90, value=0.60, step=0.05)

    st.markdown("---")

    # 2. Controls & Layout
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        if st.button("🟢 Start Live Camera", disabled=st.session_state.camera_active, use_container_width=True):
            st.session_state.camera_active = True
            st.rerun()

    with ctrl_col2:
        if st.button("🔴 Stop Camera", disabled=not st.session_state.camera_active, use_container_width=True):
            st.session_state.camera_active = False
            st.rerun()

    st.markdown(" ")

    # Display split: Left = Live Video Feed, Right = Passenger Decision Card
    feed_col, card_col = st.columns([3, 2])

    frame_placeholder = feed_col.empty()
    status_placeholder = card_col.empty()

    if not st.session_state.camera_active:
        feed_col.info("Webcam is stopped. Click **Start Live Camera** above to begin scanning passengers.")
        status_placeholder.info("Awaiting passenger scan...")
        return

    # 3. Active Camera Stream Loop
    service = get_cached_service()
    service.recognition_threshold = threshold

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        feed_col.error("Could not access default webcam. Ensure your camera is plugged in and permissions are granted.")
        st.session_state.camera_active = False
        return

    frame_count = 0
    cached_results = []

    # Stream frames while active
    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret or frame is None:
            feed_col.warning("Failed to grab video frame.")
            break

        frame_count += 1

        # Process neural network every 2 frames for smooth browser rendering
        if frame_count % 2 == 0:
            now_utc = datetime.now(timezone.utc)
            cached_results = service.process_frame(
                frame=frame,
                bus_id=current_bus,
                route=current_route,
                current_timestamp=now_utc,
                log_to_db=True
            )

        # Draw bounding boxes and HUD
        for r in cached_results:
            draw_bounding_box(frame, r, debug=False)

        # Convert OpenCV BGR to RGB for Streamlit rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update Passenger Decision Card on right panel
        if cached_results:
            latest = cached_results[0]
            with status_placeholder.container():
                # Live Check Status Indicator (Part 12)
                if latest.liveness_result:
                    liv = latest.liveness_result
                    if liv.state == LivenessState.LIVE:
                        st.info("🛡️ **LIVE CHECK**: LIVENESS PASSED")
                    elif liv.state == LivenessState.SPOOF_SUSPECTED:
                        st.error("🛡️ **LIVE CHECK**: LIVENESS FAILED")
                    elif liv.state in (LivenessState.INSUFFICIENT_DATA, LivenessState.ERROR):
                        st.warning(f"🛡️ **LIVE CHECK**: LIVENESS INCONCLUSIVE\n\n👉 *{liv.reason}*")

                if latest.allowed:
                    st.success("### 🟢 ENTRY ALLOWED")
                    st.markdown(f"**Passenger:** `{latest.student_name}`")
                    st.markdown(f"**Roll Number:** `{latest.student_id}`")
                    st.markdown(f"**Match Confidence:** `{latest.similarity:.2f}`")
                    st.markdown(f"**Route Assigned:** `{latest.route}` on Bus `{latest.bus_id}`")
                    st.caption("Pass verified. Boarding logged to database.")
                elif latest.reason_code == ReasonCode.LIVENESS_INCONCLUSIVE:
                    st.warning("### ⏳ VERIFYING PASSENGER")
                    st.markdown(f"**Prompt:** {latest.reason_message}")
                    st.caption("Please follow the head turn prompt shown above.")
                elif latest.reason_code == ReasonCode.LIVENESS_FAILED:
                    st.error("### 🔴 ENTRY DENIED")
                    st.markdown("**Status:** Anti-spoofing verification failed.")
                    st.markdown("**Reason:** Presentation attack suspected (static photo or screen detected).")
                    st.caption("Entry refused. Audit event recorded.")
                elif latest.reason_code == ReasonCode.UNKNOWN_PERSON:
                    st.warning("### 🟡 UNKNOWN PERSON")
                    st.markdown("**Status:** Face not recognized in student database.")
                    st.markdown(f"**Highest Similarity:** `{latest.similarity:.2f}` (Below `{threshold:.2f}` threshold)")
                    st.caption("Please ask passenger for physical college ID or use Manual Fallback.")
                else:
                    st.error("### 🔴 ENTRY DENIED")
                    st.markdown(f"**Passenger:** `{latest.student_name or 'Unverified'}` ({latest.student_id or 'N/A'})")
                    st.markdown(f"**Reason:** `{latest.reason_message}`")
                    st.markdown(f"**Code:** `{latest.reason_code.value}`")
                    st.caption("Entry refused. Audit event recorded.")
        else:
            with status_placeholder.container():
                st.info("Scanning for passengers in camera field of view...")

        # Small sleep prevents overwhelming the browser event loop
        time.sleep(0.03)

    cap.release()
