"""
Dashboard Home Page
Smart Bus Face Recognition and Pass Verification System
"""

import streamlit as st
import pandas as pd
from src.database.dashboard_queries import get_dashboard_summary, get_recent_activity


def render_dashboard_page():
    st.title("📊 Smart Bus Operations Dashboard")
    st.caption("Real-time summary of student enrollments, bus passes, and entrance activity.")

    # 1. Real-time Metric Cards
    try:
        stats = get_dashboard_summary()
    except Exception as e:
        st.error(f"Could not load dashboard statistics from database: {e}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Registered Students", value=stats["total_students"])
        st.metric(label="Today's Boardings (Allowed)", value=stats["today_allowed"], delta="Approved")

    with col2:
        st.metric(label="Active Bus Passes", value=stats["active_passes"])
        st.metric(label="Today's Denied Attempts", value=stats["today_denied"], delta="-Rejections", delta_color="inverse")

    with col3:
        st.metric(label="Expired Passes", value=stats["expired_passes"], delta="-Expired", delta_color="inverse")
        st.metric(label="Today's Unknown Faces", value=stats["today_unknown"], delta="-Unregistered", delta_color="inverse")

    st.markdown("---")

    # 2. Recent Boarding Activity Feed
    st.subheader("🕒 Recent Entrance Activity Feed")
    try:
        activity = get_recent_activity(limit=10)
        if activity:
            df = pd.DataFrame(activity)
            # Reorder & rename for display
            df_display = df[["timestamp", "student_name", "student_id", "bus_id", "route", "result", "reason", "similarity"]].copy()
            df_display.columns = ["Time (UTC)", "Passenger Name", "Student ID", "Bus", "Route", "Result", "Reason", "Similarity"]

            def highlight_result(val):
                if val == "APPROVED":
                    return "background-color: #d4edda; color: #155724; font-weight: bold;"
                elif val == "REJECTED":
                    return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
                return ""

            st.dataframe(
                df_display.style.map(highlight_result, subset=["Result"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No boarding logs recorded yet today. Boardings will appear here in real-time.")
    except Exception as e:
        st.error(f"Error fetching recent activity: {e}")
