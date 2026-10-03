"""
Entry Logs Audit Page
Smart Bus Face Recognition and Pass Verification System
"""

import streamlit as st
import pandas as pd
from datetime import date
from src.database.dashboard_queries import (
    get_filtered_entry_logs,
    get_distinct_buses,
    get_distinct_routes
)


def render_logs_page():
    st.title("📝 Boarding Entry Logs")
    st.caption("Immutable audit records of all successful and rejected boarding attempts.")

    # 1. Filter Controls Bar
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

    with col1:
        use_date = st.checkbox("Filter by Date", value=False)
        selected_date = st.date_input("Date:", value=date.today(), disabled=not use_date)

    with col2:
        student_query = st.text_input("Student (Name / ID):", placeholder="e.g. 21CS101")

    distinct_buses = ["All"] + get_distinct_buses()
    with col3:
        selected_bus = st.selectbox("Bus:", options=distinct_buses, index=0)

    distinct_routes = ["All"] + get_distinct_routes()
    with col4:
        selected_route = st.selectbox("Route:", options=distinct_routes, index=0)

    with col5:
        result_filter = st.selectbox("Result:", options=["ALL", "APPROVED", "REJECTED", "UNKNOWN"], index=0)

    limit = st.select_slider("Rows to Display:", options=[25, 50, 100, 250], value=50)

    # 2. Query Logs
    try:
        logs = get_filtered_entry_logs(
            start_date=selected_date if use_date else None,
            end_date=selected_date if use_date else None,
            student_query=student_query if student_query else None,
            bus_id=selected_bus,
            route=selected_route,
            result=result_filter,
            limit=limit
        )

        st.markdown(f"**Showing {len(logs)} log entry(ies):**")

        if logs:
            df = pd.DataFrame(logs)
            df_display = df[["entry_id", "timestamp", "student_name", "student_id", "bus_id", "route", "result", "reason", "similarity"]].copy()
            df_display.columns = ["Log ID", "Timestamp (UTC)", "Passenger Name", "Student ID", "Bus", "Route", "Decision", "Reason Code", "Similarity"]

            def highlight_log_row(val):
                if val == "APPROVED":
                    return "background-color: #d4edda; color: #155724; font-weight: bold;"
                elif val == "REJECTED":
                    return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
                return ""

            st.dataframe(
                df_display.style.map(highlight_log_row, subset=["Decision"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No entry logs match the active filter criteria.")
    except Exception as e:
        st.error(f"Error fetching entry logs: {e}")
