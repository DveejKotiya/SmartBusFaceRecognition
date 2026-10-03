"""
Reports & CSV Export Page
Smart Bus Face Recognition and Pass Verification System
"""

import io
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from src.database.dashboard_queries import (
    get_filtered_entry_logs,
    get_distinct_routes,
    get_distinct_buses
)


def render_reports_page():
    st.title("📈 Reports & Data Export")
    st.caption("Generate operational boarding summaries and download official audit logs in CSV format.")

    # 1. Report Mode Selection
    report_type = st.selectbox(
        "Select Report Type:",
        options=[
            "Daily Entrance Report",
            "Date-Range Audit Report",
            "Route-Specific Boarding Report",
            "Student Attendance History"
        ],
        index=0
    )

    start_d = None
    end_d = None
    route_filter = None
    student_query = None

    col1, col2 = st.columns(2)
    if report_type == "Daily Entrance Report":
        with col1:
            report_date = st.date_input("Report Date:", value=date.today())
            start_d = report_date
            end_d = report_date

    elif report_type == "Date-Range Audit Report":
        with col1:
            start_d = st.date_input("Start Date:", value=date.today() - timedelta(days=7))
        with col2:
            end_d = st.date_input("End Date:", value=date.today())

    elif report_type == "Route-Specific Boarding Report":
        routes = ["All"] + get_distinct_routes()
        with col1:
            route_filter = st.selectbox("Select Route:", options=routes, index=0)
        with col2:
            start_d = st.date_input("Since Date:", value=date.today() - timedelta(days=30))

    elif report_type == "Student Attendance History":
        with col1:
            student_query = st.text_input("Enter Roll Number / Name:", placeholder="e.g. 21CS101")

    # 2. Fetch Real Data
    try:
        data = get_filtered_entry_logs(
            start_date=start_d,
            end_date=end_d,
            student_query=student_query,
            route=route_filter,
            limit=500
        )

        if not data:
            st.info("No logs found matching this report criteria.")
            return

        df = pd.DataFrame(data)

        # 3. High-Level Summary Statistics
        st.subheader("Summary Metrics")
        m1, m2, m3 = st.columns(3)
        total_scans = len(df)
        total_approved = sum(1 for d in data if d["result"] == "APPROVED")
        total_rejected = sum(1 for d in data if d["result"] == "REJECTED")

        m1.metric("Total Scans", value=total_scans)
        m2.metric("Approved Entries", value=total_approved, delta="Allowed")
        m3.metric("Denied Attempts", value=total_rejected, delta="-Denied", delta_color="inverse")

        # 4. Preview Table
        st.markdown("### Report Data Preview")
        preview_df = df[["entry_id", "timestamp", "student_id", "student_name", "bus_id", "route", "similarity", "result", "reason"]].copy()
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        # 5. CSV Export Button
        csv_buffer = io.StringIO()
        preview_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode("utf-8")

        filename = f"smart_bus_report_{report_type.lower().replace(' ', '_')}_{date.today().isoformat()}.csv"
        st.download_button(
            label="📥 Download Official Report (CSV)",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error generating report: {e}")
