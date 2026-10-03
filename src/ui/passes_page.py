"""
Bus Passes Management Page
Smart Bus Face Recognition and Pass Verification System
"""

import streamlit as st
import pandas as pd
from datetime import date
from src.database.dashboard_queries import get_passes_directory


def render_passes_page():
    st.title("🎫 Bus Passes Directory")
    st.caption("Active, pending, and expired student bus travel authorizations.")

    # 1. Status Filter
    filter_option = st.radio(
        "Filter by Pass Status:",
        options=["ALL", "ACTIVE", "EXPIRED", "INACTIVE"],
        horizontal=True,
        index=0
    )

    try:
        passes = get_passes_directory(status_filter=filter_option)

        # Count expired in result
        expired_count = sum(1 for p in passes if p.get("is_expired") == 1 or p.get("status") == "EXPIRED")
        if expired_count > 0:
            st.warning(f"⚠️ **Attention**: {expired_count} pass(es) in this view have expired or reached their validity deadline.")

        st.markdown(f"**Total Passes Displayed: {len(passes)}**")

        if passes:
            df = pd.DataFrame(passes)
            df_display = df[["student_id", "student_name", "bus_id", "route", "start_date", "expiry_date", "status"]].copy()
            df_display.columns = ["Student ID", "Student Name", "Bus ID", "Route Code", "Valid From", "Expiry Date", "Recorded Status"]

            def format_pass_row(row):
                is_expired = (row["Recorded Status"] == "EXPIRED") or (str(row["Expiry Date"]) < date.today().isoformat())
                if is_expired:
                    return ["background-color: #fff3cd; color: #856404; font-weight: normal;"] * len(row)
                elif row["Recorded Status"] == "ACTIVE":
                    return ["background-color: #e8f5e9; color: #2e7d32; font-weight: normal;"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_display.style.apply(format_pass_row, axis=1),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No bus passes found matching the selected filter.")
    except Exception as e:
        st.error(f"Error querying bus pass records: {e}")
