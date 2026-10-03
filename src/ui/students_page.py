"""
Students Directory Page
Smart Bus Face Recognition and Pass Verification System
"""

import streamlit as st
import pandas as pd
from src.database.dashboard_queries import (
    get_students_directory,
    get_all_departments,
    get_all_semesters
)


def render_students_page():
    st.title("🎓 Student Directory")
    st.caption("Official student enrollment records, department affiliations, and pass assignments.")

    # Informational notice on source of truth
    st.info(
        "ℹ️ **Manual Curation Workflow**: Student profiles and bus passes are managed via "
        "`data/students/students.csv`. To add new students or edit details, update the CSV file and run:\n\n"
        "```bash\npython -m src.data.student_importer\n```"
    )

    # 1. Filters & Search Bar
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input("🔍 Search by Student ID or Name:", placeholder="e.g. 21CS101 or Aarav")

    departments = ["All"] + get_all_departments()
    with col2:
        selected_dept = st.selectbox("Department:", options=departments, index=0)

    semesters = ["All"] + get_all_semesters()
    with col3:
        selected_sem = st.selectbox("Semester:", options=semesters, index=0)

    # 2. Query Database
    try:
        students = get_students_directory(
            search=search_query if search_query else None,
            department=selected_dept,
            semester=selected_sem
        )

        st.markdown(f"**Found {len(students)} Student(s)**")

        if students:
            df = pd.DataFrame(students)
            df_display = df[["student_id", "name", "department", "semester", "bus_id", "route", "pass_status", "expiry_date"]].copy()
            df_display.columns = ["Student ID", "Full Name", "Department", "Semester", "Assigned Bus", "Route", "Pass Status", "Pass Expiry"]

            def highlight_status(val):
                if val == "ACTIVE":
                    return "color: #155724; font-weight: bold;"
                elif val in ("EXPIRED", "REVOKED", "NO_PASS"):
                    return "color: #721c24; font-weight: bold;"
                return ""

            st.dataframe(
                df_display.style.map(highlight_status, subset=["Pass Status"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No students matched the selected search filters.")
    except Exception as e:
        st.error(f"Error querying student directory: {e}")
