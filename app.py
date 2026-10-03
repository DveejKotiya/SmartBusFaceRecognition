"""
Main Streamlit Application Entry Point
AI-Based Face Recognition and Smart Bus Pass Verification System

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for robust imports across all environments
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Configure page metadata and wide layout
st.set_page_config(
    page_title="AI Smart Bus Entry System",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular page renderers
from src.ui.dashboard_page import render_dashboard_page
from src.ui.students_page import render_students_page
from src.ui.passes_page import render_passes_page
from src.ui.face_data_page import render_face_data_page
from src.ui.live_entry_page import render_live_entry_page
from src.ui.logs_page import render_logs_page
from src.ui.reports_page import render_reports_page
from src.ui.system_page import render_system_page


def main():
    # Sidebar Branding & Navigation
    st.sidebar.title("🚌 Smart Bus AI")
    st.sidebar.caption("Face Recognition & Pass Verification")

    pages = {
        "📊 Dashboard": render_dashboard_page,
        "🎓 Students": render_students_page,
        "🎫 Bus Passes": render_passes_page,
        "👤 Face Data": render_face_data_page,
        "📹 Live Bus Entry": render_live_entry_page,
        "📝 Entry Logs": render_logs_page,
        "📈 Reports": render_reports_page,
        "⚙️ System Status": render_system_page
    }

    selected_page_label = st.sidebar.radio(
        "Navigation",
        options=list(pages.keys()),
        index=0
    )

    st.sidebar.markdown("---")
    
    # Biometric Privacy & Security Notice in Sidebar
    st.sidebar.markdown(
        "🔒 **Biometric Privacy Notice**\n\n"
        "This terminal processes facial biometrics locally for automated bus-pass validation. "
        "No biometric data is shared externally or uploaded to cloud services."
    )
    st.sidebar.caption("Smart Bus Prototype | Stage 10 Final System")


    # Render Selected Page with Global Error Boundary
    render_fn = pages[selected_page_label]
    try:
        render_fn()
    except Exception as e:
        st.error(f"An unexpected error occurred while rendering the page: {e}")
        st.info("Please verify database connections and service health on the System Status page.")


if __name__ == "__main__":
    main()
