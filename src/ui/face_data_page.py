"""
Face Biometrics & Embedding Status Page
Smart Bus Face Recognition and Pass Verification System
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from src.database.dashboard_queries import get_students_directory
from src.vision.embedding_store import EmbeddingStore
from src.vision.build_student_embeddings import DEFAULT_FACES_DIR, SUPPORTED_IMAGE_EXTS, StudentEmbeddingBuilder


def render_face_data_page():
    st.title("👤 Face Biometrics Management")
    st.caption("Inspect enrollment photo counts, embedding generation status, and biometric synchronization.")

    st.warning(
        "🔒 **Privacy Notice**: Facial embedding vectors represent mathematical biometric signatures. "
        "In accordance with system security rules, raw 512-dimensional vectors are strictly protected and never displayed or made downloadable in this interface."
    )

    store = EmbeddingStore()
    students = get_students_directory()

    if not students:
        st.info("No registered students found in database. Please run student importer first.")
        return

    # 1. Analyze Status for Each Student
    summary_data = []
    for s in students:
        sid = s["student_id"]
        name = s["name"]

        # Check photos folder
        folder = DEFAULT_FACES_DIR / sid
        if folder.exists() and folder.is_dir():
            photos = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
            photo_count = len(photos)
        else:
            photo_count = 0

        # Check embedding store
        has_embedding = store.exists(sid)

        if photo_count > 0 and has_embedding:
            status = "READY"
        elif photo_count == 0:
            status = "MISSING_IMAGES"
        elif photo_count > 0 and not has_embedding:
            status = "MISSING_EMBEDDINGS"
        else:
            status = "ERROR"

        summary_data.append({
            "Student ID": sid,
            "Full Name": name,
            "Registered Photos": photo_count,
            "Stored Embeddings": 1 if has_embedding else 0,
            "Biometric Status": status
        })

    df = pd.DataFrame(summary_data)

    # 2. Status Metric Bar
    col1, col2, col3 = st.columns(3)
    ready_count = sum(1 for d in summary_data if d["Biometric Status"] == "READY")
    missing_photos = sum(1 for d in summary_data if d["Biometric Status"] == "MISSING_IMAGES")
    missing_embs = sum(1 for d in summary_data if d["Biometric Status"] == "MISSING_EMBEDDINGS")

    with col1:
        st.metric("Ready for Bus Entry", value=ready_count, delta="Active Biometrics")
    with col2:
        st.metric("Missing Face Photos", value=missing_photos, delta="-Photos Needed", delta_color="inverse")
    with col3:
        st.metric("Pending Embeddings", value=missing_embs, delta="-Run Builder", delta_color="inverse")

    st.markdown("---")

    # 3. Overview Table
    def color_status(val):
        if val == "READY":
            return "background-color: #d4edda; color: #155724; font-weight: bold;"
        elif val == "MISSING_IMAGES":
            return "background-color: #fff3cd; color: #856404; font-weight: bold;"
        elif val == "MISSING_EMBEDDINGS":
            return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
        return ""

    st.subheader("Student Biometric Status Table")
    st.dataframe(
        df.style.map(color_status, subset=["Biometric Status"]),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # 4. Rebuild / Generate Embeddings Tool
    st.subheader("⚙️ Biometric Store Actions")
    st.write("If you recently placed photos in `data/students/faces/<student_id>/`, click below to process and rebuild embeddings:")

    if st.button("🚀 Process Photos & Build Face Embeddings"):
        with st.spinner("Running MTCNN face detector and FaceNet embedding extractor..."):
            builder = StudentEmbeddingBuilder(faces_dir=DEFAULT_FACES_DIR)
            reports = builder.build_all()
            rebuilt_count = sum(1 for r in reports if r["status"] == "SUCCESS")
            st.success(f"Successfully processed photos! {rebuilt_count} student profile(s) updated in models/embeddings.pkl.")
            st.rerun()
