# Chapter 20: Streamlit Administration Dashboard

## 20.1 User Interface Architecture
The administrative portal (`app.py` and `src/ui/`) is implemented in Streamlit, providing transport supervisors with an intuitive web application structured into 8 dedicated modules:

1. **📊 Dashboard Overview (`dashboard_page.py`):** Real-time summary cards (Total Students, Active Passes, Today's Boardings, Denial Rate) paired with a live feed of the 10 most recent boarding attempts.
2. **🎓 Students Directory (`students_page.py`):** Comprehensive student registry with instant multi-attribute search (Roll Number, Name, Department).
3. **🎫 Bus Passes Manager (`passes_page.py`):** Pass validity tracking with color-coded status badges and expiration warnings for passes expiring within 14 days.
4. **👤 Face Data & Biometric Enrollment (`face_data_page.py`):** Visual status indicators (`READY`, `MISSING_IMAGES`, `MISSING_EMBEDDINGS`) and a one-click button to trigger the offline photo-to-embedding builder.
5. **📹 Live Bus Entry Terminal (`live_entry_page.py`):** In-browser video terminal running `BusEntryService`, displaying live bounding boxes, liveness prompt indicators, and real-time pass decisions.
6. **📝 Entry Logs Explorer (`logs_page.py`):** Searchable transit audit trail with date pickers, result filters (`APPROVED` vs `REJECTED`), and reason code filtering.
7. **📈 Reports & CSV Export (`reports_page.py`):** Aggregated visual charts (Boarding activity by hour, pass breakdown by department) and sanitized official CSV exports (9 operational columns, zero biometric vectors).
8. **⚙️ System Status & Diagnostics (`system_page.py`):** Real-time monitoring of CPU architecture, PyTorch version, GPU/CPU mode, camera sensor availability, and SQLite integrity (`PRAGMA integrity_check;`).

## 20.2 Robust Error Boundaries
The dashboard entry point wraps each page renderer in a global `try...except` boundary. If an unhandled database exception or camera disconnection occurs on any page, the application catches the error, displays an informative recovery banner, and prevents application crashes.
