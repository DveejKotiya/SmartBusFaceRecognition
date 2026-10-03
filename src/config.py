"""
Central Configuration Module
Smart Bus Face Recognition and Pass Verification System

Consolidates all system thresholds, timeouts, and liveness parameters.
Prevents hardcoded magic numbers across modules and eliminates circular imports.
"""

# =====================================================================
# ANTI-SPOOFING & LIVENESS CONFIGURATION
# =====================================================================

# Master toggle for liveness verification.
# When False, the terminal runs in DEVELOPMENT MODE with explicit warnings.
LIVENESS_ENABLED: bool = True

# When True, bus entry is strictly denied if liveness fails or is inconclusive.
LIVENESS_REQUIRED_FOR_ENTRY: bool = True

# Maximum duration allowed for passenger to complete a requested challenge (seconds)
CHALLENGE_TIMEOUT_SECONDS: float = 4.0

# Session lifetime before state is reset (seconds)
LIVENESS_TIMEOUT_SECONDS: float = 4.0

# Minimum sequential observations needed to confirm movement
MIN_FRAMES_REQUIRED: int = 3

# Head yaw asymmetry thresholds for MTCNN 5-point landmarks
# Ratio = (x_nose - min(x_left, x_right)) / |x_right - x_left|
NEUTRAL_YAW_MIN: float = 0.38
NEUTRAL_YAW_MAX: float = 0.62
TURN_LEFT_YAW_MAX: float = 0.32    # Nose shifts closer to left eye in camera view
TURN_RIGHT_YAW_MIN: float = 0.68   # Nose shifts closer to right eye in camera view

# Challenge type: "RANDOM", "TURN_LEFT", "TURN_RIGHT"
DEFAULT_CHALLENGE_TYPE: str = "RANDOM"


# =====================================================================
# CORE VISION & PASS VERIFICATION CONFIGURATION
# =====================================================================

# FaceNet VGGFace2 cosine similarity threshold
DEFAULT_RECOGNITION_THRESHOLD: float = 0.60

# Anti-duplicate boarding cooldown in seconds (5 minutes)
DEFAULT_COOLDOWN_SECONDS: int = 300

# Default transit identifiers
DEFAULT_BUS_ID: str = "BUS-12"
DEFAULT_ROUTE: str = "R-101"
