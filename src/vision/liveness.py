"""
Anti-Spoofing & Liveness Detection Module
Smart Bus Face Recognition and Pass Verification System

This module provides active challenge-response and passive motion/landmark variance
liveness detection to prevent photo and video presentation attacks (spoofing).

It operates strictly on facial geometry and landmark kinematics:
- Zero database dependencies.
- Zero identity or recognition logic.
- Evaluates whether the face in front of the camera belongs to a live, responsive human.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import random
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from src.config import (
    LIVENESS_ENABLED,
    CHALLENGE_TIMEOUT_SECONDS,
    LIVENESS_TIMEOUT_SECONDS,
    MIN_FRAMES_REQUIRED,
    NEUTRAL_YAW_MIN,
    NEUTRAL_YAW_MAX,
    TURN_LEFT_YAW_MAX,
    TURN_RIGHT_YAW_MIN,
    DEFAULT_CHALLENGE_TYPE
)


# =====================================================================
# ENUMS & DATA STRUCTURES
# =====================================================================

class LivenessState(str, Enum):
    """Possible outcomes of liveness evaluation."""
    LIVE = "LIVE"
    SPOOF_SUSPECTED = "SPOOF_SUSPECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    ERROR = "ERROR"


class ChallengeType(str, Enum):
    """Active challenge actions prompted to the passenger."""
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    PASSIVE_MOTION = "PASSIVE_MOTION"


@dataclass
class LivenessResult:
    """
    Structured outcome of a liveness evaluation.

    Attributes:
        is_live: True ONLY if liveness has been positively confirmed.
        state: State category (LIVE, SPOOF_SUSPECTED, INSUFFICIENT_DATA, ERROR).
        confidence: Confidence score (0.0 to 1.0).
        method: The specific verification method utilized.
        reason: Human-readable explanation.
        timestamp: Time of evaluation.
        challenge: Current challenge type issued to the user.
        challenge_prompt: User-facing prompt (e.g. 'Turn head slightly LEFT').
        progress: Progress ratio (0.0 to 1.0) towards completing the challenge.
    """
    is_live: bool
    state: LivenessState
    confidence: float
    method: str
    reason: str
    timestamp: datetime
    challenge: Optional[ChallengeType] = None
    challenge_prompt: Optional[str] = None
    progress: float = 0.0


@dataclass
class Observation:
    """A single frame's measurement."""
    timestamp: float
    box: Tuple[int, int, int, int]
    landmarks: np.ndarray
    yaw_ratio: float


@dataclass
class LivenessSession:
    """
    Maintains temporal tracking and challenge-response state for a single face.
    """
    session_id: str
    challenge: ChallengeType
    challenge_prompt: str
    created_at: float
    last_seen_at: float
    observations: List[Observation] = field(default_factory=list)
    has_neutral: bool = False
    has_target_movement: bool = False
    is_verified: bool = False
    verified_at: Optional[float] = None
    validity_duration_seconds: float = 3.0  # Duration a passed challenge remains valid for pass verifier


# =====================================================================
# LIVENESS DETECTOR SERVICE
# =====================================================================

class LivenessDetector:
    """
    Multi-frame challenge-response and landmark variance liveness detector.

    Uses the 5 facial landmarks extracted by MTCNN:
    0: left eye, 1: right eye, 2: nose, 3: mouth left, 4: mouth right.
    """

    def __init__(
        self,
        enabled: bool = LIVENESS_ENABLED,
        challenge_timeout: float = CHALLENGE_TIMEOUT_SECONDS,
        min_frames: int = MIN_FRAMES_REQUIRED,
        challenge_type: str = DEFAULT_CHALLENGE_TYPE
    ) -> None:
        """
        Initializes the liveness detector.

        Args:
            enabled: Master switch. If False, runs in bypass/dev mode.
            challenge_timeout: Seconds to allow passenger to respond.
            min_frames: Minimum observations required before making decisions.
            challenge_type: "RANDOM", "TURN_LEFT", or "TURN_RIGHT".
        """
        self.enabled = enabled
        self.challenge_timeout = challenge_timeout
        self.min_frames = min_frames
        self.challenge_type_config = challenge_type

        # Active tracked face sessions: session_id -> LivenessSession
        self.sessions: Dict[str, LivenessSession] = {}

    def is_enabled(self) -> bool:
        """Returns True if anti-spoofing is actively enforced."""
        return self.enabled

    def reset(self) -> None:
        """Clears all active sessions and state."""
        self.sessions.clear()

    def get_active_session_count(self) -> int:
        """Returns the number of currently tracked face sessions."""
        return len(self.sessions)

    # -----------------------------------------------------------------
    # GEOMETRIC & KINEMATIC HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def calculate_yaw_ratio(landmarks: np.ndarray) -> Optional[float]:
        """
        Calculates horizontal facial symmetry ratio from MTCNN 5-point landmarks.

        Ratio = (x_nose - min(x_left_eye, x_right_eye)) / |x_right_eye - x_left_eye|
        - Center/Neutral: ~0.45 - 0.55
        - Head turned to passenger's left (camera's left/right): drops < 0.32 or rises > 0.68
        """
        if landmarks is None or not isinstance(landmarks, (np.ndarray, list)):
            return None
        arr = np.array(landmarks, dtype=np.float32)
        if arr.shape != (5, 2):
            return None

        if np.isnan(arr).any():
            return None

        # 0: left eye, 1: right eye, 2: nose
        left_eye_x = arr[0, 0]
        right_eye_x = arr[1, 0]
        nose_x = arr[2, 0]

        min_eye_x = min(left_eye_x, right_eye_x)
        max_eye_x = max(left_eye_x, right_eye_x)
        eye_span = max_eye_x - min_eye_x

        if eye_span <= 1.0 or np.isnan(eye_span):  # Avoid division by zero
            return None

        yaw_ratio = float((nose_x - min_eye_x) / eye_span)
        if np.isnan(yaw_ratio):
            return None
        return yaw_ratio

    @staticmethod
    def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Calculates Intersection over Union between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def _match_or_create_session(
        self,
        box: Tuple[int, int, int, int],
        now_ts: float
    ) -> LivenessSession:
        """Associates incoming face detection with an ongoing session or creates a new one."""
        # 1. Prune expired or stale sessions
        stale_cutoff = now_ts - (self.challenge_timeout * 2.0)
        self.sessions = {
            sid: s for sid, s in self.sessions.items()
            if s.last_seen_at > stale_cutoff
        }

        # 2. Match existing session by bounding box spatial proximity (IoU > 0.35)
        best_sid: Optional[str] = None
        best_iou: float = 0.35

        for sid, sess in self.sessions.items():
            if sess.observations:
                last_box = sess.observations[-1].box
                iou = self.compute_iou(box, last_box)
                if iou > best_iou:
                    best_iou = iou
                    best_sid = sid

        if best_sid is not None:
            session = self.sessions[best_sid]
            session.last_seen_at = now_ts
            return session

        # 3. Create a fresh session with a challenge
        if self.challenge_type_config == "RANDOM":
            challenge = random.choice([ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT])
        elif self.challenge_type_config == "TURN_LEFT":
            challenge = ChallengeType.TURN_LEFT
        elif self.challenge_type_config == "TURN_RIGHT":
            challenge = ChallengeType.TURN_RIGHT
        else:
            challenge = ChallengeType.TURN_LEFT

        prompt = (
            "Please turn your head slightly to your LEFT"
            if challenge == ChallengeType.TURN_LEFT
            else "Please turn your head slightly to your RIGHT"
        )

        session_id = f"sess_{int(now_ts * 1000)}_{random.randint(100, 999)}"
        new_session = LivenessSession(
            session_id=session_id,
            challenge=challenge,
            challenge_prompt=prompt,
            created_at=now_ts,
            last_seen_at=now_ts
        )
        self.sessions[session_id] = new_session
        return new_session

    # -----------------------------------------------------------------
    # EVALUATION PIPELINE
    # -----------------------------------------------------------------

    def evaluate(
        self,
        box: Tuple[int, int, int, int],
        landmarks: Optional[np.ndarray],
        current_timestamp: Optional[datetime] = None
    ) -> LivenessResult:
        """
        Evaluates liveness for a detected face box and landmarks.

        Args:
            box: Bounding box tuple (x1, y1, x2, y2).
            landmarks: MTCNN 5-point facial landmarks array (shape (5, 2)).
            current_timestamp: Optional evaluation timestamp (defaults to UTC now).

        Returns:
            LivenessResult: Structured evaluation with explicit state and user prompt.
        """
        now_dt = current_timestamp if current_timestamp else datetime.now(timezone.utc)
        now_ts = now_dt.timestamp()

        # Development / Bypass Mode Check
        if not self.enabled:
            return LivenessResult(
                is_live=True,
                state=LivenessState.LIVE,
                confidence=1.0,
                method="BYPASS_DEV_MODE",
                reason="Liveness detection is disabled in development mode.",
                timestamp=now_dt,
                progress=1.0
            )

        # Validate Inputs
        if box is None or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            return LivenessResult(
                is_live=False,
                state=LivenessState.ERROR,
                confidence=0.0,
                method="CHALLENGE_HEAD_YAW",
                reason="Invalid bounding box coordinates.",
                timestamp=now_dt
            )

        if landmarks is None:
            return LivenessResult(
                is_live=False,
                state=LivenessState.INSUFFICIENT_DATA,
                confidence=0.0,
                method="CHALLENGE_HEAD_YAW",
                reason="No facial landmarks detected in frame.",
                timestamp=now_dt
            )

        try:
            yaw_ratio = self.calculate_yaw_ratio(landmarks)
            if yaw_ratio is None:
                return LivenessResult(
                    is_live=False,
                    state=LivenessState.ERROR,
                    confidence=0.0,
                    method="CHALLENGE_HEAD_YAW",
                    reason="Could not compute facial yaw symmetry ratio.",
                    timestamp=now_dt
                )
        except Exception as e:
            return LivenessResult(
                is_live=False,
                state=LivenessState.ERROR,
                confidence=0.0,
                method="CHALLENGE_HEAD_YAW",
                reason=f"Error computing facial landmarks: {e}",
                timestamp=now_dt
            )

        # Associate or start session
        session = self._match_or_create_session(box, now_ts)

        # If this session already passed liveness and is within its validity window:
        if session.is_verified and session.verified_at:
            if (now_ts - session.verified_at) <= session.validity_duration_seconds:
                return LivenessResult(
                    is_live=True,
                    state=LivenessState.LIVE,
                    confidence=0.95,
                    method="CHALLENGE_HEAD_YAW",
                    reason="Liveness challenge successfully completed.",
                    timestamp=now_dt,
                    challenge=session.challenge,
                    challenge_prompt=session.challenge_prompt,
                    progress=1.0
                )
            else:
                # Previous verification expired; reset session to require fresh challenge
                session.is_verified = False
                session.verified_at = None
                session.has_neutral = False
                session.has_target_movement = False
                session.created_at = now_ts
                session.observations.clear()

        # Add observation
        session.observations.append(
            Observation(timestamp=now_ts, box=box, landmarks=landmarks, yaw_ratio=yaw_ratio)
        )

        elapsed = now_ts - session.created_at

        # Check for static photo attack (near-zero landmark variance across frames)
        if len(session.observations) >= 6:
            recent_yaws = [obs.yaw_ratio for obs in session.observations[-6:]]
            recent_boxes = [obs.box for obs in session.observations[-6:]]
            yaw_variance = float(np.var(recent_yaws))
            box_variance = float(np.var([[b[0], b[1]] for b in recent_boxes]))

            # If face is unnaturally completely motionless across time
            if elapsed > 1.5 and yaw_variance < 1e-6 and box_variance < 1e-4:
                return LivenessResult(
                    is_live=False,
                    state=LivenessState.SPOOF_SUSPECTED,
                    confidence=0.85,
                    method="PASSIVE_VARIANCE_CHECK",
                    reason="Completely motionless static face detected — potential photo presentation attack.",
                    timestamp=now_dt,
                    challenge=session.challenge,
                    challenge_prompt=session.challenge_prompt,
                    progress=0.0
                )

        # 1. Track neutral frontal pose baseline
        if NEUTRAL_YAW_MIN <= yaw_ratio <= NEUTRAL_YAW_MAX:
            session.has_neutral = True

        # 2. Track target challenge movement
        if session.challenge == ChallengeType.TURN_LEFT:
            # Turn Left: nose shifts towards left eye
            if yaw_ratio <= TURN_LEFT_YAW_MAX:
                session.has_target_movement = True
            progress = max(0.0, min(1.0, (NEUTRAL_YAW_MAX - yaw_ratio) / (NEUTRAL_YAW_MAX - TURN_LEFT_YAW_MAX)))
        elif session.challenge == ChallengeType.TURN_RIGHT:
            # Turn Right: nose shifts towards right eye
            if yaw_ratio >= TURN_RIGHT_YAW_MIN:
                session.has_target_movement = True
            progress = max(0.0, min(1.0, (yaw_ratio - NEUTRAL_YAW_MIN) / (TURN_RIGHT_YAW_MIN - NEUTRAL_YAW_MIN)))
        else:
            progress = 0.5

        # 3. Check for Challenge Completion
        if len(session.observations) >= self.min_frames:
            # Passenger must have been seen in neutral, then performed requested turn
            if session.has_neutral and session.has_target_movement:
                session.is_verified = True
                session.verified_at = now_ts
                return LivenessResult(
                    is_live=True,
                    state=LivenessState.LIVE,
                    confidence=0.92,
                    method="CHALLENGE_HEAD_YAW",
                    reason="Head turn challenge successfully verified.",
                    timestamp=now_dt,
                    challenge=session.challenge,
                    challenge_prompt=session.challenge_prompt,
                    progress=1.0
                )

        # 4. Check for Timeout
        if elapsed > self.challenge_timeout:
            return LivenessResult(
                is_live=False,
                state=LivenessState.SPOOF_SUSPECTED,
                confidence=0.75,
                method="CHALLENGE_HEAD_YAW",
                reason=f"Liveness challenge timed out after {self.challenge_timeout:.1f}s without requested head movement.",
                timestamp=now_dt,
                challenge=session.challenge,
                challenge_prompt=session.challenge_prompt,
                progress=progress
            )

        # 5. Challenge Still In Progress
        return LivenessResult(
            is_live=False,
            state=LivenessState.INSUFFICIENT_DATA,
            confidence=0.0,
            method="CHALLENGE_HEAD_YAW",
            reason=f"{session.challenge_prompt} (Time remaining: {max(0.0, self.challenge_timeout - elapsed):.1f}s)",
            timestamp=now_dt,
            challenge=session.challenge,
            challenge_prompt=session.challenge_prompt,
            progress=progress
        )
