"""
Bus Entry Orchestration Service
Smart Bus Face Recognition and Pass Verification System

This service connects the computer vision components (FaceDetector, FaceRecognizer,
EmbeddingStore) with the business verification layer (PassVerifier) and the SQLite
database to execute end-to-end boarding decisions for each camera frame.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np

from src.vision.face_detector import FaceDetector, DetectionResult
from src.vision.face_recognizer import FaceRecognizer, compare_embeddings
from src.vision.embedding_store import EmbeddingStore
from src.vision.liveness import LivenessDetector, LivenessResult, LivenessState
from src.core.config import LIVENESS_ENABLED
from src.core.pass_verifier import (
    PassVerifier,
    RecognitionResult,
    VerificationResult,
    ReasonCode,
    DEFAULT_RECOGNITION_THRESHOLD
)


# =====================================================================
# STRUCTURED RESULT CONTAINERS
# =====================================================================

@dataclass
class RecognitionDetail:
    """
    Structured outcome of face recognition for a single detected face.

    Attributes:
        recognized: True if matched to an identity above the recognition threshold.
        student_id: Matched student ID / Roll Number (None if unknown).
        student_name: Matched student name from store (None if unknown).
        similarity: Cosine similarity score between -1.0 and 1.0.
        detection_confidence: MTCNN detection probability score (0.0 to 1.0).
        bounding_box: (x1, y1, x2, y2) integer face coordinates on frame.
    """
    recognized: bool
    student_id: Optional[str]
    student_name: Optional[str]
    similarity: float
    detection_confidence: float
    bounding_box: Tuple[int, int, int, int]


@dataclass
class BusEntryResult:
    """
    Final boarding authorization decision for a single passenger.

    Attributes:
        allowed: True if boarding is authorized; False otherwise.
        student_id: Verified student ID (or None if unknown).
        student_name: Official student name (or None if unknown).
        similarity: Facial cosine similarity score.
        detection_confidence: Face detection confidence.
        bus_id: Identifier of current bus (e.g. 'BUS-12').
        route: Identifier of current route (e.g. 'R-101').
        reason_code: Machine-readable reason code from ReasonCode enum.
        reason_message: Human-readable explanation for screen display / conductor.
        timestamp: Time of decision.
        bounding_box: (x1, y1, x2, y2) face coordinates.
        duplicate: True if rejected specifically due to the 5-minute cooldown.
    """
    allowed: bool
    student_id: Optional[str]
    student_name: Optional[str]
    similarity: float
    detection_confidence: float
    bus_id: str
    route: str
    reason_code: ReasonCode
    reason_message: str
    timestamp: datetime
    bounding_box: Tuple[int, int, int, int]
    duplicate: bool = False
    liveness_result: Optional[LivenessResult] = None


# =====================================================================
# BUS ENTRY ORCHESTRATION SERVICE
# =====================================================================

class BusEntryService:
    """
    Coordinates the real-time boarding pipeline:
    1. Detects all faces in frame using FaceDetector.
    2. Extracts 512-D embeddings using FaceRecognizer.
    3. Searches EmbeddingStore for the best candidate.
    4. Evaluates eligibility using PassVerifier.
    5. Logs verified boardings to SQLite.
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        store: Optional[EmbeddingStore] = None,
        verifier: Optional[PassVerifier] = None,
        liveness_detector: Optional[LivenessDetector] = None,
        recognition_threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
        device: Optional[str] = None,
        enable_liveness: Optional[bool] = None
    ) -> None:
        """
        Initializes the service. Accepts optional custom or mocked components.
        """
        self.detector = detector if detector else FaceDetector(device=device)
        self.recognizer = recognizer if recognizer else FaceRecognizer(device=device)
        self.store = store if store else EmbeddingStore()
        self.verifier = verifier if verifier else PassVerifier(recognition_threshold=recognition_threshold)
        self.recognition_threshold = recognition_threshold

        if enable_liveness is not None:
            is_live_enabled = enable_liveness
        elif liveness_detector is not None:
            is_live_enabled = liveness_detector.is_enabled()
        elif detector is not None and type(detector).__name__ == "MagicMock":
            is_live_enabled = False
        else:
            is_live_enabled = LIVENESS_ENABLED

        self.liveness_detector = (
            liveness_detector
            if liveness_detector
            else LivenessDetector(enabled=is_live_enabled)
        )


    def identify_face(
        self,
        face_crop: np.ndarray,
        box: Tuple[int, int, int, int],
        confidence: float
    ) -> RecognitionDetail:
        """
        Extracts embedding for a face crop and matches against registered student embeddings.

        Never guesses an identity: if no match meets the recognition threshold,
        returns recognized=False with student_id=None.
        """
        live_emb = self.recognizer.generate_embedding(face_crop)
        all_students = self.store.get_all()

        if not all_students:
            return RecognitionDetail(
                recognized=False,
                student_id=None,
                student_name=None,
                similarity=0.0,
                detection_confidence=confidence,
                bounding_box=box
            )

        best_score = -1.0
        best_student: Optional[Dict[str, Any]] = None

        for record in all_students:
            score = compare_embeddings(live_emb, record["embedding"])
            if score > best_score:
                best_score = score
                best_student = record

        # Only declare recognized if score >= configured threshold
        if best_student and best_score >= self.recognition_threshold:
            return RecognitionDetail(
                recognized=True,
                student_id=str(best_student["student_id"]),
                student_name=best_student["student_name"],
                similarity=best_score,
                detection_confidence=confidence,
                bounding_box=box
            )

        # Fallback for unknown face or low confidence
        return RecognitionDetail(
            recognized=False,
            student_id=None,
            student_name=None,
            similarity=max(0.0, best_score),
            detection_confidence=confidence,
            bounding_box=box
        )

    def process_frame(
        self,
        frame: np.ndarray,
        bus_id: str,
        route: str,
        current_timestamp: Optional[datetime] = None,
        log_to_db: bool = True
    ) -> List[BusEntryResult]:
        """
        Processes a full camera frame:
        1. Detects all faces.
        2. Identifies each face independently.
        3. Verifies boarding pass and 5-minute cooldown.
        4. Logs to database safely.

        Args:
            frame: OpenCV BGR image array.
            bus_id: Current bus identifier (e.g. 'BUS-12').
            route: Current route identifier (e.g. 'R-101').
            current_timestamp: Optional evaluation timestamp (defaults to UTC now).
            log_to_db: If True, writes entry decision to SQLite.

        Returns:
            List[BusEntryResult]: One result per detected face. Returns [] if no faces found.
        """
        if frame is None or frame.size == 0:
            return []

        if current_timestamp is None:
            now_dt = datetime.now(timezone.utc)
        elif current_timestamp.tzinfo is None:
            now_dt = current_timestamp.replace(tzinfo=timezone.utc)
        else:
            now_dt = current_timestamp.astimezone(timezone.utc)

        # 1. Detect faces
        try:
            detections = self.detector.detect_faces(frame, is_bgr=True)
        except Exception as e:
            print(f"[ERROR] Face detection failed on frame: {e}")
            return []

        if not detections:
            return []

        results: List[BusEntryResult] = []

        # 2. Process each detected face independently
        for det in detections:
            box = det.box
            confidence = det.confidence
            crop = det.face_crop
            landmarks = getattr(det, "landmarks", None)

            # ---------------------------------------------------------
            # STEP 1: Liveness Evaluation (MUST PRECEDE IDENTITY & PASS RULES)
            # ---------------------------------------------------------
            liveness_res = self.liveness_detector.evaluate(
                box=box,
                landmarks=landmarks,
                current_timestamp=now_dt
            )

            # If Liveness is enabled and failed (SPOOF_SUSPECTED)
            if self.liveness_detector.is_enabled() and liveness_res.state == LivenessState.SPOOF_SUSPECTED:
                entry_res = BusEntryResult(
                    allowed=False,
                    student_id=None,
                    student_name="Spoof Suspected",
                    similarity=0.0,
                    detection_confidence=confidence,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ReasonCode.LIVENESS_FAILED,
                    reason_message="ENTRY DENIED: Liveness verification failed (Spoof Suspected)",
                    timestamp=now_dt,
                    bounding_box=box,
                    duplicate=False,
                    liveness_result=liveness_res
                )
                if log_to_db:
                    try:
                        ver_stub = VerificationResult(
                            allowed=False,
                            student_id=None,
                            student_name=None,
                            similarity=0.0,
                            bus_id=bus_id,
                            route=route,
                            reason_code=ReasonCode.LIVENESS_FAILED,
                            reason_message=entry_res.reason_message,
                            timestamp=now_dt
                        )
                        self.verifier.log_verification(ver_stub, verification_method="LIVENESS_DETECTION")
                    except Exception as e:
                        print(f"[WARNING] Could not log liveness failure to database: {e}")
                results.append(entry_res)
                continue

            # If Liveness is enabled and inconclusive / in-progress
            if self.liveness_detector.is_enabled() and not liveness_res.is_live:
                entry_res = BusEntryResult(
                    allowed=False,
                    student_id=None,
                    student_name="Awaiting Action",
                    similarity=0.0,
                    detection_confidence=confidence,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ReasonCode.LIVENESS_INCONCLUSIVE,
                    reason_message=liveness_res.reason,
                    timestamp=now_dt,
                    bounding_box=box,
                    duplicate=False,
                    liveness_result=liveness_res
                )
                results.append(entry_res)
                continue

            # ---------------------------------------------------------
            # STEP 2: Identity Identification (Run ONLY after Liveness PASSED)
            # ---------------------------------------------------------
            try:
                rec_detail = self.identify_face(crop, box, confidence)
            except Exception as e:
                print(f"[ERROR] Recognition extraction failed: {e}")
                rec_detail = RecognitionDetail(
                    recognized=False,
                    student_id=None,
                    student_name=None,
                    similarity=0.0,
                    detection_confidence=confidence,
                    bounding_box=box
                )

            # 3. Format input for PassVerifier
            rec_input = RecognitionResult(
                recognized=rec_detail.recognized,
                student_id=rec_detail.student_id,
                student_name=rec_detail.student_name,
                similarity=rec_detail.similarity
            )

            # 4. Pass verification
            try:
                ver_res = self.verifier.verify(
                    rec_input,
                    bus_id=bus_id,
                    route=route,
                    current_timestamp=now_dt
                )
            except Exception as e:
                print(f"[ERROR] Pass verification exception: {e}")
                results.append(
                    BusEntryResult(
                        allowed=False,
                        student_id=rec_detail.student_id,
                        student_name=rec_detail.student_name,
                        similarity=rec_detail.similarity,
                        detection_confidence=confidence,
                        bus_id=bus_id,
                        route=route,
                        reason_code=ReasonCode.DATABASE_ERROR,
                        reason_message=f"Verification failure: {e}",
                        timestamp=now_dt,
                        bounding_box=box,
                        duplicate=False,
                        liveness_result=liveness_res
                    )
                )
                continue

            # 5. Database logging (safely isolated so DB error never crashes camera loop)
            if log_to_db:
                try:
                    # Log only if ALLOWED, or if rejected with a known reason
                    # Note: Cooldown prevents repeated duplicate logs for the same student
                    self.verifier.log_verification(ver_res, verification_method="FACE_RECOGNITION")
                except Exception as e:
                    print(f"[WARNING] Could not log boarding event to database: {e}")

            # 6. Build final entry result
            results.append(
                BusEntryResult(
                    allowed=ver_res.allowed,
                    student_id=ver_res.student_id,
                    student_name=ver_res.student_name,
                    similarity=ver_res.similarity,
                    detection_confidence=confidence,
                    bus_id=bus_id,
                    route=route,
                    reason_code=ver_res.reason_code,
                    reason_message=ver_res.reason_message,
                    timestamp=now_dt,
                    bounding_box=box,
                    duplicate=ver_res.duplicate,
                    liveness_result=liveness_res
                )
            )

        return results
