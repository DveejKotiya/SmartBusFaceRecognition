"""
Core Business Logic Package
Smart Bus Face Recognition and Pass Verification System
"""

from .pass_verifier import (
    PassVerifier,
    RecognitionResult,
    VerificationResult,
    ReasonCode,
    DEFAULT_RECOGNITION_THRESHOLD,
    DEFAULT_COOLDOWN_SECONDS
)
from .bus_entry_service import (
    BusEntryService,
    RecognitionDetail,
    BusEntryResult
)

__all__ = [
    "PassVerifier",
    "RecognitionResult",
    "VerificationResult",
    "ReasonCode",
    "DEFAULT_RECOGNITION_THRESHOLD",
    "DEFAULT_COOLDOWN_SECONDS",
    "BusEntryService",
    "RecognitionDetail",
    "BusEntryResult"
]
