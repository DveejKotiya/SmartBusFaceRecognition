"""
Face Detector Module
Smart Bus Face Recognition and Pass Verification System

This module uses MTCNN (Multi-task Cascaded Convolutional Networks) from facenet-pytorch
to locate human faces in images (both OpenCV frames and PIL images), crop them,
and return structured detection results.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN


@dataclass
class DetectionResult:
    """
    Structured container for a single face detection.

    Attributes:
        box (Tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2).
        confidence (float): Detection probability score between 0.0 and 1.0.
        face_crop (np.ndarray): Cropped face image in RGB format, resized to 160x160.
        landmarks (Optional[np.ndarray]): 5 facial landmarks (left_eye, right_eye, nose, mouth_l, mouth_r) as (5, 2) array.
    """
    box: Tuple[int, int, int, int]
    confidence: float
    face_crop: np.ndarray
    landmarks: Optional[np.ndarray] = None


class FaceDetector:
    """
    Face detector wrapping MTCNN.

    Supports:
    - Automatic device selection (CUDA GPU if available, otherwise CPU).
    - Detection of single and multiple faces.
    - Input formats: NumPy arrays (OpenCV BGR or RGB) and PIL Images.
    - Safe bounding box validation and 160x160 face extraction.
    """

    def __init__(
        self,
        image_size: int = 160,
        margin: int = 20,
        min_face_size: int = 30,
        confidence_threshold: float = 0.85,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Initializes the FaceDetector with MTCNN.

        Args:
            image_size: The target width and height for cropped faces (default 160x160).
            margin: Extra padding pixels around the detected face bounding box.
            min_face_size: Minimum face size in pixels to consider. Smaller faces are ignored.
            confidence_threshold: Minimum detection probability score (0.0 to 1.0).
            device: 'cuda', 'cpu', or None (auto-selects CUDA if available).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold

        # Initialize MTCNN from facenet-pytorch
        # keep_all=True allows finding multiple faces in a single frame
        self.mtcnn = MTCNN(
            image_size=self.image_size,
            margin=self.margin,
            min_face_size=self.min_face_size,
            thresholds=[0.6, 0.7, 0.7],  # Internal 3-stage MTCNN network thresholds
            factor=0.709,
            post_process=False,
            keep_all=True,
            device=self.device
        )

    def _prepare_rgb_image(
        self,
        image: Union[np.ndarray, Image.Image],
        is_bgr: bool = True
    ) -> np.ndarray:
        """
        Validates input image and converts it into a standard RGB NumPy array.

        Args:
            image: Input image as a NumPy array (OpenCV) or PIL Image.
            is_bgr: If True and input is a NumPy array, treats it as OpenCV BGR format.

        Returns:
            np.ndarray: Image array in RGB format with shape (H, W, 3) and uint8 dtype.

        Raises:
            ValueError: If image is None or has invalid dimensions.
            TypeError: If image is not a NumPy array or PIL Image.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        # If user passed a PIL Image
        if isinstance(image, Image.Image):
            rgb_image = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.size == 0 or image.ndim not in (2, 3):
                raise ValueError(f"Invalid image array with shape: {getattr(image, 'shape', None)}")

            # Handle grayscale image (2D)
            if image.ndim == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Handle 3-channel image (OpenCV default is BGR)
            elif image.shape[2] == 3:
                if is_bgr:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image.copy()
            # Handle 4-channel image (BGRA / RGBA)
            elif image.shape[2] == 4:
                if is_bgr:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                raise ValueError(f"Unsupported number of image channels: {image.shape[2]}")
        else:
            raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(image)}")

        # Ensure image has non-zero height and width
        h, w = rgb_image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Image has invalid dimensions: height={h}, width={w}")

        return rgb_image

    def detect_faces(
        self,
        image: Union[np.ndarray, Image.Image],
        is_bgr: bool = True
    ) -> List[DetectionResult]:
        """
        Detects all faces in the given image.

        Args:
            image: OpenCV frame (BGR numpy array) or PIL Image.
            is_bgr: Set to True for standard OpenCV BGR webcam frames. Set to False if RGB.

        Returns:
            List[DetectionResult]: A list of detected faces containing:
                - box: (x1, y1, x2, y2) validated integer coordinates
                - confidence: float score between 0.0 and 1.0
                - face_crop: (160, 160, 3) RGB numpy array ready for FaceNet
            Returns an empty list [] if no faces are detected.
        """
        rgb_image = self._prepare_rgb_image(image, is_bgr=is_bgr)
        height, width = rgb_image.shape[:2]

        # MTCNN detect returns bounding boxes, confidence probabilities, and 5 facial landmarks
        detect_out = self.mtcnn.detect(rgb_image, landmarks=True)
        if len(detect_out) == 3:
            boxes, probs, points = detect_out
        else:
            boxes, probs = detect_out[:2]
            points = None

        # Handle the case where no face was found
        if boxes is None or probs is None:
            return []

        results: List[DetectionResult] = []

        for idx, (box, prob) in enumerate(zip(boxes, probs)):
            # Check for invalid or low-confidence detection
            if prob is None or prob < self.confidence_threshold:
                continue

            # Validate and clamp coordinates to remain safely within image boundaries
            x1 = max(0, int(round(box[0])))
            y1 = max(0, int(round(box[1])))
            x2 = min(width, int(round(box[2])))
            y2 = min(height, int(round(box[3])))

            box_w = x2 - x1
            box_h = y2 - y1

            # Ignore invalid or tiny bounding boxes
            if box_w < self.min_face_size or box_h < self.min_face_size:
                continue

            # Crop the detected face from the RGB image
            face_crop = rgb_image[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Resize face crop to 160x160 as expected by InceptionResnetV1
            crop_resized = cv2.resize(
                face_crop,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA
            )

            face_pts = None
            if points is not None and len(points) > idx and points[idx] is not None:
                face_pts = np.array(points[idx], dtype=np.float32)

            results.append(
                DetectionResult(
                    box=(x1, y1, x2, y2),
                    confidence=float(prob),
                    face_crop=crop_resized,
                    landmarks=face_pts
                )
            )

        return results
