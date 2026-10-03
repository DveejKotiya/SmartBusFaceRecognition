"""
Face Recognizer & Embedding Extractor Module
Smart Bus Face Recognition and Pass Verification System

This module uses a pretrained deep neural network (InceptionResnetV1 trained on VGGFace2)
from facenet-pytorch to extract 512-dimensional facial embedding vectors from cropped faces.

IMPORTANT:
The model is used strictly in inference (evaluation) mode as a feature extractor.
No training or fine-tuning is performed in this module.
"""

from typing import List, Union, Optional
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization


class FaceRecognizer:
    """
    Extracts 512-dimensional deep facial embeddings and computes similarity metrics.

    Features:
    - Pretrained InceptionResnetV1 (VGGFace2).
    - Runs in evaluation mode (inference only, no gradient tracking).
    - Automatically uses CUDA GPU if available, else CPU.
    - Consistent L2-normalization of output embeddings.
    - Cosine similarity computation between face embeddings.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Initializes the FaceRecognizer with pretrained weights.

        Args:
            device: 'cuda', 'cpu', or None (auto-selects CUDA if available).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load pretrained InceptionResnetV1 model (VGGFace2 dataset weights)
        # .eval() disables dropout and batch-norm updates (inference mode)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def _prepare_face_tensor(self, face: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Validates and formats a single cropped face into a normalized (3, 160, 160) float tensor.

        Args:
            face: Cropped face as RGB NumPy array, PIL Image, or PyTorch Tensor.

        Returns:
            torch.Tensor: Normalized float tensor of shape (3, 160, 160).

        Raises:
            ValueError: If input is None, empty, or has invalid dimensions.
            TypeError: If input is not a recognized image type.
        """
        if face is None:
            raise ValueError("Input face cannot be None.")

        # Convert PIL Image to NumPy array
        if isinstance(face, Image.Image):
            face_arr = np.array(face.convert("RGB"))
        elif isinstance(face, np.ndarray):
            face_arr = face.copy()
        elif isinstance(face, torch.Tensor):
            # If already a tensor, convert to numpy for standardized preprocessing
            if face.ndim == 3 and face.shape[0] == 3:  # (3, H, W)
                face_arr = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                face_arr = face.cpu().numpy().astype(np.uint8)
        else:
            raise TypeError(f"Expected numpy.ndarray, PIL.Image, or torch.Tensor, got {type(face)}")

        # Validate array
        if face_arr.size == 0 or face_arr.ndim != 3 or face_arr.shape[2] != 3:
            raise ValueError(f"Face crop must have shape (H, W, 3), got shape {getattr(face_arr, 'shape', None)}")

        # Resize to exactly 160x160 if not already that size
        if face_arr.shape[0] != 160 or face_arr.shape[1] != 160:
            face_arr = cv2.resize(face_arr, (160, 160), interpolation=cv2.INTER_AREA)

        # Convert (160, 160, 3) NumPy array to (3, 160, 160) float tensor
        tensor = torch.from_numpy(face_arr).permute(2, 0, 1).float()

        # Apply standard FaceNet normalization: (x - 127.5) / 128.0
        normalized_tensor = fixed_image_standardization(tensor)

        return normalized_tensor

    def generate_embedding(self, face: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
        """
        Generates a normalized 512-dimensional embedding vector for a single cropped face.

        Args:
            face: Cropped face image (RGB format).

        Returns:
            np.ndarray: 1D NumPy array of shape (512,) with dtype float32, normalized to unit length.
        """
        tensor = self._prepare_face_tensor(face)
        # Add batch dimension: (1, 3, 160, 160)
        batch = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_embedding = self.model(batch)

        # Extract 1D array on CPU
        emb = raw_embedding.squeeze(0).cpu().numpy().astype(np.float32)

        # Ensure embedding is strictly L2-normalized (unit length)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb

    def generate_embeddings(
        self,
        faces: List[Union[np.ndarray, Image.Image, torch.Tensor]]
    ) -> List[np.ndarray]:
        """
        Generates embeddings for a batch or list of cropped faces.

        Args:
            faces: List of cropped face images.

        Returns:
            List[np.ndarray]: List of 512-dimensional normalized embeddings.
        """
        if not faces:
            return []

        embeddings: List[np.ndarray] = []
        for face in faces:
            embeddings.append(self.generate_embedding(face))

        return embeddings

    def compare_embeddings(
        self,
        embedding1: Union[np.ndarray, List[float]],
        embedding2: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Computes the cosine similarity between two 512-dimensional face embeddings.
        """
        return compare_embeddings(embedding1, embedding2)



def compare_embeddings(
    embedding1: Union[np.ndarray, List[float]],
    embedding2: Union[np.ndarray, List[float]]
) -> float:
    """
    Computes the cosine similarity between two 512-dimensional face embeddings.

    Cosine similarity = (A · B) / (||A|| * ||B||)

    Args:
        embedding1: First 512-dimensional embedding.
        embedding2: Second 512-dimensional embedding.

    Returns:
        float: Similarity score between -1.0 and 1.0.
               Values near 1.0 indicate high facial similarity (likely the same person).
               Values near 0.0 or negative indicate different people.

    Raises:
        ValueError: If an embedding is None, not 512-dimensional, or contains NaN/infinite values.
        TypeError: If an embedding is not an array or list.
    """
    if embedding1 is None or embedding2 is None:
        raise ValueError("Embeddings cannot be None.")

    # Convert lists to NumPy arrays
    try:
        e1 = np.asarray(embedding1, dtype=np.float32).flatten()
        e2 = np.asarray(embedding2, dtype=np.float32).flatten()
    except Exception as e:
        raise TypeError(f"Could not convert input to numeric array: {e}")

    # Validate dimensions
    if e1.shape != (512,) or e2.shape != (512,):
        raise ValueError(
            f"Embeddings must be 512-dimensional. Got shapes {e1.shape} and {e2.shape}."
        )

    # Check for NaN or Inf
    if not np.all(np.isfinite(e1)) or not np.all(np.isfinite(e2)):
        raise ValueError("Embeddings must contain finite numeric values (no NaNs or Infs).")

    # Compute cosine similarity
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    dot_product = np.dot(e1, e2)
    similarity = float(dot_product / (norm1 * norm2))

    # Clamp safely between -1.0 and 1.0 to guard against floating-point rounding
    return max(-1.0, min(1.0, similarity))
