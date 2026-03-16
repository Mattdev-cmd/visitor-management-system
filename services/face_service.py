"""
Face detection & recognition service.
Uses the *face_recognition* library (dlib under the hood) which is
lightweight enough for Raspberry Pi 5 with the HOG model.
"""

import pickle
from typing import Optional

import cv2
import face_recognition
import numpy as np

import config
from database import get_db


# ---------------------------------------------------------------------- #
#  Detection / Encoding
# ---------------------------------------------------------------------- #

def detect_faces(frame_bgr: np.ndarray):
    """Return a list of (top, right, bottom, left) bounding boxes."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model=config.FACE_RECOGNITION_MODEL)
    return locations


def encode_face(frame_bgr: np.ndarray, face_location=None) -> Optional[np.ndarray]:
    """Return the 128-d encoding for the first (or given) face in *frame_bgr*."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if face_location is not None:
        locations = [face_location]
    else:
        locations = face_recognition.face_locations(rgb, model=config.FACE_RECOGNITION_MODEL)
    if not locations:
        return None
    encodings = face_recognition.face_encodings(rgb, locations)
    return encodings[0] if encodings else None


# ---------------------------------------------------------------------- #
#  Matching
# ---------------------------------------------------------------------- #

def _load_known_faces():
    """Load all stored face encodings from the DB."""
    ids, encodings = [], []
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, face_encoding FROM visitors WHERE face_encoding IS NOT NULL"
        ).fetchall()
    for row in rows:
        enc = pickle.loads(row["face_encoding"])
        ids.append(row["id"])
        encodings.append(enc)
    return ids, encodings


def identify_visitor(frame_bgr: np.ndarray, face_location=None) -> Optional[int]:
    """Try to match a face in *frame_bgr* against the database.

    Returns the visitor ID on match, or ``None``.
    """
    encoding = encode_face(frame_bgr, face_location)
    if encoding is None:
        return None

    ids, known_encodings = _load_known_faces()
    if not known_encodings:
        return None

    distances = face_recognition.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    if distances[best_idx] <= config.FACE_RECOGNITION_TOLERANCE:
        return ids[best_idx]
    return None


def serialize_encoding(encoding: np.ndarray) -> bytes:
    """Convert a numpy encoding to bytes for DB storage."""
    return pickle.dumps(encoding)


def crop_face(frame_bgr: np.ndarray, face_location) -> np.ndarray:
    """Crop the face region from a BGR frame."""
    top, right, bottom, left = face_location
    return frame_bgr[top:bottom, left:right]
