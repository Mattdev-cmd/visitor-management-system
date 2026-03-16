"""
Configuration for Visitor Management & Emotion Recognition System.
Optimized for Raspberry Pi 5 (2 GB RAM).
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Flask ----------
SECRET_KEY = os.environ.get("VMS_SECRET_KEY", os.urandom(32).hex())
DEBUG = os.environ.get("VMS_DEBUG", "false").lower() == "true"
HOST = os.environ.get("VMS_HOST", "0.0.0.0")
PORT = int(os.environ.get("VMS_PORT", 5000))

# ---------- Database ----------
DATABASE_PATH = os.path.join(BASE_DIR, "visitors.db")

# ---------- Uploads ----------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB

# ---------- Camera ----------
# Set to "picamera" on a Raspberry Pi with a CSI camera, or "opencv" for USB / dev
CAMERA_BACKEND = os.environ.get("VMS_CAMERA_BACKEND", "opencv")
CAMERA_INDEX = int(os.environ.get("VMS_CAMERA_INDEX", 0))
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ---------- Face recognition ----------
FACE_RECOGNITION_TOLERANCE = 0.5   # lower = stricter matching
FACE_RECOGNITION_MODEL = "hog"     # "hog" (CPU-friendly) or "cnn"

# ---------- Emotion recognition ----------
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.tflite")
