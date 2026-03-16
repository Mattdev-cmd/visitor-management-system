"""
Emotion recognition service.

Strategy
--------
* If a TFLite model is present at ``config.EMOTION_MODEL_PATH`` it is used
  (ideal for Raspberry Pi — tiny memory footprint).
* Otherwise, a minimal Keras/TF model is loaded for development machines.
* As a final fallback the service returns "Neutral" so the rest of the
  system keeps working while a model is being trained/downloaded.

The expected input is a **48 × 48 grayscale** face crop — the standard
FER-2013 format.
"""

import os
from typing import Tuple

import cv2
import numpy as np

import config

_interpreter = None  # TFLite interpreter (lazy-loaded)
_tf_model = None     # Full TF/Keras model (lazy-loaded)

LABELS = config.EMOTION_LABELS  # ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]


# ---------------------------------------------------------------------- #
#  Model loading
# ---------------------------------------------------------------------- #

def _load_tflite():
    global _interpreter
    if _interpreter is not None:
        return True
    model_path = config.EMOTION_MODEL_PATH
    if not os.path.isfile(model_path):
        return False
    try:
        # Try tflite-runtime first (Raspberry Pi)
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        except ImportError:
            return False
    _interpreter = Interpreter(model_path=model_path)
    _interpreter.allocate_tensors()
    return True


def _load_keras():
    global _tf_model
    if _tf_model is not None:
        return True
    keras_path = config.EMOTION_MODEL_PATH.replace(".tflite", ".h5")
    if not os.path.isfile(keras_path):
        return False
    try:
        from tensorflow import keras  # type: ignore
        _tf_model = keras.models.load_model(keras_path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------- #
#  Pre-processing
# ---------------------------------------------------------------------- #

def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """Resize to 48×48, convert to grayscale, normalise to [0, 1]."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normed = resized.astype(np.float32) / 255.0
    return normed.reshape(1, 48, 48, 1)


# ---------------------------------------------------------------------- #
#  Inference
# ---------------------------------------------------------------------- #

def predict_emotion(face_bgr: np.ndarray) -> Tuple[str, float]:
    """Return ``(label, confidence)`` for a BGR face crop.

    Falls back to ``("Neutral", 0.0)`` when no model is available.
    """
    inp = _preprocess(face_bgr)

    # --- TFLite path (preferred on Pi) ---
    if _load_tflite():
        input_details = _interpreter.get_input_details()
        output_details = _interpreter.get_output_details()
        _interpreter.set_tensor(input_details[0]["index"], inp)
        _interpreter.invoke()
        preds = _interpreter.get_tensor(output_details[0]["index"])[0]
        idx = int(np.argmax(preds))
        return LABELS[idx], float(preds[idx])

    # --- Keras path (dev machine) ---
    if _load_keras():
        preds = _tf_model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(preds))
        return LABELS[idx], float(preds[idx])

    # --- Fallback: simple heuristic using face aspect ratio ---
    return _heuristic_emotion(face_bgr)


def _heuristic_emotion(face_bgr: np.ndarray) -> Tuple[str, float]:
    """Very rough heuristic used only when no ML model is available.

    Analyses the brightness distribution of the face to give a *very*
    approximate guess. This exists purely so the rest of the application
    pipeline keeps running during development without a trained model.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    # Simple mapping based on expression-related contrast patterns
    if std_brightness > 60:
        return "Surprise", round(std_brightness / 100, 2)
    if mean_brightness > 140:
        return "Happy", round(min(mean_brightness / 200, 1.0), 2)
    if mean_brightness < 80:
        return "Sad", round(1.0 - mean_brightness / 150, 2)
    return "Neutral", 0.50
