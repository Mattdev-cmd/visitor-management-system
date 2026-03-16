"""
Camera abstraction — supports PiCamera2 (Raspberry Pi) and OpenCV VideoCapture.
"""

import cv2
import numpy as np
import config


class Camera:
    """Unified camera interface."""

    def __init__(self):
        self._cap = None
        self._picam = None
        self._backend = config.CAMERA_BACKEND

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def open(self):
        if self._backend == "picamera":
            try:
                from picamera2 import Picamera2          # type: ignore
                self._picam = Picamera2()
                cam_config = self._picam.create_still_configuration(
                    main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)}
                )
                self._picam.configure(cam_config)
                self._picam.start()
            except ImportError:
                print("[Camera] picamera2 not available — falling back to OpenCV")
                self._backend = "opencv"
                self._open_opencv()
        else:
            self._open_opencv()

    def _open_opencv(self):
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera (OpenCV)")

    def close(self):
        if self._cap is not None:
            self._cap.release()
        if self._picam is not None:
            self._picam.close()

    # ------------------------------------------------------------------ #
    #  Capture
    # ------------------------------------------------------------------ #
    def capture_frame(self) -> np.ndarray:
        """Return a BGR numpy frame."""
        if self._backend == "picamera" and self._picam is not None:
            # picamera2 returns RGB — convert to BGR for OpenCV compat
            frame_rgb = self._picam.capture_array()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")
            return frame
        raise RuntimeError("Camera not initialised — call open() first")

    # ------------------------------------------------------------------ #
    #  MJPEG streaming generator
    # ------------------------------------------------------------------ #
    def generate_mjpeg(self):
        """Yield MJPEG frames for Flask streaming response."""
        while True:
            frame = self.capture_frame()
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
