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
        # Try PiCamera2 first if backend is picamera
        if self._backend == "picamera":
            try:
                from picamera2 import Picamera2          # type: ignore
                self._picam = Picamera2()
                cam_config = self._picam.create_still_configuration(
                    main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)}
                )
                self._picam.configure(cam_config)
                self._picam.start()
                print("[Camera] Using PiCamera2 backend.")
                return
            except Exception as e:
                print(f"[Camera] PiCamera2 failed: {e}\nFalling back to OpenCV...")
                self._backend = "opencv"
        # Try OpenCV with V4L2 backend
        try:
            self._open_opencv()
            print("[Camera] Using OpenCV backend.")
        except Exception as e:
            print(f"[Camera] OpenCV failed: {e}")
            raise RuntimeError("Cannot open camera with either PiCamera2 or OpenCV. Please check camera connection, permissions, and config.")

    def _open_opencv(self):
        # Try V4L2 backend first (best for Pi)
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not self._cap.isOpened():
            # Try default backend as fallback
            self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            if not self._cap.isOpened():
                raise RuntimeError("Cannot open camera (OpenCV, all backends)")

    def close(self):
        if self._cap is not None:
            from picamera2 import Picamera2
            import cv2

            class Camera:
                def __init__(self, width=640, height=480):
                    self.picam = Picamera2()
                    config = self.picam.create_preview_configuration(main={"size": (width, height)})
                    self.picam.configure(config)
                    self.picam.start()

                def get_frame(self):
                    frame = self.picam.capture_array()
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    return jpeg.tobytes()
            ok, frame = self._cap.read()
