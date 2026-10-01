"""MediaPipe face detection compatibility shim.

mediapipe >= 0.10.30 removed the legacy ``mp.solutions`` API. This module
returns a detector with the old ``FaceDetection.process()`` interface, backed
by the Tasks API (BlazeFace short-range) when ``solutions`` is unavailable,
with OpenCV YuNet as a fallback for far faces the short-range model misses.
"""

import os
import urllib.request
from types import SimpleNamespace

MODEL_DIR = os.path.expanduser("~/.shorts-skill/models")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")
# OpenCV YuNet catches small/far faces that BlazeFace short-range misses.
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")


def _ensure(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _ensure_model():
    return _ensure(MODEL_URL, MODEL_PATH)


def _box(xmin, ymin, width, height):
    rel = SimpleNamespace(xmin=xmin, ymin=ymin, width=width, height=height)
    return SimpleNamespace(location_data=SimpleNamespace(relative_bounding_box=rel))


class _YuNet:
    def __init__(self, min_detection_confidence):
        import cv2
        self._cv2 = cv2
        self._threshold = min_detection_confidence
        self._detector = None
        try:
            self._path = _ensure(YUNET_URL, YUNET_PATH)
        except Exception:
            self._path = None

    def detect(self, rgb):
        if self._path is None:
            return []
        h, w = rgb.shape[:2]
        if self._detector is None:
            self._detector = self._cv2.FaceDetectorYN.create(
                self._path, "", (w, h), score_threshold=self._threshold)
        self._detector.setInputSize((w, h))
        bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
        _, faces = self._detector.detect(bgr)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, fw, fh = (float(v) for v in f[:4])
            x, y = max(x, 0.0), max(y, 0.0)
            out.append(_box(x / w, y / h, min(fw, w - x) / w, min(fh, h - y) / h))
        return out


class _TasksFaceDetection:
    def __init__(self, min_detection_confidence=0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._fallback = _YuNet(min_detection_confidence)

    def process(self, rgb):
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)
        detections = [
            _box(d.bounding_box.origin_x / w, d.bounding_box.origin_y / h,
                 d.bounding_box.width / w, d.bounding_box.height / h)
            for d in result.detections
        ]
        if not detections:
            detections = self._fallback.detect(rgb)
        return SimpleNamespace(detections=detections or None)

    def close(self):
        self._detector.close()


def FaceDetection(model_selection=1, min_detection_confidence=0.5):
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        return mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )
    return _TasksFaceDetection(min_detection_confidence=min_detection_confidence)
