import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mp_face_compat


class FaceDetectionSelectionTests(unittest.TestCase):
    def test_darwin_uses_yunet_without_constructing_mediapipe_tasks(self):
        yunet = object()
        fake_mp = SimpleNamespace()
        with (
            patch.object(sys, "platform", "darwin"),
            patch.dict(sys.modules, {"mediapipe": fake_mp}),
            patch.object(mp_face_compat, "_YuNet", return_value=yunet) as yunet_factory,
            patch.object(mp_face_compat, "_TasksFaceDetection") as tasks_factory,
        ):
            detector = mp_face_compat.FaceDetection(min_detection_confidence=0.7)

        self.assertIs(detector, yunet)
        yunet_factory.assert_called_once_with(0.7)
        tasks_factory.assert_not_called()

    def test_yunet_supports_legacy_process_and_close_interface(self):
        detector = object.__new__(mp_face_compat._YuNet)
        detection = object()
        detector.detect = Mock(return_value=[detection])

        result = detector.process(object())

        self.assertEqual(result.detections, [detection])
        detector.close()


if __name__ == "__main__":
    unittest.main()
