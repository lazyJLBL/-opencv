import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autodrive_lane.calibration import CameraModel
from autodrive_lane.perception.feature_scene import analyze_scene_condition
from autodrive_lane.perception.lane_modeling import fit_lane_curve_model
from autodrive_lane.config import LaneModelConfig, SceneGuardConfig


class TestCameraProjection(unittest.TestCase):
    def test_pixel_to_ground_validity(self) -> None:
        cam = CameraModel(
            calibration=None,
            camera_height_m=1.45,
            pitch_deg=4.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            fallback_fov_deg=70.0,
        )

        points = np.array([[400.0, 600.0], [880.0, 600.0]], dtype=np.float64)
        ground, valid = cam.pixel_to_ground(points, image_size=(1280, 720))

        self.assertTrue(bool(np.all(valid)))
        self.assertGreater(float(ground[0, 1]), 0.0)
        self.assertGreater(float(ground[1, 1]), 0.0)
        self.assertLess(float(ground[0, 0]), float(ground[1, 0]))


class TestCurveModel(unittest.TestCase):
    def test_fit_lane_curve(self) -> None:
        h, w = 540, 960
        mask = np.zeros((h, w), dtype=np.uint8)

        ys = np.arange(int(h * 0.55), h)
        left_x = 0.0008 * (ys - 300) ** 2 + 220
        right_x = 0.0008 * (ys - 300) ** 2 + 620

        for x, y in zip(left_x.astype(int), ys.astype(int)):
            cv2.circle(mask, (x, y), 3, 255, -1)
        for x, y in zip(right_x.astype(int), ys.astype(int)):
            cv2.circle(mask, (x, y), 3, 255, -1)

        lane_cfg = LaneModelConfig()
        left_poly = fit_lane_curve_model(mask, "left", y_top=int(h * 0.55), y_bottom=h - 1, cfg=lane_cfg)
        right_poly = fit_lane_curve_model(mask, "right", y_top=int(h * 0.55), y_bottom=h - 1, cfg=lane_cfg)

        self.assertIsNotNone(left_poly)
        self.assertIsNotNone(right_poly)


class TestSceneGuard(unittest.TestCase):
    def test_night_scene_detection(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 20)
        scene = analyze_scene_condition(frame, SceneGuardConfig())
        self.assertEqual(scene.label, "night")


if __name__ == "__main__":
    unittest.main()
