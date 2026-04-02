import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autodrive_lane.geometry.metrics import curvature_radius_from_poly
from autodrive_lane.geometry.primitives import intersection, line_from_points, point_line_distance
from autodrive_lane.perception.lane_modeling import robust_fit_line_model


class TestGeometryPrimitives(unittest.TestCase):
    def test_line_intersection(self) -> None:
        l1 = line_from_points((0, 0), (10, 10))
        l2 = line_from_points((0, 10), (10, 0))
        self.assertIsNotNone(l1)
        self.assertIsNotNone(l2)
        p = intersection(l1, l2)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p[0], 5.0, places=5)
        self.assertAlmostEqual(p[1], 5.0, places=5)

    def test_point_line_distance(self) -> None:
        line = line_from_points((0, 0), (0, 10))
        self.assertIsNotNone(line)
        dist = point_line_distance((3, 5), line)
        self.assertAlmostEqual(abs(dist), 3.0, places=5)


class TestMetrics(unittest.TestCase):
    def test_curvature_radius(self) -> None:
        # x(y) = 0.01y^2 has known finite curvature.
        poly = np.array([0.01, 0.0, 0.0], dtype=np.float64)
        radius = curvature_radius_from_poly(poly, y_eval_m=10.0)
        self.assertIsNotNone(radius)
        self.assertGreater(radius, 10.0)


class TestRobustFit(unittest.TestCase):
    def test_robust_fit_with_outliers(self) -> None:
        rng = np.random.default_rng(0)
        inliers = []
        for x in np.linspace(100, 200, 30):
            y = -1.2 * x + 400 + rng.normal(0, 2.0)
            inliers.append([x, y, x + 5, y - 6])

        outliers = [[50, 50, 400, 300], [300, 120, 320, 500], [450, 100, 460, 520]]
        segments = np.array(inliers + outliers, dtype=np.float32)

        model = robust_fit_line_model(segments)
        self.assertIsNotNone(model)
        m, b = model
        self.assertAlmostEqual(m, -1.2, delta=0.2)
        self.assertAlmostEqual(b, 400.0, delta=40.0)


if __name__ == "__main__":
    unittest.main()
