from typing import Optional, Tuple

import numpy as np


Point = Tuple[float, float]


def line_from_points(p1: Point, p2: Point, eps: float = 1e-9) -> Optional[np.ndarray]:
    """Return normalized homogeneous line ax + by + c = 0 from two points."""
    p1_h = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
    p2_h = np.array([p2[0], p2[1], 1.0], dtype=np.float64)
    line = np.cross(p1_h, p2_h)
    norm = float(np.hypot(line[0], line[1]))
    if norm < eps:
        return None
    return line / norm


def line_from_slope_intercept(slope: float, intercept: float) -> np.ndarray:
    """Convert y = m*x + b to homogeneous line."""
    line = np.array([slope, -1.0, intercept], dtype=np.float64)
    norm = float(np.hypot(line[0], line[1]))
    if norm <= 1e-9:
        return line
    return line / norm


def intersection(line1: np.ndarray, line2: np.ndarray, eps: float = 1e-9) -> Optional[Point]:
    """Get Euclidean intersection point of two homogeneous lines."""
    p_h = np.cross(line1, line2)
    if abs(float(p_h[2])) < eps:
        return None
    return float(p_h[0] / p_h[2]), float(p_h[1] / p_h[2])


def point_line_distance(point: Point, line: np.ndarray) -> float:
    """Compute signed distance between a point and homogeneous line."""
    x, y = point
    a, b, c = line
    return float(a * x + b * y + c)


def angle_between_lines(line1: np.ndarray, line2: np.ndarray) -> float:
    """Compute angle in degrees between two lines in [0, 180]."""
    # Direction vector of ax + by + c = 0 is (b, -a)
    d1 = np.array([line1[1], -line1[0]], dtype=np.float64)
    d2 = np.array([line2[1], -line2[0]], dtype=np.float64)
    n1 = float(np.linalg.norm(d1))
    n2 = float(np.linalg.norm(d2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_v = float(np.dot(d1, d2) / (n1 * n2))
    cos_v = float(np.clip(cos_v, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_v)))


def segment_length(p1: Point, p2: Point) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return float(np.hypot(x2 - x1, y2 - y1))


def x_at_y(slope: float, intercept: float, y: float, eps: float = 1e-9) -> Optional[float]:
    """Compute x from y = m*x + b."""
    if abs(slope) < eps:
        return None
    return float((y - intercept) / slope)
