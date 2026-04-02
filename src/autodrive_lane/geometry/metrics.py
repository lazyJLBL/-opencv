from typing import Optional

import numpy as np


def lane_width_px(left_x: float, right_x: float) -> Optional[float]:
    width = float(right_x - left_x)
    if width <= 1.0:
        return None
    return width


def fit_centerline_polynomial(
    y_px: np.ndarray,
    left_x_px: np.ndarray,
    right_x_px: np.ndarray,
    xm_per_px: float,
    ym_per_px: float,
) -> Optional[np.ndarray]:
    """Fit x(y) = Ay^2 + By + C in meter space for lane centerline."""
    if len(y_px) < 3:
        return None
    center_x_px = 0.5 * (left_x_px + right_x_px)
    y_m = y_px * ym_per_px
    x_m = center_x_px * xm_per_px
    return np.polyfit(y_m, x_m, deg=2)


def curvature_radius_from_poly(poly_coeff: np.ndarray, y_eval_m: float) -> Optional[float]:
    """Compute curvature radius for x(y) polynomial in meter coordinates."""
    if poly_coeff is None or len(poly_coeff) != 3:
        return None
    a, b, _ = poly_coeff
    denom = abs(2.0 * a)
    if denom < 1e-9:
        # Near-zero second derivative means almost straight centerline.
        return 1e6
    return float(((1.0 + (2.0 * a * y_eval_m + b) ** 2) ** 1.5) / denom)


def lateral_offset_m(
    image_width_px: int,
    left_x_bottom_px: float,
    right_x_bottom_px: float,
    xm_per_px: float,
) -> Optional[float]:
    lane_center_x = 0.5 * (left_x_bottom_px + right_x_bottom_px)
    camera_center_x = 0.5 * image_width_px
    return float((camera_center_x - lane_center_x) * xm_per_px)


def lane_width_m_from_ground(left_ground: np.ndarray, right_ground: np.ndarray) -> Optional[float]:
    if left_ground is None or right_ground is None:
        return None
    if len(left_ground) == 0 or len(right_ground) == 0:
        return None

    distances = np.linalg.norm(right_ground - left_ground, axis=1)
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        return None
    return float(np.median(distances))


def fit_centerline_ground_poly(center_ground: np.ndarray) -> Optional[np.ndarray]:
    """Fit X(Z)=aZ^2+bZ+c in ground plane coordinates."""
    if center_ground is None or len(center_ground) < 3:
        return None

    x = center_ground[:, 0]
    z = center_ground[:, 1]
    valid = np.isfinite(x) & np.isfinite(z)
    if int(np.sum(valid)) < 3:
        return None

    return np.polyfit(z[valid], x[valid], deg=2)


def curvature_radius_xz(poly_coeff: np.ndarray, z_eval_m: float, straight_radius_m: float = 1e6) -> Optional[float]:
    if poly_coeff is None or len(poly_coeff) != 3:
        return None
    a, b, _ = poly_coeff
    denom = abs(2.0 * a)
    if denom < 1e-9:
        return float(straight_radius_m)
    return float(((1.0 + (2.0 * a * z_eval_m + b) ** 2) ** 1.5) / denom)


def lateral_offset_from_ground(left_bottom_ground: np.ndarray, right_bottom_ground: np.ndarray) -> Optional[float]:
    if left_bottom_ground is None or right_bottom_ground is None:
        return None

    lane_center_x = 0.5 * (float(left_bottom_ground[0]) + float(right_bottom_ground[0]))
    # Camera is assumed at X=0 in world coordinates.
    return float(-lane_center_x)
