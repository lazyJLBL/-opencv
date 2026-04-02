from typing import Optional, Tuple

import cv2
import numpy as np

from autodrive_lane.config import HoughConfig, LaneModelConfig, ROIConfig


# =========================
# ROI + 线段提取（合并原 line_segments.py）
# =========================
def build_roi_polygon(
    width: int,
    height: int,
    cfg: ROIConfig,
    vanishing_point: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """根据消失点动态构建道路 ROI。"""
    top_y = int(height * cfg.top_y_ratio)
    left_top_x = int(width * cfg.left_top_x_ratio)
    right_top_x = int(width * cfg.right_top_x_ratio)

    if vanishing_point is not None:
        vx, vy = vanishing_point
        vp_window = int(width * 0.11)
        left_top_x = int(np.clip(vx - vp_window, 0, width - 1))
        right_top_x = int(np.clip(vx + vp_window, 0, width - 1))
        top_y = int(np.clip(vy + 20, height * cfg.min_top_y_ratio, height * cfg.max_top_y_ratio))

    return np.array(
        [
            [int(width * cfg.left_bottom_x_ratio), height - 1],
            [left_top_x, top_y],
            [right_top_x, top_y],
            [int(width * cfg.right_bottom_x_ratio), height - 1],
        ],
        dtype=np.int32,
    )


def apply_roi_mask(binary: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(binary, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(binary, mask)


def detect_segments(roi_binary: np.ndarray, cfg: HoughConfig) -> np.ndarray:
    lines = cv2.HoughLinesP(
        roi_binary,
        rho=cfg.rho,
        theta=cfg.theta,
        threshold=cfg.threshold,
        minLineLength=cfg.min_line_length,
        maxLineGap=cfg.max_line_gap,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def split_left_right(segments: np.ndarray, image_width: int, min_abs_slope: float) -> Tuple[np.ndarray, np.ndarray]:
    if segments.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    center_x = 0.5 * image_width
    left_segments = []
    right_segments = []

    for x1, y1, x2, y2 in segments:
        dx = float(x2 - x1)
        if abs(dx) < 1e-6:
            continue
        slope = float((y2 - y1) / dx)
        if abs(slope) < min_abs_slope:
            continue

        x_mid = 0.5 * (x1 + x2)
        if slope < 0 and x_mid < center_x * 1.05:
            left_segments.append([x1, y1, x2, y2])
        elif slope > 0 and x_mid > center_x * 0.95:
            right_segments.append([x1, y1, x2, y2])

    left_arr = np.array(left_segments, dtype=np.float32) if left_segments else np.empty((0, 4), dtype=np.float32)
    right_arr = np.array(right_segments, dtype=np.float32) if right_segments else np.empty((0, 4), dtype=np.float32)
    return left_arr, right_arr


# =========================
# 鲁棒直线拟合（合并原 robust_fit.py）
# =========================
def _collect_points_from_segments(segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if segments.size == 0:
        return np.array([]), np.array([]), np.array([])

    x_list = []
    y_list = []
    w_list = []
    for x1, y1, x2, y2 in segments:
        length = float(np.hypot(x2 - x1, y2 - y1))
        weight = max(length, 1.0)
        x_list.extend([x1, x2])
        y_list.extend([y1, y2])
        w_list.extend([weight, weight])

    return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64), np.array(w_list, dtype=np.float64)


def robust_fit_line_model(
    segments: np.ndarray,
    residual_threshold_px: float = 18.0,
    max_iter: int = 4,
    min_points: int = 6,
) -> Optional[Tuple[float, float]]:
    """鲁棒拟合直线模型 y=mx+b。"""
    if segments.size == 0:
        return None

    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    dx = segments[:, 2] - segments[:, 0]
    segments = segments[np.abs(dx) > 1e-6]
    if len(segments) == 0:
        return None

    slopes = (segments[:, 3] - segments[:, 1]) / (segments[:, 2] - segments[:, 0])
    median_slope = float(np.median(slopes))
    filtered = segments[(np.sign(slopes) == np.sign(median_slope)) & (np.abs(slopes - median_slope) < 1.2)]
    if len(filtered) >= 2:
        segments = filtered

    x, y, w = _collect_points_from_segments(segments)
    if len(x) < min_points:
        return None

    best_mask = None
    best_score = -1.0
    best_model = None

    for seg in segments:
        x1, y1, x2, y2 = seg
        seg_dx = x2 - x1
        if abs(seg_dx) < 1e-6:
            continue
        m_candidate = (y2 - y1) / seg_dx
        b_candidate = y1 - m_candidate * x1

        residual = np.abs(y - (m_candidate * x + b_candidate))
        mask = residual <= residual_threshold_px
        score = float(np.sum(w[mask]))
        if score > best_score and int(np.sum(mask)) >= min_points:
            best_score = score
            best_mask = mask
            best_model = (float(m_candidate), float(b_candidate))

    if best_model is None:
        m, b = np.polyfit(x, y, deg=1, w=w)
        best_mask = np.ones_like(x, dtype=bool)
    else:
        m, b = best_model

    for _ in range(max_iter):
        residual = np.abs(y - (m * x + b))
        next_mask = residual <= residual_threshold_px
        if int(np.sum(next_mask)) < min_points:
            break
        if np.array_equal(next_mask, best_mask):
            break
        best_mask = next_mask
        m, b = np.polyfit(x[best_mask], y[best_mask], deg=1, w=w[best_mask])

    return float(m), float(b)


# =========================
# 曲线车道拟合（合并原 curve_model.py）
# =========================
def robust_fit_curve(y: np.ndarray, x: np.ndarray, weights: Optional[np.ndarray], cfg: LaneModelConfig) -> Optional[np.ndarray]:
    if len(x) < cfg.min_points:
        return None

    if weights is None:
        weights = np.ones_like(x, dtype=np.float64)

    mask = np.ones_like(x, dtype=bool)
    poly = np.polyfit(y, x, deg=2, w=weights)

    for _ in range(4):
        pred_x = np.polyval(poly, y)
        residual = np.abs(x - pred_x)
        next_mask = residual <= cfg.residual_threshold_px
        if int(np.sum(next_mask)) < cfg.min_points:
            break
        if np.array_equal(next_mask, mask):
            break
        mask = next_mask
        poly = np.polyfit(y[mask], x[mask], deg=2, w=weights[mask])

    return poly.astype(np.float64)


def fit_lane_curve_model(
    binary_mask: np.ndarray,
    side: str,
    y_top: int,
    y_bottom: int,
    cfg: LaneModelConfig,
    prior_curve: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None

    _, width = binary_mask.shape[:2]
    valid_y = (ys >= int(y_top)) & (ys <= int(y_bottom))
    side_mask = xs < int(width * 0.55) if side == "left" else xs > int(width * 0.45)

    mask = valid_y & side_mask
    ys_sel = ys[mask].astype(np.float64)
    xs_sel = xs[mask].astype(np.float64)
    if len(xs_sel) < cfg.min_points:
        return None

    if prior_curve is not None:
        pred = np.polyval(prior_curve, ys_sel)
        near_prior = np.abs(xs_sel - pred) <= cfg.search_margin_px
        if int(np.sum(near_prior)) >= cfg.min_points:
            ys_sel = ys_sel[near_prior]
            xs_sel = xs_sel[near_prior]

    y_span = max(float(y_bottom - y_top), 1.0)
    weights = 0.5 + (ys_sel - y_top) / y_span
    return robust_fit_curve(ys_sel, xs_sel, weights=weights, cfg=cfg)


def evaluate_curve_x(curve: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.polyval(curve, y)


def sample_curve_points(curve: np.ndarray, y_top: float, y_bottom: float, num: int = 40) -> np.ndarray:
    y_values = np.linspace(y_bottom, y_top, num=num, dtype=np.float64)
    x_values = evaluate_curve_x(curve, y_values)
    return np.stack([x_values, y_values], axis=1).astype(np.float32)


def curve_to_line_model(curve: np.ndarray, y_top: float, y_bottom: float) -> Optional[Tuple[float, float]]:
    x1 = float(np.polyval(curve, y_bottom))
    x2 = float(np.polyval(curve, y_top))
    y1 = float(y_bottom)
    y2 = float(y_top)
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return None
    slope = (y2 - y1) / dx
    intercept = y1 - slope * x1
    return float(slope), float(intercept)


def shift_curve(curve: np.ndarray, delta_x: float) -> np.ndarray:
    shifted = np.array(curve, dtype=np.float64)
    shifted[2] += float(delta_x)
    return shifted
