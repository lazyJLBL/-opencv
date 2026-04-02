from typing import Optional, Tuple

import numpy as np

from autodrive_lane.config import TemporalFilterConfig


class EMAFilter:
    """指数滑动平均滤波器。"""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.state: Optional[np.ndarray] = None

    def update(self, value: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if value is None:
            return self.state

        arr = np.asarray(value, dtype=np.float64)
        if self.state is None:
            self.state = arr
        else:
            self.state = self.alpha * arr + (1.0 - self.alpha) * self.state
        return self.state


class LaneTemporalSmoother:
    """车道几何和指标时序平滑器。"""

    def __init__(self, cfg: TemporalFilterConfig):
        self.cfg = cfg
        self.left_line_filter = EMAFilter(cfg.line_alpha)
        self.right_line_filter = EMAFilter(cfg.line_alpha)
        self.vp_filter = EMAFilter(cfg.line_alpha)

        self.left_curve_filter = EMAFilter(cfg.curve_alpha)
        self.right_curve_filter = EMAFilter(cfg.curve_alpha)

        self.width_filter = EMAFilter(cfg.metric_alpha)
        self.curvature_filter = EMAFilter(cfg.metric_alpha)
        self.offset_filter = EMAFilter(cfg.metric_alpha)

        self.last_lane_width_px: Optional[float] = None

    def update_lines(
        self,
        left_model: Optional[Tuple[float, float]],
        right_model: Optional[Tuple[float, float]],
        vanishing_point: Optional[Tuple[float, float]],
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        left_state = self.left_line_filter.update(np.array(left_model) if left_model is not None else None)
        right_state = self.right_line_filter.update(np.array(right_model) if right_model is not None else None)
        vp_state = self.vp_filter.update(np.array(vanishing_point) if vanishing_point is not None else None)

        left_out = tuple(left_state.tolist()) if left_state is not None else None
        right_out = tuple(right_state.tolist()) if right_state is not None else None
        vp_out = tuple(vp_state.tolist()) if vp_state is not None else None
        return left_out, right_out, vp_out

    def update_curves(
        self,
        left_curve: Optional[np.ndarray],
        right_curve: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        left_state = self.left_curve_filter.update(left_curve)
        right_state = self.right_curve_filter.update(right_curve)
        return left_state, right_state

    def update_metrics(
        self,
        lane_width_px: Optional[float],
        curvature_m: Optional[float],
        offset_m: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        validated_width = self._validate_lane_width(lane_width_px)

        width_state = self.width_filter.update(np.array([validated_width]) if validated_width is not None else None)
        curvature_state = self.curvature_filter.update(np.array([curvature_m]) if curvature_m is not None else None)
        offset_state = self.offset_filter.update(np.array([offset_m]) if offset_m is not None else None)

        width_out = float(width_state[0]) if width_state is not None else None
        curvature_out = float(curvature_state[0]) if curvature_state is not None else None
        offset_out = float(offset_state[0]) if offset_state is not None else None
        return width_out, curvature_out, offset_out

    def _validate_lane_width(self, lane_width_px: Optional[float]) -> Optional[float]:
        if lane_width_px is None:
            return self.last_lane_width_px

        if self.last_lane_width_px is None:
            self.last_lane_width_px = lane_width_px
            return lane_width_px

        jump_ratio = abs(lane_width_px - self.last_lane_width_px) / max(self.last_lane_width_px, 1e-6)
        if jump_ratio > self.cfg.max_lane_width_jump_ratio:
            return self.last_lane_width_px

        self.last_lane_width_px = lane_width_px
        return lane_width_px
