from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from autodrive_lane.calibration import CameraModel
from autodrive_lane.config import PipelineConfig
from autodrive_lane.geometry import (
    build_default_ipm,
    curvature_radius_xz,
    fit_centerline_ground_poly,
    intersection,
    lane_width_m_from_ground,
    lane_width_px,
    lateral_offset_from_ground,
    line_from_slope_intercept,
    warp,
)
from autodrive_lane.perception.feature_scene import (
    analyze_scene_condition,
    preprocess_lane_binary,
    should_degrade_mode,
)
from autodrive_lane.perception.lane_change_fsm import LaneChangeDecision, LaneChangeStateMachine
from autodrive_lane.perception.lane_modeling import (
    apply_roi_mask,
    build_roi_polygon,
    curve_to_line_model,
    detect_segments,
    evaluate_curve_x,
    fit_lane_curve_model,
    robust_fit_line_model,
    sample_curve_points,
    shift_curve,
    split_left_right,
)
from autodrive_lane.perception.overlay_renderer import render_overlay_frame
from autodrive_lane.perception.temporal_smoother import LaneTemporalSmoother


@dataclass
class LaneFrameResult:
    """单帧输出结果。"""

    frame_index: int
    input_frame: np.ndarray
    overlay_frame: np.ndarray
    binary_mask: np.ndarray
    roi_masked: np.ndarray
    bev_mask: np.ndarray
    roi_polygon: np.ndarray
    left_model: Optional[Tuple[float, float]]
    right_model: Optional[Tuple[float, float]]
    left_points: Optional[np.ndarray]
    right_points: Optional[np.ndarray]
    vanishing_point: Optional[Tuple[float, float]]
    lane_width_px: Optional[float]
    lane_width_m: Optional[float]
    curvature_m: Optional[float]
    offset_m: Optional[float]
    lane_change_state: str
    lane_change_confidence: float
    scene_label: str
    degraded: bool
    degradation_reasons: list[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


class LaneGeometryPipeline:
    """车道几何主流水线（重构版）。"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config if config is not None else PipelineConfig()

        self.camera_model = CameraModel.from_calibration_path(
            calibration_path=self.cfg.camera.calibration_path,
            camera_height_m=self.cfg.camera.camera_height_m,
            pitch_deg=self.cfg.camera.pitch_deg,
            yaw_deg=self.cfg.camera.yaw_deg,
            roll_deg=self.cfg.camera.roll_deg,
            fallback_fov_deg=self.cfg.camera.fallback_fov_deg,
        )
        self.temporal = LaneTemporalSmoother(self.cfg.temporal)
        self.lane_change_fsm = LaneChangeStateMachine(self.cfg.lane_change_fsm)

    def process(self, frame_bgr: np.ndarray, frame_index: int = 0) -> LaneFrameResult:
        # 1) 相机去畸变
        working_frame = self.camera_model.undistort(frame_bgr) if self.cfg.camera.enable_undistort else frame_bgr

        # 2) 场景识别 + 自适应预处理
        scene = analyze_scene_condition(working_frame, self.cfg.scene_guard)
        binary = preprocess_lane_binary(working_frame, self.cfg.preprocess, scene_label=scene.label)

        height, width = working_frame.shape[:2]
        y_bottom = float(height - 1)

        # 3) 线段先验与消失点
        roi_polygon = build_roi_polygon(width, height, self.cfg.roi)
        roi_masked = apply_roi_mask(binary, roi_polygon)

        segments = detect_segments(roi_masked, self.cfg.hough)
        left_segments, right_segments = split_left_right(
            segments,
            image_width=width,
            min_abs_slope=self.cfg.hough.min_abs_slope,
        )

        left_line_raw = robust_fit_line_model(left_segments)
        right_line_raw = robust_fit_line_model(right_segments)
        vp_raw = self._compute_vanishing_point(left_line_raw, right_line_raw)
        left_line, right_line, vp = self.temporal.update_lines(left_line_raw, right_line_raw, vp_raw)

        # 4) 依据消失点刷新 ROI 后做曲线拟合
        roi_polygon = build_roi_polygon(width, height, self.cfg.roi, vanishing_point=vp)
        y_top = float(np.min(roi_polygon[:, 1]))
        roi_masked = apply_roi_mask(binary, roi_polygon)

        left_prior = self.temporal.left_curve_filter.state
        right_prior = self.temporal.right_curve_filter.state

        left_curve_raw = fit_lane_curve_model(
            roi_masked,
            side="left",
            y_top=int(y_top),
            y_bottom=int(y_bottom),
            cfg=self.cfg.lane_model,
            prior_curve=left_prior,
        )
        right_curve_raw = fit_lane_curve_model(
            roi_masked,
            side="right",
            y_top=int(y_top),
            y_bottom=int(y_bottom),
            cfg=self.cfg.lane_model,
            prior_curve=right_prior,
        )

        inferred_reasons: list[str] = []
        lane_width_prior = self.temporal.last_lane_width_px
        if left_curve_raw is None and right_curve_raw is not None and lane_width_prior is not None:
            left_curve_raw = shift_curve(right_curve_raw, -lane_width_prior)
            inferred_reasons.append("left_inferred_from_right")
        if right_curve_raw is None and left_curve_raw is not None and lane_width_prior is not None:
            right_curve_raw = shift_curve(left_curve_raw, lane_width_prior)
            inferred_reasons.append("right_inferred_from_left")

        left_curve, right_curve = self.temporal.update_curves(left_curve_raw, right_curve_raw)

        left_points = sample_curve_points(left_curve, y_top=y_top, y_bottom=y_bottom) if left_curve is not None else None
        right_points = sample_curve_points(right_curve, y_top=y_top, y_bottom=y_bottom) if right_curve is not None else None

        left_model = curve_to_line_model(left_curve, y_top=y_top, y_bottom=y_bottom) if left_curve is not None else left_line
        right_model = curve_to_line_model(right_curve, y_top=y_top, y_bottom=y_bottom) if right_curve is not None else right_line

        # 5) 几何指标估计
        lane_w_px = None
        lane_w_m = None
        curvature = None
        offset = None
        bev_mask = np.zeros_like(roi_masked)

        if left_curve is not None and right_curve is not None and left_points is not None and right_points is not None:
            lane_w_px, lane_w_m, curvature, offset = self._estimate_metrics(
                y_bottom=y_bottom,
                y_top=y_top,
                image_size=(width, height),
                left_curve=left_curve,
                right_curve=right_curve,
            )

            src_quad = np.array([left_points[0], left_points[1], right_points[1], right_points[0]], dtype=np.float32)
            if self._valid_quad(src_quad, width, height):
                h_mat, _, _ = build_default_ipm(src_quad, (width, height))
                bev_mask = warp(roi_masked, h_mat, (width, height))

        lane_w_px, curvature, offset = self.temporal.update_metrics(lane_w_px, curvature, offset)

        # 6) 失效检测 + 变道状态机
        degraded, degradation_reasons = should_degrade_mode(
            scene=scene,
            segment_count=int(len(segments)),
            left_curve_found=left_curve_raw is not None,
            right_curve_found=right_curve_raw is not None,
            cfg=self.cfg.scene_guard,
        )
        degradation_reasons.extend(inferred_reasons)

        lane_change: LaneChangeDecision = self.lane_change_fsm.update(offset_m=offset, degraded=degraded)

        overlay = render_overlay_frame(
            frame_bgr=working_frame,
            roi_polygon=roi_polygon,
            left_points=left_points,
            right_points=right_points,
            vanishing_point=vp,
            lane_width_px=lane_w_px,
            lane_width_m=lane_w_m,
            curvature_m=curvature,
            offset_m=offset,
            scene_label=scene.label,
            lane_change_state=lane_change.state,
            lane_change_confidence=lane_change.confidence,
            degraded=degraded,
            degradation_reasons=degradation_reasons,
        )

        debug = {
            "segment_count": int(len(segments)),
            "left_segment_count": int(len(left_segments)),
            "right_segment_count": int(len(right_segments)),
            "scene_brightness": scene.brightness,
            "scene_contrast": scene.contrast,
            "scene_glare_ratio": scene.glare_ratio,
            "scene_edge_density": scene.edge_density,
            "camera_calibrated": self.camera_model.calibration is not None,
            "lane_change_left_score": lane_change.left_score,
            "lane_change_right_score": lane_change.right_score,
        }

        return LaneFrameResult(
            frame_index=frame_index,
            input_frame=working_frame,
            overlay_frame=overlay,
            binary_mask=binary,
            roi_masked=roi_masked,
            bev_mask=bev_mask,
            roi_polygon=roi_polygon,
            left_model=left_model,
            right_model=right_model,
            left_points=left_points,
            right_points=right_points,
            vanishing_point=vp,
            lane_width_px=lane_w_px,
            lane_width_m=lane_w_m,
            curvature_m=curvature,
            offset_m=offset,
            lane_change_state=lane_change.state,
            lane_change_confidence=lane_change.confidence,
            scene_label=scene.label,
            degraded=degraded,
            degradation_reasons=degradation_reasons,
            debug=debug,
        )

    @staticmethod
    def _compute_vanishing_point(
        left_model: Optional[Tuple[float, float]],
        right_model: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if left_model is None or right_model is None:
            return None
        left_line = line_from_slope_intercept(left_model[0], left_model[1])
        right_line = line_from_slope_intercept(right_model[0], right_model[1])
        return intersection(left_line, right_line)

    def _estimate_metrics(
        self,
        y_bottom: float,
        y_top: float,
        image_size: Tuple[int, int],
        left_curve: np.ndarray,
        right_curve: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        width, height = image_size

        left_bottom_x = float(np.polyval(left_curve, y_bottom))
        right_bottom_x = float(np.polyval(right_curve, y_bottom))
        lane_w_px = lane_width_px(left_bottom_x, right_bottom_x)
        if lane_w_px is None:
            return None, None, None, None

        y_samples = np.linspace(y_top, y_bottom, 55, dtype=np.float64)
        left_x = evaluate_curve_x(left_curve, y_samples)
        right_x = evaluate_curve_x(right_curve, y_samples)

        left_uv = np.stack([left_x, y_samples], axis=1)
        right_uv = np.stack([right_x, y_samples], axis=1)

        left_ground, left_valid = self.camera_model.pixel_to_ground(left_uv, image_size=(width, height))
        right_ground, right_valid = self.camera_model.pixel_to_ground(right_uv, image_size=(width, height))
        valid = left_valid & right_valid
        if int(np.sum(valid)) < 8:
            return lane_w_px, None, None, None

        left_ground = left_ground[valid]
        right_ground = right_ground[valid]

        lane_w_m = lane_width_m_from_ground(left_ground, right_ground)
        center_ground = 0.5 * (left_ground + right_ground)
        center_poly = fit_centerline_ground_poly(center_ground)
        z_eval = float(np.nanmax(center_ground[:, 1])) if len(center_ground) > 0 else 0.0
        curvature = curvature_radius_xz(
            center_poly,
            z_eval_m=z_eval,
            straight_radius_m=self.cfg.metric.straight_curvature_m,
        )

        offset = lateral_offset_from_ground(left_ground[-1], right_ground[-1])
        return lane_w_px, lane_w_m, curvature, offset

    @staticmethod
    def _valid_quad(quad: np.ndarray, width: int, height: int) -> bool:
        if quad.shape != (4, 2):
            return False
        if np.any(np.isnan(quad)):
            return False
        min_x, min_y = np.min(quad[:, 0]), np.min(quad[:, 1])
        max_x, max_y = np.max(quad[:, 0]), np.max(quad[:, 1])
        if max_x - min_x < 20 or max_y - min_y < 20:
            return False
        if max_x < -width or min_x > 2 * width:
            return False
        if max_y < -height or min_y > 2 * height:
            return False
        return True
