from typing import Tuple

import cv2
import numpy as np

from autodrive_lane.perception.lane_pipeline import LaneFrameResult


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def _metric_panel(result: LaneFrameResult, shape: Tuple[int, int, int]) -> np.ndarray:
    panel = np.zeros(shape, dtype=np.uint8)

    rows = [
        "Geometry Dashboard",
        f"Frame: {result.frame_index}",
        f"Scene: {result.scene_label}",
        f"Lane state: {result.lane_change_state}",
        f"Lane conf: {result.lane_change_confidence:.2f}",
        f"Segments: {result.debug.get('segment_count', 0)}",
        f"Left/Right: {result.debug.get('left_segment_count', 0)} / {result.debug.get('right_segment_count', 0)}",
        f"Camera calibrated: {result.debug.get('camera_calibrated', False)}",
        f"Lane width(px): {result.lane_width_px:.1f}" if result.lane_width_px is not None else "Lane width(px): NA",
        f"Lane width(m): {result.lane_width_m:.2f}" if result.lane_width_m is not None else "Lane width(m): NA",
        f"Curvature(m): {result.curvature_m:.1f}" if result.curvature_m is not None else "Curvature(m): NA",
        f"Offset(m): {result.offset_m:+.2f}" if result.offset_m is not None else "Offset(m): NA",
        f"Degraded: {result.degraded}",
    ]

    if result.degradation_reasons:
        rows.append("Reasons: " + ",".join(result.degradation_reasons[:2]))

    for idx, text in enumerate(rows):
        color = (0, 220, 255) if idx == 0 else (230, 230, 230)
        cv2.putText(
            panel,
            text,
            (25, 40 + idx * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9 if idx == 0 else 0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    return panel


def compose_dashboard(result: LaneFrameResult) -> np.ndarray:
    """Compose a 2x2 dashboard frame for demonstration and analysis."""
    overlay = result.overlay_frame
    h, w = overlay.shape[:2]

    binary_bgr = _to_bgr(result.binary_mask)
    bev_bgr = _to_bgr(result.bev_mask)
    panel = _metric_panel(result, (h, w, 3))

    top = np.hstack([overlay, panel])
    bottom = np.hstack([binary_bgr, bev_bgr])
    return np.vstack([top, bottom])
