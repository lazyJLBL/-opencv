from typing import Optional

import cv2
import numpy as np


def render_overlay_frame(
    frame_bgr: np.ndarray,
    roi_polygon: np.ndarray,
    left_points: Optional[np.ndarray],
    right_points: Optional[np.ndarray],
    vanishing_point: Optional[tuple[float, float]],
    lane_width_px: Optional[float],
    lane_width_m: Optional[float],
    curvature_m: Optional[float],
    offset_m: Optional[float],
    scene_label: str,
    lane_change_state: str,
    lane_change_confidence: float,
    degraded: bool,
    degradation_reasons: list[str],
) -> np.ndarray:
    """渲染车道检测叠加图。"""
    overlay = frame_bgr.copy()

    cv2.polylines(overlay, [roi_polygon.astype(np.int32)], True, (255, 160, 0), 2)

    if left_points is not None:
        cv2.polylines(overlay, [np.int32(left_points)], False, (0, 220, 255), 6)

    if right_points is not None:
        cv2.polylines(overlay, [np.int32(right_points)], False, (0, 220, 255), 6)

    if left_points is not None and right_points is not None:
        lane_poly = np.array(np.vstack([left_points, right_points[::-1]]), dtype=np.int32)
        fill = overlay.copy()
        cv2.fillPoly(fill, [lane_poly], (40, 140, 40))
        overlay = cv2.addWeighted(fill, 0.25, overlay, 0.75, 0)

    if vanishing_point is not None:
        vx, vy = vanishing_point
        if np.isfinite(vx) and np.isfinite(vy):
            cv2.circle(overlay, (int(vx), int(vy)), 5, (20, 20, 255), -1)

    text_rows = [
        f"Lane width(px): {lane_width_px:.1f}" if lane_width_px is not None else "Lane width(px): NA",
        f"Lane width(m): {lane_width_m:.2f}" if lane_width_m is not None else "Lane width(m): NA",
        f"Curvature(m): {curvature_m:.1f}" if curvature_m is not None else "Curvature(m): NA",
        f"Offset(m): {offset_m:+.2f}" if offset_m is not None else "Offset(m): NA",
        f"Scene: {scene_label}",
        f"Lane change: {lane_change_state} ({lane_change_confidence:.2f})",
    ]

    for idx, text in enumerate(text_rows):
        cv2.putText(
            overlay,
            text,
            (20, 35 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if degraded:
        reason_text = ",".join(degradation_reasons[:3]) if degradation_reasons else "unstable"
        cv2.putText(
            overlay,
            f"DEGRADED MODE: {reason_text}",
            (20, overlay.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (40, 60, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay
