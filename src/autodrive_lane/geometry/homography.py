from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def build_default_ipm(
    src_quad: Sequence[Sequence[float]],
    image_size: Tuple[int, int],
    dst_margin_ratio: float = 0.2,
    dst_top_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build perspective transform maps for inverse perspective mapping (IPM)."""
    width, height = image_size
    src = np.array(src_quad, dtype=np.float32)

    margin_x = int(width * dst_margin_ratio)
    top_y = int(height * dst_top_ratio)
    dst = np.array(
        [
            [margin_x, height - 1],
            [margin_x, top_y],
            [width - margin_x, top_y],
            [width - margin_x, height - 1],
        ],
        dtype=np.float32,
    )

    h_mat = cv2.getPerspectiveTransform(src, dst)
    h_inv = cv2.getPerspectiveTransform(dst, src)
    return h_mat, h_inv, dst


def warp(image: np.ndarray, homography: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Warp image with a perspective transform matrix."""
    return cv2.warpPerspective(image, homography, output_size, flags=cv2.INTER_LINEAR)


def transform_points(points: np.ndarray, homography: np.ndarray) -> Optional[np.ndarray]:
    """Transform Nx2 points with homography and return Nx2 output."""
    if points is None or len(points) == 0:
        return None
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography)
    return transformed.reshape(-1, 2)
