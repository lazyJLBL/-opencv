import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    """相机标定参数容器。"""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    reprojection_error: Optional[float] = None

    @classmethod
    def from_json(cls, path: str) -> "CameraCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            camera_matrix=np.array(payload["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(payload["dist_coeffs"], dtype=np.float64),
            image_size=tuple(payload["image_size"]),
            reprojection_error=payload.get("reprojection_error"),
        )

    def to_json(self, path: str) -> None:
        payload = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "reprojection_error": self.reprojection_error,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class CameraModel:
    """相机几何模型：负责去畸变和像素到地面坐标投影。"""

    def __init__(
        self,
        calibration: Optional[CameraCalibration] = None,
        camera_height_m: float = 1.45,
        pitch_deg: float = 3.0,
        yaw_deg: float = 0.0,
        roll_deg: float = 0.0,
        fallback_fov_deg: float = 68.0,
    ):
        self.calibration = calibration
        self.camera_height_m = float(camera_height_m)
        self.pitch_deg = float(pitch_deg)
        self.yaw_deg = float(yaw_deg)
        self.roll_deg = float(roll_deg)
        self.fallback_fov_deg = float(fallback_fov_deg)
        self._intrinsic_cache: Optional[Tuple[int, int, np.ndarray]] = None

    @classmethod
    def from_calibration_path(
        cls,
        calibration_path: Optional[str],
        camera_height_m: float,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float,
        fallback_fov_deg: float,
    ) -> "CameraModel":
        calibration = None
        if calibration_path:
            path = Path(calibration_path)
            if path.exists():
                calibration = CameraCalibration.from_json(str(path))
        return cls(
            calibration=calibration,
            camera_height_m=camera_height_m,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            fallback_fov_deg=fallback_fov_deg,
        )

    def undistort(self, frame_bgr: np.ndarray) -> np.ndarray:
        """若提供标定参数则执行去畸变，否则直接返回原图。"""
        if self.calibration is None:
            return frame_bgr
        return cv2.undistort(frame_bgr, self.calibration.camera_matrix, self.calibration.dist_coeffs)

    def get_intrinsic(self, width: int, height: int) -> np.ndarray:
        if self.calibration is not None:
            return self.calibration.camera_matrix

        if self._intrinsic_cache is not None:
            c_w, c_h, c_k = self._intrinsic_cache
            if c_w == width and c_h == height:
                return c_k

        # 无标定文件时，按视场角构造一个近似内参矩阵。
        fov = np.deg2rad(self.fallback_fov_deg)
        fx = 0.5 * width / np.tan(0.5 * fov)
        fy = fx
        cx = width * 0.5
        cy = height * 0.5
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._intrinsic_cache = (width, height, k)
        return k

    def pixel_to_ground(self, points_uv: np.ndarray, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """将像素点投影到地面平面，返回地面坐标(X,Z)和有效掩码。"""
        width, height = image_size
        k = self.get_intrinsic(width, height)
        k_inv = np.linalg.inv(k)

        points_uv = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
        ones = np.ones((len(points_uv), 1), dtype=np.float64)
        uv1 = np.hstack([points_uv, ones])

        rays_cam = (k_inv @ uv1.T).T
        r_cw = self._camera_to_world_rotation()
        rays_world = (r_cw @ rays_cam.T).T

        y_dir = rays_world[:, 1]
        valid = np.abs(y_dir) > 1e-8

        t = np.full(len(points_uv), np.nan, dtype=np.float64)
        t[valid] = -self.camera_height_m / y_dir[valid]
        valid = valid & (t > 0.0)

        origin = np.array([0.0, self.camera_height_m, 0.0], dtype=np.float64)
        ground = np.full((len(points_uv), 2), np.nan, dtype=np.float64)
        if np.any(valid):
            xyz = origin + rays_world[valid] * t[valid][:, None]
            ground[valid, 0] = xyz[:, 0]
            ground[valid, 1] = xyz[:, 2]

        return ground, valid

    def _camera_to_world_rotation(self) -> np.ndarray:
        pitch = np.deg2rad(self.pitch_deg)
        yaw = np.deg2rad(self.yaw_deg)
        roll = np.deg2rad(self.roll_deg)

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(-pitch), -np.sin(-pitch)],
                [0.0, np.sin(-pitch), np.cos(-pitch)],
            ],
            dtype=np.float64,
        )
        ry = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        rz = np.array(
            [
                [np.cos(roll), -np.sin(roll), 0.0],
                [np.sin(roll), np.cos(roll), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # 相机坐标系(x右,y下,z前)转换到世界坐标系(X右,Y上,Z前)。
        base = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return rz @ ry @ rx @ base
