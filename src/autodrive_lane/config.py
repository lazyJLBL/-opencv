from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PreprocessConfig:
    """图像预处理参数。"""

    gaussian_ksize: int = 5
    canny_low: int = 50
    canny_high: int = 140
    sobel_ksize: int = 3
    grad_thresh: Tuple[int, int] = (25, 255)
    white_l_thresh: int = 190
    yellow_hsv_low: Tuple[int, int, int] = (15, 70, 70)
    yellow_hsv_high: Tuple[int, int, int] = (40, 255, 255)
    morph_kernel: int = 5


@dataclass
class SceneGuardConfig:
    """场景识别与失效检测阈值。"""

    night_brightness: float = 72.0
    night_contrast: float = 48.0
    backlight_glare_ratio: float = 0.12
    backlight_contrast: float = 62.0
    rain_edge_density: float = 0.19
    rain_contrast: float = 58.0
    min_segment_for_confident: int = 6


@dataclass
class ROIConfig:
    """道路 ROI 几何约束。"""

    top_y_ratio: float = 0.62
    left_bottom_x_ratio: float = 0.08
    left_top_x_ratio: float = 0.45
    right_top_x_ratio: float = 0.55
    right_bottom_x_ratio: float = 0.92
    min_top_y_ratio: float = 0.45
    max_top_y_ratio: float = 0.78


@dataclass
class HoughConfig:
    """霍夫线段提取参数。"""

    rho: float = 1.0
    theta: float = 3.141592653589793 / 180.0
    threshold: int = 25
    min_line_length: int = 25
    max_line_gap: int = 20
    min_abs_slope: float = 0.3


@dataclass
class LaneModelConfig:
    """车道曲线模型与拟合参数。"""

    search_margin_px: int = 80
    residual_threshold_px: float = 20.0
    min_points: int = 120


@dataclass
class TemporalFilterConfig:
    """时序滤波参数。"""

    line_alpha: float = 0.25
    curve_alpha: float = 0.3
    metric_alpha: float = 0.2
    max_lane_width_jump_ratio: float = 0.35


@dataclass
class LaneChangeFSMConfig:
    """变道状态机参数。"""

    evidence_decay: float = 0.88
    velocity_gain: float = 2.0
    displacement_gain: float = 1.2
    prepare_threshold: float = 0.45
    change_threshold: float = 0.68
    recover_threshold: float = 0.32
    decision_margin: float = 0.08
    min_prepare_frames: int = 4
    min_change_frames: int = 6
    min_recover_frames: int = 5


@dataclass
class MetricConfig:
    """物理尺度指标参数。"""

    straight_curvature_m: float = 1e6


@dataclass
class CameraConfig:
    """相机模型与安装参数。"""

    calibration_path: str = ""
    enable_undistort: bool = True
    camera_height_m: float = 1.45
    pitch_deg: float = 3.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    fallback_fov_deg: float = 68.0


@dataclass
class PipelineConfig:
    """整条感知流水线的总配置。"""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    scene_guard: SceneGuardConfig = field(default_factory=SceneGuardConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    hough: HoughConfig = field(default_factory=HoughConfig)
    lane_model: LaneModelConfig = field(default_factory=LaneModelConfig)
    temporal: TemporalFilterConfig = field(default_factory=TemporalFilterConfig)
    lane_change_fsm: LaneChangeFSMConfig = field(default_factory=LaneChangeFSMConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
