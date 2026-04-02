from dataclasses import dataclass

import cv2
import numpy as np

from autodrive_lane.config import PreprocessConfig, SceneGuardConfig


@dataclass
class SceneCondition:
    """场景质量分析结果。"""

    label: str
    brightness: float
    contrast: float
    glare_ratio: float
    edge_density: float
    reasons: list[str]


def analyze_scene_condition(frame_bgr: np.ndarray, cfg: SceneGuardConfig) -> SceneCondition:
    """识别夜晚/逆光/雨天等高风险场景。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    glare_ratio = float(np.mean(v_channel > 245))

    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.mean(edges > 0))

    reasons: list[str] = []
    is_night = brightness < cfg.night_brightness and contrast < cfg.night_contrast
    is_backlight = glare_ratio > cfg.backlight_glare_ratio and contrast < cfg.backlight_contrast
    is_rain_like = edge_density > cfg.rain_edge_density and contrast < cfg.rain_contrast

    label = "normal"
    if is_night:
        label = "night"
        reasons.append("low_brightness")
    if is_backlight:
        label = "backlight"
        reasons.append("strong_glare")
    if is_rain_like and label == "normal":
        label = "rain"
        reasons.append("dense_noisy_edges")

    return SceneCondition(
        label=label,
        brightness=brightness,
        contrast=contrast,
        glare_ratio=glare_ratio,
        edge_density=edge_density,
        reasons=reasons,
    )


def preprocess_lane_binary(frame_bgr: np.ndarray, cfg: PreprocessConfig, scene_label: str) -> np.ndarray:
    """按场景自适应预处理，输出车道候选二值图。"""
    canny_low = cfg.canny_low
    canny_high = cfg.canny_high
    white_l_thresh = cfg.white_l_thresh
    morph_kernel = cfg.morph_kernel

    working = frame_bgr
    if scene_label == "night":
        canny_low = max(20, int(cfg.canny_low * 0.65))
        canny_high = max(canny_low + 20, int(cfg.canny_high * 0.75))
        white_l_thresh = max(120, cfg.white_l_thresh - 40)
    elif scene_label == "backlight":
        white_l_thresh = max(150, cfg.white_l_thresh - 25)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        working = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    elif scene_label == "rain":
        canny_low = int(cfg.canny_low * 1.15)
        canny_high = int(cfg.canny_high * 1.2)
        morph_kernel = max(7, cfg.morph_kernel)

    blurred = cv2.GaussianBlur(working, (cfg.gaussian_ksize, cfg.gaussian_ksize), 0)

    hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    white_mask = (l_channel >= white_l_thresh) & (s_channel <= 200)

    yellow_mask = cv2.inRange(
        hsv,
        np.array(cfg.yellow_hsv_low, dtype=np.uint8),
        np.array(cfg.yellow_hsv_high, dtype=np.uint8),
    ) > 0

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cfg.sobel_ksize)
    abs_sobel = np.absolute(sobel_x)
    max_val = np.max(abs_sobel)
    if max_val < 1e-6:
        scaled = np.zeros_like(abs_sobel, dtype=np.uint8)
    else:
        scaled = np.uint8(255 * abs_sobel / max_val)

    grad_mask = (scaled >= cfg.grad_thresh[0]) & (scaled <= cfg.grad_thresh[1])
    canny = cv2.Canny(gray, canny_low, canny_high) > 0

    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[(white_mask | yellow_mask | grad_mask | canny)] = 255

    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def should_degrade_mode(
    scene: SceneCondition,
    segment_count: int,
    left_curve_found: bool,
    right_curve_found: bool,
    cfg: SceneGuardConfig,
) -> tuple[bool, list[str]]:
    """根据场景风险和模型稳定性决定是否降级。"""
    reasons: list[str] = []

    if scene.label != "normal":
        reasons.append(f"scene_{scene.label}")
    if not left_curve_found or not right_curve_found:
        reasons.append("lane_model_missing")
    if scene.label != "normal" and segment_count < cfg.min_segment_for_confident:
        reasons.append("low_feature_support")

    return len(reasons) > 0, reasons
