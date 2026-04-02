from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CalibrationFrameStat:
    """单帧标定统计信息。"""

    image_path: str
    reprojection_error: float
    inlier: bool


def _mad_threshold(values: np.ndarray, sigma: float = 2.5) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma < 1e-9:
        return median + 1e-6
    return median + sigma * robust_sigma


def build_calibration_quality_report(image_paths: list[str], errors: np.ndarray, sigma: float = 2.5) -> dict[str, Any]:
    """根据重投影误差计算分布统计和坏帧剔除结果。"""
    errors = np.asarray(errors, dtype=np.float64)
    threshold = _mad_threshold(errors, sigma=sigma)
    inlier_mask = errors <= threshold

    frames = [
        CalibrationFrameStat(
            image_path=str(path),
            reprojection_error=float(err),
            inlier=bool(ok),
        )
        for path, err, ok in zip(image_paths, errors, inlier_mask)
    ]

    report = {
        "threshold": float(threshold),
        "total_frames": int(len(errors)),
        "inlier_frames": int(np.sum(inlier_mask)),
        "outlier_frames": int(np.sum(~inlier_mask)),
        "error_stats": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "std": float(np.std(errors)),
            "p90": float(np.percentile(errors, 90)),
            "p95": float(np.percentile(errors, 95)),
            "max": float(np.max(errors)),
        },
        "frames": [frame.__dict__ for frame in frames],
    }
    return report


def draw_calibration_quality_figure(errors: np.ndarray, threshold: float, save_path: str) -> None:
    """自动生成重投影误差可视化图。"""
    errors = np.asarray(errors, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(errors, bins=20, color="#4C78A8", alpha=0.85)
    axes[0].axvline(threshold, color="#E45756", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    axes[0].set_title("Reprojection Error Histogram")
    axes[0].set_xlabel("error (pixel)")
    axes[0].set_ylabel("count")
    axes[0].legend()

    sorted_errors = np.sort(errors)
    axes[1].plot(np.arange(1, len(sorted_errors) + 1), sorted_errors, color="#72B7B2", linewidth=2)
    axes[1].axhline(threshold, color="#E45756", linestyle="--", linewidth=2)
    axes[1].set_title("Sorted Frame Errors")
    axes[1].set_xlabel("frame rank")
    axes[1].set_ylabel("error (pixel)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
