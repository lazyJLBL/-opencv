import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autodrive_lane.config import CameraConfig, PipelineConfig
from autodrive_lane.perception import LaneGeometryPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lane geometry metrics on a video")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "data" / "video.mp4"),
    )
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--save", type=str, default=str(ROOT / "outputs" / "metrics.json"))
    parser.add_argument("--calibration", type=str, default="", help="Path to camera calibration json")
    parser.add_argument("--camera-height", type=float, default=1.45)
    parser.add_argument("--pitch-deg", type=float, default=3.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--fov-deg", type=float, default=68.0)
    parser.add_argument("--disable-undistort", action="store_true")
    return parser.parse_args()


def safe_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "p95": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")

    pipeline_cfg = PipelineConfig(
        camera=CameraConfig(
            calibration_path=args.calibration,
            enable_undistort=not args.disable_undistort,
            camera_height_m=args.camera_height,
            pitch_deg=args.pitch_deg,
            yaw_deg=args.yaw_deg,
            roll_deg=args.roll_deg,
            fallback_fov_deg=args.fov_deg,
        )
    )
    pipeline = LaneGeometryPipeline(config=pipeline_cfg)

    total = 0
    lane_detected = 0
    widths = []
    widths_m = []
    curvatures = []
    offsets = []
    lane_change_confidences = []
    scene_counts = {"normal": 0, "night": 0, "backlight": 0, "rain": 0}
    degraded_frames = 0
    lane_change_counts = {
        "keep_lane": 0,
        "prepare_left": 0,
        "prepare_right": 0,
        "changing_left": 0,
        "changing_right": 0,
        "recovering": 0,
    }

    while capture.isOpened() and total < args.max_frames:
        ret, frame = capture.read()
        if not ret:
            break

        result = pipeline.process(frame, frame_index=total)
        total += 1

        if result.left_model is not None and result.right_model is not None:
            lane_detected += 1

        if result.lane_width_px is not None:
            widths.append(float(result.lane_width_px))
        if result.lane_width_m is not None:
            widths_m.append(float(result.lane_width_m))
        if result.curvature_m is not None:
            curvatures.append(float(result.curvature_m))
        if result.offset_m is not None:
            offsets.append(float(result.offset_m))
        lane_change_confidences.append(float(result.lane_change_confidence))

        if result.degraded:
            degraded_frames += 1

        scene_counts[result.scene_label] = scene_counts.get(result.scene_label, 0) + 1
        lane_change_counts[result.lane_change_state] = lane_change_counts.get(result.lane_change_state, 0) + 1

    capture.release()

    report = {
        "total_frames": total,
        "lane_detected_frames": lane_detected,
        "detection_rate": lane_detected / max(total, 1),
        "lane_width_px": safe_stats(widths),
        "lane_width_m": safe_stats(widths_m),
        "curvature_m": safe_stats(curvatures),
        "offset_m": safe_stats(offsets),
        "scene_counts": scene_counts,
        "degraded_frame_ratio": degraded_frames / max(total, 1),
        "lane_change_counts": lane_change_counts,
        "lane_change_confidence": safe_stats(lane_change_confidences),
        "camera_calibration_used": bool(args.calibration),
    }

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
