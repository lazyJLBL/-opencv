import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autodrive_lane.config import CameraConfig, PipelineConfig
from autodrive_lane.perception import LaneGeometryPipeline
from autodrive_lane.visualization import compose_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry-driven lane perception demo")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "data" / "video.mp4"),
        help="Input video path",
    )
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs" / "demo_output.mp4"))
    parser.add_argument("--show", action="store_true", help="Display real-time window")
    parser.add_argument("--max-frames", type=int, default=-1, help="Process only first N frames")
    parser.add_argument("--calibration", type=str, default="", help="Path to camera calibration json")
    parser.add_argument("--camera-height", type=float, default=1.45, help="Camera height in meters")
    parser.add_argument("--pitch-deg", type=float, default=3.0, help="Camera pitch angle in degrees")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Camera yaw angle in degrees")
    parser.add_argument("--roll-deg", type=float, default=0.0, help="Camera roll angle in degrees")
    parser.add_argument("--fov-deg", type=float, default=68.0, help="Fallback horizontal FOV in degrees")
    parser.add_argument("--disable-undistort", action="store_true", help="Disable frame undistortion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    ret, first_frame = capture.read()
    if not ret:
        raise RuntimeError("Input video has no readable frames")

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
    first_result = pipeline.process(first_frame, frame_index=0)
    first_dashboard = compose_dashboard(first_result)
    out_h, out_w = first_dashboard.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    writer.write(first_dashboard)
    if args.show:
        cv2.imshow("Lane Geometry Dashboard", first_dashboard)

    frame_index = 1
    while capture.isOpened():
        if args.max_frames > 0 and frame_index >= args.max_frames:
            break

        ret, frame = capture.read()
        if not ret:
            break

        result = pipeline.process(frame, frame_index=frame_index)
        dashboard = compose_dashboard(result)
        writer.write(dashboard)

        if args.show:
            cv2.imshow("Lane Geometry Dashboard", dashboard)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_index % 50 == 0:
            print(f"processed frames: {frame_index}")

        frame_index += 1

    capture.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
