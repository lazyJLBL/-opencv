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

from autodrive_lane.calibration import CameraCalibration
from autodrive_lane.calibration.quality_report import (
    build_calibration_quality_report,
    draw_calibration_quality_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate camera from chessboard images")
    parser.add_argument("--images", type=str, required=True, help="Glob pattern, e.g. data/calib/*.jpg")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners per chessboard row")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners per chessboard col")
    parser.add_argument("--square-size", type=float, default=0.025, help="Chessboard square size in meters")
    parser.add_argument("--save", type=str, default=str(ROOT / "outputs" / "camera_calibration.json"))
    parser.add_argument("--quality-report", type=str, default=str(ROOT / "outputs" / "calibration_quality_report.json"))
    parser.add_argument("--quality-figure", type=str, default=str(ROOT / "outputs" / "calibration_quality.png"))
    parser.add_argument("--reject-sigma", type=float, default=2.5, help="Outlier rejection sigma for MAD threshold")
    parser.add_argument("--preview", action="store_true", help="Visualize detected corners")
    return parser.parse_args()


def _compute_frame_errors(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """计算每一帧重投影误差分布。"""
    errors = []
    for objp, corners, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        err = cv2.norm(corners, proj, cv2.NORM_L2) / max(len(proj), 1)
        errors.append(float(err))
    return np.array(errors, dtype=np.float64)


def main() -> None:
    args = parse_args()

    image_paths = sorted(Path().glob(args.images))
    if not image_paths:
        raise FileNotFoundError(f"No chessboard images matched: {args.images}")

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    valid_paths: list[str] = []
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        ok, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
        if not ok:
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)
        valid_paths.append(str(path))

        if args.preview:
            vis = image.copy()
            cv2.drawChessboardCorners(vis, (args.cols, args.rows), corners_refined, ok)
            cv2.imshow("corners", vis)
            cv2.waitKey(120)

    cv2.destroyAllWindows()

    if len(obj_points) < 6:
        raise RuntimeError(f"Not enough valid calibration images. Found: {len(obj_points)}")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )

    frame_errors = _compute_frame_errors(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs)
    report = build_calibration_quality_report(valid_paths, frame_errors, sigma=args.reject_sigma)
    threshold = float(report["threshold"])

    # 按误差阈值剔除坏帧，并再次标定获得更稳定参数。
    inlier_mask = frame_errors <= threshold
    if int(np.sum(inlier_mask)) >= 6 and int(np.sum(~inlier_mask)) > 0:
        obj_points = [obj for obj, ok in zip(obj_points, inlier_mask) if ok]
        img_points = [img for img, ok in zip(img_points, inlier_mask) if ok]
        valid_paths = [path for path, ok in zip(valid_paths, inlier_mask) if ok]

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            image_size,
            None,
            None,
        )
        frame_errors = _compute_frame_errors(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs)
        report = build_calibration_quality_report(valid_paths, frame_errors, sigma=args.reject_sigma)
        threshold = float(report["threshold"])

    report["mad_sigma"] = float(args.reject_sigma)
    report["final_reprojection_error"] = float(ret)

    calib = CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        reprojection_error=float(ret),
    )
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    calib.to_json(str(save_path))

    report_path = Path(args.quality_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_path = Path(args.quality_figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    draw_calibration_quality_figure(frame_errors, threshold=threshold, save_path=str(figure_path))

    print(f"Calibration images used: {len(obj_points)}")
    print(f"Reprojection error: {ret:.6f}")
    print(f"Quality report saved to: {report_path}")
    print(f"Quality figure saved to: {figure_path}")
    print(f"Saved calibration to: {save_path}")


if __name__ == "__main__":
    main()
