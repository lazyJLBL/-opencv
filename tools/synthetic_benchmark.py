import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autodrive_lane.perception import LaneGeometryPipeline


def generate_synthetic_frame(height: int, width: int, frame_idx: int) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (35, 35, 35)

    y_bottom = height - 1
    y_top = int(height * 0.58)

    center_shift = int(20 * np.sin(frame_idx / 25.0))
    left_bottom = (int(width * 0.28) + center_shift, y_bottom)
    left_top = (int(width * 0.46) + center_shift, y_top)
    right_bottom = (int(width * 0.72) + center_shift, y_bottom)
    right_top = (int(width * 0.54) + center_shift, y_top)

    cv2.line(img, left_bottom, left_top, (255, 255, 255), 8)
    cv2.line(img, right_bottom, right_top, (255, 255, 255), 8)

    # Add random distractors to test robustness.
    rng = np.random.default_rng(frame_idx + 2026)
    for _ in range(40):
        x1, y1 = int(rng.integers(0, width)), int(rng.integers(y_top, height))
        x2, y2 = int(rng.integers(0, width)), int(rng.integers(y_top, height))
        color = int(rng.integers(40, 120))
        cv2.line(img, (x1, y1), (x2, y2), (color, color, color), 1)

    return img


def main() -> None:
    pipeline = LaneGeometryPipeline()

    detect_ok = 0
    offsets = []

    total_frames = 200
    for idx in range(total_frames):
        frame = generate_synthetic_frame(height=540, width=960, frame_idx=idx)
        result = pipeline.process(frame, frame_index=idx)
        if result.left_model is not None and result.right_model is not None:
            detect_ok += 1
        if result.offset_m is not None:
            offsets.append(result.offset_m)

    detect_rate = detect_ok / total_frames
    offset_std = float(np.std(offsets)) if offsets else float("nan")

    print(f"Synthetic detect rate: {detect_rate:.3f}")
    print(f"Offset std (m): {offset_std:.4f}")


if __name__ == "__main__":
    main()
