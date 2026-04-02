from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from autodrive_lane.config import LaneChangeFSMConfig


@dataclass
class LaneChangeDecision:
    """变道状态机输出。"""

    state: str
    confidence: float
    left_score: float
    right_score: float


class LaneChangeStateMachine:
    """基于时序证据的变道状态机，替代单阈值判定。"""

    def __init__(self, cfg: LaneChangeFSMConfig):
        self.cfg = cfg
        self.state = "keep_lane"
        self.state_frames = 0

        self.left_score = 0.0
        self.right_score = 0.0

        self.prev_offset: Optional[float] = None
        self.offset_history: deque[float] = deque(maxlen=20)

    def update(self, offset_m: Optional[float], degraded: bool) -> LaneChangeDecision:
        self.state_frames += 1

        if offset_m is None:
            self._decay_scores(multiplier=0.92)
            return LaneChangeDecision(
                state=self.state,
                confidence=float(max(self.left_score, self.right_score)),
                left_score=float(self.left_score),
                right_score=float(self.right_score),
            )

        self.offset_history.append(float(offset_m))
        velocity = 0.0 if self.prev_offset is None else float(offset_m - self.prev_offset)
        self.prev_offset = float(offset_m)

        baseline = float(np.median(self.offset_history))
        displacement = float(offset_m - baseline)

        left_evidence = np.tanh(max(0.0, -velocity) * self.cfg.velocity_gain + max(0.0, -displacement) * self.cfg.displacement_gain)
        right_evidence = np.tanh(max(0.0, velocity) * self.cfg.velocity_gain + max(0.0, displacement) * self.cfg.displacement_gain)

        if degraded:
            # 降级模式下，证据增益降低，避免误触发变道。
            left_evidence *= 0.6
            right_evidence *= 0.6

        decay = self.cfg.evidence_decay
        self.left_score = decay * self.left_score + (1.0 - decay) * float(left_evidence)
        self.right_score = decay * self.right_score + (1.0 - decay) * float(right_evidence)

        self._transition(velocity=velocity)

        confidence = self._state_confidence()
        return LaneChangeDecision(
            state=self.state,
            confidence=float(confidence),
            left_score=float(self.left_score),
            right_score=float(self.right_score),
        )

    def _transition(self, velocity: float) -> None:
        dominant_left = self.left_score > self.right_score + self.cfg.decision_margin
        dominant_right = self.right_score > self.left_score + self.cfg.decision_margin

        if self.state == "keep_lane":
            if dominant_left and self.left_score > self.cfg.prepare_threshold:
                self._set_state("prepare_left")
            elif dominant_right and self.right_score > self.cfg.prepare_threshold:
                self._set_state("prepare_right")
            return

        if self.state == "prepare_left":
            if not dominant_left and self.left_score < self.cfg.recover_threshold:
                self._set_state("recovering")
            elif self.state_frames >= self.cfg.min_prepare_frames and self.left_score > self.cfg.change_threshold:
                self._set_state("changing_left")
            return

        if self.state == "prepare_right":
            if not dominant_right and self.right_score < self.cfg.recover_threshold:
                self._set_state("recovering")
            elif self.state_frames >= self.cfg.min_prepare_frames and self.right_score > self.cfg.change_threshold:
                self._set_state("changing_right")
            return

        if self.state == "changing_left":
            if self.state_frames >= self.cfg.min_change_frames and (self.left_score < self.cfg.recover_threshold or velocity > 0.0):
                self._set_state("recovering")
            return

        if self.state == "changing_right":
            if self.state_frames >= self.cfg.min_change_frames and (self.right_score < self.cfg.recover_threshold or velocity < 0.0):
                self._set_state("recovering")
            return

        if self.state == "recovering":
            if self.state_frames >= self.cfg.min_recover_frames and max(self.left_score, self.right_score) < self.cfg.prepare_threshold * 0.65:
                self._set_state("keep_lane")

    def _state_confidence(self) -> float:
        if self.state in {"prepare_left", "changing_left"}:
            return self.left_score
        if self.state in {"prepare_right", "changing_right"}:
            return self.right_score
        if self.state == "recovering":
            return max(self.left_score, self.right_score) * 0.7
        return max(self.left_score, self.right_score) * 0.5

    def _set_state(self, new_state: str) -> None:
        if self.state != new_state:
            self.state = new_state
            self.state_frames = 0

    def _decay_scores(self, multiplier: float) -> None:
        self.left_score *= multiplier
        self.right_score *= multiplier
