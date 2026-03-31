from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Any, Iterable

from constants import AIM_ANCHOR_X_RATIO, AIM_ANCHOR_Y_RATIO


@dataclass
class DetectionSample:
    observed_at: float
    received_at: float
    bbox_center_x: float
    bbox_center_y: float
    observed_x: float
    observed_y: float


@dataclass
class TrackState:
    track_id: int
    created_at: float
    last_observed_at: float
    last_detected_at: float
    history: deque[DetectionSample]
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    aim_velocity_x: float = 0.0
    aim_velocity_y: float = 0.0
    aim_accel_x: float = 0.0
    aim_accel_y: float = 0.0
    box_width: float = 0.0
    box_height: float = 0.0
    bbox_center_x: float = 0.0
    bbox_center_y: float = 0.0
    observed_x: float = 0.0
    observed_y: float = 0.0
    smoothed_aim_x: float = 0.0
    smoothed_aim_y: float = 0.0
    last_conf: float = 0.0
    cls_id: int = -1
    label: str = ""

    @property
    def latest(self) -> DetectionSample | None:
        return self.history[-1] if self.history else None


@dataclass
class TargetTracker:
    selection_mode: str = "highest_confidence"
    history_size: int = 14
    stale_after_ms: float = 85.0
    hold_after_miss_ms: float = 240.0
    max_prediction_age_ms: float = 420.0
    reacquire_age_ms: float = 360.0
    base_gate_px: float = 32.0
    gate_size_factor: float = 0.40
    gate_velocity_factor: float = 0.040
    gate_stale_factor: float = 0.11
    box_smoothing: float = 0.26
    aim_smoothing: float = 0.35
    velocity_smoothing: float = 0.34
    acceleration_smoothing: float = 0.20
    hysteresis_switch_margin: float = 0.22
    max_prediction_dt_s: float = 0.50

    _state: TrackState | None = field(default=None, init=False)
    _next_track_id: int = field(default=1, init=False)

    def update_detections(
        self,
        detections: Iterable[dict[str, Any]],
        roi_center: tuple[float, float],
        screen_center: tuple[float, float],
        captured_at: float,
        detected_at: float,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        now_ts = time.perf_counter() if now is None else now
        enriched = [
            self._enrich_detection(det, roi_center=roi_center, screen_center=screen_center)
            for det in detections
        ]

        if not enriched:
            return self.get_active_target(now=now_ts)

        observation_ts = min(captured_at, detected_at)
        if self._state is None:
            self._start_track(self._pick_initial_target(enriched), observation_ts, detected_at)
            return self.get_active_target(now=now_ts)

        predicted_obs_x, predicted_obs_y = self._predict_observed_point(observation_ts)
        matched = self._match_to_track(enriched, predicted_obs_x, predicted_obs_y, detected_at)

        if matched is not None:
            self._update_track_from_detection(matched, observation_ts, detected_at)
            return self.get_active_target(now=now_ts)

        candidate = self._pick_initial_target(enriched)
        if self._should_switch_track(candidate, predicted_obs_x, predicted_obs_y, detected_at):
            self._start_track(candidate, observation_ts, detected_at)

        return self.get_active_target(now=now_ts)

    def get_active_target(self, now: float | None = None) -> dict[str, Any] | None:
        state = self._state
        if state is None:
            return None

        now_ts = time.perf_counter() if now is None else now
        stale_ms = (now_ts - state.last_detected_at) * 1000.0
        prediction_age_ms = (now_ts - state.last_observed_at) * 1000.0

        if prediction_age_ms > self.max_prediction_age_ms:
            self._state = None
            return None

        predicted_obs_x, predicted_obs_y = self._predict_observed_point(now_ts)
        predicted_bbox_x, predicted_bbox_y = self._predict_bbox_center(now_ts)

        half_w = max(5.0, state.box_width * 0.5)
        half_h = max(5.0, state.box_height * 0.5)
        age_ms = (now_ts - state.created_at) * 1000.0

        return {
            "x1": int(round(predicted_bbox_x - half_w)),
            "y1": int(round(predicted_bbox_y - half_h)),
            "x2": int(round(predicted_bbox_x + half_w)),
            "y2": int(round(predicted_bbox_y + half_h)),
            "bbox_center_x": predicted_bbox_x,
            "bbox_center_y": predicted_bbox_y,
            "center_x": predicted_bbox_x,
            "center_y": predicted_bbox_y,
            "observed_x": state.observed_x,
            "observed_y": state.observed_y,
            "predicted_x": predicted_obs_x,
            "predicted_y": predicted_obs_y,
            "aim_x": state.smoothed_aim_x,
            "aim_y": state.smoothed_aim_y,
            "raw_aim_x": predicted_obs_x,
            "raw_aim_y": predicted_obs_y,
            "conf": state.last_conf,
            "cls_id": state.cls_id,
            "label": state.label,
            "track_id": state.track_id,
            "velocity_x": state.aim_velocity_x,
            "velocity_y": state.aim_velocity_y,
            "accel_x": state.aim_accel_x,
            "accel_y": state.aim_accel_y,
            "age_ms": age_ms,
            "stale_ms": stale_ms,
            "latency_compensation_ms": max(0.0, prediction_age_ms),
            "predicted": stale_ms > self.stale_after_ms,
            "fresh": stale_ms <= self.stale_after_ms,
            "capture_timestamp": state.last_observed_at,
            "detect_timestamp": state.last_detected_at,
            "anchor_x_ratio": AIM_ANCHOR_X_RATIO,
            "anchor_y_ratio": AIM_ANCHOR_Y_RATIO,
        }

    def _start_track(self, det: dict[str, Any], observed_at: float, detected_at: float):
        history: deque[DetectionSample] = deque(maxlen=self.history_size)
        history.append(
            DetectionSample(
                observed_at=observed_at,
                received_at=detected_at,
                bbox_center_x=det["bbox_center_x"],
                bbox_center_y=det["bbox_center_y"],
                observed_x=det["observed_x"],
                observed_y=det["observed_y"],
            )
        )
        self._state = TrackState(
            track_id=self._next_track_id,
            created_at=detected_at,
            last_observed_at=observed_at,
            last_detected_at=detected_at,
            history=history,
            box_width=det["box_width"],
            box_height=det["box_height"],
            bbox_center_x=det["bbox_center_x"],
            bbox_center_y=det["bbox_center_y"],
            observed_x=det["observed_x"],
            observed_y=det["observed_y"],
            smoothed_aim_x=det["observed_x"],
            smoothed_aim_y=det["observed_y"],
            last_conf=float(det["conf"]),
            cls_id=int(det["cls_id"]),
            label=str(det.get("label", "")),
        )
        self._next_track_id += 1

    def _update_track_from_detection(self, det: dict[str, Any], observed_at: float, detected_at: float):
        state = self._state
        if state is None:
            return

        prev_aim_vx = state.aim_velocity_x
        prev_aim_vy = state.aim_velocity_y
        prev_bbox_vx = state.velocity_x
        prev_bbox_vy = state.velocity_y

        latest = state.latest
        state.history.append(
            DetectionSample(
                observed_at=observed_at,
                received_at=detected_at,
                bbox_center_x=det["bbox_center_x"],
                bbox_center_y=det["bbox_center_y"],
                observed_x=det["observed_x"],
                observed_y=det["observed_y"],
            )
        )
        state.last_observed_at = observed_at
        state.last_detected_at = detected_at

        state.box_width = (1.0 - self.box_smoothing) * state.box_width + self.box_smoothing * det["box_width"]
        state.box_height = (1.0 - self.box_smoothing) * state.box_height + self.box_smoothing * det["box_height"]
        state.bbox_center_x = det["bbox_center_x"]
        state.bbox_center_y = det["bbox_center_y"]
        state.observed_x = det["observed_x"]
        state.observed_y = det["observed_y"]

        state.smoothed_aim_x = (1.0 - self.aim_smoothing) * state.smoothed_aim_x + self.aim_smoothing * det["observed_x"]
        state.smoothed_aim_y = (1.0 - self.aim_smoothing) * state.smoothed_aim_y + self.aim_smoothing * det["observed_y"]

        state.last_conf = float(det["conf"])
        state.cls_id = int(det["cls_id"])
        state.label = str(det.get("label", ""))

        if latest is None:
            return

        dt = max(1e-3, observed_at - latest.observed_at)

        raw_bbox_vx = (det["bbox_center_x"] - latest.bbox_center_x) / dt
        raw_bbox_vy = (det["bbox_center_y"] - latest.bbox_center_y) / dt
        state.velocity_x = (1.0 - self.velocity_smoothing) * state.velocity_x + self.velocity_smoothing * raw_bbox_vx
        state.velocity_y = (1.0 - self.velocity_smoothing) * state.velocity_y + self.velocity_smoothing * raw_bbox_vy

        raw_aim_vx = (det["observed_x"] - latest.observed_x) / dt
        raw_aim_vy = (det["observed_y"] - latest.observed_y) / dt
        state.aim_velocity_x = (1.0 - self.velocity_smoothing) * state.aim_velocity_x + self.velocity_smoothing * raw_aim_vx
        state.aim_velocity_y = (1.0 - self.velocity_smoothing) * state.aim_velocity_y + self.velocity_smoothing * raw_aim_vy

        raw_bbox_ax = (state.velocity_x - prev_bbox_vx) / dt
        raw_bbox_ay = (state.velocity_y - prev_bbox_vy) / dt
        raw_aim_ax = (state.aim_velocity_x - prev_aim_vx) / dt
        raw_aim_ay = (state.aim_velocity_y - prev_aim_vy) / dt

        state.accel_x = (1.0 - self.acceleration_smoothing) * state.accel_x + self.acceleration_smoothing * raw_bbox_ax
        state.accel_y = (1.0 - self.acceleration_smoothing) * state.accel_y + self.acceleration_smoothing * raw_bbox_ay
        state.aim_accel_x = (1.0 - self.acceleration_smoothing) * state.aim_accel_x + self.acceleration_smoothing * raw_aim_ax
        state.aim_accel_y = (1.0 - self.acceleration_smoothing) * state.aim_accel_y + self.acceleration_smoothing * raw_aim_ay

    def _predict_observed_point(self, ts: float) -> tuple[float, float]:
        state = self._state
        if state is None:
            return 0.0, 0.0
        dt = max(0.0, min(self.max_prediction_dt_s, ts - state.last_observed_at))
        px = state.observed_x + state.aim_velocity_x * dt + 0.5 * state.aim_accel_x * dt * dt
        py = state.observed_y + state.aim_velocity_y * dt + 0.5 * state.aim_accel_y * dt * dt
        return px, py

    def _predict_bbox_center(self, ts: float) -> tuple[float, float]:
        state = self._state
        if state is None:
            return 0.0, 0.0
        dt = max(0.0, min(self.max_prediction_dt_s, ts - state.last_observed_at))
        px = state.bbox_center_x + state.velocity_x * dt + 0.5 * state.accel_x * dt * dt
        py = state.bbox_center_y + state.velocity_y * dt + 0.5 * state.accel_y * dt * dt
        return px, py

    def _match_to_track(
        self,
        detections: list[dict[str, Any]],
        predicted_obs_x: float,
        predicted_obs_y: float,
        detected_at: float,
    ) -> dict[str, Any] | None:
        state = self._state
        if state is None:
            return None

        stale_ms = (detected_at - state.last_detected_at) * 1000.0
        gate = self._compute_gate_radius(state=state, stale_ms=stale_ms)

        scored: list[tuple[float, dict[str, Any]]] = []
        for det in detections:
            distance = hypot(det["observed_x"] - predicted_obs_x, det["observed_y"] - predicted_obs_y)
            if distance > gate:
                continue
            distance_score = distance / max(1.0, gate)
            conf_score = 1.0 - float(det["conf"])
            center_bias = det["distance_to_screen_center"] / max(120.0, gate * 4.0)
            score = (distance_score * 0.64) + (conf_score * 0.26) + (center_bias * 0.10)
            scored.append((score, det))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _compute_gate_radius(self, state: TrackState, stale_ms: float) -> float:
        speed = hypot(state.aim_velocity_x, state.aim_velocity_y)
        box_scale = max(state.box_width, state.box_height)
        gate = self.base_gate_px
        gate += box_scale * self.gate_size_factor
        gate += speed * self.gate_velocity_factor
        gate += max(0.0, stale_ms) * self.gate_stale_factor
        return max(22.0, gate)

    def _should_switch_track(
        self,
        candidate: dict[str, Any],
        predicted_obs_x: float,
        predicted_obs_y: float,
        detected_at: float,
    ) -> bool:
        state = self._state
        if state is None:
            return True

        stale_ms = (detected_at - state.last_detected_at) * 1000.0
        if stale_ms >= self.reacquire_age_ms:
            return True

        if stale_ms <= self.hold_after_miss_ms:
            return False

        candidate_distance = hypot(candidate["observed_x"] - predicted_obs_x, candidate["observed_y"] - predicted_obs_y)
        scale = max(20.0, max(state.box_width, state.box_height))
        candidate_score = (candidate_distance / scale) + (1.0 - float(candidate["conf"]))
        keep_score = (stale_ms / max(1.0, self.max_prediction_age_ms)) + (1.0 - state.last_conf)
        return candidate_score + self.hysteresis_switch_margin < keep_score

    def _pick_initial_target(self, detections: list[dict[str, Any]]) -> dict[str, Any]:
        if self.selection_mode == "nearest_center":
            return min(detections, key=lambda det: (det["distance_to_roi_center"], -det["conf"]))
        return max(detections, key=lambda det: (det["conf"], -det["distance_to_roi_center"]))

    @staticmethod
    def _enrich_detection(
        det: dict[str, Any],
        roi_center: tuple[float, float],
        screen_center: tuple[float, float],
    ) -> dict[str, Any]:
        x1 = float(det["x1"])
        y1 = float(det["y1"])
        x2 = float(det["x2"])
        y2 = float(det["y2"])

        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        bbox_center_x = x1 + (box_width * 0.5)
        bbox_center_y = y1 + (box_height * 0.5)

        observed_x = x1 + (box_width * AIM_ANCHOR_X_RATIO)
        observed_y = y1 + (box_height * AIM_ANCHOR_Y_RATIO)

        return {
            **det,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "box_width": box_width,
            "box_height": box_height,
            "bbox_center_x": bbox_center_x,
            "bbox_center_y": bbox_center_y,
            "observed_x": observed_x,
            "observed_y": observed_y,
            "distance_to_roi_center": hypot(observed_x - roi_center[0], observed_y - roi_center[1]),
            "distance_to_screen_center": hypot(observed_x - screen_center[0], observed_y - screen_center[1]),
        }
