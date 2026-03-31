from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Any, Iterable


@dataclass
class DetectionSample:
    timestamp: float
    center_x: float
    center_y: float


@dataclass
class TrackState:
    track_id: int
    created_at: float
    last_seen_at: float
    history: deque[DetectionSample]
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    box_width: float = 0.0
    box_height: float = 0.0
    last_conf: float = 0.0
    cls_id: int = -1
    label: str = ""
    predicted_x: float = 0.0
    predicted_y: float = 0.0

    def append_sample(self, timestamp: float, center_x: float, center_y: float):
        self.history.append(DetectionSample(timestamp=timestamp, center_x=center_x, center_y=center_y))
        self.last_seen_at = timestamp

    @property
    def latest(self) -> DetectionSample | None:
        return self.history[-1] if self.history else None


@dataclass
class TargetTracker:
    selection_mode: str = "highest_confidence"
    history_size: int = 10
    max_prediction_age_ms: float = 320.0
    stale_after_ms: float = 90.0
    hold_after_miss_ms: float = 220.0
    reacquire_age_ms: float = 420.0
    base_gate_px: float = 34.0
    gate_size_factor: float = 0.45
    gate_velocity_factor: float = 0.045
    gate_stale_factor: float = 0.10
    box_smoothing: float = 0.28
    velocity_smoothing: float = 0.38
    acceleration_smoothing: float = 0.22
    hysteresis_switch_margin: float = 0.25
    max_prediction_dt_s: float = 0.45

    _state: TrackState | None = field(default=None, init=False)
    _next_track_id: int = field(default=1, init=False)

    def update_detections(
        self,
        detections: Iterable[dict[str, Any]],
        roi_center: tuple[float, float],
        screen_center: tuple[float, float],
        now: float | None = None,
    ) -> dict[str, Any] | None:
        now_ts = time.perf_counter() if now is None else now
        enriched = [
            self._enrich_detection(det, roi_center=roi_center, screen_center=screen_center)
            for det in detections
        ]

        if not enriched:
            return self.get_active_target(now=now_ts)

        if self._state is None:
            self._start_track(self._pick_initial_target(enriched), now_ts)
            return self.get_active_target(now=now_ts)

        predicted_x, predicted_y = self._predict_center(now_ts)
        matched = self._match_to_track(enriched, predicted_x, predicted_y, now_ts)

        if matched is not None:
            self._update_track_from_detection(matched, now_ts)
            return self.get_active_target(now=now_ts)

        candidate = self._pick_initial_target(enriched)
        if self._should_switch_track(candidate, predicted_x, predicted_y, now_ts):
            self._start_track(candidate, now_ts)

        return self.get_active_target(now=now_ts)

    def get_active_target(self, now: float | None = None) -> dict[str, Any] | None:
        state = self._state
        if state is None:
            return None

        now_ts = time.perf_counter() if now is None else now
        stale_ms = (now_ts - state.last_seen_at) * 1000.0

        if stale_ms > self.max_prediction_age_ms:
            self._state = None
            return None

        predicted_x, predicted_y = self._predict_center(now_ts)
        state.predicted_x = predicted_x
        state.predicted_y = predicted_y

        half_w = max(5.0, state.box_width * 0.5)
        half_h = max(5.0, state.box_height * 0.5)
        age_ms = (now_ts - state.created_at) * 1000.0

        return {
            "x1": int(round(predicted_x - half_w)),
            "y1": int(round(predicted_y - half_h)),
            "x2": int(round(predicted_x + half_w)),
            "y2": int(round(predicted_y + half_h)),
            "center_x": predicted_x,
            "center_y": predicted_y,
            "raw_center_x": state.latest.center_x if state.latest else predicted_x,
            "raw_center_y": state.latest.center_y if state.latest else predicted_y,
            "conf": state.last_conf,
            "cls_id": state.cls_id,
            "label": state.label,
            "track_id": state.track_id,
            "velocity_x": state.velocity_x,
            "velocity_y": state.velocity_y,
            "accel_x": state.accel_x,
            "accel_y": state.accel_y,
            "age_ms": age_ms,
            "stale_ms": stale_ms,
            "predicted": stale_ms > self.stale_after_ms,
            "fresh": stale_ms <= self.stale_after_ms,
            "prediction_ms": stale_ms,
        }

    def _start_track(self, det: dict[str, Any], now_ts: float):
        history: deque[DetectionSample] = deque(maxlen=self.history_size)
        history.append(DetectionSample(timestamp=now_ts, center_x=det["center_x"], center_y=det["center_y"]))
        self._state = TrackState(
            track_id=self._next_track_id,
            created_at=now_ts,
            last_seen_at=now_ts,
            history=history,
            box_width=det["box_width"],
            box_height=det["box_height"],
            last_conf=det["conf"],
            cls_id=int(det["cls_id"]),
            label=str(det.get("label", "")),
            predicted_x=det["center_x"],
            predicted_y=det["center_y"],
        )
        self._next_track_id += 1

    def _update_track_from_detection(self, det: dict[str, Any], now_ts: float):
        state = self._state
        if state is None:
            return

        previous_velocity_x = state.velocity_x
        previous_velocity_y = state.velocity_y
        previous = state.latest

        state.append_sample(now_ts, det["center_x"], det["center_y"])
        state.box_width = (1.0 - self.box_smoothing) * state.box_width + self.box_smoothing * det["box_width"]
        state.box_height = (1.0 - self.box_smoothing) * state.box_height + self.box_smoothing * det["box_height"]
        state.last_conf = float(det["conf"])
        state.cls_id = int(det["cls_id"])
        state.label = str(det.get("label", ""))

        if previous is not None:
            dt = max(1e-3, now_ts - previous.timestamp)
            raw_vx = (det["center_x"] - previous.center_x) / dt
            raw_vy = (det["center_y"] - previous.center_y) / dt

            state.velocity_x = (1.0 - self.velocity_smoothing) * state.velocity_x + self.velocity_smoothing * raw_vx
            state.velocity_y = (1.0 - self.velocity_smoothing) * state.velocity_y + self.velocity_smoothing * raw_vy

            raw_ax = (state.velocity_x - previous_velocity_x) / dt
            raw_ay = (state.velocity_y - previous_velocity_y) / dt
            state.accel_x = (1.0 - self.acceleration_smoothing) * state.accel_x + self.acceleration_smoothing * raw_ax
            state.accel_y = (1.0 - self.acceleration_smoothing) * state.accel_y + self.acceleration_smoothing * raw_ay

        state.predicted_x = det["center_x"]
        state.predicted_y = det["center_y"]

    def _predict_center(self, now_ts: float) -> tuple[float, float]:
        state = self._state
        if state is None:
            return 0.0, 0.0

        latest = state.latest
        if latest is None:
            return state.predicted_x, state.predicted_y

        dt = max(0.0, min(self.max_prediction_dt_s, now_ts - state.last_seen_at))
        predicted_x = latest.center_x + state.velocity_x * dt + 0.5 * state.accel_x * dt * dt
        predicted_y = latest.center_y + state.velocity_y * dt + 0.5 * state.accel_y * dt * dt
        return predicted_x, predicted_y

    def _match_to_track(
        self,
        detections: list[dict[str, Any]],
        predicted_x: float,
        predicted_y: float,
        now_ts: float,
    ) -> dict[str, Any] | None:
        state = self._state
        if state is None:
            return None

        stale_ms = (now_ts - state.last_seen_at) * 1000.0
        gate = self._compute_gate_radius(state=state, stale_ms=stale_ms)

        scored: list[tuple[float, dict[str, Any]]] = []
        for det in detections:
            distance = hypot(det["center_x"] - predicted_x, det["center_y"] - predicted_y)
            if distance > gate:
                continue
            distance_score = distance / max(1.0, gate)
            conf_score = 1.0 - float(det["conf"])
            center_bias = det["distance_to_screen_center"] / max(120.0, gate * 4.0)
            score = (distance_score * 0.62) + (conf_score * 0.28) + (center_bias * 0.10)
            scored.append((score, det))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _compute_gate_radius(self, state: TrackState, stale_ms: float) -> float:
        speed = hypot(state.velocity_x, state.velocity_y)
        box_scale = max(state.box_width, state.box_height)
        gate = self.base_gate_px
        gate += box_scale * self.gate_size_factor
        gate += speed * self.gate_velocity_factor
        gate += max(0.0, stale_ms) * self.gate_stale_factor
        return max(24.0, gate)

    def _should_switch_track(
        self,
        candidate: dict[str, Any],
        predicted_x: float,
        predicted_y: float,
        now_ts: float,
    ) -> bool:
        state = self._state
        if state is None:
            return True

        stale_ms = (now_ts - state.last_seen_at) * 1000.0
        if stale_ms >= self.reacquire_age_ms:
            return True

        if stale_ms <= self.hold_after_miss_ms:
            return False

        candidate_distance = hypot(candidate["center_x"] - predicted_x, candidate["center_y"] - predicted_y)
        scale = max(24.0, max(state.box_width, state.box_height))
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
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        return {
            **det,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": center_x,
            "center_y": center_y,
            "box_width": max(0, x2 - x1),
            "box_height": max(0, y2 - y1),
            "distance_to_roi_center": hypot(center_x - roi_center[0], center_y - roi_center[1]),
            "distance_to_screen_center": hypot(center_x - screen_center[0], center_y - screen_center[1]),
        }
