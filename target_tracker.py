from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Any, Iterable


@dataclass
class TrackState:
    track_id: int
    history: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=8))
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    box_width: float = 0.0
    box_height: float = 0.0
    last_conf: float = 0.0
    cls_id: int = -1
    label: str = ""
    created_at: float = 0.0
    last_seen_at: float = 0.0
    predicted_x: float = 0.0
    predicted_y: float = 0.0


@dataclass
class TargetTracker:
    selection_mode: str = "highest_confidence"
    history_size: int = 8
    max_prediction_age_ms: float = 250.0
    reacquire_age_ms: float = 380.0
    stale_after_ms: float = 120.0
    base_gate_px: float = 40.0
    gate_size_factor: float = 0.55
    gate_velocity_factor: float = 0.075
    switch_score_margin: float = 0.22
    velocity_smoothing: float = 0.35

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
            best = self._pick_initial_target(enriched)
            self._start_new_track(best, now_ts)
            return self.get_active_target(now=now_ts)

        predicted_x, predicted_y = self._predict_center(now_ts)
        matched = self._match_to_current_track(enriched, predicted_x, predicted_y)

        if matched is None:
            candidate = self._pick_initial_target(enriched)
            if self._should_switch_target(candidate, predicted_x, predicted_y, now_ts):
                self._start_new_track(candidate, now_ts)
            return self.get_active_target(now=now_ts)

        self._update_track(matched, now_ts)
        return self.get_active_target(now=now_ts)

    def get_active_target(self, now: float | None = None) -> dict[str, Any] | None:
        if self._state is None:
            return None

        now_ts = time.perf_counter() if now is None else now
        stale_ms = (now_ts - self._state.last_seen_at) * 1000.0
        age_ms = (now_ts - self._state.created_at) * 1000.0

        if stale_ms > self.max_prediction_age_ms:
            self._state = None
            return None

        pred_x, pred_y = self._predict_center(now_ts)
        self._state.predicted_x = pred_x
        self._state.predicted_y = pred_y

        half_w = max(4.0, self._state.box_width / 2.0)
        half_h = max(4.0, self._state.box_height / 2.0)

        target = {
            "x1": int(round(pred_x - half_w)),
            "y1": int(round(pred_y - half_h)),
            "x2": int(round(pred_x + half_w)),
            "y2": int(round(pred_y + half_h)),
            "center_x": pred_x,
            "center_y": pred_y,
            "conf": self._state.last_conf,
            "cls_id": self._state.cls_id,
            "label": self._state.label,
            "predicted": stale_ms > 1.0,
            "track_id": self._state.track_id,
            "velocity_x": self._state.velocity_x,
            "velocity_y": self._state.velocity_y,
            "age_ms": age_ms,
            "stale_ms": stale_ms,
            "prediction_ms": max(0.0, stale_ms),
        }
        return target

    def _start_new_track(self, det: dict[str, Any], now_ts: float):
        state = TrackState(track_id=self._next_track_id)
        self._next_track_id += 1
        state.history = deque(maxlen=self.history_size)
        state.created_at = now_ts
        state.last_seen_at = now_ts
        state.box_width = det["box_width"]
        state.box_height = det["box_height"]
        state.last_conf = det["conf"]
        state.cls_id = int(det["cls_id"])
        state.label = str(det.get("label", ""))
        state.predicted_x = det["center_x"]
        state.predicted_y = det["center_y"]
        state.history.append((now_ts, det["center_x"], det["center_y"]))
        self._state = state

    def _update_track(self, det: dict[str, Any], now_ts: float):
        state = self._state
        if state is None:
            return

        previous = state.history[-1] if state.history else None
        state.history.append((now_ts, det["center_x"], det["center_y"]))
        state.last_seen_at = now_ts
        state.box_width = (state.box_width * 0.7) + (det["box_width"] * 0.3)
        state.box_height = (state.box_height * 0.7) + (det["box_height"] * 0.3)
        state.last_conf = det["conf"]
        state.cls_id = int(det["cls_id"])
        state.label = str(det.get("label", ""))

        if previous is not None:
            dt = max(1e-3, now_ts - previous[0])
            vx = (det["center_x"] - previous[1]) / dt
            vy = (det["center_y"] - previous[2]) / dt
            state.velocity_x = (state.velocity_x * (1.0 - self.velocity_smoothing)) + (
                vx * self.velocity_smoothing
            )
            state.velocity_y = (state.velocity_y * (1.0 - self.velocity_smoothing)) + (
                vy * self.velocity_smoothing
            )

        state.predicted_x = det["center_x"]
        state.predicted_y = det["center_y"]

    def _predict_center(self, now_ts: float) -> tuple[float, float]:
        state = self._state
        if state is None or not state.history:
            return 0.0, 0.0

        _, last_x, last_y = state.history[-1]
        dt = max(0.0, now_ts - state.last_seen_at)
        return (last_x + state.velocity_x * dt, last_y + state.velocity_y * dt)

    def _match_to_current_track(
        self,
        detections: list[dict[str, Any]],
        predicted_x: float,
        predicted_y: float,
    ) -> dict[str, Any] | None:
        state = self._state
        if state is None:
            return None

        speed = hypot(state.velocity_x, state.velocity_y)
        gate = self.base_gate_px
        gate += max(state.box_width, state.box_height) * self.gate_size_factor
        gate += speed * self.gate_velocity_factor

        in_gate = []
        for det in detections:
            distance = hypot(det["center_x"] - predicted_x, det["center_y"] - predicted_y)
            if distance <= gate:
                score = distance - (det["conf"] * 22.0)
                in_gate.append((score, det))

        if not in_gate:
            return None

        in_gate.sort(key=lambda item: item[0])
        return in_gate[0][1]

    def _should_switch_target(
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
        candidate_distance = hypot(candidate["center_x"] - predicted_x, candidate["center_y"] - predicted_y)

        if stale_ms >= self.reacquire_age_ms:
            return True

        track_score = (stale_ms * 0.004) + (1.0 - min(1.0, state.last_conf))
        candidate_score = (candidate_distance / max(20.0, max(state.box_width, state.box_height))) + (
            1.0 - candidate["conf"]
        )
        return candidate_score + self.switch_score_margin < track_score

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
            "distance_to_screen_center": hypot(
                center_x - screen_center[0],
                center_y - screen_center[1],
            ),
        }
