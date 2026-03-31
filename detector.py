from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from config import AppSettings
from constants import CONF, IOU, MAX_DET, SHOW_LABELS
from target_tracker import TargetTracker


class DetectorWorker(QObject):
    detections_ready = Signal(list)
    log_ready = Signal(str)
    finished_one = Signal()

    def __init__(self, settings: AppSettings, yolo_cls, mss_mod):
        super().__init__()
        self.settings = settings
        self._yolo_cls = yolo_cls
        self._mss_mod = mss_mod

        self.model = None
        self.names: Any = None
        self.sct = None
        self.fallback_done = False
        self.target_tracker = TargetTracker(selection_mode=settings.target_selection_mode)
        self._last_status_log = 0.0
        self._last_status_key: tuple[Any, ...] | None = None

    def _ensure_ready(self):
        if self.sct is None:
            self.sct = self._mss_mod.mss()

        if self.model is None:
            self.log_ready.emit(f"Modell wird geladen: {self.settings.model_path}")
            self.model = self._yolo_cls(str(self.settings.model_path))
            self.names = getattr(self.model, "names", None)
            self.log_ready.emit("Modell geladen.")
            self._warmup()

    def _warmup(self):
        try:
            dummy = np.zeros((self.settings.imgsz, self.settings.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy,
                conf=CONF,
                iou=IOU,
                imgsz=self.settings.imgsz,
                device=self.settings.device_string,
                half=self.settings.use_half,
                max_det=1,
                classes=self.settings.team_classes,
                verbose=False,
            )
            self.log_ready.emit("Modell-Warmup abgeschlossen.")
        except Exception as exc:  # noqa: BLE001
            self.log_ready.emit(f"Warmup-Hinweis: {exc}")

    def _predict_once(self, frame: np.ndarray):
        return self.model.predict(
            source=frame,
            conf=CONF,
            iou=IOU,
            imgsz=self.settings.imgsz,
            device=self.settings.device_string,
            half=self.settings.use_half,
            max_det=MAX_DET,
            classes=self.settings.team_classes,
            augment=False,
            verbose=False,
        )

    def _parse_results(self, results) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not results:
            return out

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.float().cpu().tolist()
        confs = boxes.conf.float().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, class_ids):
            if isinstance(self.names, dict):
                name = self.names.get(cls_id, str(cls_id))
            elif isinstance(self.names, list) and 0 <= cls_id < len(self.names):
                name = self.names[cls_id]
            else:
                name = str(cls_id)

            label = f"{name} {conf:.2f}" if SHOW_LABELS else ""
            out.append(
                {
                    "x1": x1 + self.settings.offset_x,
                    "y1": y1 + self.settings.offset_y,
                    "x2": x2 + self.settings.offset_x,
                    "y2": y2 + self.settings.offset_y,
                    "cls_id": cls_id,
                    "label": label,
                    "conf": float(conf),
                }
            )

        return out

    def _select_active_target(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        target = self.target_tracker.select_best_target(
            detections=detections,
            roi_center=self.settings.roi_center,
            screen_center=self.settings.screen_center,
        )
        if target is None:
            return []
        return [target]

    def _emit_target_status(self, active_target: list[dict[str, Any]]):
        now = time.perf_counter()
        if active_target:
            det = active_target[0]
            status_key: tuple[Any, ...] = (
                True,
                round(det["center_x"], 1),
                round(det["center_y"], 1),
                round(det["conf"], 2),
            )
            text = (
                f"Aktives Ziel: ja | Regel={self.settings.target_mode_name} | "
                f"Center=({det['center_x']:.1f}, {det['center_y']:.1f}) | "
                f"Conf={det['conf']:.2f}"
            )
        else:
            status_key = (False,)
            text = f"Aktives Ziel: nein | Regel={self.settings.target_mode_name}"

        should_log = status_key != self._last_status_key or (now - self._last_status_log) >= 1.0
        if should_log:
            self.log_ready.emit(text)
            self._last_status_log = now
            self._last_status_key = status_key

    def _grab_frame(self) -> np.ndarray:
        shot = self.sct.grab(self.settings.capture_region)
        frame = np.asarray(shot, dtype=np.uint8)[:, :, :3]
        return np.ascontiguousarray(frame)

    @Slot()
    def detect_once(self):
        start = time.perf_counter()
        try:
            self._ensure_ready()
            frame = self._grab_frame()
            detections = self._parse_results(self._predict_once(frame))
            active_target = self._select_active_target(detections)
            self.detections_ready.emit(active_target)
            self._emit_target_status(active_target)
        except Exception as exc:  # noqa: BLE001
            if (not self.fallback_done) and self.settings.device_string != "cpu":
                self.fallback_done = True
                self.log_ready.emit(
                    f"Gerät '{self.settings.device_name}' fehlgeschlagen, CPU-Fallback aktiv. Fehler: {exc}"
                )
                self.settings.device_name = "CPU (Fallback)"
                self.settings.device_string = "cpu"
                self.settings.use_half = False
                self.model = None
                self.names = None
                try:
                    self._ensure_ready()
                    frame = self._grab_frame()
                    detections = self._parse_results(self._predict_once(frame))
                    active_target = self._select_active_target(detections)
                    self.detections_ready.emit(active_target)
                    self._emit_target_status(active_target)
                except Exception as second_exc:  # noqa: BLE001
                    self.log_ready.emit(f"Erkennungsfehler nach CPU-Fallback: {second_exc}")
            else:
                self.log_ready.emit(f"Erkennungsfehler: {exc}")
        finally:
            self.log_ready.emit(f"Detection-Zyklus: {(time.perf_counter() - start) * 1000:.1f} ms")
            self.finished_one.emit()
