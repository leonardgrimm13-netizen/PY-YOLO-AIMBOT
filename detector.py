from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from config import AppSettings
from constants import CONF, DETECT_FPS, IOU, MAX_DET, SHOW_LABELS


@dataclass
class FramePacket:
    frame: np.ndarray
    captured_at: float


class DetectorWorker(QObject):
    detections_ready = Signal(list, dict)
    log_ready = Signal(str)
    stopped = Signal()

    def __init__(self, settings: AppSettings, yolo_cls, mss_mod):
        super().__init__()
        self.settings = settings
        self._yolo_cls = yolo_cls
        self._mss_mod = mss_mod

        self.model = None
        self.names: Any = None
        self.sct = None
        self.fallback_done = False

        self._running = False
        self._stop_event = threading.Event()
        self._latest_packet: FramePacket | None = None
        self._frame_interval = 1.0 / max(1.0, DETECT_FPS)

        self._last_status_log = 0.0
        self._last_status_key: tuple[Any, ...] | None = None
        self._last_emit_time = 0.0
        self._smoothed_det_hz = 0.0

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

    def _capture_latest_frame(self, now: float) -> None:
        shot = self.sct.grab(self.settings.capture_region)
        frame = np.asarray(shot, dtype=np.uint8)[:, :, :3]
        self._latest_packet = FramePacket(frame=np.ascontiguousarray(frame), captured_at=now)

    def _consume_latest_frame(self) -> FramePacket | None:
        packet = self._latest_packet
        self._latest_packet = None
        return packet

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

    def _update_detection_frequency(self, emit_ts: float):
        if self._last_emit_time > 0.0:
            dt = max(1e-3, emit_ts - self._last_emit_time)
            hz = 1.0 / dt
            if self._smoothed_det_hz <= 0.0:
                self._smoothed_det_hz = hz
            else:
                self._smoothed_det_hz = (0.80 * self._smoothed_det_hz) + (0.20 * hz)
        self._last_emit_time = emit_ts

    def _emit_target_status(self, detections: list[dict[str, Any]], latency_ms: float, infer_ms: float):
        now = time.perf_counter()
        count = len(detections)
        top_conf = max((det.get("conf", 0.0) for det in detections), default=0.0)
        hz = self._smoothed_det_hz
        status_key: tuple[Any, ...] = (
            count,
            round(top_conf, 2),
            int(latency_ms // 10),
            int(infer_ms // 10),
            int(hz),
        )
        text = (
            f"Detections: {count} | Regel={self.settings.target_mode_name} | "
            f"Top-Conf={top_conf:.2f} | Inference={infer_ms:.1f} ms | E2E={latency_ms:.1f} ms | "
            f"Det-Hz={hz:.1f}"
        )

        should_log = status_key != self._last_status_key or (now - self._last_status_log) >= 1.0
        if should_log:
            self.log_ready.emit(text)
            self._last_status_log = now
            self._last_status_key = status_key

    def _handle_detection_error(self, exc: Exception) -> bool:
        if self.fallback_done or self.settings.device_string == "cpu":
            self.log_ready.emit(f"Erkennungsfehler: {exc}")
            return False

        self.fallback_done = True
        self.log_ready.emit(
            f"Gerät '{self.settings.device_name}' fehlgeschlagen, CPU-Fallback aktiv. Fehler: {exc}"
        )
        self.settings.device_name = "CPU (Fallback)"
        self.settings.device_string = "cpu"
        self.settings.use_half = False
        self.model = None
        self.names = None
        return True

    @Slot()
    def run_loop(self):
        self._stop_event.clear()
        self._running = True
        self._latest_packet = None

        try:
            self._ensure_ready()
            next_capture_due = time.perf_counter()

            while not self._stop_event.is_set():
                now = time.perf_counter()
                if now >= next_capture_due:
                    self._capture_latest_frame(now)
                    if self._stop_event.is_set():
                        break
                    next_capture_due = now + self._frame_interval

                packet = self._consume_latest_frame()
                if packet is None:
                    self._stop_event.wait(0.001)
                    continue

                inference_start = time.perf_counter()
                try:
                    detections = self._parse_results(self._predict_once(packet.frame))
                    if self._stop_event.is_set():
                        break
                except Exception as exc:  # noqa: BLE001
                    if self._handle_detection_error(exc):
                        self._ensure_ready()
                        continue
                    self._stop_event.wait(0.003)
                    continue

                now_done = time.perf_counter()
                infer_ms = (now_done - inference_start) * 1000.0
                latency_ms = (now_done - packet.captured_at) * 1000.0
                self._update_detection_frequency(now_done)

                meta = {
                    "captured_at": packet.captured_at,
                    "detected_at": now_done,
                    "inference_ms": infer_ms,
                    "latency_ms": latency_ms,
                    "detection_hz": self._smoothed_det_hz,
                }
                self.detections_ready.emit(detections, meta)
                self._emit_target_status(detections=detections, latency_ms=latency_ms, infer_ms=infer_ms)
        except Exception as exc:  # noqa: BLE001
            self.log_ready.emit(f"Detector-Loop beendet mit Fehler: {exc}")
        finally:
            self._running = False
            if self.sct is not None:
                try:
                    self.sct.close()
                except Exception:  # noqa: BLE001
                    pass
                self.sct = None
            self.stopped.emit()

    @Slot()
    def stop(self):
        if self._stop_event.is_set():
            return
        self.log_ready.emit("Detector-Stop angefordert.")
        self._stop_event.set()
