import sys
import time
from pathlib import Path

import numpy as np
import mss

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO

from PySide6.QtCore import Qt, QTimer, QRect, QObject, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QLabel,
    QPushButton,
    QComboBox,
    QPlainTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
)


# =========================================================
# FESTE EINSTELLUNGEN
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model.pt"

TARGET_FPS = 15          # Overlay-FPS fest
DETECT_FPS = 4.0         # Erkennung pro Sekunde
CONF = 0.45
IOU = 0.50
MAX_DET = 5

SHOW_LABELS = True
SHOW_FPS = False
LINE_WIDTH = 3
FONT_SIZE = 12

# ROI immer aktiv, mittig
ROI_WIDTH = 1280
ROI_HEIGHT = 720


# =========================================================
# HILFSFUNKTIONEN
# =========================================================
def quality_to_imgsz(name: str) -> int:
    mapping = {
        "Schnell": 640,
        "Standard": 960,
        "Genau": 1280,
    }
    return mapping.get(name, 960)


def team_to_classes(name: str):
    mapping = {
        "Beide": None,
        "Orange": [0],
        "Blau": [1],
    }
    return mapping.get(name, None)


def make_center_roi(monitor):
    width = min(ROI_WIDTH, monitor["width"])
    height = min(ROI_HEIGHT, monitor["height"])

    x = max(0, (monitor["width"] - width) // 2)
    y = max(0, (monitor["height"] - height) // 2)

    capture_region = {
        "left": monitor["left"] + x,
        "top": monitor["top"] + y,
        "width": width,
        "height": height,
    }
    return capture_region, x, y


def list_available_devices():
    devices = []

    devices.append(("Auto (empfohlen)", {
        "kind": "auto",
        "name": "Auto (empfohlen)",
        "device": "auto",
    }))

    devices.append(("CPU", {
        "kind": "torch",
        "name": "CPU",
        "device": "cpu",
    }))

    if torch is None:
        return devices

    # NVIDIA CUDA
    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = f"CUDA {i}"
                try:
                    _ = torch.empty(1, device=f"cuda:{i}")
                    devices.append((f"NVIDIA CUDA {i}: {name}", {
                        "kind": "torch",
                        "name": f"NVIDIA CUDA {i}: {name}",
                        "device": f"cuda:{i}",
                    }))
                except Exception:
                    pass
    except Exception:
        pass

    # Intel XPU
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            count = torch.xpu.device_count()
            for i in range(count):
                try:
                    name = torch.xpu.get_device_name(i)
                except Exception:
                    name = f"XPU {i}"
                try:
                    _ = torch.empty(1, device=f"xpu:{i}")
                    devices.append((f"Intel XPU {i}: {name}", {
                        "kind": "torch",
                        "name": f"Intel XPU {i}: {name}",
                        "device": f"xpu:{i}",
                    }))
                except Exception:
                    pass
    except Exception:
        pass

    # Apple MPS
    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                try:
                    _ = torch.empty(1, device="mps")
                    devices.append(("Apple MPS", {
                        "kind": "torch",
                        "name": "Apple MPS",
                        "device": "mps",
                    }))
                except Exception:
                    pass
    except Exception:
        pass

    return devices


def resolve_auto_device(devices):
    """
    Bevorzugung:
    1. CUDA
    2. XPU
    3. MPS
    4. CPU
    """
    # Erst echte Geräte außer Auto
    real_devices = [info for _, info in devices if info["kind"] != "auto"]

    for prefix in ("cuda:", "xpu:", "mps", "cpu"):
        for info in real_devices:
            dev = info["device"]
            if prefix.endswith(":"):
                if dev.startswith(prefix):
                    return info
            else:
                if dev == prefix:
                    return info

    return {
        "kind": "torch",
        "name": "CPU",
        "device": "cpu",
    }


def should_use_half(device_string: str) -> bool:
    return str(device_string).startswith("cuda:")


# =========================================================
# WORKER
# =========================================================
class DetectorWorker(QObject):
    detections_ready = Signal(list)
    log_ready = Signal(str)
    finished_one = Signal()

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

        self.model = None
        self.names = None
        self.sct = None

        self.fallback_done = False

    def _ensure_ready(self):
        if self.sct is None:
            self.sct = mss.mss()

        if self.model is None:
            self.log_ready.emit(f"Modell wird geladen: {self.settings['model_path']}")
            self.model = YOLO(str(self.settings["model_path"]))
            self.names = self.model.names
            self.log_ready.emit("Modell geladen.")

    def _run_predict(self, frame):
        return self.model.predict(
            source=frame,
            conf=CONF,
            iou=IOU,
            imgsz=self.settings["imgsz"],
            device=self.settings["device_string"],
            half=self.settings["use_half"],
            max_det=MAX_DET,
            classes=self.settings["team_classes"],
            augment=False,
            verbose=False,
        )

    @Slot()
    def detect_once(self):
        try:
            self._ensure_ready()

            shot = self.sct.grab(self.settings["capture_region"])
            frame = np.asarray(shot)[:, :, :3]

            results = self._run_predict(frame)

            r = results[0]
            boxes = r.boxes
            out = []

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.int().cpu().tolist()
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

                    out.append({
                        "x1": x1 + self.settings["offset_x"],
                        "y1": y1 + self.settings["offset_y"],
                        "x2": x2 + self.settings["offset_x"],
                        "y2": y2 + self.settings["offset_y"],
                        "cls_id": cls_id,
                        "label": label,
                    })

            self.detections_ready.emit(out)

        except Exception as e:
            # Einmal automatisch auf CPU zurückfallen
            if not self.fallback_done and self.settings["device_string"] != "cpu":
                self.fallback_done = True
                self.log_ready.emit(
                    f"Gerät '{self.settings['device_name']}' fehlgeschlagen. "
                    f"Wechsle automatisch auf CPU. Fehler: {e}"
                )
                self.settings["device_name"] = "CPU (Fallback)"
                self.settings["device_string"] = "cpu"
                self.settings["use_half"] = False
                self.model = None
                self.names = None
                try:
                    self._ensure_ready()
                    shot = self.sct.grab(self.settings["capture_region"])
                    frame = np.asarray(shot)[:, :, :3]
                    results = self._run_predict(frame)

                    r = results[0]
                    boxes = r.boxes
                    out = []

                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.int().cpu().tolist()
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

                            out.append({
                                "x1": x1 + self.settings["offset_x"],
                                "y1": y1 + self.settings["offset_y"],
                                "x2": x2 + self.settings["offset_x"],
                                "y2": y2 + self.settings["offset_y"],
                                "cls_id": cls_id,
                                "label": label,
                            })

                    self.detections_ready.emit(out)
                except Exception as e2:
                    self.log_ready.emit(f"Erkennungsfehler nach CPU-Fallback: {e2}")
            else:
                self.log_ready.emit(f"Erkennungsfehler: {e}")

        finally:
            self.finished_one.emit()


# =========================================================
# OVERLAY
# =========================================================
class OverlayWindow(QWidget):
    request_detect = Signal()

    def __init__(self, settings, log_callback=None):
        super().__init__()
        self.settings = settings
        self.log_callback = log_callback

        self.detections = []
        self.overlay_fps = 0.0
        self.last_paint_time = time.time()
        self.last_detect_request = 0.0
        self.detect_interval = 1.0 / max(0.1, DETECT_FPS)
        self.detector_busy = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.setGeometry(
            settings["screen_left"],
            settings["screen_top"],
            settings["screen_width"],
            settings["screen_height"]
        )

        self.detector_thread = QThread()
        self.detector = DetectorWorker(settings)
        self.detector.moveToThread(self.detector_thread)

        self.request_detect.connect(self.detector.detect_once)
        self.detector.detections_ready.connect(self.on_detections_ready)
        self.detector.log_ready.connect(self.on_log)
        self.detector.finished_one.connect(self.on_detector_finished)

        self.detector_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / TARGET_FPS))

        self._log("Overlay gestartet.")
        self._log(f"Bildschirm: {settings['screen_name']}")
        self._log(f"Qualität: {settings['quality_name']} ({settings['imgsz']})")
        self._log(f"Team: {settings['team_name']}")
        self._log(f"Gerät: {settings['device_name']}")
        self._log(f"ROI aktiv: {settings['capture_region']}")

    def _log(self, text):
        if self.log_callback:
            self.log_callback(text)

    @Slot()
    def tick(self):
        now = time.time()
        dt = now - self.last_paint_time
        self.last_paint_time = now

        if dt > 0:
            self.overlay_fps = 1.0 / dt

        if (now - self.last_detect_request >= self.detect_interval) and (not self.detector_busy):
            self.last_detect_request = now
            self.detector_busy = True
            self.request_detect.emit()

        self.update()

    @Slot(list)
    def on_detections_ready(self, detections):
        self.detections = detections

    @Slot()
    def on_detector_finished(self):
        self.detector_busy = False

    @Slot(str)
    def on_log(self, text):
        self._log(text)

    def color_for_class(self, cls_id):
        palette = [
            QColor(255, 140, 0),   # orange
            QColor(0, 120, 255),   # blau
            QColor(255, 0, 0),
            QColor(0, 255, 0),
            QColor(255, 0, 255),
            QColor(0, 255, 255),
            QColor(255, 255, 0),
            QColor(180, 80, 255),
        ]
        return palette[cls_id % len(palette)]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Arial", FONT_SIZE))

        for det in self.detections:
            color = self.color_for_class(det["cls_id"])
            pen = QPen(color)
            pen.setWidth(LINE_WIDTH)
            painter.setPen(pen)

            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

            if det["label"]:
                metrics = painter.fontMetrics()
                text = det["label"]
                text_w = metrics.horizontalAdvance(text) + 10
                text_h = metrics.height() + 6

                label_x = x1
                label_y = max(0, y1 - text_h)

                painter.fillRect(label_x, label_y, text_w, text_h, QColor(0, 0, 0, 180))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(label_x + 5, label_y + text_h - 6, text)

        if SHOW_FPS:
            txt = f"FPS: {self.overlay_fps:.1f}"
            painter.fillRect(10, 10, 100, 28, QColor(0, 0, 0, 180))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(18, 30, txt)

        painter.end()

    def stop_overlay(self):
        self.timer.stop()
        self.detector_thread.quit()
        self.detector_thread.wait()
        self.close()

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.detector_thread.quit()
            self.detector_thread.wait()
        except Exception:
            pass
        super().closeEvent(event)


# =========================================================
# GUI
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Overlay")
        self.resize(650, 520)

        self.overlay = None
        self.available_devices = list_available_devices()

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)

        info_group = QGroupBox("Einfach starten")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(QLabel("Lege deine Datei 'model.pt' in denselben Ordner wie dieses Script."))
        info_layout.addWidget(QLabel(f"Erwarteter Modellpfad: {MODEL_PATH}"))

        settings_group = QGroupBox("Einstellungen")
        settings_layout = QVBoxLayout(settings_group)

        self.screen_combo = QComboBox()
        self.fill_screens()

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Schnell", "Standard", "Genau"])
        self.quality_combo.setCurrentText("Standard")

        self.team_combo = QComboBox()
        self.team_combo.addItems(["Beide", "Orange", "Blau"])
        self.team_combo.setCurrentText("Beide")

        self.device_combo = QComboBox()
        self.fill_devices()

        refresh_btn = QPushButton("Geräte neu prüfen")
        refresh_btn.clicked.connect(self.refresh_devices)

        settings_layout.addWidget(QLabel("Bildschirm"))
        settings_layout.addWidget(self.screen_combo)

        settings_layout.addWidget(QLabel("Erkennungsmodus"))
        settings_layout.addWidget(self.quality_combo)

        settings_layout.addWidget(QLabel("Team"))
        settings_layout.addWidget(self.team_combo)

        settings_layout.addWidget(QLabel("Gerät"))
        settings_layout.addWidget(self.device_combo)
        settings_layout.addWidget(refresh_btn)

        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Starten")
        self.stop_btn = QPushButton("Stoppen")
        self.exit_btn = QPushButton("Beenden")

        self.start_btn.clicked.connect(self.start_overlay)
        self.stop_btn.clicked.connect(self.stop_overlay)
        self.exit_btn.clicked.connect(self.close)

        self.stop_btn.setEnabled(False)

        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addWidget(self.exit_btn)

        log_group = QGroupBox("Status")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)

        root.addWidget(info_group)
        root.addWidget(settings_group)
        root.addLayout(button_row)
        root.addWidget(log_group)

        self.log("Programm bereit.")
        self.log(f"Gefundene Geräte: {len(self.available_devices)}")

    def fill_screens(self):
        self.screen_combo.clear()
        screens = QApplication.screens()
        for i, s in enumerate(screens):
            geo = s.geometry()
            name = s.name() or f"Screen {i + 1}"
            self.screen_combo.addItem(f"{i}: {name} ({geo.width()}x{geo.height()})", i)

    def fill_devices(self):
        self.device_combo.clear()
        for display_name, info in self.available_devices:
            self.device_combo.addItem(display_name, info)

    def refresh_devices(self):
        self.available_devices = list_available_devices()
        self.fill_devices()
        self.log(f"Geräte neu geprüft. Gefunden: {len(self.available_devices)}")

    def log(self, text):
        self.log_box.appendPlainText(text)

    def build_settings(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Die Datei 'model.pt' wurde nicht gefunden.\n"
                f"Erwarteter Ort: {MODEL_PATH}"
            )

        screen_index = self.screen_combo.currentData()
        qt_screen = QApplication.screens()[screen_index]
        geom = qt_screen.geometry()

        with mss.mss() as sct:
            mon = sct.monitors[screen_index + 1]

        capture_region, offset_x, offset_y = make_center_roi(mon)

        quality_name = self.quality_combo.currentText()
        imgsz = quality_to_imgsz(quality_name)

        team_name = self.team_combo.currentText()
        team_classes = team_to_classes(team_name)

        selected_info = self.device_combo.currentData()
        if selected_info["kind"] == "auto":
            selected_info = resolve_auto_device(self.available_devices)

        device_name = selected_info["name"]
        device_string = selected_info["device"]

        return {
            "model_path": MODEL_PATH,
            "screen_name": self.screen_combo.currentText(),
            "quality_name": quality_name,
            "imgsz": imgsz,
            "team_name": team_name,
            "team_classes": team_classes,
            "device_name": device_name,
            "device_string": device_string,
            "use_half": should_use_half(device_string),
            "screen_left": geom.x(),
            "screen_top": geom.y(),
            "screen_width": geom.width(),
            "screen_height": geom.height(),
            "capture_region": capture_region,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }

    def start_overlay(self):
        if self.overlay is not None:
            self.log("Overlay läuft bereits.")
            return

        try:
            settings = self.build_settings()
            self.overlay = OverlayWindow(settings, log_callback=self.log)
            self.overlay.showFullScreen()

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.log("Overlay-Fenster erstellt.")

        except Exception as e:
            self.log(f"Start-Fehler: {e}")

    def stop_overlay(self):
        if self.overlay is None:
            return

        try:
            self.overlay.stop_overlay()
        except Exception as e:
            self.log(f"Stop-Fehler: {e}")

        self.overlay = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log("Overlay gestoppt.")

    def closeEvent(self, event):
        try:
            self.stop_overlay()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()