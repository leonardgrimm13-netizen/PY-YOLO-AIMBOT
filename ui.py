from __future__ import annotations

import time

from PySide6.QtCore import QThread, QTimer, Qt, QRect, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import AppSettings, make_center_roi, quality_to_imgsz, team_to_classes, validate_model_path
from constants import (
    APP_TITLE,
    FONT_SIZE,
    LINE_WIDTH,
    MODEL_PATH,
    QUALITY_TO_IMGSZ,
    SHOW_FPS,
    TARGET_FPS,
    TARGET_SELECTION_MODES,
    TEAM_TO_CLASSES,
)
from detector import DetectorWorker
from aim import AimController
from devices import DeviceInfo, list_available_devices, resolve_auto_device, should_use_half
from target_tracker import TargetTracker


class OverlayWindow(QWidget):
    _request_worker_stop = Signal()

    def __init__(self, settings: AppSettings, log_callback, yolo_cls, mss_mod):
        super().__init__()
        self.settings = settings
        self.log_callback = log_callback

        self.detections = []
        self.overlay_fps = 0.0
        self.last_paint = time.perf_counter()
        self.last_detection_meta = None
        self.aim_controller = AimController()
        self.target_tracker = TargetTracker(selection_mode=settings.target_selection_mode)
        self.active_target = None

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
            settings.screen_left,
            settings.screen_top,
            settings.screen_width,
            settings.screen_height,
        )

        self.detector_thread = QThread(self)
        self.detector = DetectorWorker(settings, yolo_cls=yolo_cls, mss_mod=mss_mod)
        self.detector.moveToThread(self.detector_thread)
        self.detector_thread.started.connect(self.detector.run_loop)
        self.detector.detections_ready.connect(self.on_detections_ready)
        self.detector.log_ready.connect(self.log_callback)
        self.detector.stopped.connect(self.detector_thread.quit)
        self.detector_thread.finished.connect(self.detector.deleteLater)
        self.detector_thread.finished.connect(self._on_detector_thread_finished)
        self._request_worker_stop.connect(self.detector.stop, Qt.ConnectionType.QueuedConnection)
        self.detector_thread.start()

        self._stop_in_progress = False
        self._stopped = False
        self._close_requested = False

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / TARGET_FPS))

        self._log_startup_summary()

    def _log_startup_summary(self):
        self.log_callback("Overlay gestartet.")
        self.log_callback(f"Bildschirm: {self.settings.screen_name}")
        self.log_callback(f"Modell: {self.settings.model_path}")
        self.log_callback(f"Qualität: {self.settings.quality_name} ({self.settings.imgsz})")
        self.log_callback(f"Team: {self.settings.team_name} -> Klassen {self.settings.team_classes}")
        self.log_callback(
            f"Zielauswahl: {self.settings.target_mode_name} ({self.settings.target_selection_mode})"
        )
        self.log_callback(f"Gerät: {self.settings.device_name} ({self.settings.device_string})")
        self.log_callback(f"ROI aktiv und mittig: {self.settings.capture_region}")

    @Slot()
    def tick(self):
        now = time.perf_counter()
        dt = now - self.last_paint
        self.last_paint = now
        if dt > 0:
            self.overlay_fps = 1.0 / dt

        self.active_target = self.target_tracker.get_active_target(now=now)
        if self.active_target is not None:
            self.aim_controller.aim_target(self.active_target, self.settings.screen_center, now=now)
            self.detections = [self.active_target]
        else:
            self.detections = []

        self.update()

    @Slot(list, dict)
    def on_detections_ready(self, detections, meta):
        now_ts = time.perf_counter()
        self.last_detection_meta = meta
        self.target_tracker.update_detections(
            detections=detections,
            roi_center=self.settings.roi_center,
            screen_center=self.settings.screen_center,
            captured_at=float(meta.get("captured_at", now_ts)),
            detected_at=float(meta.get("detected_at", now_ts)),
            now=now_ts,
        )

    def color_for_class(self, cls_id: int) -> QColor:
        palette = [
            QColor(255, 140, 0),
            QColor(0, 120, 255),
            QColor(255, 0, 0),
            QColor(0, 255, 0),
            QColor(255, 0, 255),
            QColor(0, 255, 255),
            QColor(255, 255, 0),
            QColor(180, 80, 255),
        ]
        return palette[cls_id % len(palette)]

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Arial", FONT_SIZE))

        for det in self.detections:
            pen = QPen(self.color_for_class(det["cls_id"]))
            pen.setWidth(LINE_WIDTH)
            painter.setPen(pen)
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
            center_x = int(det["bbox_center_x"])
            center_y = int(det["bbox_center_y"])
            aim_x = int(det.get("aim_x", center_x))
            aim_y = int(det.get("aim_y", center_y))

            screen_pen = QPen(QColor(255, 255, 255, 180))
            screen_pen.setWidth(max(1, LINE_WIDTH - 1))
            painter.setPen(screen_pen)
            painter.drawLine(
                int(self.settings.screen_center[0]),
                int(self.settings.screen_center[1]),
                aim_x,
                aim_y,
            )

            painter.setPen(pen)
            painter.drawEllipse(center_x - 3, center_y - 3, 6, 6)
            painter.drawEllipse(aim_x - 4, aim_y - 4, 8, 8)
            painter.drawLine(aim_x - 8, aim_y, aim_x + 8, aim_y)
            painter.drawLine(aim_x, aim_y - 8, aim_x, aim_y + 8)

            label = det.get("label", "")
            if label:
                metrics = painter.fontMetrics()
                text_w = metrics.horizontalAdvance(label) + 10
                text_h = metrics.height() + 6
                label_x = x1
                label_y = max(0, y1 - text_h)
                painter.fillRect(label_x, label_y, text_w, text_h, QColor(0, 0, 0, 180))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(label_x + 5, label_y + text_h - 6, label)

        if SHOW_FPS:
            painter.fillRect(10, 10, 160, 28, QColor(0, 0, 0, 180))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(18, 30, f"Overlay FPS: {self.overlay_fps:.1f}")

        painter.end()

    def stop_overlay(self) -> bool:
        if self._stopped:
            return True
        if self._stop_in_progress:
            return not self.detector_thread.isRunning()

        self._stop_in_progress = True
        self._close_requested = True
        self.timer.stop()
        self.log_callback("Stoppen angefordert …")
        self._request_worker_stop.emit()
        if not self.detector_thread.wait(5000):
            self.log_callback(
                "Warnung: Detector-Thread reagiert nicht rechtzeitig beim Stoppen. "
                "Bitte nach kurzer Zeit erneut versuchen."
            )
            self._stop_in_progress = False
            return False
        if not self._stopped:
            self._on_detector_thread_finished()
        return self._stopped

    def closeEvent(self, event):  # noqa: N802
        if self._stopped:
            super().closeEvent(event)
            return
        self._close_requested = True
        self.stop_overlay()
        event.ignore()

    @Slot()
    def _on_detector_thread_finished(self):
        self._stopped = True
        self._stop_in_progress = False
        self.log_callback("Detector-Thread sauber beendet.")
        if self._close_requested:
            self.close()


class MainWindow(QMainWindow):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self.overlay = None
        self._overlay_stopping = False
        self.setWindowTitle(APP_TITLE)
        self.resize(680, 560)

        self.available_devices: list[DeviceInfo] = []

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
        self.quality_combo = QComboBox()
        self.team_combo = QComboBox()
        self.target_mode_combo = QComboBox()
        self.device_combo = QComboBox()

        self.fill_screens()
        self.quality_combo.addItems(list(QUALITY_TO_IMGSZ.keys()))
        self.quality_combo.setCurrentText("Standard")
        self.team_combo.addItems(list(TEAM_TO_CLASSES.keys()))
        self.team_combo.setCurrentText("Beide")
        self.target_mode_combo.addItems(list(TARGET_SELECTION_MODES.keys()))
        self.target_mode_combo.setCurrentText("Höchste Konfidenz")

        self.refresh_btn = QPushButton("Geräte neu prüfen")
        self.refresh_btn.clicked.connect(self.refresh_devices)

        settings_layout.addWidget(QLabel("Bildschirm"))
        settings_layout.addWidget(self.screen_combo)
        settings_layout.addWidget(QLabel("Erkennungsmodus"))
        settings_layout.addWidget(self.quality_combo)
        settings_layout.addWidget(QLabel("Team"))
        settings_layout.addWidget(self.team_combo)
        settings_layout.addWidget(QLabel("Zielauswahl"))
        settings_layout.addWidget(self.target_mode_combo)
        settings_layout.addWidget(QLabel("Gerät"))
        settings_layout.addWidget(self.device_combo)
        settings_layout.addWidget(self.refresh_btn)

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
        self.refresh_devices()

    def log(self, text: str):
        self.log_box.appendPlainText(text)

    def fill_screens(self):
        self.screen_combo.clear()
        for i, screen in enumerate(QApplication.screens()):
            geo = screen.geometry()
            name = screen.name() or f"Screen {i + 1}"
            self.screen_combo.addItem(f"{i}: {name} ({geo.width()}x{geo.height()})", i)

    def refresh_devices(self):
        self.available_devices, logs = list_available_devices()
        self.device_combo.clear()
        for info in self.available_devices:
            suffix = "" if info.kind == "auto" else f" [{info.note}]"
            self.device_combo.addItem(f"{info.display_name}{suffix}", info.to_ui_data())
        self.log(f"Geräte neu geprüft. Gefunden: {len(self.available_devices)}")
        for line in logs:
            self.log(line)

    def _get_monitor_for_screen(self, screen_index: int):
        with self.runtime.mss.mss() as sct:
            monitors = sct.monitors
            if screen_index + 1 >= len(monitors):
                raise RuntimeError(f"Monitorindex {screen_index} außerhalb gültiger Range {len(monitors)-1}")
            return monitors[screen_index + 1]

    def _validate_team_mapping(self):
        expected = {"Orange": [0], "Blau": [1]}
        if self.team_combo.currentText() in expected:
            self.log(
                "Hinweis: Team-Filter nimmt an, dass Klasse 0=Orange und Klasse 1=Blau ist. "
                "Bitte prüfen, ob dein Modell diese Klassenbelegung nutzt."
            )

    def build_settings(self) -> AppSettings:
        validate_model_path(MODEL_PATH)
        screen_index = self.screen_combo.currentData()
        qt_screen = QApplication.screens()[screen_index]
        geom = qt_screen.geometry()
        monitor = self._get_monitor_for_screen(screen_index)
        capture_region, offset_x, offset_y = make_center_roi(monitor)

        quality_name = self.quality_combo.currentText()
        team_name = self.team_combo.currentText()
        target_mode_name = self.target_mode_combo.currentText()
        selected_info = self.device_combo.currentData()

        auto_resolved = False
        if selected_info["kind"] == "auto":
            resolved = resolve_auto_device(self.available_devices)
            selected_info = resolved.to_ui_data()
            auto_resolved = True

        if auto_resolved:
            self.log(f"Auto-Modus auf reales Gerät aufgelöst: {selected_info['name']}")

        self._validate_team_mapping()

        return AppSettings(
            model_path=MODEL_PATH,
            screen_name=self.screen_combo.currentText(),
            quality_name=quality_name,
            imgsz=quality_to_imgsz(quality_name),
            team_name=team_name,
            team_classes=team_to_classes(team_name),
            target_mode_name=target_mode_name,
            target_selection_mode=TARGET_SELECTION_MODES[target_mode_name],
            device_name=selected_info["name"],
            device_string=selected_info["device"],
            use_half=should_use_half(selected_info["device"]),
            screen_left=geom.x(),
            screen_top=geom.y(),
            screen_width=geom.width(),
            screen_height=geom.height(),
            capture_region=capture_region,
            offset_x=offset_x,
            offset_y=offset_y,
        )

    def start_overlay(self):
        if self.overlay is not None or self._overlay_stopping:
            self.log("Overlay läuft bereits.")
            return

        try:
            settings = self.build_settings()
            self.overlay = OverlayWindow(
                settings,
                log_callback=self.log,
                yolo_cls=self.runtime.YOLO,
                mss_mod=self.runtime.mss,
            )
            self.overlay.destroyed.connect(self._on_overlay_destroyed)
            # show() + Geometrie ist für Multi-Monitor stabiler als showFullScreen().
            self.overlay.show()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.log("Overlay-Fenster erstellt.")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Start-Fehler: {exc}")

    def stop_overlay(self):
        if self.overlay is None:
            return
        if self._overlay_stopping:
            self.log("Stoppen läuft bereits …")
            return

        self._overlay_stopping = True
        self.stop_btn.setEnabled(False)
        try:
            stopped = self.overlay.stop_overlay()
        except Exception as exc:  # noqa: BLE001
            stopped = False
            self.log(f"Stop-Fehler: {exc}")

        if stopped:
            self.start_btn.setEnabled(True)
            self.log("Overlay gestoppt.")
        else:
            self.stop_btn.setEnabled(self.overlay is not None)
        self._overlay_stopping = False

    def closeEvent(self, event):  # noqa: N802
        if self.overlay is not None:
            self.stop_overlay()
            if self.overlay is not None:
                event.ignore()
                return
        super().closeEvent(event)

    @Slot()
    def _on_overlay_destroyed(self):
        self.overlay = None
        self._overlay_stopping = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
