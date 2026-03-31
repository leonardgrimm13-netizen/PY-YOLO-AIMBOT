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
    DETECT_FPS,
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
from devices import DeviceInfo, list_available_devices, resolve_auto_device, should_use_half


class OverlayWindow(QWidget):
    request_detect = Signal()

    def __init__(self, settings: AppSettings, log_callback, yolo_cls, mss_mod):
        super().__init__()
        self.settings = settings
        self.log_callback = log_callback

        self.detections = []
        self.overlay_fps = 0.0
        self.last_paint = time.perf_counter()
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
            settings.screen_left,
            settings.screen_top,
            settings.screen_width,
            settings.screen_height,
        )

        self.detector_thread = QThread()
        self.detector = DetectorWorker(settings, yolo_cls=yolo_cls, mss_mod=mss_mod)
        self.detector.moveToThread(self.detector_thread)
        self.request_detect.connect(self.detector.detect_once)
        self.detector.detections_ready.connect(self.on_detections_ready)
        self.detector.log_ready.connect(self.log_callback)
        self.detector.finished_one.connect(self.on_detector_finished)
        self.detector_thread.start()

        self.timer = QTimer(self)
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

        if (now - self.last_detect_request) >= self.detect_interval and not self.detector_busy:
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
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
            center_x = int(det["center_x"])
            center_y = int(det["center_y"])

            screen_pen = QPen(QColor(255, 255, 255, 180))
            screen_pen.setWidth(max(1, LINE_WIDTH - 1))
            painter.setPen(screen_pen)
            painter.drawLine(
                int(self.settings.screen_center[0]),
                int(self.settings.screen_center[1]),
                center_x,
                center_y,
            )

            painter.setPen(pen)
            painter.drawEllipse(center_x - 4, center_y - 4, 8, 8)
            painter.drawLine(center_x - 8, center_y, center_x + 8, center_y)
            painter.drawLine(center_x, center_y - 8, center_x, center_y + 8)

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
            painter.fillRect(10, 10, 130, 28, QColor(0, 0, 0, 180))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(18, 30, f"Overlay FPS: {self.overlay_fps:.1f}")

        painter.end()

    def stop_overlay(self):
        self.timer.stop()
        self.detector_thread.quit()
        if not self.detector_thread.wait(3000):
            self.log_callback("Warnung: Detector-Thread reagiert nicht rechtzeitig beim Stoppen.")
        self.close()

    def closeEvent(self, event):  # noqa: N802
        try:
            self.timer.stop()
            self.detector_thread.quit()
            self.detector_thread.wait(3000)
        except Exception:
            pass
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self.overlay = None
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
        if self.overlay is not None:
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
        try:
            self.overlay.stop_overlay()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Stop-Fehler: {exc}")
        self.overlay = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log("Overlay gestoppt.")

    def closeEvent(self, event):  # noqa: N802
        self.stop_overlay()
        super().closeEvent(event)
