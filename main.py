from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class RuntimeDeps:
    QApplication: object
    YOLO: object
    mss: object


def _dependency_error_message(missing: list[str]) -> str:
    install_hint = (
        "Fehlende Abhängigkeiten erkannt.\n\n"
        f"Fehlt: {', '.join(missing)}\n\n"
        "Installation (Windows, PowerShell):\n"
        "1) py -3 -m venv .venv\n"
        "2) .\\.venv\\Scripts\\Activate.ps1\n"
        "3) pip install -r requirements.txt\n\n"
        "Alternativ einzeln:\n"
        "pip install ultralytics PySide6 mss numpy torch"
    )
    return install_hint


def _show_fatal_dialog(message: str):
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Startfehler", message)
    except Exception:
        print(message)


def load_runtime_dependencies() -> RuntimeDeps:
    missing = []

    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append("numpy")

    try:
        import mss
    except Exception:
        missing.append("mss")
        mss = None

    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")

    try:
        from ultralytics import YOLO
    except Exception:
        missing.append("ultralytics")
        YOLO = None

    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        missing.append("PySide6")
        QApplication = None

    if missing:
        raise RuntimeError(_dependency_error_message(missing))

    return RuntimeDeps(QApplication=QApplication, YOLO=YOLO, mss=mss)


def main():
    try:
        runtime = load_runtime_dependencies()
    except Exception as exc:  # noqa: BLE001
        _show_fatal_dialog(str(exc))
        return 1

    from ui import MainWindow

    app = runtime.QApplication(sys.argv)
    window = MainWindow(runtime=runtime)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
