"""Zentraler Starter mit vorgeschaltetem Repo-Updater."""

from __future__ import annotations

import os
import sys

from main import main
from update import run_prelaunch_update

RESTART_ENV = "PY_YOLO_AIMBOT_AFTER_UPDATE"


def _restart_self() -> None:
    os.environ[RESTART_ENV] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    update_result = run_prelaunch_update()
    if update_result.changed and os.environ.get(RESTART_ENV) != "1":
        _restart_self()
    main()
