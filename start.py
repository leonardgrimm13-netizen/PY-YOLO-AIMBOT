"""Kompatibilitäts-Launcher für bestehende Startgewohnheiten."""

from __future__ import annotations

import sys

from update import restart_application, run_prelaunch_update


def _launch_main() -> int:
    from main import main

    return main()


if __name__ == "__main__":
    update_result = run_prelaunch_update()
    if update_result.restart_required:
        if restart_application():
            raise SystemExit(0)
        raise SystemExit(1)

    raise SystemExit(_launch_main())
