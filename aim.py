from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Any


@dataclass(frozen=True)
class AimSuggestion:
    """Reine Berechnungshilfe ohne jegliche Eingabe-/Maussteuerung."""

    target_center_x: float
    target_center_y: float
    screen_center_x: float
    screen_center_y: float
    delta_x: float
    delta_y: float
    pixel_distance: float
    angle_deg: float


def build_aim_suggestion(active_target: dict[str, Any] | None, screen_center: tuple[float, float]) -> AimSuggestion | None:
    """Berechnet die Richtung zur Zielmitte aus dem aktiven Tracking-Ziel.

    Diese Funktion sendet keine Inputs und steuert keine Maus/Tastatur.
    """
    if not active_target:
        return None

    target_center_x = float(active_target["center_x"])
    target_center_y = float(active_target["center_y"])
    screen_center_x, screen_center_y = float(screen_center[0]), float(screen_center[1])

    delta_x = target_center_x - screen_center_x
    delta_y = target_center_y - screen_center_y

    return AimSuggestion(
        target_center_x=target_center_x,
        target_center_y=target_center_y,
        screen_center_x=screen_center_x,
        screen_center_y=screen_center_y,
        delta_x=delta_x,
        delta_y=delta_y,
        pixel_distance=hypot(delta_x, delta_y),
        angle_deg=degrees(atan2(delta_y, delta_x)),
    )
