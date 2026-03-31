from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Iterable


@dataclass(frozen=True)
class TargetTracker:
    selection_mode: str = "highest_confidence"

    def select_best_target(
        self,
        detections: Iterable[dict[str, Any]],
        roi_center: tuple[float, float],
        screen_center: tuple[float, float],
    ) -> dict[str, Any] | None:
        enriched = [
            self._enrich_detection(det, roi_center=roi_center, screen_center=screen_center)
            for det in detections
        ]
        if not enriched:
            return None

        if self.selection_mode == "nearest_center":
            # Primär: Nähe zur ROI-Mitte; Tiebreaker: höhere Konfidenz.
            return min(enriched, key=lambda det: (det["distance_to_roi_center"], -det["conf"]))

        # Standard: höchste Konfidenz; Tiebreaker: Nähe zur ROI-Mitte.
        return max(enriched, key=lambda det: (det["conf"], -det["distance_to_roi_center"]))

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
