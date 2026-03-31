from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from constants import MODEL_PATH, QUALITY_TO_IMGSZ, ROI_HEIGHT, ROI_WIDTH, TEAM_TO_CLASSES


@dataclass
class AppSettings:
    model_path: Path
    screen_name: str
    quality_name: str
    imgsz: int
    team_name: str
    team_classes: Optional[list[int]]
    target_mode_name: str
    target_selection_mode: str
    device_name: str
    device_string: str
    use_half: bool
    screen_left: int
    screen_top: int
    screen_width: int
    screen_height: int
    capture_region: dict
    offset_x: int
    offset_y: int

    @property
    def roi_center(self) -> tuple[float, float]:
        return (
            self.capture_region["left"] + (self.capture_region["width"] / 2.0),
            self.capture_region["top"] + (self.capture_region["height"] / 2.0),
        )

    @property
    def screen_center(self) -> tuple[float, float]:
        return (
            self.screen_left + (self.screen_width / 2.0),
            self.screen_top + (self.screen_height / 2.0),
        )


def quality_to_imgsz(name: str) -> int:
    return QUALITY_TO_IMGSZ.get(name, QUALITY_TO_IMGSZ["Standard"])


def team_to_classes(name: str) -> Optional[list[int]]:
    return TEAM_TO_CLASSES.get(name, None)


def make_center_roi(monitor: dict) -> tuple[dict, int, int]:
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


def validate_model_path(path: Path = MODEL_PATH) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Die Datei 'model.pt' wurde nicht gefunden. "
            f"Erwarteter Ort: {path}"
        )
