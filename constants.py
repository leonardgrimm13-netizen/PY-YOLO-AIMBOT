from pathlib import Path

APP_TITLE = "YOLO Overlay"
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model.pt"

# Feste Laufzeitparameter
TARGET_FPS = 60
DETECT_FPS = 18.0
CONF = 0.45
IOU = 0.50
# Robustere Strategie: Modell darf mehrere Kandidaten liefern, Tracker reduziert sauber auf genau 1 aktives Ziel.
MAX_DET = 10

SHOW_LABELS = True
SHOW_FPS = False
LINE_WIDTH = 3
FONT_SIZE = 12

# ROI immer aktiv und mittig
ROI_WIDTH = 1280
ROI_HEIGHT = 720

# Interner Zielpunkt innerhalb der Box (0.0 = links/oben, 1.0 = rechts/unten)
AIM_ANCHOR_X_RATIO = 0.50
AIM_ANCHOR_Y_RATIO = 0.43

QUALITY_TO_IMGSZ = {
    "Schnell": 640,
    "Standard": 960,
    "Genau": 1280,
}

TEAM_TO_CLASSES = {
    "Beide": None,
    "Orange": [0],
    "Blau": [1],
}

TARGET_SELECTION_MODES = {
    "Höchste Konfidenz": "highest_confidence",
    "Nächste zur Mitte": "nearest_center",
}
