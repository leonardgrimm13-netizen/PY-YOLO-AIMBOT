from pathlib import Path

APP_TITLE = "YOLO Overlay"
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model.pt"

# Feste Laufzeitparameter
TARGET_FPS = 15
DETECT_FPS = 4.0
CONF = 0.45
IOU = 0.50
MAX_DET = 5

SHOW_LABELS = True
SHOW_FPS = False
LINE_WIDTH = 3
FONT_SIZE = 12

# ROI immer aktiv und mittig
ROI_WIDTH = 1280
ROI_HEIGHT = 720

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
