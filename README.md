# PY-YOLO-AIMBOT (GUI Overlay)

Einfache Desktop-GUI für YOLO-Erkennung mit transparentem Overlay. Fokus: stabile Erkennung, klare Bedienung, robuste Geräteauswahl und sauberer Fallback.

## Voraussetzungen
- Python 3.10+
- Windows empfohlen (Multi-Monitor + Overlay getestet auf Windows-Logik)
- `model.pt` muss **im selben Ordner** wie `1.py`/`main.py` liegen
- `model.pt` muss **im selben Ordner** wie `start.py`/`main.py` liegen

## Installation
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Start
```powershell
python 1.py
python start.py
```
(Alternativ: `python main.py`)

## GUI-Optionen
- **Bildschirm**: Zielmonitor für Overlay
- **Erkennungsmodus**:
  - Schnell = 640
  - Standard = 960
  - Genau = 1280
- **Team**:
  - Beide = alle Klassen
  - Orange = nur Klasse 0
  - Blau = nur Klasse 1
- **Zielauswahl**:
  - Höchste Konfidenz (Standard)
  - Nächste zur Mitte (ROI-Zentrum)
- **Gerät**:
  - Auto (empfohlen)
  - CPU
  - NVIDIA CUDA (falls testbar)
  - Intel XPU (falls testbar)
  - Apple MPS (falls testbar)

## Wichtige Hinweise
- ROI ist immer aktiv, immer mittig und nicht als GUI-Option sichtbar.
- Das Modell kann intern mehrere Roh-Detections liefern; visualisiert wird immer genau **ein** aktives Ziel.
- Im Overlay werden nur aktive Zielbox, Zielmittelpunkt und eine Hilfslinie vom Bildschirmzentrum gezeichnet.
- Im Status-Log werden Zielstatus, Mittelpunkt, Konfidenz und Auswahlregel ausgegeben.
- `aim.py` liefert nur optionale Richtungs-/Delta-Berechnungen zur Zielmitte (read-only) und steuert **keine** Eingaben.
- Bei Gerätefehlern wird automatisch sauber auf CPU zurückgefallen.
- Wenn Teamfilter nicht passt, muss die Klassenbelegung deines Modells geprüft werden (erwartet: 0=Orange, 1=Blau).

## Bekannte Limits zur Gerätekompatibilität
- Geräte werden ehrlich als erkannt/getestet gelistet; manche Backends sind erkannt, aber nicht stabil mit allen `.pt`-Modellen.
- Halbpräzision (`half=True`) wird nur auf CUDA verwendet.
- Auto-Modus bevorzugt getestete Beschleuniger (CUDA > XPU > MPS > CPU).
