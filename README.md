# PY-YOLO-AIMBOT (GUI Overlay)

Einfache Desktop-GUI für YOLO-Erkennung mit transparentem Overlay. Fokus: stabile Erkennung, klare Bedienung, robuste Geräteauswahl und sauberer Fallback.

## Voraussetzungen
- Python 3.10+
- Windows empfohlen (Multi-Monitor + Overlay getestet auf Windows-Logik)
- `model.pt` muss **im selben Ordner** wie `start.py`/`main.py` liegen

## Installation
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Start (empfohlen)
```powershell
python start.py
```

`start.py` bleibt der zentrale Einstiegspunkt und führt **immer zuerst** den Pre-Launch-Updater aus. Erst danach startet die App (`main.main()`).

## Alternative ohne Updater
```powershell
python main.py
```
`main.py` startet direkt die GUI ohne Repo-Synchronisation. Das ist nützlich für lokales Debugging.

## Repo-Updater (ohne Manifest)
Der Updater arbeitet ohne `update_manifest.json` und synchronisiert direkt gegen dieses GitHub-Repository:

- Owner: `leonardgrimm13-netizen`
- Repo: `PY-YOLO-AIMBOT`
- Branch: `main`
- Primäre Prüfmethode: GitHub Trees API (`recursive=1`)

### Verhalten beim Start
1. `update.py` lädt den aktuellen Remote-Tree (Dateiliste + Blob-SHAs).
2. Neue Dateien im Repo werden lokal erstellt.
3. Geänderte Dateien werden lokal ersetzt (atomar über Temp-Dateien).
4. Verwaltete Dateien, die im Repo gelöscht wurden, werden lokal ebenfalls gelöscht.
5. Lokal fehlende/beschädigte verwaltete Dateien werden erneut geladen.
6. Ist GitHub nicht erreichbar, wird der lokale App-Start **nicht** blockiert.

### Wichtige Sicherheitsregeln
- Es wird **kein externer Server** und **kein Drittanbieter-Paket** für Updates genutzt.
- Gelöscht werden nur Dateien, die zuvor als „vom Updater verwaltet“ im lokalen State (`.update_state.json`) erfasst wurden.
- Nutzerdateien außerhalb der verwalteten Repo-Dateiliste bleiben unberührt.
- Nach tatsächlich angewendeten Änderungen startet `start.py` den Prozess einmal sauber neu, ohne Endlosschleife.

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
- `aim.py` steuert automatisch relative Mausbewegungen (Windows), sodass der erkannte Zielmittelpunkt zur Bildschirmmitte geführt wird.
- Bei Gerätefehlern wird automatisch sauber auf CPU zurückgefallen.
- Stoppen/Schließen ist robust umgesetzt: Der Detector-Thread wird zuerst sauber beendet, danach werden Overlay und Fenster geschlossen.
- Wenn Teamfilter nicht passt, muss die Klassenbelegung deines Modells geprüft werden (erwartet: 0=Orange, 1=Blau).

## Bekannte Limits zur Gerätekompatibilität
- Geräte werden ehrlich als erkannt/getestet gelistet; manche Backends sind erkannt, aber nicht stabil mit allen `.pt`-Modellen.
- Halbpräzision (`half=True`) wird nur auf CUDA verwendet.
- Auto-Modus bevorzugt getestete Beschleuniger (CUDA > XPU > MPS > CPU).
