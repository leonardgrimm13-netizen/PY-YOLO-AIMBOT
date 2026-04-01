from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

OWNER = "leonardgrimm13-netizen"
REPO = "PY-YOLO-AIMBOT"
BRANCH = "main"

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / ".update_state.json"
MANIFEST_NAME = "update_manifest.json"
MANIFEST_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{MANIFEST_NAME}"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}"

SMALL_FILE_THRESHOLD = 5 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 1024 * 512
REQUEST_TIMEOUT = 12
MAX_WORKERS = 4


@dataclass
class UpdatePlan:
    manifest: dict[str, Any]
    files_to_update: list[dict[str, Any]]
    state: dict[str, Any]
    state_updates: dict[str, Any]


@dataclass
class UpdateResult:
    restart_required: bool
    updated_files: list[str]
    requirements_changed: bool


class UpdateError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"[UPDATE] {msg}")


def _log_check(msg: str) -> None:
    print(f"[UPDATE][CHECK] {msg}")


def _log_download(msg: str) -> None:
    print(f"[UPDATE][DOWNLOAD] {msg}")


def _log_apply(msg: str) -> None:
    print(f"[UPDATE][APPLY] {msg}")


def _log_error(msg: str) -> None:
    print(f"[UPDATE][ERROR] {msg}")


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log_error(f"Konnte State-Datei nicht lesen: {exc}")
        return {}


def _save_state(state: dict[str, Any]) -> None:
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, STATE_PATH)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_target(path_str: str) -> Path:
    candidate = (BASE_DIR / path_str).resolve()
    if BASE_DIR not in candidate.parents and candidate != BASE_DIR:
        raise UpdateError(f"Unsicherer Pfad im Manifest: {path_str}")
    return candidate


def _http_get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes, dict[str, str]]:
    req = request.Request(url, headers=headers or {})
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            status = getattr(resp, "status", 200)
            data = resp.read()
            resp_headers = {k: v for k, v in resp.headers.items()}
            return status, data, resp_headers
    except error.HTTPError as exc:
        if exc.code == 304:
            return 304, b"", {k: v for k, v in exc.headers.items()}
        raise


def _fetch_manifest(state: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    headers: dict[str, str] = {}
    if state.get("manifest_etag"):
        headers["If-None-Match"] = state["manifest_etag"]
    if state.get("manifest_last_modified"):
        headers["If-Modified-Since"] = state["manifest_last_modified"]

    status, body, resp_headers = _http_get(MANIFEST_URL, headers=headers)
    state_updates = {
        "last_check": int(time.time()),
        "manifest_etag": resp_headers.get("ETag", state.get("manifest_etag")),
        "manifest_last_modified": resp_headers.get("Last-Modified", state.get("manifest_last_modified")),
    }

    if status == 304:
        _log_check("Manifest unverändert (HTTP 304).")
        return None, state_updates

    manifest = json.loads(body.decode("utf-8"))
    return manifest, state_updates


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("repo_owner") != OWNER or manifest.get("repo_name") != REPO:
        raise UpdateError("Manifest verweist auf ein anderes Repository.")
    if manifest.get("branch") != BRANCH:
        raise UpdateError("Manifest-Branch passt nicht zur Updater-Konfiguration.")


def _needs_update(entry: dict[str, Any]) -> bool:
    path = _safe_target(entry["path"])
    if not path.exists():
        return True

    expected_size = int(entry["size"])
    local_size = path.stat().st_size
    if local_size != expected_size:
        return True

    expected_sha = str(entry["sha256"]).lower()
    local_sha = _sha256_file(path).lower()
    return local_sha != expected_sha


def check_for_updates() -> UpdatePlan | None:
    state = _load_state()
    try:
        manifest, state_updates = _fetch_manifest(state)
    except Exception as exc:  # noqa: BLE001
        _log_error(f"Manifest-Check fehlgeschlagen: {exc}")
        _save_state({**state, "last_check": int(time.time()), "last_error": str(exc)})
        return None

    if manifest is None:
        _save_state({**state, **state_updates, "last_error": None})
        return UpdatePlan(manifest={}, files_to_update=[], state=state, state_updates=state_updates)

    try:
        _validate_manifest(manifest)
        files = manifest.get("files", [])
        files_to_update = [entry for entry in files if _needs_update(entry)]
    except Exception as exc:  # noqa: BLE001
        _log_error(f"Manifest ungültig oder Dateiprüfung fehlgeschlagen: {exc}")
        _save_state({**state, **state_updates, "last_error": str(exc)})
        return None

    _save_state({**state, **state_updates, "last_error": None})
    _log_check(f"{len(files_to_update)} Datei(en) benötigen ein Update.")
    return UpdatePlan(manifest=manifest, files_to_update=files_to_update, state=state, state_updates=state_updates)


def _download_to_temp(entry: dict[str, Any]) -> tuple[str, Path]:
    rel_path = entry["path"]
    target = _safe_target(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE_URL}/{rel_path}"

    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)

    digest = hashlib.sha256()
    total = 0

    try:
        req = request.Request(url)
        with request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp, tmp_path.open("wb") as tmp_file:
            while True:
                chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                tmp_file.write(chunk)
                digest.update(chunk)
                total += len(chunk)

        expected_size = int(entry["size"])
        expected_sha = str(entry["sha256"]).lower()
        if total != expected_size:
            raise UpdateError(f"Größe stimmt nicht für {rel_path}: {total} != {expected_size}")
        if digest.hexdigest().lower() != expected_sha:
            raise UpdateError(f"SHA256 stimmt nicht für {rel_path}")

        return rel_path, tmp_path
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def apply_updates(plan: UpdatePlan) -> UpdateResult:
    if not plan.files_to_update:
        return UpdateResult(restart_required=False, updated_files=[], requirements_changed=False)

    small_files = [f for f in plan.files_to_update if int(f["size"]) <= SMALL_FILE_THRESHOLD and f["path"] != "model.pt"]
    large_files = [f for f in plan.files_to_update if f not in small_files]

    downloads: dict[str, Path] = {}
    errors: list[str] = []

    if small_files:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(small_files))) as ex:
            future_map = {ex.submit(_download_to_temp, entry): entry["path"] for entry in small_files}
            for future in as_completed(future_map):
                path = future_map[future]
                try:
                    rel_path, tmp_path = future.result()
                    downloads[rel_path] = tmp_path
                    _log_download(f"Geladen: {rel_path}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{path}: {exc}")

    for entry in large_files:
        rel = entry["path"]
        try:
            rel_path, tmp_path = _download_to_temp(entry)
            downloads[rel_path] = tmp_path
            _log_download(f"Geladen: {rel_path}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rel}: {exc}")

    if errors:
        for tmp in downloads.values():
            tmp.unlink(missing_ok=True)
        for message in errors:
            _log_error(f"Download fehlgeschlagen: {message}")
        raise UpdateError("Mindestens ein Download fehlgeschlagen. Update wird verworfen.")

    update_order = sorted(downloads.keys(), key=lambda p: (p == "update.py", p))
    backups: list[tuple[Path, Path | None, bool]] = []
    applied: list[tuple[Path, Path | None, bool]] = []

    try:
        for rel_path in update_order:
            target = _safe_target(rel_path)
            tmp_path = downloads[rel_path]
            existed = target.exists()
            backup_path: Path | None = None

            if existed:
                fd, backup_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".bak", dir=str(target.parent))
                os.close(fd)
                backup_path = Path(backup_name)
                with target.open("rb") as src, backup_path.open("wb") as dst:
                    while True:
                        chunk = src.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        dst.write(chunk)

            backups.append((target, backup_path, existed))
            os.replace(tmp_path, target)
            applied.append((target, backup_path, existed))
            _log_apply(f"Ersetzt: {rel_path}")

        for _target, backup, _existed in backups:
            if backup:
                backup.unlink(missing_ok=True)

    except Exception as exc:  # noqa: BLE001
        _log_error(f"Fehler beim Anwenden: {exc}. Rolle Änderungen zurück.")
        for target, backup, existed in reversed(applied):
            try:
                if backup and backup.exists():
                    os.replace(backup, target)
                elif not existed and target.exists():
                    target.unlink(missing_ok=True)
            except Exception as rollback_exc:  # noqa: BLE001
                _log_error(f"Rollback fehlgeschlagen für {target}: {rollback_exc}")
        for _target, backup, _existed in backups:
            if backup and backup.exists():
                backup.unlink(missing_ok=True)
        for tmp in downloads.values():
            tmp.unlink(missing_ok=True)
        raise

    requirements_changed = "requirements.txt" in downloads
    return UpdateResult(restart_required=True, updated_files=update_order, requirements_changed=requirements_changed)


def restart_application() -> bool:
    try:
        subprocess.Popen([sys.executable, *sys.argv], cwd=str(BASE_DIR))
        _log("Anwendung wird neu gestartet.")
        return True
    except Exception as exc:  # noqa: BLE001
        _log_error(f"Neustart fehlgeschlagen: {exc}")
        return False


def run_prelaunch_update() -> UpdateResult:
    _log("Starte Pre-Launch Updateprüfung.")
    plan = check_for_updates()
    if plan is None:
        _log("Updateprüfung übersprungen, lokaler Start wird fortgesetzt.")
        return UpdateResult(restart_required=False, updated_files=[], requirements_changed=False)

    if not plan.files_to_update:
        _log("Keine Updates erforderlich.")
        return UpdateResult(restart_required=False, updated_files=[], requirements_changed=False)

    try:
        result = apply_updates(plan)
        new_state = {
            **plan.state,
            **plan.state_updates,
            "last_successful_update": int(time.time()),
            "manifest_version": plan.manifest.get("manifest_version"),
            "app_version": plan.manifest.get("app_version"),
            "manifest_commit": plan.manifest.get("commit_sha"),
            "last_error": None,
        }
        _save_state(new_state)

        if result.requirements_changed:
            _log("requirements.txt wurde aktualisiert. Ein Dependency-Sync wird empfohlen.")

        _log(f"Update erfolgreich ({len(result.updated_files)} Datei(en)).")
        return result
    except Exception as exc:  # noqa: BLE001
        _log_error(f"Update fehlgeschlagen: {exc}")
        state = _load_state()
        state.update({"last_error": str(exc), "last_failed_update": int(time.time())})
        _save_state(state)
        return UpdateResult(restart_required=False, updated_files=[], requirements_changed=False)
