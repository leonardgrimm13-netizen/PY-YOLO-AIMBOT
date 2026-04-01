from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from urllib import error, parse, request

OWNER = "leonardgrimm13-netizen"
REPO = "PY-YOLO-AIMBOT"
BRANCH = "main"

TREE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}"
USER_AGENT = "PY-YOLO-AIMBOT-Updater/1.0"

ROOT_DIR = Path(__file__).resolve().parent
STATE_PATH = ROOT_DIR / ".update_state.json"

SMALL_FILE_THRESHOLD = 2 * 1024 * 1024
MAX_PARALLEL_DOWNLOADS = 4

IGNORE_PREFIXES = (
    ".git/",
    ".venv/",
    "venv/",
    "__pycache__/",
)
IGNORE_EXACT = {
    ".update_state.json",
}
IGNORE_PATTERNS = (
    "*.tmp",
    "*.bak",
    "*.pyc",
    "*.pyo",
)


@dataclass(frozen=True)
class RemoteFile:
    path: str
    blob_sha: str
    size: int


@dataclass(frozen=True)
class LocalMeta:
    size: int
    sha256: str


@dataclass(frozen=True)
class UpdateResult:
    success: bool
    changed: bool
    tree_sha: str | None = None


class UpdateError(Exception):
    pass


def _log(scope: str, message: str) -> None:
    print(f"[UPDATE]{scope} {message}")


def _is_ignored(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized in IGNORE_EXACT:
        return True
    if any(normalized.startswith(prefix) for prefix in IGNORE_PREFIXES):
        return True
    if any(fnmatch(normalized, pattern) for pattern in IGNORE_PATTERNS):
        return True
    return False


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {
            "managed_files": {},
            "last_error": None,
            "etag": None,
            "last_tree_sha": None,
            "last_successful_check": None,
            "last_successful_update": None,
        }
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "managed_files": {},
            "last_error": "State konnte nicht gelesen werden, wurde zurückgesetzt.",
            "etag": None,
            "last_tree_sha": None,
            "last_successful_check": None,
            "last_successful_update": None,
        }


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _request_json(url: str, etag: str | None = None) -> tuple[int, dict[str, Any] | None, str | None]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    }
    if etag:
        headers["If-None-Match"] = etag
    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.status, payload, resp.headers.get("ETag")
    except error.HTTPError as exc:
        if exc.code == 304:
            return 304, None, etag
        msg = exc.read().decode("utf-8", errors="ignore")
        raise UpdateError(f"GitHub HTTP {exc.code}: {msg[:300]}") from exc
    except error.URLError as exc:
        raise UpdateError(f"GitHub nicht erreichbar: {exc}") from exc


def _build_remote_files(tree_payload: dict[str, Any]) -> dict[str, RemoteFile]:
    tree = tree_payload.get("tree", [])
    remote: dict[str, RemoteFile] = {}
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path = entry.get("path")
        blob_sha = entry.get("sha")
        size = int(entry.get("size") or 0)
        if not path or not blob_sha:
            continue
        if _is_ignored(path):
            continue
        remote[path] = RemoteFile(path=path, blob_sha=blob_sha, size=size)
    return remote


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _local_meta(path: Path) -> LocalMeta | None:
    if not path.exists() or not path.is_file():
        return None
    return LocalMeta(size=path.stat().st_size, sha256=_sha256_file(path))


def _download_one(remote_file: RemoteFile, staging_dir: Path) -> tuple[str, Path, LocalMeta]:
    quoted = parse.quote(remote_file.path)
    url = f"{RAW_BASE_URL}/{quoted}"
    req = request.Request(url, headers={"User-Agent": USER_AGENT})

    staging_path = staging_dir / remote_file.path
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = staging_path.with_suffix(staging_path.suffix + ".tmp")

    digest = hashlib.sha256()
    size = 0

    try:
        with request.urlopen(req, timeout=30) as resp, tmp_path.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        os.replace(tmp_path, staging_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise UpdateError(f"Download fehlgeschlagen ({remote_file.path}): {exc}") from exc

    return remote_file.path, staging_path, LocalMeta(size=size, sha256=digest.hexdigest())


def _apply_files(downloaded: list[tuple[str, Path, LocalMeta]]) -> None:
    for rel_path, staged_path, _ in downloaded:
        target = ROOT_DIR / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_path, target)


def _delete_removed(removed: list[str]) -> None:
    for rel_path in removed:
        path = ROOT_DIR / rel_path
        if path.exists() and path.is_file() and not _is_ignored(rel_path):
            _log("[DELETE]", f"Entferne {rel_path}")
            path.unlink(missing_ok=True)
            parent = path.parent
            while parent != ROOT_DIR:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent


def _should_download(
    rel_path: str,
    remote_file: RemoteFile,
    previous: dict[str, Any],
) -> bool:
    local_path = ROOT_DIR / rel_path
    previous_remote_sha = previous.get("remote_blob_sha")
    previous_local_sha = previous.get("local_sha256")
    previous_size = previous.get("size")

    local = _local_meta(local_path)
    if local is None:
        return True

    if previous_remote_sha != remote_file.blob_sha:
        return True

    if previous_local_sha and previous_size is not None:
        if local.size != int(previous_size):
            return True
        if local.sha256 != previous_local_sha:
            return True

    return False


def run_prelaunch_update() -> UpdateResult:
    state = _load_state()
    _log("[CHECK]", "Prüfe Remote-Repositoryzustand …")

    try:
        status, payload, etag = _request_json(TREE_URL, etag=state.get("etag"))
    except UpdateError as exc:
        state["last_error"] = str(exc)
        _save_state(state)
        _log("[ERROR]", f"{exc}. Lokaler Start wird fortgesetzt.")
        return UpdateResult(success=False, changed=False)

    if status == 304:
        _log("[TREE]", "Remote-Tree unverändert (ETag 304).")
        state["last_successful_check"] = int(time.time())
        state["last_error"] = None
        _save_state(state)
        return UpdateResult(success=True, changed=False, tree_sha=state.get("last_tree_sha"))

    if not payload:
        state["last_error"] = "Leere Antwort beim Tree-Request"
        _save_state(state)
        return UpdateResult(success=False, changed=False)

    if payload.get("truncated") is True:
        _log("[TREE]", "Warnung: GitHub Tree-Antwort ist abgeschnitten.")

    tree_sha = payload.get("sha")
    remote_files = _build_remote_files(payload)
    _log("[TREE]", f"{len(remote_files)} verwaltete Dateien im Remote-Tree gefunden.")

    managed_before: dict[str, Any] = state.get("managed_files", {})
    removed_files = sorted(path for path in managed_before.keys() if path not in remote_files and not _is_ignored(path))

    download_targets: list[RemoteFile] = []
    for rel_path, remote_file in remote_files.items():
        prev = managed_before.get(rel_path, {})
        if _should_download(rel_path, remote_file, prev):
            download_targets.append(remote_file)

    if not download_targets and not removed_files:
        _log("[CHECK]", "Keine Änderungen erforderlich.")
        state["last_successful_check"] = int(time.time())
        state["last_tree_sha"] = tree_sha
        state["etag"] = etag
        state["last_error"] = None
        _save_state(state)
        return UpdateResult(success=True, changed=False, tree_sha=tree_sha)

    _log("[DOWNLOAD]", f"{len(download_targets)} Datei(en) werden geladen, {len(removed_files)} Datei(en) werden entfernt.")

    downloaded: list[tuple[str, Path, LocalMeta]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="repo-sync-") as tmp_dir:
            staging_dir = Path(tmp_dir)
            small_files = [f for f in download_targets if f.size <= SMALL_FILE_THRESHOLD]
            large_files = [f for f in download_targets if f.size > SMALL_FILE_THRESHOLD]

            for remote_file in large_files:
                _log("[DOWNLOAD]", f"(stream) {remote_file.path}")
                downloaded.append(_download_one(remote_file, staging_dir))

            if small_files:
                workers = min(MAX_PARALLEL_DOWNLOADS, len(small_files))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_download_one, f, staging_dir): f for f in small_files}
                    for future in as_completed(futures):
                        downloaded.append(future.result())

            _apply_files(downloaded)
    except UpdateError as exc:
        state["last_error"] = str(exc)
        _save_state(state)
        _log("[ERROR]", f"{exc}. Änderungen wurden nicht angewendet.")
        return UpdateResult(success=False, changed=False)

    _delete_removed(removed_files)

    new_managed: dict[str, Any] = {}
    for rel_path, remote_file in remote_files.items():
        local = _local_meta(ROOT_DIR / rel_path)
        if local is None:
            continue
        new_managed[rel_path] = {
            "path": rel_path,
            "remote_blob_sha": remote_file.blob_sha,
            "local_sha256": local.sha256,
            "size": local.size,
        }

    now = int(time.time())
    state["managed_files"] = new_managed
    state["last_tree_sha"] = tree_sha
    state["etag"] = etag
    state["last_successful_check"] = now
    state["last_successful_update"] = now
    state["last_error"] = None

    if "requirements.txt" in {path for path, _, _ in downloaded}:
        _log("[APPLY]", "requirements.txt wurde aktualisiert. Bitte Abhängigkeiten bei Bedarf manuell installieren.")

    _save_state(state)
    _log("[APPLY]", "Synchronisierung erfolgreich abgeschlossen.")
    return UpdateResult(success=True, changed=True, tree_sha=tree_sha)


if __name__ == "__main__":
    result = run_prelaunch_update()
    raise SystemExit(0 if result.success else 1)
