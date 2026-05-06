"""Shared helpers for path handling and temp directory management."""
from __future__ import annotations

import atexit
import hashlib
import multiprocessing
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_NETWORK_PATH_PREFIXES: tuple[str, ...] = (
    "/n/",
    "/net/",
    "/gpfs/",
    "/lustre/",
    "/panfs/",
    "/afs/",
)
_FALLBACK_DIRNAME = "calibrated-temp"


def allows_remote_tempdirs() -> bool:
    """Return True if remote temp paths are allowed."""

    return os.environ.get("QA_ALLOW_REMOTE_TEMP", "0") == "1"


def looks_remote_path(path: Path) -> bool:
    normalized = str(path)
    return any(normalized.startswith(prefix) for prefix in _NETWORK_PATH_PREFIXES)


def safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        return


def cleanup_stray_tempdirs(root: Path) -> None:
    for candidate in root.glob("pymp-*"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)


def _register_tempdir_cleanup(path: Path) -> None:
    def _cleanup() -> None:
        parent = getattr(multiprocessing, "parent_process", None)
        if parent is not None and parent() is not None:
            return
        if multiprocessing.current_process().name != "MainProcess":
            return
        safe_rmtree(path)

    atexit.register(_cleanup)


def _tempdir_supports_unix_sockets(directory: Path) -> tuple[bool, str | None]:
    suffix = hashlib.sha1(os.urandom(16)).hexdigest()[:8]
    probe = directory / f".sock-{os.getpid()}-{suffix}"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(probe))
    except OSError as exc:  # noqa: PERF203 - clarity beats micro-opts
        return False, f"unable to bind UNIX socket in {directory}: {exc}"
    finally:
        try:
            sock.close()
        finally:
            try:
                probe.unlink()
            except FileNotFoundError:
                pass
    return True, None


def configure_temp_directory(base_dir: Path, run_name: str) -> Path:
    temp_base = base_dir.expanduser().resolve()
    digest = hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:12]
    system_tmp = Path(tempfile.gettempdir()).resolve()
    fallback_base = (system_tmp / _FALLBACK_DIRNAME).resolve()
    prefer_local = looks_remote_path(temp_base) and not allows_remote_tempdirs()
    candidates: list[Path]
    if prefer_local and fallback_base != temp_base:
        candidates = [fallback_base, temp_base]
    else:
        candidates = [temp_base]
        if fallback_base != temp_base:
            candidates.append(fallback_base)
    temp_root: Path | None = None
    fallback_reason: str | None = None
    for candidate_base in candidates:
        try:
            candidate_base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            fallback_reason = f"failed to create {candidate_base}: {exc}"
            continue
        candidate_root = (candidate_base / f"run-{digest}").resolve()
        shutil.rmtree(candidate_root, ignore_errors=True)
        try:
            candidate_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            fallback_reason = f"failed to create {candidate_root}: {exc}"
            continue
        supports_sockets, reason = _tempdir_supports_unix_sockets(candidate_root)
        if supports_sockets:
            temp_root = candidate_root
            if candidate_base != temp_base:
                message = fallback_reason or "primary temp root unusable"
                print(
                    f"[tempdir] Falling back to {temp_root} ({message}).",
                    file=sys.stderr,
                )
            break
        fallback_reason = reason or "unknown temp directory failure"
        shutil.rmtree(candidate_root, ignore_errors=True)
    if temp_root is None:
        raise RuntimeError(
            f"Unable to configure a writable temp directory under {temp_base}: {fallback_reason}"
        )
    marker = temp_root / "run_name.txt"
    marker.write_text(run_name + "\n", encoding="utf-8")
    for variable in ("TMPDIR", "TMP", "TEMP"):
        os.environ[variable] = str(temp_root)
    tempfile.tempdir = str(temp_root)
    multiprocessing.current_process()._config["tempdir"] = str(temp_root)
    _register_tempdir_cleanup(temp_root)
    return temp_root
