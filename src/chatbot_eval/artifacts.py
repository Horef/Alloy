from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings
from .models import ChatbotResult, EvaluationRecord, SilverQuestion


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for supplied in paths:
        resolved = supplied.resolve()
        files = [resolved] if resolved.is_file() else sorted(path for path in resolved.rglob("*") if path.is_file())
        for path in files:
            inventory.append({
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    return inventory


def evaluation_fingerprint(question: SilverQuestion, result: ChatbotResult | None = None) -> str:
    value: dict[str, Any] = {
        "question": {
            "id": question.id,
            "question": question.question,
            "expected_answer": question.expected_answer,
            "answerable": question.answerable,
            "expected_behavior": question.expected_behavior.value,
            "reference_claims": question.reference_claims,
            "sources": [source.model_dump(mode="json") for source in question.sources],
        }
    }
    if result is not None:
        value["premade_result"] = result.model_dump(mode="json")
    return _json_hash(value)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


class EvaluationCheckpoint:
    def __init__(self, path: Path, fingerprints: dict[str, str], *, resume: bool):
        self.path = path
        self.fingerprints = fingerprints
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            atomic_write_text(self.path, "")

    def load(self) -> dict[str, EvaluationRecord]:
        if not self.path.exists():
            return {}
        completed: dict[str, EvaluationRecord] = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                record = EvaluationRecord.model_validate(item["record"])
                fingerprint = str(item["input_fingerprint"])
            except Exception as exc:
                raise ValueError(f"Invalid checkpoint entry at line {line_number}: {exc}") from exc
            identifier = record.question.id
            if identifier not in self.fingerprints:
                raise ValueError(f"Checkpoint contains question {identifier!r}, which is not in the current input")
            if fingerprint != self.fingerprints[identifier]:
                raise ValueError(f"Checkpoint input mismatch for question {identifier!r}; start a new run")
            completed[identifier] = record
        return completed

    def append(self, record: EvaluationRecord) -> None:
        item = {
            "input_fingerprint": self.fingerprints[record.question.id],
            "record": record.model_dump(mode="json"),
            "saved_at": _now(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _implementation_hash() -> str:
    package_root = Path(__file__).resolve().parent
    values = [
        {"file": path.name, "sha256": file_sha256(path)}
        for path in sorted(package_root.glob("*.py"))
    ]
    return _json_hash(values)


class RunManifest:
    def __init__(
        self,
        output_dir: Path,
        *,
        command: str,
        settings: Settings,
        inputs: list[Path],
        parameters: dict[str, Any],
    ):
        safe_settings = asdict(settings)
        safe_settings.pop("api_key", None)
        self.path = output_dir / "run_manifest.json"
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "module_version": __version__,
            "command": command,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "git_commit": _git_commit(),
            "git_worktree_dirty": _git_dirty(),
            "implementation_sha256": _implementation_hash(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "settings": safe_settings,
            "parameters": parameters,
            "inputs": input_inventory(inputs),
            "results": {},
        }

    def __enter__(self) -> "RunManifest":
        atomic_write_text(self.path, json.dumps(self.data, ensure_ascii=False, indent=2))
        return self

    def complete(self, **results: Any) -> None:
        self.data["status"] = "completed"
        self.data["results"] = results

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.data["finished_at"] = _now()
        if exc is not None:
            self.data["status"] = "failed"
            self.data["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        atomic_write_text(self.path, json.dumps(self.data, ensure_ascii=False, indent=2))
        return False
