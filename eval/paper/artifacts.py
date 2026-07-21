"""Strict prediction, resume, manifest, checksum, and packaging contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

from decode_utils import response_leak_flags

from .protocol import canonical_json, sha256_json


TECHNICAL_FAILURES = {"missing", "empty", "generation_error", "decode_leak", "duplicate"}
REFERENCE_FIELDS = {
    "answer",
    "answers",
    "caption",
    "gold",
    "gold_answer",
    "gold_label",
    "ground_truth",
    "label",
    "labels",
    "allowed_labels",
    "reference",
    "references",
    "reference_answer",
    "reference_caption",
    "reference_label",
    "reference_text",
    "target",
    "targets",
    "target_label",
}
RAW_DATA_FIELDS = {
    "annotation",
    "annotations",
    "conversation",
    "conversations",
    "dataset_row",
    "dataset_rows",
    "gated_source",
    "gated_source_data",
    "image",
    "image_bytes",
    "image_data",
    "image_path",
    "image_relpath",
    "locked_files",
    "messages",
    "raw_annotation",
    "raw_image",
    "raw_reference",
    "raw_source",
    "snapshot_files",
    "snapshot_path",
    "source_annotation",
    "source_file",
    "source_files",
    "source_id",
    "source_metadata",
    "source_object_id",
    "source_path",
    "source_record",
    "source_row",
    "object_id",
    "path",
    "records_file",
}

# Instance-level fields derived from the gated AstroVLBench snapshot.  These
# are not needed to reproduce any public aggregate and can reveal the hidden
# source grouping or row identity even after the canonical IDs are removed.
_GATED_SENSITIVE_FIELDS = {
    "asib_group",
    "cluster_id",
    "group",
    "row_index",
    "row_number",
}

# These output fields are produced by the evaluated model rather than supplied by
# a benchmark.  Keep them in the shareable audit bundle even though their names
# contain ``label`` or ``answer``.
PUBLIC_MODEL_OUTPUT_FIELDS = {
    "model_label",
    "predicted_answer",
    "predicted_label",
    "prediction_label",
}

_HASH_FIELD_MARKERS = ("sha256", "sha512", "checksum", "digest", "_hash")
_PUBLIC_RAW_IMAGE_SUFFIXES = {
    ".bmp",
    ".fit",
    ".fits",
    ".fts",
    ".gif",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_PUBLIC_FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".bin",
    ".db",
    ".feather",
    ".gz",
    ".h5",
    ".hdf5",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pth",
    ".pt",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".zip",
}
_PUBLIC_SOURCE_DIRECTORIES = {
    "annotations",
    "datasets",
    "gated",
    "images",
    "locked_snapshot",
    "raw",
    "references",
    "source_data",
    "source_files",
}
_PUBLIC_STRUCTURED_REDACTION_SUFFIXES = {".csv", ".json", ".jsonl", ".tsv"}
_PUBLIC_SENSITIVE_FILENAME_TOKENS = {
    "annotation",
    "annotations",
    "answer",
    "answers",
    "gated",
    "label",
    "labels",
    "reference",
    "references",
    "source",
}


class ArtifactError(RuntimeError):
    """Raised when a run cannot satisfy the frozen paper-artifact contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(value: Any) -> str:
    return sha256_bytes(str(value or "").encode("utf-8"))


def read_jsonl(path: str | Path, tolerate_truncated_tail: bool = False) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    lines = file_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if tolerate_truncated_tail and index == len(lines):
                break
            raise ArtifactError(f"{file_path}:{index}: invalid JSONL: {exc}") from exc
        if not isinstance(row, dict):
            raise ArtifactError(f"{file_path}:{index}: every JSONL row must be an object")
        rows.append(row)
    return rows


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def record_fingerprint(record: Mapping[str, Any], model_protocol_hash: str) -> str:
    return sha256_json(
        {
            "model_protocol_hash": model_protocol_hash,
            "id": record.get("id"),
            "image_sha256": record.get("image_sha256"),
            "prompt_sha256": record.get("prompt_sha256") or text_hash(record.get("prompt")),
            "reference_sha256": record.get("reference_sha256")
            or text_hash(record.get("reference")),
        }
    )


def technical_status(row: Mapping[str, Any], prompt: str | None = None) -> tuple[str, list[str]]:
    if row.get("error"):
        return "generation_error", []
    response = str(row.get("response") or "").strip()
    if not response:
        return "empty", []
    flags = list(row.get("leak_flags") or row.get("leak_flag") or [])
    if not flags:
        flags = response_leak_flags(response, prompt)
    if flags:
        return "decode_leak", sorted(set(str(flag) for flag in flags))
    return "ok", []


def build_attempt(
    *,
    record: Mapping[str, Any],
    model_label: str,
    model_revision: str,
    backend: str,
    run_id: str,
    suite_protocol_hash: str,
    model_protocol_hash: str,
    response: str = "",
    raw_response: str = "",
    generated_token_ids: Sequence[int] | None = None,
    prompt_token_count: int | None = None,
    termination_reason: str | None = None,
    rendered_prompt: str | None = None,
    latency_seconds: float | None = None,
    error: Mapping[str, Any] | str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    prompt = str(record.get("prompt") or "")
    reference = str(record.get("reference") or "")
    row: Dict[str, Any] = {
        **dict(record),
        "model_label": model_label,
        "model_revision": model_revision,
        "backend": backend,
        "run_id": run_id,
        "suite_protocol_hash": suite_protocol_hash,
        "model_protocol_hash": model_protocol_hash,
        "record_fingerprint": record_fingerprint(record, model_protocol_hash),
        "prompt_sha256": record.get("prompt_sha256") or text_hash(prompt),
        "reference_sha256": record.get("reference_sha256") or text_hash(reference),
        "rendered_prompt_sha256": text_hash(rendered_prompt or prompt),
        "raw_response": raw_response or response,
        "response": response,
        "generated_token_ids": list(generated_token_ids or []),
        "prompt_token_count": prompt_token_count,
        "completion_token_count": len(generated_token_ids or []),
        "termination_reason": termination_reason,
        "latency_seconds": latency_seconds,
        "error": error,
        "created_at_utc": utc_now(),
    }
    if rendered_prompt is not None:
        row["rendered_prompt"] = rendered_prompt
    if extra:
        row.update(extra)
    status, flags = technical_status(row, prompt)
    row["status"] = status
    row["leak_flags"] = flags
    return row


@dataclass
class ValidationReport:
    expected: int
    successful: int
    missing: list[str]
    extra: list[str]
    failed: Dict[str, str]
    duplicate_successes: list[str]

    @property
    def complete(self) -> bool:
        return not self.missing and not self.extra and not self.failed and not self.duplicate_successes

    def as_dict(self) -> Dict[str, Any]:
        return {
            "expected": self.expected,
            "successful": self.successful,
            "missing": self.missing,
            "extra": self.extra,
            "failed": self.failed,
            "duplicate_successes": self.duplicate_successes,
            "complete": self.complete,
        }


class PredictionStore:
    """Append attempts safely and finalize one fingerprint-matched success per ID."""

    def __init__(
        self,
        output_dir: str | Path,
        expected_records: Sequence[Mapping[str, Any]],
        model_protocol_hash: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_path = self.output_dir / "attempts.jsonl"
        self.predictions_path = self.output_dir / "predictions.jsonl"
        self.report_path = self.output_dir / "completion_report.json"
        self.expected = [dict(row) for row in expected_records]
        self.expected_by_id: Dict[str, Dict[str, Any]] = {}
        for row in self.expected:
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ArtifactError("Every expected record requires a non-empty id")
            if row_id in self.expected_by_id:
                raise ArtifactError(f"Duplicate expected record id: {row_id}")
            self.expected_by_id[row_id] = row
        self.model_protocol_hash = model_protocol_hash

    def attempts(self) -> list[dict[str, Any]]:
        return read_jsonl(self.attempts_path, tolerate_truncated_tail=True)

    def append(self, row: MutableMapping[str, Any]) -> None:
        row_id = str(row.get("id") or "")
        if row_id not in self.expected_by_id:
            raise ArtifactError(f"Attempt has unexpected id {row_id!r}")
        expected_fingerprint = record_fingerprint(
            self.expected_by_id[row_id], self.model_protocol_hash
        )
        if row.get("model_protocol_hash") != self.model_protocol_hash:
            raise ArtifactError(f"Attempt {row_id} has a mismatched model protocol hash")
        if row.get("record_fingerprint") != expected_fingerprint:
            raise ArtifactError(f"Attempt {row_id} has a mismatched record fingerprint")
        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.attempts_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def successes(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for row in self.attempts():
            row_id = str(row.get("id") or "")
            expected = self.expected_by_id.get(row_id)
            if expected is None:
                continue
            if row.get("model_protocol_hash") != self.model_protocol_hash:
                continue
            if row.get("record_fingerprint") != record_fingerprint(
                expected, self.model_protocol_hash
            ):
                continue
            status, _ = technical_status(row, str(expected.get("prompt") or ""))
            if status == "ok":
                result[row_id] = row
        return result

    def pending_records(self) -> list[Dict[str, Any]]:
        done = set(self.successes())
        return [row for row in self.expected if str(row["id"]) not in done]

    def validate(self) -> ValidationReport:
        attempts = self.attempts()
        successes: Dict[str, Dict[str, Any]] = {}
        success_counts: Dict[str, int] = {}
        latest_failure: Dict[str, str] = {}
        extra = sorted(
            {
                str(row.get("id") or "")
                for row in attempts
                if str(row.get("id") or "") not in self.expected_by_id
            }
        )
        for row in attempts:
            row_id = str(row.get("id") or "")
            expected = self.expected_by_id.get(row_id)
            if expected is None:
                continue
            fingerprint_ok = (
                row.get("model_protocol_hash") == self.model_protocol_hash
                and row.get("record_fingerprint")
                == record_fingerprint(expected, self.model_protocol_hash)
            )
            if not fingerprint_ok:
                latest_failure[row_id] = "protocol_mismatch"
                continue
            status, _ = technical_status(row, str(expected.get("prompt") or ""))
            if status == "ok":
                successes[row_id] = row
                success_counts[row_id] = success_counts.get(row_id, 0) + 1
                latest_failure.pop(row_id, None)
            else:
                latest_failure[row_id] = status
        missing = [row_id for row_id in self.expected_by_id if row_id not in successes]
        failed = {row_id: latest_failure[row_id] for row_id in missing if row_id in latest_failure}
        duplicate_successes = sorted(
            row_id for row_id, count in success_counts.items() if count > 1
        )
        # Multiple successful retry attempts are retained for audit but canonicalization chooses
        # the latest. They are not a dataset duplicate, so report them without failing completion.
        report = ValidationReport(
            expected=len(self.expected),
            successful=len(successes),
            missing=missing,
            extra=extra,
            failed=failed,
            duplicate_successes=[],
        )
        write_json_atomic(
            self.report_path,
            {**report.as_dict(), "successful_retry_ids": duplicate_successes},
        )
        return report

    def finalize(self, allow_partial: bool = False) -> ValidationReport:
        report = self.validate()
        if not report.complete and not allow_partial:
            raise ArtifactError(
                "Prediction set is incomplete: "
                f"missing={len(report.missing)}, extra={len(report.extra)}, "
                f"failed={len(report.failed)}"
            )
        successes = self.successes()
        canonical = [
            successes[str(record["id"])]
            for record in self.expected
            if str(record["id"]) in successes
        ]
        write_jsonl_atomic(self.predictions_path, canonical)
        return report


def git_state(repo_root: str | Path) -> Dict[str, Any]:
    root = Path(repo_root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "status": status.splitlines(),
            "dirty_patch_sha256": sha256_bytes(diff),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": repr(exc)}


def environment_manifest(repo_root: str | Path) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git": git_state(repo_root),
        "retrieval": False,
    }
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
        )
        manifest["pip_freeze"] = freeze.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        manifest["pip_freeze_error"] = repr(exc)
    try:
        output = subprocess.check_output(["nvidia-smi", "-q"], text=True, stderr=subprocess.STDOUT)
        manifest["nvidia_smi_sha256"] = text_hash(output)
        manifest["nvidia_smi"] = output
    except (OSError, subprocess.CalledProcessError) as exc:
        manifest["nvidia_smi_error"] = repr(exc)
    return manifest


def checksum_tree(root: str | Path, exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
    base = Path(root)
    excluded = set(exclude)
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if relative in excluded:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def write_checksum_file(root: str | Path, name: str = "CHECKSUMS.sha256") -> Path:
    base = Path(root)
    target = base / name
    rows = checksum_tree(base, exclude={name})
    target.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows), encoding="ascii"
    )
    return target


def _normalise_field_name(key: Any) -> str:
    """Return a stable key form for case- and separator-insensitive checks."""

    return re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")


def _is_hash_field(key: str) -> bool:
    return any(marker in key for marker in _HASH_FIELD_MARKERS)


def _is_sensitive_public_field(key: Any) -> bool:
    """Identify benchmark evidence while retaining model outputs and hashes.

    The exact sets cover the canonical schemas.  Prefix/suffix checks make the
    redaction fail closed for common variants such as ``Reference-Label`` or
    ``human_annotation_text`` that a future adapter may introduce.
    """

    field = _normalise_field_name(key)
    if field in PUBLIC_MODEL_OUTPUT_FIELDS:
        return False
    if field == "record_fingerprint" or field.startswith(
        ("reference_", "gold_", "ground_truth_", "answer_")
    ):
        return True
    if field in {"image_id", "image_sha256", "normalized_reference_sha256"}:
        return True
    if _is_hash_field(field):
        return False
    if field in REFERENCE_FIELDS or field in RAW_DATA_FIELDS:
        return True
    if field.startswith(("reference_", "gold_", "ground_truth_", "answer_")):
        return True
    if field.endswith(("_reference", "_answer", "_annotation", "_annotations")):
        return True
    if field == "label" or field == "labels" or field.startswith("allowed_label"):
        return True
    if field.endswith("_label") and field not in PUBLIC_MODEL_OUTPUT_FIELDS:
        return True
    if "annotation" in field:
        return True
    if "reference" in field or "ground_truth" in field:
        return True
    if field.startswith(("raw_image", "raw_source", "gated_source")):
        return True
    if field.startswith("image_") and field not in {"image_id", "image_sha256"}:
        return True
    if field.endswith(("_path", "_dir", "_root")):
        return True
    if field in {"files", "file_inventory", "snapshot_inventory"}:
        return True
    return False


def redact_value(value: Any, *, gated: bool = False) -> Any:
    if isinstance(value, Mapping):
        dataset = str(value.get("dataset") or value.get("benchmark") or "").casefold()
        current_gated = gated or "astrovlbench" in dataset
        redacted: Dict[Any, Any] = {}
        for key, item in value.items():
            field = _normalise_field_name(key)
            if _is_sensitive_public_field(key):
                continue
            if current_gated and field in _GATED_SENSITIVE_FIELDS:
                continue
            if current_gated and field in {"id", "sample_id"}:
                continue
            if field == "error" and item:
                if isinstance(item, Mapping):
                    redacted["error_type"] = str(item.get("type") or "generation_error")
                else:
                    redacted["error_type"] = "generation_error"
                continue
            redacted[key] = redact_value(item, gated=current_gated)
        return redacted
    if isinstance(value, list):
        return [redact_value(item, gated=gated) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, gated=gated) for item in value)
    return value


def redact_json_file(source: Path, destination: Path) -> None:
    gated = "astrovlbench" in source.as_posix().casefold()
    try:
        if source.suffix.casefold() == ".jsonl":
            write_jsonl_atomic(
                destination,
                (redact_value(row, gated=gated) for row in read_jsonl(source)),
            )
        else:
            write_json_atomic(
                destination,
                redact_value(
                    json.loads(source.read_text(encoding="utf-8")), gated=gated
                ),
            )
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Cannot safely redact malformed JSON file {source}: {exc}") from exc


def redact_delimited_file(source: Path, destination: Path) -> None:
    """Copy a CSV/TSV while removing columns that contain source evidence."""

    delimiter = "\t" if source.suffix.casefold() == ".tsv" else ","
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ArtifactError(f"Cannot safely redact headerless delimited file: {source}")
        gated = "astrovlbench" in source.as_posix().casefold()
        fields = [
            field
            for field in reader.fieldnames
            if not _is_sensitive_public_field(field)
            and not (gated and _normalise_field_name(field) in _GATED_SENSITIVE_FIELDS)
            and not (gated and _normalise_field_name(field) in {"id", "sample_id"})
        ]
        rows = [{field: row.get(field, "") for field in fields} for row in reader]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _public_file_allowed(relative: Path) -> bool:
    parts = tuple(part.casefold() for part in relative.parts)
    if any(part in _PUBLIC_SOURCE_DIRECTORIES for part in parts[:-1]):
        return False
    suffix = relative.suffix.casefold()
    if suffix in _PUBLIC_FORBIDDEN_SUFFIXES:
        return False
    stem_tokens = set(_normalise_field_name(relative.stem).split("_"))
    if (
        suffix not in _PUBLIC_STRUCTURED_REDACTION_SUFFIXES
        and stem_tokens & _PUBLIC_SENSITIVE_FILENAME_TOKENS
    ):
        return False
    if suffix in _PUBLIC_RAW_IMAGE_SUFFIXES:
        # Generated paper PNGs are intentionally shareable.  Raw dataset PNGs
        # are not, even if an adapter placed one at the output root.
        return suffix == ".png" and bool(parts) and parts[0] == "reports"
    return True


def create_bundle(
    source_root: str | Path,
    destination: str | Path,
    *,
    public: bool,
) -> Path:
    """Create a gzipped tar bundle; public mode redacts source data and references."""

    source = Path(source_root)
    target = Path(destination)
    if not source.is_dir():
        raise ArtifactError(f"Bundle source directory does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target_resolved = target.resolve()
    with tempfile.TemporaryDirectory(prefix="paper-eval-bundle-") as tmp:
        stage = Path(tmp) / source.name
        stage.mkdir()
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source)
            if path.is_symlink():
                raise ArtifactError(f"Refusing to package symlinked file: {path}")
            if path.resolve() == target_resolved:
                continue
            if any(part.casefold() == "bundles" for part in relative.parts[:-1]):
                continue
            if public and not _public_file_allowed(relative):
                continue
            out = stage / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            if public and path.suffix.casefold() in {".json", ".jsonl"}:
                # Security boundary: malformed structured output must fail the
                # public package rather than being copied verbatim.
                redact_json_file(path, out)
                continue
            if public and path.suffix.casefold() in {".csv", ".tsv"}:
                redact_delimited_file(path, out)
                continue
            shutil.copy2(path, out)
        write_checksum_file(stage)
        handle, temporary_target = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(handle)
        try:
            with tarfile.open(temporary_target, "w:gz") as archive:
                archive.add(stage, arcname=stage.name, recursive=True)
            os.replace(temporary_target, target)
        finally:
            if os.path.exists(temporary_target):
                os.unlink(temporary_target)
    return target
