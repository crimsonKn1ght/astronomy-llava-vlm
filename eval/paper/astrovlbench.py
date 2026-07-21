"""Locked, image-only adapter for the gated AstroVLBench release.

The adapter intentionally has no import-time dependency on ``huggingface_hub``.
Downloading is an explicit operation which requires ``HF_TOKEN``; once a snapshot
has been locked, discovery and evaluation use only local files and the lock
manifest.  This prevents a mutable Hub branch or a partially changed local copy
from silently changing a paper result.

The public dataset card documents five task directories.  This module reads those
directories directly instead of using a hosted dataset builder, retains the
underlying source identifier for clustered statistics, and materializes *image
only* records.  In particular, Task 3 catalog photometry and redshift columns and
the Task 4/5 numerical CSV products are never included in model inputs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence


DATASET_NAME = "AstroVLBench"
DEFAULT_REPO_ID = "XiaomanZhang/AstroVLBench"
LOCK_SCHEMA_VERSION = 1
PROMPT_SOURCE_FILES = tuple(f"code/task{i}/llm.py" for i in range(1, 6))

# These numbers are audit references from the public dataset card, not slicing
# limits.  The locked files determine the actual denominator.
PUBLIC_REFERENCE_COUNTS: Mapping[str, int] = {
    "task1": 557,
    "task2.first": 606,
    "task2.nvss": 833,
    "task3": 168,
    "task4": 142,
    "task5.q1": 700,
    "task5.q2": 500,
    "task5.q3": 400,
}

PUBLIC_REFERENCE_CONFLICTS = (
    "task2.first: the dataset-card directory description reports 606 FIRST images, "
    "while the paper Methods describes 833 matched FIRST/NVSS sources; the locked "
    "snapshot, not either prose statement, determines the evaluation denominator",
)

DOCUMENTED_LAYOUT: Mapping[str, str] = {
    "task1": "data/Task1_QSOHost/image_labels.csv",
    "task2.first": "data/Task2_RadioMorph/MiraBest_F/metadata.jsonl",
    "task2.nvss": "data/Task2_RadioMorph/MiraBest_N/metadata.jsonl",
    "task3": "data/Task3_SED/nirsed_v2_catalog.csv",
    "task4": "data/Task4_LightCurve/manifest.csv",
    "task5": "data/Task5_SpecType/ASIB_v1_selection_with_snr.csv",
}

ALLOWED_LABELS: Mapping[str, tuple[str, ...]] = {
    "task1": ("AGN", "Galaxy"),
    "task2.first": ("FRI", "FRII"),
    "task2.nvss": ("FRI", "FRII"),
    "task3": ("Type-1 AGN", "Type-2 AGN", "Galaxy"),
    "task4": ("AGN", "SNIa", "TDE", "RRL", "Mira"),
    "task5.q1": ("Yes", "No"),
    "task5.q2": ("Yes", "No"),
    "task5.q3": ("Star-Forming", "Composite", "Seyfert", "LINER"),
}

_LABEL_ALIASES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "task1": {
        "AGN": ("agn", "active galactic nucleus", "qso", "quasar"),
        "Galaxy": ("galaxy", "inactive galaxy", "host galaxy"),
    },
    "task2": {
        "FRI": ("fri", "fr i", "fr 1", "fanaroff riley i", "fanaroff riley type i"),
        "FRII": ("frii", "fr ii", "fr 2", "fanaroff riley ii", "fanaroff riley type ii"),
    },
    "task3": {
        "Type-1 AGN": ("type 1 agn", "type i agn", "type1 agn", "type 1", "type i", "type1", "agn1"),
        "Type-2 AGN": ("type 2 agn", "type ii agn", "type2 agn", "type 2", "type ii", "type2", "agn2"),
        "Galaxy": ("galaxy", "inactive galaxy", "normal galaxy"),
    },
    "task4": {
        "AGN": ("agn", "active galactic nucleus"),
        "SNIa": ("snia", "sn ia", "type ia supernova", "type 1a supernova"),
        "TDE": ("tde", "tidal disruption event"),
        "RRL": ("rrl", "rr lyrae", "rr lyrae variable"),
        "Mira": ("mira", "mira variable"),
    },
    "task5.q1": {
        "Yes": ("yes", "true", "both present", "present"),
        "No": ("no", "false", "not both present", "absent"),
    },
    "task5.q2": {
        "Yes": ("yes", "true", "blagn", "broad line agn", "broad-line agn"),
        "No": (
            "no",
            "false",
            "not blagn",
            "not a blagn",
            "not broad line agn",
            "not a broad line agn",
            "narrow line agn",
            "narrow-line agn",
        ),
    },
    "task5.q3": {
        "Star-Forming": ("star forming", "star-forming", "star formation", "hii"),
        "Composite": ("composite",),
        "Seyfert": ("seyfert",),
        "LINER": ("liner", "low ionization nuclear emission line region"),
    },
}

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
_IMAGE_FIELDS = (
    "image_path",
    "image",
    "filename",
    "file_name",
    "image_name",
    "img_path",
    "img",
    "figure",
    "figure_path",
    "plot_path",
)
_SOURCE_ID_FIELDS = (
    "source_object_id",
    "object_id",
    "source_id",
    "target_id",
    "objid",
    "objectid",
    "name",
    "id",
)
_LABEL_FIELDS = (
    "label",
    "class",
    "target",
    "source_type",
    "object_type",
    "obj_type",
    "agn_type",
    "classification",
    "type",
)
_GROUP_FIELDS = ("group", "group_label", "subgroup", "class", "category", "label")


class AstroVLBenchError(RuntimeError):
    """Base exception for lock, schema, and materialization failures."""


class LockValidationError(AstroVLBenchError):
    """The local snapshot does not match its immutable manifest."""


class SchemaError(AstroVLBenchError):
    """A documented dataset file cannot be interpreted without guessing."""


class PromptExtractionError(AstroVLBenchError):
    """The official guided prompt could not be recovered from source."""


@dataclass(frozen=True)
class AstroVLBenchRecord:
    """One model input and its paper-evaluation identity."""

    sample_id: str
    source_object_id: str
    task: str
    subtask: str | None
    image_path: str
    image_relpath: str
    prompt: str
    reference_label: str
    allowed_labels: tuple[str, ...]
    metadata: Mapping[str, str]

    @property
    def task_key(self) -> str:
        return self.task if self.subtask is None else f"{self.task}.{self.subtask}"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_labels"] = list(self.allowed_labels)
        value["metadata"] = dict(self.metadata)
        value["task_key"] = self.task_key
        # Canonical aliases consumed by the shared prediction worker.  Keep the
        # explicit benchmark names above as well so classification analysis does
        # not have to reverse-map generic caption fields.
        value.update(
            {
                "id": self.sample_id,
                "dataset": "astrovlbench",
                "split": "test",
                "record_type": "classification",
                "reference": self.reference_label,
                "image": self.image_relpath,
                "image_id": self.image_relpath,
                "prompt_sha256": sha256_text(self.prompt),
                "reference_sha256": sha256_text(self.reference_label),
                "image_sha256": _cached_image_sha256(self.image_path),
            }
        )
        return value


@dataclass(frozen=True)
class ParseResult:
    raw_response: str
    label: str | None
    valid: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryResult:
    records: tuple[AstroVLBenchRecord, ...]
    expected_counts: Mapping[str, int]
    public_reference_counts: Mapping[str, int]
    discrepancies: tuple[str, ...]
    source_audit: Mapping[str, Any]

    def report_dict(self) -> dict[str, Any]:
        return {
            "expected_counts_from_locked_snapshot": dict(self.expected_counts),
            "public_reference_counts": dict(self.public_reference_counts),
            "discrepancies": list(self.discrepancies),
            "source_audit": dict(self.source_audit),
            "total_materialized_records": len(self.records),
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def _cached_image_sha256(path: str) -> str:
    return sha256_file(Path(path))


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _normal_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def _canonical_commit(value: str) -> str:
    commit = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise AstroVLBenchError(
            f"Resolved revision must be an immutable 40-character Git SHA, got {value!r}"
        )
    return commit


def _require_hf_token(token: str | None = None, env_name: str = "HF_TOKEN") -> str:
    resolved = token or os.environ.get(env_name)
    if not resolved or not resolved.strip():
        raise AstroVLBenchError(
            f"AstroVLBench is gated. Accept its access conditions and set {env_name} "
            "before locking the snapshot."
        )
    return resolved.strip()


def _iter_snapshot_files(snapshot_dir: Path, excluded: Iterable[Path] = ()) -> Iterator[Path]:
    root = snapshot_dir.resolve()
    excluded_resolved = {path.resolve() for path in excluded}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.resolve() in excluded_resolved:
            continue
        relative = path.relative_to(root)
        # Hub local-dir metadata is transport state, not dataset content.
        if any(part in {".git", ".cache", ".huggingface"} for part in relative.parts):
            continue
        yield path


def _literal(node: ast.AST, symbols: Mapping[str, Any]) -> Any:
    """Resolve static prompt literals without importing untrusted benchmark code."""

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in symbols:
        return symbols[node.id]
    if isinstance(node, ast.Dict):
        return {
            _literal(key, symbols): _literal(value, symbols)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal(item, symbols) for item in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left, symbols), _literal(node.right, symbols)
        return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                value = _literal(part.value, symbols)
                if not isinstance(value, (str, int, float)):
                    raise ValueError("non-static f-string value")
                parts.append(str(value))
            else:
                raise ValueError("unsupported f-string component")
        return "".join(parts)
    raise ValueError(f"non-static expression: {type(node).__name__}")


def _prompt_candidates_from_object(value: Any, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            candidates.extend(_prompt_candidates_from_object(item, path + (str(key),)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            candidates.extend(_prompt_candidates_from_object(item, path + (str(index),)))
    elif isinstance(value, str):
        normalized_path = tuple(_normal_text(part) for part in path)
        if any("guided" in part or "physical" in part for part in normalized_path):
            question = next((part for part in normalized_path if part in {"q1", "q2", "q3"}), "default")
            candidates.append((question, value.strip()))
    return candidates


def extract_guided_prompts_from_source(path: Path) -> Mapping[str, str]:
    """Statically extract official guided/physical prompt strings from ``llm.py``.

    The release scripts conventionally place prompts in literal dictionaries.  We
    intentionally do not execute those scripts because they may import clients or
    read credentials.  Ambiguous candidates are resolved deterministically by
    selecting the longest guided string; every selected value and its source file
    are captured in the immutable lock manifest.
    """

    source = path.read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise PromptExtractionError(f"Cannot parse official prompt source {path}: {exc}") from exc

    symbols: dict[str, Any] = {}
    pending: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_node = node.value
            if value_node is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    pending.append((target.id, value_node))

    for _ in range(len(pending) + 1):
        changed = False
        for name, value_node in pending:
            if name in symbols:
                continue
            try:
                symbols[name] = _literal(value_node, symbols)
                changed = True
            except (KeyError, TypeError, ValueError):
                pass
        if not changed:
            break

    candidates: list[tuple[str, str]] = []
    for name, value in symbols.items():
        candidates.extend(_prompt_candidates_from_object(value, (name,)))
        normalized_name = _normal_text(name)
        if isinstance(value, str) and "prompt" in normalized_name and any(
            token in normalized_name for token in ("guided", "physical")
        ):
            candidates.append(("default", value.strip()))

    # Support direct ``if prompt_type == 'guided': ...`` code.  Task 5
    # implementations sometimes put Q1/Q2/Q3 in outer branches, so retain that
    # context rather than collapsing three different prompts into one default.
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        try:
            test_text = ast.unparse(node.test).casefold()
        except Exception:
            test_text = ""
        subtree_text = " ".join(
            ast.unparse(child.test).casefold()
            for child in ast.walk(node)
            if isinstance(child, ast.If)
        )
        if "guided" not in subtree_text and "physical" not in subtree_text:
            continue
        question = next(
            (
                key
                for key in ("q1", "q2", "q3")
                if re.search(rf"[\"']{key}[\"']", test_text)
            ),
            "default",
        )
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            value_node: ast.AST | None = None
            if isinstance(child, ast.Return) and child.value is not None:
                value_node = child.value
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                names = [target.id.casefold() for target in targets if isinstance(target, ast.Name)]
                if any("prompt" in name or "message" in name for name in names):
                    value_node = child.value
            if value_node is not None:
                try:
                    value = _literal(value_node, symbols)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, str):
                    candidates.append((question, value.strip()))

    selected: dict[str, str] = {}
    for key in {candidate[0] for candidate in candidates}:
        options = sorted(
            {text for candidate_key, text in candidates if candidate_key == key and text},
            key=lambda text: (len(text), text),
        )
        if options:
            selected[key] = options[-1]
    if not selected:
        raise PromptExtractionError(
            f"No static guided/physical prompt found in {path}. The upstream prompt code "
            "changed; inspect and extend the static extractor before evaluation."
        )
    return selected


def extract_official_guided_prompts(snapshot_dir: Path) -> Mapping[str, Mapping[str, str]]:
    root = snapshot_dir.resolve()
    prompts: dict[str, Mapping[str, str]] = {}
    for index, relative in enumerate(PROMPT_SOURCE_FILES, 1):
        path = root / PurePosixPath(relative)
        if not path.is_file():
            raise PromptExtractionError(f"Missing official prompt source: {relative}")
        extracted = dict(extract_guided_prompts_from_source(path))
        if index == 5:
            missing = [question for question in ("q1", "q2", "q3") if question not in extracted]
            if missing and "default" not in extracted:
                raise PromptExtractionError(
                    f"{relative} has no prompt for {', '.join(missing)} and no default prompt"
                )
        elif "default" not in extracted:
            # Some sources key their only prompt by a task name; retaining exactly
            # one candidate is unambiguous.
            if len(extracted) == 1:
                extracted = {"default": next(iter(extracted.values()))}
            else:
                raise PromptExtractionError(f"{relative} has no unambiguous default guided prompt")
        prompts[f"task{index}"] = extracted
    return prompts


def create_lock_manifest(
    snapshot_dir: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    requested_revision: str = "main",
    commit_sha: str,
    exclude_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build a deterministic manifest for an already downloaded local snapshot."""

    root = snapshot_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AstroVLBench snapshot directory not found: {root}")
    commit = _canonical_commit(commit_sha)
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _iter_snapshot_files(root, exclude_paths)
    ]
    if not files:
        raise LockValidationError(f"Snapshot contains no dataset files: {root}")
    by_path = {entry["path"]: entry for entry in files}
    missing_prompt_sources = [path for path in PROMPT_SOURCE_FILES if path not in by_path]
    if missing_prompt_sources:
        raise LockValidationError(
            "Snapshot is missing official prompt source files: " + ", ".join(missing_prompt_sources)
        )
    prompts = extract_official_guided_prompts(root)
    prompt_sources = [dict(by_path[path]) for path in PROMPT_SOURCE_FILES]
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "commit_sha": commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_contract": {
            "prompt_type": "guided",
            "modality": "image",
            "few_shot": False,
            "task3_redshift_mode": "without",
        },
        "files": files,
        "official_prompt_sources": prompt_sources,
        "official_guided_prompts": prompts,
        "official_guided_prompt_sha256": {
            task: {key: sha256_text(value) for key, value in task_prompts.items()}
            for task, task_prompts in prompts.items()
        },
    }


def write_lock_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_lock_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockValidationError(f"Cannot read AstroVLBench lock manifest {path}: {exc}") from exc
    required = {
        "schema_version",
        "dataset",
        "repo_id",
        "commit_sha",
        "files",
        "official_prompt_sources",
        "official_guided_prompts",
    }
    missing = sorted(required - value.keys()) if isinstance(value, dict) else sorted(required)
    if not isinstance(value, dict) or missing:
        raise LockValidationError(f"Lock manifest is missing fields: {', '.join(missing)}")
    if value["schema_version"] != LOCK_SCHEMA_VERSION or value["dataset"] != DATASET_NAME:
        raise LockValidationError("Unsupported AstroVLBench lock schema or dataset name")
    _canonical_commit(str(value["commit_sha"]))
    if not isinstance(value["files"], list) or not value["files"]:
        raise LockValidationError("Lock manifest has no file inventory")
    return value


def validate_lock_manifest(
    snapshot_dir: Path,
    manifest_or_path: Mapping[str, Any] | Path,
    *,
    allow_untracked: Iterable[Path] = (),
) -> dict[str, Any]:
    """Verify file set, sizes, hashes, prompt sources, and prompt text hashes."""

    root = snapshot_dir.resolve()
    manifest_path = manifest_or_path.resolve() if isinstance(manifest_or_path, Path) else None
    manifest = read_lock_manifest(manifest_or_path) if manifest_path is not None else dict(manifest_or_path)
    _canonical_commit(str(manifest.get("commit_sha", "")))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise LockValidationError("Lock file inventory must be a list")
    locked: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not {"path", "size", "sha256"} <= entry.keys():
            raise LockValidationError("Malformed file entry in lock manifest")
        relative = PurePosixPath(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise LockValidationError(f"Unsafe file path in lock manifest: {relative}")
        if relative.as_posix() in locked:
            raise LockValidationError(f"Duplicate file in lock manifest: {relative}")
        locked[relative.as_posix()] = entry

    ignored = list(allow_untracked)
    if manifest_path is not None and manifest_path.is_relative_to(root):
        ignored.append(manifest_path)
    actual = {
        path.relative_to(root).as_posix(): path
        for path in _iter_snapshot_files(root, ignored)
    }
    missing = sorted(set(locked) - set(actual))
    extra = sorted(set(actual) - set(locked))
    if missing or extra:
        raise LockValidationError(
            f"Snapshot file set differs from lock (missing={missing}, unexpected={extra})"
        )
    for relative, entry in locked.items():
        path = actual[relative]
        if path.stat().st_size != int(entry["size"]):
            raise LockValidationError(f"Size mismatch for locked file {relative}")
        digest = sha256_file(path)
        if digest != str(entry["sha256"]).lower():
            raise LockValidationError(f"SHA-256 mismatch for locked file {relative}")

    prompt_sources = manifest.get("official_prompt_sources", [])
    prompt_paths = {str(entry.get("path")) for entry in prompt_sources if isinstance(entry, Mapping)}
    if prompt_paths != set(PROMPT_SOURCE_FILES):
        raise LockValidationError("Lock does not identify exactly the five official prompt source files")
    extracted = extract_official_guided_prompts(root)
    if extracted != manifest.get("official_guided_prompts"):
        raise LockValidationError("Official guided prompt extraction differs from the lock manifest")
    hashes = {
        task: {key: sha256_text(value) for key, value in values.items()}
        for task, values in extracted.items()
    }
    if "official_guided_prompt_sha256" in manifest and hashes != manifest[
        "official_guided_prompt_sha256"
    ]:
        raise LockValidationError("Official guided prompt text hash mismatch")
    return manifest


def resolve_and_lock_snapshot(
    destination: Path,
    lock_path: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
    token: str | None = None,
    token_env: str = "HF_TOKEN",
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve, download, hash, and lock the gated dataset at an immutable SHA."""

    auth_token = _require_hf_token(token, token_env)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise AstroVLBenchError(
            "Locking AstroVLBench requires huggingface_hub. Install it in the setup environment."
        ) from exc

    info = HfApi().dataset_info(repo_id=repo_id, revision=revision, token=auth_token)
    commit = _canonical_commit(str(info.sha))
    destination.mkdir(parents=True, exist_ok=True)
    resolved_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=commit,
            token=auth_token,
            local_dir=str(destination),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    ).resolve()
    excluded = [lock_path] if lock_path.resolve().is_relative_to(resolved_dir) else []
    manifest = create_lock_manifest(
        resolved_dir,
        repo_id=repo_id,
        requested_revision=revision,
        commit_sha=commit,
        exclude_paths=excluded,
    )
    write_lock_manifest(manifest, lock_path)
    validate_lock_manifest(resolved_dir, manifest, allow_untracked=excluded)
    return manifest


def lock_huggingface_snapshot(
    repo_id: str,
    output_dir: Path,
    token: str | None = None,
    cache_dir: Path | None = None,
    revision: str = "main",
) -> Path:
    """Orchestrator-facing one-call snapshot lock.

    ``output_dir`` is a self-contained bundle with a ``snapshot`` directory and
    ``astrovlbench.lock.json``.  Returning the lock path lets later CPU-only jobs
    discover and validate the snapshot without carrying additional state.
    """

    bundle = Path(output_dir).resolve()
    snapshot_dir = bundle / "snapshot"
    lock_path = bundle / "astrovlbench.lock.json"
    manifest = resolve_and_lock_snapshot(
        snapshot_dir,
        lock_path,
        repo_id=repo_id,
        revision=revision,
        token=token,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
    )
    manifest["snapshot_relpath"] = "snapshot"
    write_lock_manifest(manifest, lock_path)
    validate_lock_manifest(snapshot_dir, lock_path)
    return lock_path


def discover_documented_layout(snapshot_dir: Path) -> Mapping[str, Path]:
    root = snapshot_dir.resolve()
    resolved = {key: root / PurePosixPath(relative) for key, relative in DOCUMENTED_LAYOUT.items()}
    missing = [f"{key}: {path.relative_to(root).as_posix()}" for key, path in resolved.items() if not path.is_file()]
    if missing:
        raise SchemaError("AstroVLBench snapshot is missing documented task files: " + "; ".join(missing))
    return resolved


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise SchemaError(f"CSV has no header: {path}")
        rows = [{str(key): ("" if value is None else str(value).strip()) for key, value in row.items()} for row in reader]
    if not rows:
        raise SchemaError(f"CSV has no data rows: {path}")
    return rows


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SchemaError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise SchemaError(f"JSONL has no data rows: {path}")
    return rows


def _row_value(row: Mapping[str, Any], candidates: Sequence[str], *, required: bool = False) -> str | None:
    by_normal = {_normal_key(key): value for key, value in row.items()}
    for candidate in candidates:
        value = by_normal.get(_normal_key(candidate))
        if value is not None and str(value).strip():
            return str(value).strip()
    if required:
        raise SchemaError(
            f"None of the required fields {list(candidates)} occur in row with fields {list(row)}"
        )
    return None


class _ImageIndex:
    def __init__(self, task_dir: Path, snapshot_root: Path):
        self.task_dir = task_dir.resolve()
        self.snapshot_root = snapshot_root.resolve()
        self.by_name: dict[str, list[Path]] = {}
        self.by_stem: dict[str, list[Path]] = {}
        for path in self.task_dir.rglob("*"):
            if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES:
                self.by_name.setdefault(path.name.casefold(), []).append(path.resolve())
                self.by_stem.setdefault(path.stem.casefold(), []).append(path.resolve())
        if not self.by_name:
            raise SchemaError(f"No supported images found below {self.task_dir}")

    def resolve(self, value: str, preferred_parents: Sequence[str] = ()) -> Path:
        cleaned = value.strip().replace("\\", "/")
        candidate_path = PurePosixPath(cleaned)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise SchemaError(f"Unsafe image reference {value!r} in {self.task_dir}")
        direct_candidates = [self.task_dir / candidate_path, self.snapshot_root / candidate_path]
        for candidate in direct_candidates:
            if candidate.is_file() and candidate.suffix.casefold() in _IMAGE_SUFFIXES:
                resolved = candidate.resolve()
                if not resolved.is_relative_to(self.snapshot_root):
                    raise SchemaError(f"Image reference escapes snapshot: {value!r}")
                return resolved
        name_matches = self.by_name.get(candidate_path.name.casefold(), [])
        if not name_matches and not candidate_path.suffix:
            name_matches = self.by_stem.get(candidate_path.name.casefold(), [])
        if len(name_matches) > 1 and preferred_parents:
            normalized_parents = {_normal_key(value) for value in preferred_parents}
            preferred = [
                path
                for path in name_matches
                if any(_normal_key(part) in normalized_parents for part in path.parent.parts)
            ]
            if len(preferred) == 1:
                return preferred[0]
        if len(name_matches) == 1:
            return name_matches[0]
        if not name_matches:
            raise SchemaError(f"No image matches {value!r} below {self.task_dir}")
        raise SchemaError(f"Ambiguous image reference {value!r}: {[str(path) for path in name_matches]}")


def _canonical_label(task_key: str, value: str) -> str:
    alias_key = "task2" if task_key.startswith("task2.") else task_key
    aliases = _LABEL_ALIASES[alias_key]
    normalized = _normal_text(value)
    matches = [canonical for canonical, values in aliases.items() if normalized in {_normal_text(v) for v in values + (canonical,)}]
    if len(matches) != 1:
        raise SchemaError(f"Unknown or ambiguous {task_key} ground-truth label {value!r}")
    return matches[0]


def _source_id(row: Mapping[str, Any], image: Path, survey: str | None = None) -> str:
    value = _row_value(row, _SOURCE_ID_FIELDS) or image.stem
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if survey:
        normalized = re.sub(r"^(?:mirabest_?[fn]|first|nvss)_+", "", normalized)
        normalized = re.sub(r"_+(?:first|nvss)$", "", normalized)
    if not normalized:
        raise SchemaError(f"Cannot derive a source object ID from {value!r}")
    return normalized


def _sample_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not token:
        raise SchemaError(f"Cannot create a sample ID from {value!r}")
    return token


def _prompt(prompts: Mapping[str, Mapping[str, str]], task: str, subtask: str | None = None) -> str:
    values = prompts.get(task)
    if not isinstance(values, Mapping):
        raise LockValidationError(f"Lock manifest has no official prompt mapping for {task}")
    key = subtask or "default"
    result = values.get(key, values.get("default"))
    if not isinstance(result, str) or not result.strip():
        raise LockValidationError(f"Lock manifest has no official guided prompt for {task}.{key}")
    return result


def _record(
    root: Path,
    *,
    sample_id: str,
    source_id: str,
    task: str,
    subtask: str | None,
    image: Path,
    prompt: str,
    label: str,
    metadata: Mapping[str, str],
) -> AstroVLBenchRecord:
    task_key = task if subtask is None else f"{task}.{subtask}"
    return AstroVLBenchRecord(
        sample_id=sample_id,
        source_object_id=source_id,
        task=task,
        subtask=subtask,
        image_path=str(image.resolve()),
        image_relpath=image.resolve().relative_to(root.resolve()).as_posix(),
        prompt=prompt,
        reference_label=label,
        allowed_labels=ALLOWED_LABELS[task_key],
        metadata=dict(metadata),
    )


def _materialize_simple_csv(
    root: Path,
    source_file: Path,
    task: str,
    prompts: Mapping[str, Mapping[str, str]],
) -> list[AstroVLBenchRecord]:
    task_dir = source_file.parent
    index = _ImageIndex(task_dir, root)
    records: list[AstroVLBenchRecord] = []
    for row_number, row in enumerate(_read_csv_rows(source_file), 2):
        image_ref = _row_value(row, _IMAGE_FIELDS) or _row_value(row, _SOURCE_ID_FIELDS, required=True)
        assert image_ref is not None
        label_value = _row_value(row, _LABEL_FIELDS)
        image = index.resolve(image_ref, (label_value,)) if label_value else index.resolve(image_ref)
        if label_value is None:
            label_value = image.parent.name
        label = _canonical_label(task, label_value)
        source_id = _source_id(row, image)
        records.append(
            _record(
                root,
                sample_id=f"astrovlbench_{task}_{_sample_token(source_id)}",
                source_id=source_id,
                task=task,
                subtask=None,
                image=image,
                prompt=_prompt(prompts, task),
                label=label,
                metadata={
                    "modality": "image",
                    "prompt_type": "guided",
                    "source_row": str(row_number),
                },
            )
        )
    return records


def _materialize_task2(
    root: Path,
    source_file: Path,
    subtask: str,
    prompts: Mapping[str, Mapping[str, str]],
) -> list[AstroVLBenchRecord]:
    task_dir = source_file.parent
    index = _ImageIndex(task_dir, root)
    records: list[AstroVLBenchRecord] = []
    for row_number, row in enumerate(_read_jsonl_rows(source_file), 1):
        image_ref = _row_value(row, _IMAGE_FIELDS, required=True)
        assert image_ref is not None
        label_value = _row_value(row, _LABEL_FIELDS)
        image = index.resolve(image_ref, (label_value,)) if label_value else index.resolve(image_ref)
        label_value = label_value or image.parent.name
        label = _canonical_label(f"task2.{subtask}", label_value)
        source_id = _source_id(row, image, survey=subtask)
        records.append(
            _record(
                root,
                sample_id=f"astrovlbench_task2_{subtask}_{_sample_token(source_id)}",
                source_id=f"radio_{source_id}",
                task="task2",
                subtask=subtask,
                image=image,
                prompt=_prompt(prompts, "task2"),
                label=label,
                metadata={
                    "modality": "image",
                    "prompt_type": "guided",
                    "survey": subtask.upper(),
                    "source_row": str(row_number),
                },
            )
        )
    return records


def _normalize_group(value: str) -> str:
    normalized = _normal_key(value)
    if normalized.startswith("group"):
        normalized = normalized[5:]
    normalized = normalized.upper()
    if normalized not in {"A", "B", "C1", "C2", "C3", "C4", "D"}:
        raise SchemaError(f"Unknown Task 5 ASIB group {value!r}")
    return normalized


_TASK5_LABELS: Mapping[str, Mapping[str, str]] = {
    "q1": {"A": "No", "B": "No", "C1": "Yes", "C2": "Yes", "C3": "Yes", "C4": "Yes", "D": "Yes"},
    "q2": {"C1": "No", "C2": "No", "C3": "No", "C4": "No", "D": "Yes"},
    "q3": {"C1": "Star-Forming", "C2": "Composite", "C3": "Seyfert", "C4": "LINER"},
}


def _materialize_task5(
    root: Path,
    source_file: Path,
    prompts: Mapping[str, Mapping[str, str]],
) -> tuple[list[AstroVLBenchRecord], Mapping[str, int]]:
    task_dir = source_file.parent
    index = _ImageIndex(task_dir, root)
    records: list[AstroVLBenchRecord] = []
    group_counts: Counter[str] = Counter()
    for row_number, row in enumerate(_read_csv_rows(source_file), 2):
        group_value = _row_value(row, _GROUP_FIELDS, required=True)
        assert group_value is not None
        group = _normalize_group(group_value)
        group_counts[group] += 1
        image_ref = _row_value(row, _IMAGE_FIELDS) or _row_value(row, _SOURCE_ID_FIELDS, required=True)
        assert image_ref is not None
        image = index.resolve(image_ref, (group, f"Group_{group}"))
        source_id = _source_id(row, image)
        cluster_id = f"spectrum_{source_id}"
        for question, label_by_group in _TASK5_LABELS.items():
            if group not in label_by_group:
                continue
            records.append(
                _record(
                    root,
                    sample_id=f"astrovlbench_task5_{question}_{_sample_token(source_id)}",
                    source_id=cluster_id,
                    task="task5",
                    subtask=question,
                    image=image,
                    prompt=_prompt(prompts, "task5", question),
                    label=label_by_group[group],
                    metadata={
                        "modality": "image",
                        "prompt_type": "guided",
                        "asib_group": group,
                        "source_row": str(row_number),
                    },
                )
            )
    return records, dict(sorted(group_counts.items()))


def _assert_unique_complete(records: Sequence[AstroVLBenchRecord]) -> None:
    counts = Counter(record.sample_id for record in records)
    duplicate_ids = sorted(sample_id for sample_id, count in counts.items() if count != 1)
    if duplicate_ids:
        raise SchemaError(f"Duplicate materialized sample IDs: {duplicate_ids[:10]}")
    for record in records:
        if not Path(record.image_path).is_file():
            raise SchemaError(f"Materialized image is missing: {record.image_path}")
        if record.reference_label not in record.allowed_labels:
            raise SchemaError(f"Label {record.reference_label!r} is not allowed for {record.task_key}")


def discover_records(
    snapshot_dir: Path,
    lock_manifest: Mapping[str, Any] | Path,
) -> DiscoveryResult:
    """Validate a locked snapshot and materialize every documented image task."""

    root = snapshot_dir.resolve()
    manifest = validate_lock_manifest(root, lock_manifest)
    layout = discover_documented_layout(root)
    prompts = manifest["official_guided_prompts"]

    records: list[AstroVLBenchRecord] = []
    records.extend(_materialize_simple_csv(root, layout["task1"], "task1", prompts))
    records.extend(_materialize_task2(root, layout["task2.first"], "first", prompts))
    records.extend(_materialize_task2(root, layout["task2.nvss"], "nvss", prompts))
    records.extend(_materialize_simple_csv(root, layout["task3"], "task3", prompts))
    records.extend(_materialize_simple_csv(root, layout["task4"], "task4", prompts))
    task5_records, task5_groups = _materialize_task5(root, layout["task5"], prompts)
    records.extend(task5_records)
    records.sort(key=lambda record: record.sample_id)
    _assert_unique_complete(records)

    counts = Counter(record.task_key for record in records)
    expected_counts = {key: counts.get(key, 0) for key in PUBLIC_REFERENCE_COUNTS}
    count_discrepancies = tuple(
        f"{key}: locked snapshot has {expected_counts[key]} records; public dataset card reports {reference}"
        for key, reference in PUBLIC_REFERENCE_COUNTS.items()
        if expected_counts[key] != reference
    )
    discrepancies = tuple(PUBLIC_REFERENCE_CONFLICTS) + count_discrepancies
    first_sources = {record.source_object_id for record in records if record.task_key == "task2.first"}
    nvss_sources = {record.source_object_id for record in records if record.task_key == "task2.nvss"}
    source_audit = {
        "task2_first_source_objects": len(first_sources),
        "task2_nvss_source_objects": len(nvss_sources),
        "task2_matched_source_objects": len(first_sources & nvss_sources),
        "task2_first_without_nvss": len(first_sources - nvss_sources),
        "task2_nvss_without_first": len(nvss_sources - first_sources),
        "task5_group_counts": task5_groups,
        "task5_unique_source_objects": len(
            {record.source_object_id for record in records if record.task == "task5"}
        ),
    }
    return DiscoveryResult(
        records=tuple(records),
        expected_counts=expected_counts,
        public_reference_counts=dict(PUBLIC_REFERENCE_COUNTS),
        discrepancies=discrepancies,
        source_audit=source_audit,
    )


def materialize_locked_records(lock_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a bundle lock and return JSON-ready records plus audit report."""

    resolved_lock = Path(lock_path).resolve()
    manifest = read_lock_manifest(resolved_lock)
    relative = manifest.get("snapshot_relpath", "snapshot")
    relative_path = PurePosixPath(str(relative))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise LockValidationError(f"Unsafe snapshot_relpath in lock: {relative!r}")
    snapshot_dir = resolved_lock.parent / relative_path
    if not snapshot_dir.is_dir():
        # Local/manual locks are often stored immediately beside snapshot files.
        # Use that layout only when the documented data directory is present.
        adjacent = resolved_lock.parent
        if (adjacent / "data").is_dir() and (adjacent / "code").is_dir():
            snapshot_dir = adjacent
        else:
            raise LockValidationError(
                f"Locked AstroVLBench snapshot not found at {snapshot_dir}; "
                "keep the lock beside its snapshot bundle"
            )
    result = discover_records(snapshot_dir, resolved_lock)
    records = []
    for index, record in enumerate(result.records, 1):
        value = record.to_dict()
        value["record_index"] = index
        records.append(value)
    return records, result.report_dict()


def _alias_patterns(task_key: str) -> Mapping[str, tuple[str, ...]]:
    alias_key = "task2" if task_key.startswith("task2.") else task_key
    if alias_key not in _LABEL_ALIASES:
        raise ValueError(f"Unsupported AstroVLBench task key: {task_key}")
    return _LABEL_ALIASES[alias_key]


def parse_label_response(response: str, task_key: str) -> ParseResult:
    """Strictly parse a label while retaining invalid/ambiguous raw responses.

    Exact answers and a unique alias anywhere in a response are accepted.  If a
    rationale mentions two possible classes, the result is deliberately invalid
    instead of choosing whichever class happened to occur first.
    """

    raw = "" if response is None else str(response)
    if not raw.strip():
        return ParseResult(raw, None, False, "empty_response")

    # The official prompts request {"answer": ..., "reason": ...}.  Parse the
    # answer field alone when present so a perfectly clear answer is not made
    # ambiguous merely because its explanation discusses another class.
    candidate_text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate_text = fenced.group(1)
    used_json = False
    if candidate_text.startswith("{"):
        try:
            payload = json.loads(candidate_text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping) and "answer" in payload:
            candidate_text = str(payload["answer"])
            used_json = True

    normalized = _normal_text(candidate_text)
    if not normalized:
        return ParseResult(raw, None, False, "empty_answer_field" if used_json else "empty_response")
    # Common non-JSON wrappers do not change the semantic answer.
    normalized = re.sub(r"^(?:final\s+answer|answer|classification|class|label)\s+", "", normalized)

    aliases_by_label = _alias_patterns(task_key)
    exact = {
        canonical
        for canonical, aliases in aliases_by_label.items()
        if normalized in {_normal_text(alias) for alias in aliases + (canonical,)}
    }
    if len(exact) == 1:
        reason = "json_answer" if used_json else "exact_allowed_label"
        return ParseResult(raw, next(iter(exact)), True, reason)
    if len(exact) > 1:
        return ParseResult(raw, None, False, "multiple_allowed_labels")

    occurrences: list[tuple[int, int, str]] = []
    for canonical, aliases in aliases_by_label.items():
        for alias in aliases + (canonical,):
            token = _normal_text(alias)
            for match in re.finditer(
                rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized
            ):
                occurrences.append((match.start(), match.end(), canonical))
    # Suppress a shorter alias when it is wholly contained in a longer, more
    # specific alias (e.g. BLAGN inside "not a BLAGN").
    retained = [
        occurrence
        for occurrence in occurrences
        if not any(
            other[0] <= occurrence[0]
            and other[1] >= occurrence[1]
            and (other[1] - other[0]) > (occurrence[1] - occurrence[0])
            for other in occurrences
        )
    ]
    matches = {canonical for _, _, canonical in retained}
    if len(matches) == 1:
        reason = "json_answer" if used_json else "unique_allowed_label"
        return ParseResult(raw, next(iter(matches)), True, reason)
    if not matches:
        return ParseResult(raw, None, False, "no_allowed_label")
    return ParseResult(raw, None, False, "multiple_allowed_labels")


def parse_label(
    record: AstroVLBenchRecord | Mapping[str, Any], response: str
) -> ParseResult:
    """Parse one response using the record's declared task/subtask contract."""

    if isinstance(record, AstroVLBenchRecord):
        task_key = record.task_key
    else:
        explicit = record.get("task_key")
        if explicit:
            task_key = str(explicit)
        else:
            task = str(record.get("task", ""))
            subtask = record.get("subtask")
            if not task:
                raise ValueError("AstroVLBench record has no task/task_key")
            task_key = task if not subtask else f"{task}.{subtask}"
    return parse_label_response(response, task_key)


def hierarchical_project_aggregate(scores: Mapping[str, float]) -> dict[str, Any]:
    """Apply the predeclared survey/question hierarchy to one score metric."""

    required = set(PUBLIC_REFERENCE_COUNTS)
    missing = sorted(required - scores.keys())
    if missing:
        raise ValueError(f"Missing AstroVLBench component scores: {missing}")
    clean: dict[str, float] = {}
    for key in required:
        value = float(scores[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"AstroVLBench score {key} must be finite and within [0, 1]")
        clean[key] = value
    task_scores = {
        "task1": clean["task1"],
        "task2": (clean["task2.first"] + clean["task2.nvss"]) / 2.0,
        "task3": clean["task3"],
        "task4": clean["task4"],
        "task5": (clean["task5.q1"] + clean["task5.q2"] + clean["task5.q3"]) / 3.0,
    }
    return {
        "component_scores": {key: clean[key] for key in sorted(clean)},
        "top_level_task_scores": task_scores,
        "project_macro_average": sum(task_scores.values()) / len(task_scores),
        "aggregation": "equal surveys within Task 2; equal questions within Task 5; equal five top-level tasks",
    }


aggregate_hierarchical_scores = hierarchical_project_aggregate


def _write_discovery(result: DiscoveryResult, records_path: Path, report_path: Path) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in result.records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.report_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock = subparsers.add_parser("lock", help="Download and immutably lock the gated snapshot.")
    lock.add_argument("--snapshot-dir", type=Path, required=True)
    lock.add_argument("--lock-file", type=Path, required=True)
    lock.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    lock.add_argument("--revision", default="main")
    lock.add_argument("--token-env", default="HF_TOKEN")

    validate = subparsers.add_parser("validate", help="Validate a local snapshot against its lock.")
    validate.add_argument("--snapshot-dir", type=Path, required=True)
    validate.add_argument("--lock-file", type=Path, required=True)

    discover = subparsers.add_parser("discover", help="Materialize locked image-only records.")
    discover.add_argument("--snapshot-dir", type=Path, required=True)
    discover.add_argument("--lock-file", type=Path, required=True)
    discover.add_argument("--records", type=Path, required=True)
    discover.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "lock":
        manifest = resolve_and_lock_snapshot(
            args.snapshot_dir,
            args.lock_file,
            repo_id=args.repo_id,
            revision=args.revision,
            token_env=args.token_env,
        )
        print(f"Locked {manifest['repo_id']} at {manifest['commit_sha']} -> {args.lock_file}")
        return 0
    if args.command == "validate":
        manifest = validate_lock_manifest(args.snapshot_dir, args.lock_file)
        print(f"Validated {len(manifest['files'])} files at {manifest['commit_sha']}")
        return 0
    if args.command == "discover":
        result = discover_records(args.snapshot_dir, args.lock_file)
        _write_discovery(result, args.records, args.report)
        print(f"Materialized {len(result.records)} records; discrepancies={len(result.discrepancies)}")
        return 0
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AstroVLBenchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
