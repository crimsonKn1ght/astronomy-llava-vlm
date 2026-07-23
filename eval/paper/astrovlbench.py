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
PINNED_RELEASE_REVISION = "d1708958d4d1dda45c078eb2f4d6db3e6fa96286"
LOCK_SCHEMA_VERSION = 2
PROMPT_SOURCE_FILES = tuple(f"code/task{i}/llm.py" for i in range(1, 6))
PROMPT_COMPOSITION_ID = "official-system-plus-user-v1"
PROMPT_SEPARATOR = "\n\n--- Official user instruction ---\n"

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

PINNED_RELEASE_COUNTS: Mapping[str, int] = {
    "task1": 557,
    "task2.first": 605,
    "task2.nvss": 833,
    "task3": 168,
    "task4": 142,
    "task5.q1": 700,
    "task5.q2": 500,
    "task5.q3": 400,
}
PINNED_RELEASE_TOTAL = 3905
PINNED_FIRST_RAW_ROWS = 833
PINNED_FIRST_EXCLUSIONS: Mapping[str, int] = {
    "missing_in_pinned_snapshot": 227,
    "upstream_zero_byte_image": 1,
}
PINNED_FIRST_VALID_LABELS: Mapping[str, int] = {"FRI": 397, "FRII": 208}
PINNED_FIRST_EXCLUDED_LABELS: Mapping[str, int] = {"FRII": 228}

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
    "fig_path",
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


def _parse_static_source(path: Path) -> tuple[ast.Module, dict[str, Any]]:
    """Parse prompt code and resolve only top-level literal assignments."""

    source = path.read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise PromptExtractionError(f"Cannot parse official prompt source {path}: {exc}") from exc
    symbols: dict[str, Any] = {}
    pending: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if node.value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                pending.append((target.id, node.value))
    for _ in range(len(pending) + 1):
        changed = False
        for name, value_node in pending:
            if name in symbols:
                continue
            try:
                symbols[name] = _literal(value_node, symbols)
                changed = True
            except (KeyError, TypeError, ValueError):
                continue
        if not changed:
            break
    return tree, symbols


def _required_string(symbols: Mapping[str, Any], name: str, path: Path) -> str:
    value = symbols.get(name)
    if not isinstance(value, str) or not value.strip():
        raise PromptExtractionError(f"{path} has no static non-empty {name}")
    return value.strip()


def _function(tree: ast.Module, name: str, path: Path) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise PromptExtractionError(f"{path} must define exactly one {name} function")
    return matches[0]


def _literal_return(
    tree: ast.Module,
    symbols: Mapping[str, Any],
    function_name: str,
    bindings: Mapping[str, Any],
    path: Path,
) -> str:
    function = _function(tree, function_name, path)
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or returns[0].value is None:
        raise PromptExtractionError(f"{path}:{function_name} must have one static return")
    try:
        value = _literal(returns[0].value, {**symbols, **bindings})
    except (KeyError, TypeError, ValueError) as exc:
        raise PromptExtractionError(
            f"Cannot statically render {path}:{function_name}: {exc}"
        ) from exc
    if not isinstance(value, str) or not value.strip():
        raise PromptExtractionError(f"{path}:{function_name} returned no prompt text")
    return value.strip()


def _task3_without_redshift_prompt(
    tree: ast.Module, symbols: Mapping[str, Any], path: Path
) -> str:
    function = _function(tree, "build_image_prompt", path)
    assignments: dict[str, list[str]] = {"redshift_block": [], "redshift_instruction": []}
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        for name in names:
            if name not in assignments:
                continue
            try:
                value = _literal(node.value, symbols)
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(value, str):
                assignments[name].append(value)
    redshift_blocks = [
        value for value in assignments["redshift_block"] if "not provided" in value.casefold()
    ]
    redshift_instructions = [
        value for value in assignments["redshift_instruction"] if "do not assume" in value.casefold()
    ]
    if len(set(redshift_blocks)) != 1 or len(set(redshift_instructions)) != 1:
        raise PromptExtractionError(
            f"{path}:build_image_prompt has no unambiguous without-redshift branch"
        )
    template = _required_string(symbols, "SYSTEM_PROMPT_IMAGE", path)
    try:
        rendered = template.format(
            REDSHIFT_BLOCK=redshift_blocks[0],
            REDSHIFT_INSTRUCTION=redshift_instructions[0],
        )
    except (KeyError, ValueError) as exc:
        raise PromptExtractionError(f"Cannot render Task 3 image prompt: {exc}") from exc
    return rendered.strip()


def _compose_prompt(system_prompt: str, user_text: str) -> str:
    system = system_prompt.strip()
    user = user_text.strip()
    if not system or not user:
        raise PromptExtractionError("Official system and user prompt components must be non-empty")
    return system + PROMPT_SEPARATOR + user


def extract_official_prompt_components(
    snapshot_dir: Path,
) -> Mapping[str, Mapping[str, Mapping[str, str]]]:
    """Extract the pinned release prompts without importing or executing its code."""

    root = snapshot_dir.resolve()
    parsed: dict[int, tuple[Path, ast.Module, dict[str, Any]]] = {}
    for index, relative in enumerate(PROMPT_SOURCE_FILES, 1):
        path = root / PurePosixPath(relative)
        if not path.is_file():
            raise PromptExtractionError(f"Missing official prompt source: {relative}")
        tree, symbols = _parse_static_source(path)
        parsed[index] = (path, tree, symbols)

    path1, _, symbols1 = parsed[1]
    path2, tree2, symbols2 = parsed[2]
    path3, tree3, symbols3 = parsed[3]
    path4, _, symbols4 = parsed[4]
    path5, _, symbols5 = parsed[5]
    raw: dict[str, dict[str, tuple[str, str]]] = {
        "task1": {
            "default": (
                _required_string(symbols1, "SYSTEM_PROMPT_GUIDED", path1),
                _required_string(symbols1, "USER_TEXT", path1),
            )
        },
        "task2": {
            "first": (
                _literal_return(
                    tree2, symbols2, "build_prompt_guided", {"survey": "FIRST"}, path2
                ),
                _required_string(symbols2, "USER_TEXT", path2),
            ),
            "nvss": (
                _literal_return(
                    tree2, symbols2, "build_prompt_guided", {"survey": "NVSS"}, path2
                ),
                _required_string(symbols2, "USER_TEXT", path2),
            ),
        },
        "task3": {
            "default": (
                _task3_without_redshift_prompt(tree3, symbols3, path3),
                _required_string(symbols3, "USER_TEXT", path3),
            )
        },
        "task4": {
            "default": (
                _required_string(symbols4, "SYSTEM_PROMPT_IMAGE", path4),
                _required_string(symbols4, "USER_TEXT_IMAGE", path4),
            )
        },
        "task5": {
            question: (
                _required_string(symbols5, f"SYSTEM_PROMPT_{question.upper()}", path5),
                _required_string(symbols5, f"USER_TEXT_{question.upper()}", path5),
            )
            for question in ("q1", "q2", "q3")
        },
    }
    return {
        task: {
            key: {
                "system": system,
                "user": user,
                "composed": _compose_prompt(system, user),
            }
            for key, (system, user) in values.items()
        }
        for task, values in raw.items()
    }


def extract_official_guided_prompts(snapshot_dir: Path) -> Mapping[str, Mapping[str, str]]:
    components = extract_official_prompt_components(snapshot_dir)
    return {
        task: {key: values["composed"] for key, values in task_values.items()}
        for task, task_values in components.items()
    }


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
    components = extract_official_prompt_components(root)
    prompts = {
        task: {key: value["composed"] for key, value in values.items()}
        for task, values in components.items()
    }
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
            "prompt_composition_id": PROMPT_COMPOSITION_ID,
        },
        "snapshot_inventory_sha256": sha256_text(
            json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
        "files": files,
        "official_prompt_sources": prompt_sources,
        "prompt_composition": {
            "id": PROMPT_COMPOSITION_ID,
            "separator": PROMPT_SEPARATOR,
        },
        "official_prompt_components": components,
        "official_guided_prompts": prompts,
        "official_prompt_component_sha256": {
            task: {
                key: {
                    component: sha256_text(text)
                    for component, text in values.items()
                }
                for key, values in task_components.items()
            }
            for task, task_components in components.items()
        },
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
        "official_prompt_components",
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
    components = extract_official_prompt_components(root)
    extracted = {
        task: {key: values["composed"] for key, values in task_values.items()}
        for task, task_values in components.items()
    }
    if components != manifest.get("official_prompt_components"):
        raise LockValidationError("Official prompt components differ from the lock manifest")
    if extracted != manifest.get("official_guided_prompts"):
        raise LockValidationError("Official guided prompt extraction differs from the lock manifest")
    composition = manifest.get("prompt_composition")
    if composition != {"id": PROMPT_COMPOSITION_ID, "separator": PROMPT_SEPARATOR}:
        raise LockValidationError("Unsupported official prompt composition contract")
    component_hashes = {
        task: {
            key: {
                component: sha256_text(text)
                for component, text in values.items()
            }
            for key, values in task_components.items()
        }
        for task, task_components in components.items()
    }
    if component_hashes != manifest.get("official_prompt_component_sha256"):
        raise LockValidationError("Official prompt component hash mismatch")
    hashes = {
        task: {key: sha256_text(value) for key, value in values.items()}
        for task, values in extracted.items()
    }
    if "official_guided_prompt_sha256" in manifest and hashes != manifest[
        "official_guided_prompt_sha256"
    ]:
        raise LockValidationError("Official guided prompt text hash mismatch")
    return manifest


def validate_protocol_release_contract(
    manifest_or_path: Mapping[str, Any] | Path,
    dataset_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a raw snapshot lock to the immutable inventory declared by a protocol."""

    manifest = (
        read_lock_manifest(manifest_or_path)
        if isinstance(manifest_or_path, Path)
        else dict(manifest_or_path)
    )
    expected_revision = str(dataset_config.get("locked_revision") or "")
    expected_repo = str(dataset_config.get("source_repo") or "")
    observed_files = len(manifest["files"])
    observed_bytes = sum(int(entry["size"]) for entry in manifest["files"])
    observed_inventory = str(manifest.get("snapshot_inventory_sha256") or "")
    failures = []
    if manifest.get("repo_id") != expected_repo:
        failures.append(f"repo={manifest.get('repo_id')!r}/{expected_repo!r}")
    if manifest.get("commit_sha") != expected_revision:
        failures.append(f"revision={manifest.get('commit_sha')!r}/{expected_revision!r}")
    if observed_files != int(dataset_config.get("expected_snapshot_files") or 0):
        failures.append(
            f"files={observed_files}/{dataset_config.get('expected_snapshot_files')}"
        )
    if observed_bytes != int(dataset_config.get("expected_snapshot_bytes") or 0):
        failures.append(
            f"bytes={observed_bytes}/{dataset_config.get('expected_snapshot_bytes')}"
        )
    if observed_inventory != str(
        dataset_config.get("expected_snapshot_inventory_sha256") or ""
    ):
        failures.append(
            "inventory="
            f"{observed_inventory}/{dataset_config.get('expected_snapshot_inventory_sha256')}"
        )
    if failures:
        raise LockValidationError(
            "AstroVLBench snapshot does not match the protocol-pinned release: "
            + ", ".join(failures)
        )
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


def lock_local_snapshot(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = PINNED_RELEASE_REVISION,
) -> Path:
    """Adopt an existing ``hf download`` directory without modifying or copying it."""

    snapshot = Path(snapshot_dir).resolve()
    bundle = Path(output_dir).resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    lock_path = bundle / "astrovlbench.lock.json"
    manifest = create_lock_manifest(
        snapshot,
        repo_id=repo_id,
        requested_revision=revision,
        commit_sha=revision,
    )
    try:
        relative = snapshot.relative_to(bundle)
    except ValueError:
        manifest["snapshot_path"] = str(snapshot)
    else:
        manifest["snapshot_relpath"] = relative.as_posix()
    write_lock_manifest(manifest, lock_path)
    validate_lock_manifest(snapshot, lock_path)
    return lock_path


def _snapshot_from_lock(lock_path: Path, manifest: Mapping[str, Any]) -> Path:
    absolute = manifest.get("snapshot_path")
    if absolute:
        snapshot = Path(str(absolute))
        if not snapshot.is_absolute():
            raise LockValidationError("snapshot_path in the lock must be absolute")
        return snapshot.resolve()
    relative = manifest.get("snapshot_relpath", "snapshot")
    relative_path = PurePosixPath(str(relative))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise LockValidationError(f"Unsafe snapshot_relpath in lock: {relative!r}")
    return (Path(lock_path).resolve().parent / relative_path).resolve()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _verify_decodable_image(
    path: Path,
) -> tuple[bool, str | None, tuple[int, int] | None, str | None]:
    if not path.is_file():
        return False, "missing_in_pinned_snapshot", None, None
    if path.stat().st_size == 0:
        return False, "upstream_zero_byte_image", None, None
    try:
        from PIL import Image

        with Image.open(path) as image:
            size = image.size
            mode = image.mode
            image.verify()
        return True, None, size, mode
    except Exception as exc:
        return False, f"unreadable_image:{type(exc).__name__}:{exc}", None, None


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


def build_first_overlay(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    revision: str,
) -> dict[str, Any]:
    """Build the deterministic FIRST correction overlay and full image-QA report."""

    snapshot = Path(snapshot_dir).resolve()
    output = Path(output_dir).resolve()
    if output == snapshot or output.is_relative_to(snapshot):
        raise LockValidationError("The correction overlay must be outside the immutable snapshot")
    relative_metadata = PurePosixPath(DOCUMENTED_LAYOUT["task2.first"])
    raw_metadata = snapshot / relative_metadata
    rows = _read_jsonl_rows(raw_metadata)
    filenames = [str(row.get("filename") or "") for row in rows]
    if any(not value for value in filenames):
        raise SchemaError("FIRST metadata contains a blank filename")
    duplicate_filenames = sorted(
        name for name, count in Counter(filenames).items() if count != 1
    )
    if duplicate_filenames:
        raise SchemaError(
            f"FIRST metadata contains duplicate filenames: {duplicate_filenames[:10]}"
        )

    valid_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    valid_dimensions: Counter[str] = Counter()
    valid_modes: Counter[str] = Counter()
    first_root = raw_metadata.parent
    for source_line, source in enumerate(rows, 1):
        row = dict(source)
        filename = str(row["filename"])
        relative = PurePosixPath(filename.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise SchemaError(f"Unsafe FIRST image reference: {filename!r}")
        image_path = first_root / relative
        valid, reason, size, mode = _verify_decodable_image(image_path)
        if valid:
            valid_rows.append(row)
            assert size is not None and mode is not None
            valid_dimensions[f"{size[0]}x{size[1]}"] += 1
            valid_modes[mode] += 1
        else:
            exclusions.append(
                {
                    "source_line": source_line,
                    "filename": filename,
                    "label": str(row.get("label") or ""),
                    "source_split": str(row.get("source_split") or ""),
                    "reason": reason,
                    "local_size_bytes": (
                        image_path.stat().st_size if image_path.is_file() else None
                    ),
                }
            )

    overlay_metadata = output / relative_metadata
    exclusions_path = (
        output / "data/Task2_RadioMorph/MiraBest_F/exclusions.jsonl"
    )
    _atomic_write_text(overlay_metadata, _jsonl_text(valid_rows))
    _atomic_write_text(exclusions_path, _jsonl_text(exclusions))

    image_paths = sorted(
        path
        for path in (snapshot / "data").rglob("*")
        if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
    )
    invalid_images: list[dict[str, Any]] = []
    image_counts: Counter[str] = Counter()
    image_hashes: dict[str, list[str]] = {}
    for path in image_paths:
        relative = path.relative_to(snapshot / "data")
        image_counts[relative.parts[0]] += 1
        valid, reason, _, _ = _verify_decodable_image(path)
        if not valid:
            invalid_images.append(
                {
                    "path": relative.as_posix(),
                    "reason": reason,
                    "size_bytes": path.stat().st_size,
                }
            )
        digest = sha256_file(path)
        image_hashes.setdefault(digest, []).append(relative.as_posix())
    duplicate_groups = sorted(
        (members for members in image_hashes.values() if len(members) > 1),
        key=lambda members: members[0],
    )

    reason_counts = Counter(str(row["reason"]) for row in exclusions)
    valid_labels = Counter(str(row.get("label") or "") for row in valid_rows)
    excluded_labels = Counter(str(row.get("label") or "") for row in exclusions)
    if revision == PINNED_RELEASE_REVISION:
        observed = {
            "raw_rows": len(rows),
            "valid_rows": len(valid_rows),
            "excluded_rows": len(exclusions),
        }
        expected = {
            "raw_rows": PINNED_FIRST_RAW_ROWS,
            "valid_rows": PINNED_RELEASE_COUNTS["task2.first"],
            "excluded_rows": sum(PINNED_FIRST_EXCLUSIONS.values()),
        }
        if observed != expected:
            raise SchemaError(f"Pinned FIRST row counts changed: {observed}, expected {expected}")
        if dict(reason_counts) != dict(PINNED_FIRST_EXCLUSIONS):
            raise SchemaError(
                f"Pinned FIRST exclusion reasons changed: {dict(reason_counts)}"
            )
        if dict(valid_labels) != dict(PINNED_FIRST_VALID_LABELS):
            raise SchemaError(f"Pinned FIRST valid labels changed: {dict(valid_labels)}")
        if dict(excluded_labels) != dict(PINNED_FIRST_EXCLUDED_LABELS):
            raise SchemaError(
                f"Pinned FIRST excluded labels changed: {dict(excluded_labels)}"
            )
        expected_invalid = {
            "Task2_RadioMorph/MiraBest_F/images/FRII/"
            "200_198.125+019.841_0.3720_0030.50.png"
        }
        observed_invalid = {str(row["path"]) for row in invalid_images}
        if observed_invalid != expected_invalid:
            raise SchemaError(
                f"Pinned release invalid-image set changed: {sorted(observed_invalid)}"
            )
        if duplicate_groups:
            raise SchemaError(
                f"Pinned release unexpectedly contains duplicate image hashes: {duplicate_groups[:3]}"
            )

    report = {
        "schema_version": 1,
        "repair_policy": (
            "non-destructive overlay; retain only FIRST metadata rows whose pinned "
            "image exists, is non-empty, and passes Pillow verification"
        ),
        "source": {
            "repo_id": DEFAULT_REPO_ID,
            "revision": revision,
            "snapshot_path": str(snapshot),
            "raw_first_metadata": relative_metadata.as_posix(),
            "raw_first_metadata_sha256": sha256_file(raw_metadata),
        },
        "first_metadata": {
            "raw_rows": len(rows),
            "valid_rows": len(valid_rows),
            "excluded_rows": len(exclusions),
            "valid_label_counts": dict(sorted(valid_labels.items())),
            "excluded_reason_counts": dict(sorted(reason_counts.items())),
            "excluded_label_counts": dict(sorted(excluded_labels.items())),
            "valid_dimensions": dict(sorted(valid_dimensions.items())),
            "valid_modes": dict(sorted(valid_modes.items())),
            "overlay_metadata": overlay_metadata.relative_to(output).as_posix(),
            "overlay_metadata_sha256": sha256_file(overlay_metadata),
            "exclusions_ledger": exclusions_path.relative_to(output).as_posix(),
            "exclusions_sha256": sha256_file(exclusions_path),
        },
        "full_image_audit": {
            "image_files": len(image_paths),
            "readable_images": len(image_paths) - len(invalid_images),
            "invalid_images": invalid_images,
            "counts_by_task": dict(sorted(image_counts.items())),
            "duplicate_sha256_groups": duplicate_groups,
        },
    }
    _atomic_write_text(
        output / "repair_report.json",
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return report


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


def _prompt(
    manifest: Mapping[str, Any], task: str, subtask: str | None = None
) -> tuple[str, Mapping[str, str]]:
    prompts = manifest.get("official_guided_prompts")
    components = manifest.get("official_prompt_components")
    if not isinstance(prompts, Mapping) or not isinstance(components, Mapping):
        raise LockValidationError("Lock manifest has no official prompt contract")
    values = prompts.get(task)
    if not isinstance(values, Mapping):
        raise LockValidationError(f"Lock manifest has no official prompt mapping for {task}")
    key = subtask or "default"
    result = values.get(key, values.get("default"))
    if not isinstance(result, str) or not result.strip():
        raise LockValidationError(f"Lock manifest has no official guided prompt for {task}.{key}")
    task_components = components.get(task)
    component = (
        task_components.get(key, task_components.get("default"))
        if isinstance(task_components, Mapping)
        else None
    )
    if not isinstance(component, Mapping):
        raise LockValidationError(f"Lock manifest has no prompt components for {task}.{key}")
    system = component.get("system")
    user = component.get("user")
    composed = component.get("composed")
    if not all(isinstance(value, str) and value.strip() for value in (system, user, composed)):
        raise LockValidationError(f"Malformed prompt components for {task}.{key}")
    if composed != result:
        raise LockValidationError(f"Composed prompt differs for {task}.{key}")
    return result, {
        "prompt_composition_id": PROMPT_COMPOSITION_ID,
        "system_prompt_sha256": sha256_text(str(system)),
        "user_prompt_sha256": sha256_text(str(user)),
        "composed_prompt_sha256": sha256_text(result),
    }


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
    manifest: Mapping[str, Any],
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
        prompt, prompt_evidence = _prompt(manifest, task)
        records.append(
            _record(
                root,
                sample_id=f"astrovlbench_{task}_{_sample_token(source_id)}",
                source_id=source_id,
                task=task,
                subtask=None,
                image=image,
                prompt=prompt,
                label=label,
                metadata={
                    "modality": "image",
                    "prompt_type": "guided",
                    "source_row": str(row_number),
                    **prompt_evidence,
                },
            )
        )
    return records


def _materialize_task2(
    root: Path,
    source_file: Path,
    subtask: str,
    manifest: Mapping[str, Any],
    *,
    image_task_dir: Path | None = None,
) -> list[AstroVLBenchRecord]:
    task_dir = image_task_dir or source_file.parent
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
        prompt, prompt_evidence = _prompt(manifest, "task2", subtask)
        records.append(
            _record(
                root,
                sample_id=f"astrovlbench_task2_{subtask}_{_sample_token(source_id)}",
                source_id=f"radio_{source_id}",
                task="task2",
                subtask=subtask,
                image=image,
                prompt=prompt,
                label=label,
                metadata={
                    "modality": "image",
                    "prompt_type": "guided",
                    "survey": subtask.upper(),
                    "source_row": str(row_number),
                    **prompt_evidence,
                },
            )
        )
    return records


_TASK3_IMAGE_PREFIX: Mapping[str, str] = {
    "Type-1 AGN": "Type1AGN",
    "Type-2 AGN": "Type2AGN",
    "Galaxy": "Galaxy",
}


def _materialize_task3(
    root: Path,
    source_file: Path,
    manifest: Mapping[str, Any],
) -> list[AstroVLBenchRecord]:
    index = _ImageIndex(source_file.parent, root)
    records: list[AstroVLBenchRecord] = []
    prompt, prompt_evidence = _prompt(manifest, "task3")
    for row_number, row in enumerate(_read_csv_rows(source_file), 2):
        source_value = _row_value(row, ("targetid",), required=True)
        label_value = _row_value(row, ("class",), required=True)
        assert source_value is not None and label_value is not None
        label = _canonical_label("task3", label_value)
        image = index.resolve(f"{_TASK3_IMAGE_PREFIX[label]}_{source_value}.png")
        source_id = _source_id(row, image)
        records.append(
            _record(
                root,
                sample_id=f"astrovlbench_task3_{_sample_token(source_id)}",
                source_id=source_id,
                task="task3",
                subtask=None,
                image=image,
                prompt=prompt,
                label=label,
                metadata={
                    "modality": "image",
                    "prompt_type": "guided",
                    "redshift_mode": "without",
                    "source_row": str(row_number),
                    **prompt_evidence,
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
    manifest: Mapping[str, Any],
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
        target_id = _row_value(row, ("TARGETID",), required=True)
        assert target_id is not None
        image = index.resolve(f"spectrum_{target_id}.png")
        source_id = _source_id(row, image)
        cluster_id = f"spectrum_{source_id}"
        for question, label_by_group in _TASK5_LABELS.items():
            if group not in label_by_group:
                continue
            prompt, prompt_evidence = _prompt(manifest, "task5", question)
            records.append(
                _record(
                    root,
                    sample_id=f"astrovlbench_task5_{question}_{_sample_token(source_id)}",
                    source_id=cluster_id,
                    task="task5",
                    subtask=question,
                    image=image,
                    prompt=prompt,
                    label=label_by_group[group],
                    metadata={
                        "modality": "image",
                        "prompt_type": "guided",
                        "asib_group": group,
                        "source_row": str(row_number),
                        **prompt_evidence,
                    },
                )
            )
    return records, dict(sorted(group_counts.items()))


def _assert_unique_complete(records: Sequence[AstroVLBenchRecord]) -> None:
    counts = Counter(record.sample_id for record in records)
    duplicate_ids = sorted(sample_id for sample_id, count in counts.items() if count != 1)
    if duplicate_ids:
        raise SchemaError(f"Duplicate materialized sample IDs: {duplicate_ids[:10]}")
    for image_path in sorted({record.image_path for record in records}):
        image = Path(image_path)
        valid, reason, _, _ = _verify_decodable_image(image)
        if not valid:
            raise SchemaError(f"Materialized image is invalid ({reason}): {image}")
    for record in records:
        if record.reference_label not in record.allowed_labels:
            raise SchemaError(f"Label {record.reference_label!r} is not allowed for {record.task_key}")


def discover_records(
    snapshot_dir: Path,
    lock_manifest: Mapping[str, Any] | Path,
    *,
    overlay_dir: Path | None = None,
) -> DiscoveryResult:
    """Validate a locked snapshot and materialize every documented image task."""

    root = snapshot_dir.resolve()
    manifest = validate_lock_manifest(root, lock_manifest)
    layout = discover_documented_layout(root)
    overlay = Path(overlay_dir).resolve() if overlay_dir is not None else None
    first_metadata = (
        overlay / PurePosixPath(DOCUMENTED_LAYOUT["task2.first"])
        if overlay is not None
        else layout["task2.first"]
    )
    if not first_metadata.is_file():
        raise SchemaError(
            "Prepared FIRST overlay metadata is missing; run AstroVLBench preparation first"
        )

    records: list[AstroVLBenchRecord] = []
    records.extend(_materialize_simple_csv(root, layout["task1"], "task1", manifest))
    records.extend(
        _materialize_task2(
            root,
            first_metadata,
            "first",
            manifest,
            image_task_dir=layout["task2.first"].parent,
        )
    )
    records.extend(_materialize_task2(root, layout["task2.nvss"], "nvss", manifest))
    records.extend(_materialize_task3(root, layout["task3"], manifest))
    records.extend(_materialize_simple_csv(root, layout["task4"], "task4", manifest))
    task5_records, task5_groups = _materialize_task5(root, layout["task5"], manifest)
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
    if str(manifest.get("commit_sha")) == PINNED_RELEASE_REVISION:
        observed_counts = {key: counts.get(key, 0) for key in PINNED_RELEASE_COUNTS}
        if observed_counts != dict(PINNED_RELEASE_COUNTS):
            raise SchemaError(
                f"Pinned AstroVLBench component counts changed: {observed_counts}"
            )
        if len(records) != PINNED_RELEASE_TOTAL:
            raise SchemaError(
                f"Pinned AstroVLBench total is {len(records)}, expected {PINNED_RELEASE_TOTAL}"
            )
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
    """Prepare the derived overlay and return JSON-ready records plus audit evidence."""

    resolved_lock = Path(lock_path).resolve()
    manifest = read_lock_manifest(resolved_lock)
    snapshot_dir = _snapshot_from_lock(resolved_lock, manifest)
    if not snapshot_dir.is_dir():
        raise LockValidationError(f"Locked AstroVLBench snapshot not found: {snapshot_dir}")
    validate_lock_manifest(snapshot_dir, resolved_lock)
    inventory_before = str(manifest.get("snapshot_inventory_sha256") or "")
    if not inventory_before:
        raise LockValidationError("AstroVLBench lock has no snapshot inventory hash")

    overlay_dir = resolved_lock.parent / "overlay"
    repair = build_first_overlay(
        snapshot_dir,
        overlay_dir,
        revision=str(manifest["commit_sha"]),
    )
    # This second validation proves that preparation did not mutate the raw snapshot.
    validated_after = validate_lock_manifest(snapshot_dir, resolved_lock)
    inventory_after = str(validated_after.get("snapshot_inventory_sha256") or "")
    if inventory_after != inventory_before:
        raise LockValidationError("Raw snapshot inventory changed during preparation")

    result = discover_records(
        snapshot_dir,
        resolved_lock,
        overlay_dir=overlay_dir,
    )
    records: list[dict[str, Any]] = []
    for index, record in enumerate(result.records, 1):
        value = record.to_dict()
        value["record_index"] = index
        records.append(value)
    report = result.report_dict()
    report.update(
        {
            "snapshot_inventory_sha256_before": inventory_before,
            "snapshot_inventory_sha256_after": inventory_after,
            "raw_snapshot_unchanged": True,
            "repair_report": repair,
            "prompt_composition": manifest["prompt_composition"],
        }
    )

    records_path = resolved_lock.parent / "records.jsonl"
    adapter_report_path = resolved_lock.parent / "adapter_report.json"
    repair_report_path = overlay_dir / "repair_report.json"
    _atomic_write_text(records_path, _jsonl_text(records))
    _atomic_write_text(
        adapter_report_path,
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    preparation_script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_astrovlbench.py"
    manifest = dict(manifest)
    manifest["overlay_relpath"] = "overlay"
    manifest["derived_artifacts"] = {
        "repair_report": {
            "path": repair_report_path.relative_to(resolved_lock.parent).as_posix(),
            "sha256": sha256_file(repair_report_path),
        },
        "adapter_report": {
            "path": adapter_report_path.relative_to(resolved_lock.parent).as_posix(),
            "sha256": sha256_file(adapter_report_path),
        },
        "records": {
            "path": records_path.relative_to(resolved_lock.parent).as_posix(),
            "sha256": sha256_file(records_path),
            "count": len(records),
        },
        "preparation_implementation": {
            "adapter_path": Path(__file__).name,
            "adapter_sha256": sha256_file(Path(__file__)),
            "script_path": "scripts/prepare_astrovlbench.py",
            "script_sha256": (
                sha256_file(preparation_script) if preparation_script.is_file() else None
            ),
        },
    }
    write_lock_manifest(manifest, resolved_lock)
    validate_lock_manifest(snapshot_dir, resolved_lock)
    return records, report


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
