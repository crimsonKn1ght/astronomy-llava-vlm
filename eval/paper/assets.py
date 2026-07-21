"""Pinned Hugging Face asset resolution and checkpoint verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence

from .artifacts import sha256_file, write_json_atomic
from .protocol import PaperProtocol


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class AssetError(RuntimeError):
    pass


def snapshot_commit(path: str | Path) -> str | None:
    candidate = Path(path).resolve()
    for part in reversed(candidate.parts):
        if COMMIT_RE.fullmatch(part.lower()):
            return part.lower()
    return None


def verify_snapshot_revision(path: str | Path, expected_revision: str) -> str:
    resolved = snapshot_commit(path)
    if resolved is None:
        metadata = Path(path) / ".paper-eval-revision"
        if metadata.is_file():
            resolved = metadata.read_text(encoding="ascii").strip().lower()
    if resolved != expected_revision.lower():
        raise AssetError(
            f"Resolved snapshot revision is {resolved!r}, expected {expected_revision.lower()}"
        )
    return resolved


def download_snapshot(
    *,
    repo_id: str,
    revision: str,
    cache_dir: str | Path,
    repo_type: str = "model",
    allow_patterns: Sequence[str] | None = None,
    token: str | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AssetError("huggingface_hub is required to download pinned evaluation assets") from exc
    path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            cache_dir=str(cache_dir),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            token=token,
            resume_download=True,
        )
    )
    verify_snapshot_revision(path, revision)
    return path


def _zip_member_is_link(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def safe_extract_zip(archive: str | Path, destination: str | Path) -> Path:
    source = Path(archive)
    target = Path(destination)
    marker = target / ".paper-eval-extracted"
    digest = sha256_file(source)
    if marker.is_file() and marker.read_text(encoding="ascii").strip() == digest:
        return target
    if target.exists() and any(target.iterdir()):
        raise AssetError(f"Checkpoint extraction directory is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper-eval-zip-", dir=target.parent) as tmp:
        temp = Path(tmp)
        with zipfile.ZipFile(source, "r") as bundle:
            for info in bundle.infolist():
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts or _zip_member_is_link(info):
                    raise AssetError(f"Unsafe checkpoint ZIP member: {info.filename!r}")
            bundle.extractall(temp)
        target.mkdir(parents=True, exist_ok=True)
        for child in temp.iterdir():
            shutil.move(str(child), target / child.name)
        marker.write_text(digest + "\n", encoding="ascii")
    return target


def find_unique(root: str | Path, relative_suffix: str) -> Path:
    suffix = relative_suffix.replace("\\", "/")
    matches = [
        path
        for path in Path(root).rglob(Path(suffix).name)
        if path.as_posix().endswith(suffix)
    ]
    if len(matches) != 1:
        raise AssetError(f"Expected one *{suffix} below {root}, found {len(matches)}")
    return matches[0]


def verify_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise AssetError(f"Missing {label}: {file_path}")
    observed = sha256_file(file_path)
    if observed != expected_sha256.lower():
        raise AssetError(f"{label} SHA-256 is {observed}, expected {expected_sha256.lower()}")
    return file_path


def zip_member_sha256(archive: str | Path, relative_suffix: str) -> str:
    """Hash one uniquely named regular ZIP member without trusting an extraction."""

    suffix = relative_suffix.replace("\\", "/")
    with zipfile.ZipFile(archive, "r") as bundle:
        matches = [
            info
            for info in bundle.infolist()
            if not info.is_dir()
            and not _zip_member_is_link(info)
            and PurePosixPath(info.filename).as_posix().endswith(suffix)
        ]
        if len(matches) != 1:
            raise AssetError(
                f"Expected one *{suffix} member in {archive}, found {len(matches)}"
            )
        digest = hashlib.sha256()
        with bundle.open(matches[0], "r") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


@dataclass
class AssetRegistry:
    root: Path
    entries: Dict[str, Dict[str, Any]]
    protocol_sha256: str | None = None
    shared_assets: Dict[str, Dict[str, Any]] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "AssetRegistry":
        file_path = Path(path)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return cls(
            file_path.parent,
            data["entries"],
            data.get("protocol_sha256"),
            data.get("shared_assets") or {},
        )

    def model_path(self, label: str) -> Path:
        entry = self.entries.get(label)
        if not entry or not entry.get("model_path"):
            raise AssetError(f"Asset manifest has no model path for {label}")
        return Path(entry["model_path"])

    def checkpoint_path(self, label: str) -> Path:
        entry = self.entries.get(label)
        if not entry or not entry.get("checkpoint_path"):
            raise AssetError(f"Asset manifest has no checkpoint path for {label}")
        return Path(entry["checkpoint_path"])

    def shared_path(self, key: str) -> Path:
        entry = (self.shared_assets or {}).get(key)
        if not entry or not entry.get("snapshot_path"):
            raise AssetError(f"Asset manifest has no shared snapshot path for {key}")
        return Path(entry["snapshot_path"])

    def validate_model(self, label: str, protocol: PaperProtocol) -> None:
        if self.protocol_sha256 != protocol.fingerprint:
            raise AssetError(
                "Asset manifest protocol hash does not match the loaded paper protocol"
            )
        model = protocol.data["models"].get(label)
        entry = self.entries.get(label)
        if model is None or entry is None:
            raise AssetError(f"Asset manifest has no locked entry for {label}")
        for key in ("repo_id", "revision", "backend"):
            if entry.get(key) != model.get(key):
                raise AssetError(f"Asset manifest {label}.{key} does not match the protocol")
        snapshot = entry.get("snapshot_path")
        if not snapshot:
            raise AssetError(f"Asset manifest entry for {label} has no resolved snapshot")
        verify_snapshot_revision(snapshot, str(model["revision"]))
        if label == "astraq_stage1":
            verify_file(entry["archive_path"], model["checkpoint_sha256"], "Stage-1 archive")
            archived_connector_sha256 = zip_member_sha256(
                entry["archive_path"], "connector.safetensors"
            )
            if entry.get("connector_sha256") != archived_connector_sha256:
                raise AssetError(
                    "Stage-1 manifest connector hash does not match the pinned archive member"
                )
            connector = verify_file(
                entry["connector_path"], archived_connector_sha256, "Stage-1 connector"
            )
            if connector.parent != self.checkpoint_path(label):
                raise AssetError("Stage-1 connector path does not match checkpoint_path")
        elif label == "astraq_stage2":
            verify_file(entry["connector_path"], model["connector_sha256"], "Stage-2 connector")
            verify_file(entry["lora_path"], model["lora_sha256"], "Stage-2 LoRA adapter")
            verify_file(
                entry["lora_config_path"],
                entry["lora_config_sha256"],
                "Stage-2 LoRA configuration",
            )

        if model["backend"] == "astraq":
            shared = self.shared_assets or {}
            for name, expected in protocol.data["base_models"].items():
                locked = shared.get(f"base_models.{name}")
                if not locked or locked.get("revision") != expected["revision"]:
                    raise AssetError(f"Asset manifest is missing pinned shared asset {name}")
                verify_snapshot_revision(locked["snapshot_path"], expected["revision"])
        elif model["backend"] == "astrollava":
            expected = model["vision_encoder"]
            locked = (self.shared_assets or {}).get("models.astrollava.vision_encoder")
            if not locked or locked.get("revision") != expected["revision"]:
                raise AssetError("Asset manifest is missing AstroLLaVA's pinned vision encoder")
            verify_snapshot_revision(locked["snapshot_path"], expected["revision"])


def materialize_model_assets(
    protocol: PaperProtocol,
    model_labels: Iterable[str],
    asset_root: str | Path,
    cache_dir: str | Path,
    *,
    dry_run: bool = False,
    suites: Iterable[str] = (),
) -> Path:
    root = Path(asset_root).resolve()
    manifest_path = root / "asset_manifest.json"
    entries: Dict[str, Dict[str, Any]] = {}
    shared_assets: Dict[str, Dict[str, Any]] = {}
    selected_labels = list(model_labels)
    selected_suites = set(suites)
    for label in selected_labels:
        model = protocol.data["models"][label]
        entry: Dict[str, Any] = {
            "repo_id": model["repo_id"],
            "revision": model["revision"],
            "backend": model["backend"],
        }
        if dry_run:
            entry["planned"] = True
            entries[label] = entry
            continue
        patterns = None
        if label == "astraq_stage1":
            patterns = [model["checkpoint_file"]]
        elif label == "astraq_stage2":
            patterns = [f"{model['checkpoint_prefix']}/**"]
        snapshot = download_snapshot(
            repo_id=model["repo_id"],
            revision=model["revision"],
            cache_dir=cache_dir,
            allow_patterns=patterns,
        )
        entry["snapshot_path"] = str(snapshot)
        entry["resolved_revision"] = verify_snapshot_revision(snapshot, model["revision"])
        if label == "astraq_stage1":
            archive = snapshot / model["checkpoint_file"]
            verify_file(archive, model["checkpoint_sha256"], "Stage-1 epoch-3 archive")
            extracted = safe_extract_zip(archive, root / label / "checkpoint")
            connector = find_unique(extracted, "connector.safetensors")
            connector_sha256 = zip_member_sha256(archive, "connector.safetensors")
            verify_file(connector, connector_sha256, "Stage-1 extracted connector")
            checkpoint = connector.parent
            entry.update(
                {
                    "archive_path": str(archive),
                    "archive_sha256": sha256_file(archive),
                    "checkpoint_path": str(checkpoint),
                    "connector_path": str(connector),
                    "connector_sha256": connector_sha256,
                }
            )
        elif label == "astraq_stage2":
            checkpoint = snapshot / model["checkpoint_prefix"]
            connector = verify_file(
                checkpoint / "connector.safetensors", model["connector_sha256"], "Stage-2 connector"
            )
            lora = verify_file(
                checkpoint / "lora" / "adapter_model.safetensors",
                model["lora_sha256"],
                "Stage-2 LoRA adapter",
            )
            lora_config = checkpoint / "lora" / "adapter_config.json"
            if not lora_config.is_file():
                raise AssetError(f"Missing Stage-2 LoRA configuration: {lora_config}")
            entry.update(
                {
                    "checkpoint_path": str(checkpoint),
                    "connector_path": str(connector),
                    "lora_path": str(lora),
                    "lora_config_path": str(lora_config),
                    "connector_sha256": sha256_file(connector),
                    "lora_sha256": sha256_file(lora),
                    "lora_config_sha256": sha256_file(lora_config),
                }
            )
        else:
            entry["model_path"] = str(snapshot)
        entries[label] = entry
    shared_specs: list[tuple[str, Mapping[str, Any]]] = []
    if any(protocol.data["models"][label]["backend"] == "astraq" for label in selected_labels):
        shared_specs.extend(
            (f"base_models.{name}", spec)
            for name, spec in protocol.data["base_models"].items()
        )
    if "internal" in selected_suites:
        shared_specs.extend(
            (f"scorers.{name}", spec)
            for name, spec in protocol.data["scorers"].items()
        )
    if any(protocol.data["models"][label]["backend"] == "astrollava" for label in selected_labels):
        shared_specs.append(
            (
                "models.astrollava.vision_encoder",
                protocol.data["models"]["astrollava"]["vision_encoder"],
            )
        )
    for key, spec in shared_specs:
        shared_entry = {"repo_id": spec["repo_id"], "revision": spec["revision"]}
        if dry_run:
            shared_entry["planned"] = True
        else:
            snapshot = download_snapshot(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                cache_dir=cache_dir,
            )
            shared_entry.update(
                {
                    "snapshot_path": str(snapshot),
                    "resolved_revision": verify_snapshot_revision(snapshot, spec["revision"]),
                }
            )
        shared_assets[key] = shared_entry
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        existing_entries: Dict[str, Any] = {}
        existing_shared: Dict[str, Any] = {}
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("protocol_sha256") == protocol.fingerprint:
                existing_entries.update(existing.get("entries") or {})
                existing_shared.update(existing.get("shared_assets") or {})
        existing_entries.update(entries)
        existing_shared.update(shared_assets)
        write_json_atomic(
            manifest_path,
            {
                "protocol_sha256": protocol.fingerprint,
                "entries": existing_entries,
                "shared_assets": existing_shared,
            },
        )
    return manifest_path
