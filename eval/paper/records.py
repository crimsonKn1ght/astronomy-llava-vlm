"""Canonical record schema shared by dataset adapters and model workers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .artifacts import sha256_file, text_hash, write_jsonl_atomic


class RecordError(RuntimeError):
    pass


def _pair(row: Mapping[str, Any]) -> tuple[str, str]:
    prompt = ""
    reference = ""
    for turn in row.get("conversations") or []:
        role = str(turn.get("from") or "").lower()
        if role == "human":
            prompt = str(turn.get("value") or "")
        elif role in {"gpt", "assistant", "astrollava"}:
            reference = str(turn.get("value") or "")
    return re.sub(r"\s+", " ", prompt.replace("<image>", " ")).strip(), reference.strip()


def canonicalize_llava(
    records_json: str | Path,
    image_dir: str | Path,
    dataset: str,
    *,
    require_images: bool = True,
) -> list[Dict[str, Any]]:
    rows = json.loads(Path(records_json).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise RecordError(f"{records_json} must contain a JSON list")
    root = Path(image_dir).resolve()
    result: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(rows, 1):
        row_id = str(source.get("id") or f"{dataset}_{index:04d}")
        if row_id in seen:
            raise RecordError(f"Duplicate record id: {row_id}")
        seen.add(row_id)
        prompt, reference = _pair(source)
        image = Path(str(source.get("image") or "")).name
        image_path = root / image
        if not prompt or not reference or not image:
            raise RecordError(f"Malformed record {row_id}")
        if require_images and not image_path.is_file():
            raise RecordError(f"Missing image for {row_id}: {image_path}")
        reserved = {"conversations", "prompt", "reference", "image_path"}
        metadata = {key: value for key, value in source.items() if key not in reserved}
        item: Dict[str, Any] = {
            **metadata,
            "id": row_id,
            "record_index": index,
            "dataset": dataset,
            "split": str(source.get("split") or "test"),
            "record_type": str(source.get("record_type") or "caption"),
            "image": image,
            "image_id": str(source.get("image_id") or image),
            "source_object_id": str(source.get("source_object_id") or image),
            "image_path": str(image_path),
            "prompt": prompt,
            "reference": reference,
            "prompt_sha256": source.get("prompt_sha256") or text_hash(prompt),
            "reference_sha256": source.get("reference_sha256") or text_hash(reference),
        }
        if image_path.is_file():
            item["image_sha256"] = sha256_file(image_path)
        result.append(item)
    return result


def validate_unique_records(records: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        row_id = str(record.get("id") or "")
        if not row_id:
            raise RecordError("Canonical record missing id")
        if row_id in seen:
            raise RecordError(f"Duplicate canonical record id: {row_id}")
        seen.add(row_id)
        for key in ("image_path", "prompt", "reference", "image_id", "source_object_id"):
            if record.get(key) in (None, ""):
                raise RecordError(f"Canonical record {row_id} missing {key}")


def write_records(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    validate_unique_records(records)
    target = Path(path)
    write_jsonl_atomic(target, records)
    return target
