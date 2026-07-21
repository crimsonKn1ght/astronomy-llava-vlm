"""Frozen internal split preparation, lineage, and image-leakage auditing."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .artifacts import sha256_file, text_hash, write_json_atomic, write_jsonl_atomic


class InternalDataError(RuntimeError):
    pass


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _last_pair(row: Mapping[str, Any]) -> tuple[str, str]:
    prompt = ""
    reference = ""
    for turn in row.get("conversations") or []:
        role = str(turn.get("from") or "").lower()
        value = str(turn.get("value") or "")
        if role == "human":
            prompt = value
        elif role in {"gpt", "assistant", "astrollava"}:
            reference = value
    return prompt.replace("<image>", " ").strip(), reference.strip()


def infer_record_type(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("record_type") or "").lower()
    if explicit in {"caption", "qa"}:
        return explicit
    return "qa" if "_qa" in str(row.get("id") or "").lower() else "caption"


def extract_frozen_test(
    archive_path: str | Path,
    member: str,
    output_path: str | Path,
    expected_sha256: str,
) -> Path:
    archive = Path(archive_path)
    target = Path(output_path)
    if not archive.is_file():
        raise InternalDataError(f"Frozen internal artifact is missing: {archive}")
    with zipfile.ZipFile(archive, "r") as bundle:
        names = bundle.namelist()
        if member not in names:
            raise InternalDataError(f"{archive} does not contain {member!r}")
        data = bundle.read(member)
    digest = hashlib.sha256(data).hexdigest()
    if digest.lower() != expected_sha256.lower():
        raise InternalDataError(
            f"Frozen {member} SHA-256 is {digest}, expected {expected_sha256.lower()}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def canonical_records(
    records_json: str | Path,
    image_dir: str | Path | None = None,
    require_images: bool = False,
) -> list[Dict[str, Any]]:
    raw = json.loads(Path(records_json).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise InternalDataError("Internal test JSON must contain a list")
    root = Path(image_dir) if image_dir else None
    records: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(raw, 1):
        row_id = str(row.get("id") or f"record_{index}")
        if row_id in seen:
            raise InternalDataError(f"Duplicate internal record id: {row_id}")
        seen.add(row_id)
        image_name = PurePosixPath(str(row.get("image") or "")).name
        prompt, reference = _last_pair(row)
        if not image_name or not prompt or not reference:
            raise InternalDataError(f"Malformed internal record {row_id}")
        image_path = root / image_name if root else Path(image_name)
        if require_images and not image_path.is_file():
            raise InternalDataError(f"Missing internal image: {image_path}")
        item: Dict[str, Any] = {
            "id": row_id,
            "record_index": index,
            "dataset": "internal",
            "split": "test",
            "record_type": infer_record_type(row),
            "image": image_name,
            "image_id": image_name,
            "source_object_id": image_name,
            "image_path": str(image_path),
            "prompt": re.sub(r"\s+", " ", prompt).strip(),
            "reference": reference,
            "prompt_sha256": text_hash(re.sub(r"\s+", " ", prompt).strip()),
            "reference_sha256": text_hash(reference),
            "source_corpus": "UniverseTBD/AstroLLaVA_convos",
            "reference_provenance": (
                "documented_gpt4_generated_qa"
                if infer_record_type(row) == "qa"
                else "documented_human_written_caption"
            ),
        }
        if image_path.is_file():
            item["image_sha256"] = sha256_file(image_path)
        records.append(item)
    return records


def validate_frozen_counts(records: Sequence[Mapping[str, Any]], expected: Mapping[str, int]) -> None:
    counts = Counter(str(row["record_type"]) for row in records)
    images = {str(row["image_id"]) for row in records}
    observed = {
        "expected_records": len(records),
        "expected_images": len(images),
        "expected_caption_records": counts["caption"],
        "expected_qa_records": counts["qa"],
    }
    for key, value in observed.items():
        if int(expected[key]) != value:
            raise InternalDataError(f"Internal {key} is {value}, expected {expected[key]}")


def write_split_outputs(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    source_revision: str,
    test_json_sha256: str,
    builder_revision: str,
    builder_arguments: Mapping[str, Any],
) -> Dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "records.jsonl"
    csv_path = root / "split_manifest.csv"
    lineage_path = root / "data_lineage.json"
    manifest_path = root / "manifest.json"
    write_jsonl_atomic(jsonl_path, records)
    fields = [
        "id",
        "image_id",
        "record_type",
        "source_corpus",
        "split",
        "reference_provenance",
        "image_sha256",
        "prompt_sha256",
        "reference_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in fields})
    counts = Counter(str(row["record_type"]) for row in records)
    lineage = {
        "source_dataset": "UniverseTBD/AstroLLaVA_convos",
        "source_revision": source_revision,
        "caption_records": counts["caption"],
        "qa_records": counts["qa"],
        "caption_provenance": "Documented by the source dataset as human-written captions.",
        "qa_provenance": "Documented by the source dataset/build pipeline as GPT-4-generated conversations.",
        "rag_used_in_training": False,
        "rag_used_in_evaluation": False,
        "external_datasets_used_for_training": False,
        "limitations": [
            "The source metadata does not provide an independently verified author field for every turn.",
            "Foundation-model pretraining data cannot be exhaustively audited from released weights.",
        ],
    }
    write_json_atomic(lineage_path, lineage)
    manifest = {
        "source_revision": source_revision,
        "test_json_sha256": test_json_sha256.lower(),
        "builder_git_revision": builder_revision,
        "builder_arguments": dict(builder_arguments),
        "records": len(records),
        "images": len({str(row["image_id"]) for row in records}),
        "caption_records": counts["caption"],
        "qa_records": counts["qa"],
        "records_jsonl_sha256": sha256_file(jsonl_path),
        "split_manifest_sha256": sha256_file(csv_path),
    }
    write_json_atomic(manifest_path, manifest)
    return {
        "records": jsonl_path,
        "split_manifest": csv_path,
        "lineage": lineage_path,
        "manifest": manifest_path,
    }


def decoded_pixel_hash(path: str | Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        payload = (
            f"RGB:{rgb.width}x{rgb.height}:".encode("ascii") + np.asarray(rgb, dtype=np.uint8).tobytes()
        )
    return hashlib.sha256(payload).hexdigest()


def _dct_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float64)
    scale0 = math.sqrt(1.0 / size)
    scale = math.sqrt(2.0 / size)
    for k in range(size):
        alpha = scale0 if k == 0 else scale
        for n in range(size):
            matrix[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2 * size))
    return matrix


_DCT32 = _dct_matrix(32)


def phash64(path: str | Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    coefficients = _DCT32 @ pixels @ _DCT32.T
    low = coefficients[:8, :8].copy()
    values = low.flatten()
    median = float(np.median(values[1:]))
    result = 0
    for index, value in enumerate(values):
        if value > median:
            result |= 1 << index
    return result


def image_index(records: Sequence[Mapping[str, Any]], image_dir: str | Path) -> list[Dict[str, Any]]:
    root = Path(image_dir)
    unique = sorted({str(row.get("image") or row.get("image_id")) for row in records})
    index: list[Dict[str, Any]] = []
    for name in unique:
        path = root / name
        if not path.is_file():
            raise InternalDataError(f"Missing image for leakage audit: {path}")
        index.append(
            {
                "image": name,
                "byte_sha256": sha256_file(path),
                "pixel_sha256": decoded_pixel_hash(path),
                "phash64": f"{phash64(path):016x}",
            }
        )
    return index


def audit_image_overlap(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    image_dir: str | Path,
    output_dir: str | Path,
    likely_threshold: int = 4,
    sensitivity_threshold: int = 8,
) -> Dict[str, Any]:
    train_index = image_index(train_records, image_dir)
    test_index = image_index(test_records, image_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_byte = defaultdict(list)
    train_pixel = defaultdict(list)
    for item in train_index:
        train_byte[item["byte_sha256"]].append(item["image"])
        train_pixel[item["pixel_sha256"]].append(item["image"])
    collisions: list[Dict[str, Any]] = []
    train_hashes = [(int(item["phash64"], 16), item["image"]) for item in train_index]
    for test in test_index:
        for match in train_byte.get(test["byte_sha256"], []):
            collisions.append({"type": "exact_bytes", "distance": 0, "train": match, "test": test["image"]})
        for match in train_pixel.get(test["pixel_sha256"], []):
            collisions.append({"type": "exact_pixels", "distance": 0, "train": match, "test": test["image"]})
        test_hash = int(test["phash64"], 16)
        for train_hash, train_name in train_hashes:
            distance = (test_hash ^ train_hash).bit_count()
            if distance <= sensitivity_threshold:
                collisions.append(
                    {
                        "type": "phash_likely" if distance <= likely_threshold else "phash_sensitivity",
                        "distance": distance,
                        "train": train_name,
                        "test": test["image"],
                    }
                )
    collisions.sort(key=lambda row: (row["distance"], row["test"], row["train"], row["type"]))
    write_jsonl_atomic(output / "image_overlap_candidates.jsonl", collisions)
    write_jsonl_atomic(output / "train_image_index.jsonl", train_index)
    write_jsonl_atomic(output / "test_image_index.jsonl", test_index)
    report = {
        "train_images": len(train_index),
        "test_images": len(test_index),
        "likely_threshold": likely_threshold,
        "sensitivity_threshold": sensitivity_threshold,
        "exact_byte_pairs": sum(row["type"] == "exact_bytes" for row in collisions),
        "exact_pixel_pairs": sum(row["type"] == "exact_pixels" for row in collisions),
        "phash_likely_pairs": sum(row["type"] == "phash_likely" for row in collisions),
        "phash_sensitivity_pairs": sum(row["type"] == "phash_sensitivity" for row in collisions),
        "candidate_file": "image_overlap_candidates.jsonl",
    }
    write_json_atomic(output / "image_overlap_report.json", report)
    return report


def audit_text_overlap(
    train_records: Sequence[Mapping[str, Any]], test_records: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    train_refs = defaultdict(list)
    for row in train_records:
        train_refs[normalize_text(row.get("reference") or _last_pair(row)[1])].append(str(row.get("id")))
    matches = []
    for row in test_records:
        normalized = normalize_text(row.get("reference") or _last_pair(row)[1])
        if normalized and normalized in train_refs:
            matches.append(
                {
                    "test_id": str(row.get("id")),
                    "train_ids": train_refs[normalized],
                    "normalized_reference_sha256": text_hash(normalized),
                }
            )
    return {"test_records": len(test_records), "exact_normalized_reference_matches": len(matches), "matches": matches}
