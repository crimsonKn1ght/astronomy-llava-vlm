"""Download, validate, audit, and adapt the DeepSDO Description dataset.

The KASI release contains tab-separated caption files and a shared image folder.
This adapter preserves the official splits, emits only explicitly selected splits,
and records the release quirks needed for a defensible external evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, List, Mapping, MutableSequence, Sequence


DATASET_URL = "http://swds.kasi.re.kr/sdo/kasi_deepsdo_desc_dataset.tar.gz"
ARCHIVE_BYTES = 36_555_441
# Hash of the archive served by KASI on 2026-07-21. KASI does not publish a hash.
ARCHIVE_SHA256 = "508382874f62510add0ce925a35fe51d58e0673b79ac559ebcfcae84adfd139e"
EXPECTED_SPLIT_COUNTS = {"train": 847, "val": 102, "test": 102}
EXPECTED_VALID_SPLIT_COUNTS = {"train": 846, "val": 102, "test": 102}
ANNOTATION_SHA256 = {
    "train": "05e64de84020137d74a6d06496ac3ffad9a20cf5153d6f2bbe99119a3bae98f5",
    "val": "4eb7f387fe0e50cc3ce690f33b18d4db8e1b7bf852fc3ffbe7a268afe591dde7",
    "test": "7716d84fe14b68c1f578b931b209ee6077cf9b6df82263ef968031ebc5030518",
}
CAPTION_PROMPT = "Describe this solar image."
CAPTION_NORMALIZATION = "NFKC+smart-apostrophe+casefold+whitespace-v1"
TOPIC_MAPPING_PROVENANCE = (
    "Derived descriptive strata from the pinned official test ordering and reference-caption "
    "semantics; these are not official DeepSDO class labels."
)
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_DOWNLOAD_RETRIES = 4
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_DOWNLOAD_TIMEOUT = 60


@dataclass(frozen=True)
class CaptionRow:
    image: str
    caption: str
    line_number: int = field(default=0, compare=False)


@dataclass(frozen=True)
class TopicStratum:
    key: str
    label: str
    start_index: int
    end_index: int

    @property
    def count(self) -> int:
        return self.end_index - self.start_index + 1


@dataclass(frozen=True)
class ImageMetadata:
    timestamp_utc: str
    instrument: str
    channel: str
    wavelength_angstrom: int | None
    collapsed_modality: str


TEST_TOPIC_STRATA: tuple[TopicStratum, ...] = (
    TopicStratum("sunspots", "Sunspots", 1, 8),
    TopicStratum("flares", "Flares", 9, 26),
    TopicStratum("prominences", "Prominences", 27, 35),
    TopicStratum("prominence_eruptions", "Prominence eruptions", 36, 49),
    TopicStratum("coronal_holes", "Coronal holes", 50, 60),
    TopicStratum("coronal_loops", "Coronal loops", 61, 72),
    TopicStratum("filaments", "Filaments", 73, 78),
    TopicStratum("active_regions", "Active regions", 79, 89),
    TopicStratum("eclipses_transits", "Eclipses/transits", 90, 102),
)
EXPECTED_TEST_TOPIC_COUNTS = {topic.key: topic.count for topic in TEST_TOPIC_STRATA}
EXPECTED_TEST_CHANNEL_COUNTS = {
    "AIA/94": 1,
    "AIA/131": 1,
    "AIA/171": 29,
    "AIA/193": 13,
    "AIA/211": 4,
    "AIA/304": 46,
    "HMI/Ic": 8,
}
EXPECTED_TEST_MODALITY_COUNTS = {
    "hmi_continuum": 8,
    "aia_304": 46,
    "aia_other_euv": 48,
}
EXPECTED_TRAIN_ANOMALIES = {
    ("malformed_annotation", 397, None),
    ("missing_annotated_image", 137, "20130515_014431_SDO_AIA_304_512.jpg"),
    ("missing_annotated_image", 405, "20100527_184602_SDO_AIA_171_512.jpg"),
}
EXPECTED_NORMALIZED_TEST_TRAIN_OVERLAP = 100

_IMAGE_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_SDO_"
    r"(?:(?P<aia>AIA)_(?P<aia_channel>94|131|171|193|211|304)|"
    r"(?P<hmi>HMI)_(?P<hmi_channel>Ic))_512\.jpg$"
)
_CONTENT_RANGE_RE = re.compile(r"^bytes (?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+)$")


class IncompleteDownloadError(IOError):
    """The server ended a response before the pinned archive was complete."""


def sha256_file(path: Path, chunk_size: int = DOWNLOAD_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def normalize_caption(caption: str) -> str:
    normalized = unicodedata.normalize("NFKC", caption)
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    return " ".join(normalized.casefold().split())


def validate_archive(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"DeepSDO archive not found: {path}")
    size = path.stat().st_size
    if size != ARCHIVE_BYTES:
        raise ValueError(
            f"DeepSDO archive is {size:,} bytes; expected {ARCHIVE_BYTES:,}. "
            "The file is retained; rerun with --download to resume a .part file or inspect it."
        )
    digest = sha256_file(path)
    if digest != ARCHIVE_SHA256:
        raise ValueError(
            f"DeepSDO archive SHA-256 is {digest}, expected {ARCHIVE_SHA256}. "
            "The file is retained because the upstream archive may have changed."
        )
    return digest


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return int(status or 200)


def _validate_content_range(response: object, offset: int) -> None:
    header = getattr(response, "headers", {}).get("Content-Range")
    match = _CONTENT_RANGE_RE.fullmatch(header or "")
    if not match:
        raise ValueError(f"Range response has invalid Content-Range: {header!r}")
    start, end, total = (int(match.group(name)) for name in ("start", "end", "total"))
    if start != offset or end < start or total != ARCHIVE_BYTES or end >= total:
        raise ValueError(
            "Range response does not match the requested/pinned archive: "
            f"requested offset={offset}, Content-Range={header!r}"
        )


def _append_response(response: object, partial: Path, offset: int) -> None:
    with partial.open("ab") as output:
        current = offset
        while current < ARCHIVE_BYTES:
            chunk = response.read(min(DOWNLOAD_CHUNK_BYTES, ARCHIVE_BYTES - current))
            if not chunk:
                break
            output.write(chunk)
            output.flush()
            current += len(chunk)
        if current == ARCHIVE_BYTES and response.read(1):
            raise ValueError("DeepSDO server returned more bytes than the pinned archive size")
    if current != ARCHIVE_BYTES:
        raise IncompleteDownloadError(
            f"DeepSDO download stopped at {current:,}/{ARCHIVE_BYTES:,} bytes"
        )


def _consume_matching_prefix(response: object, partial: Path, offset: int) -> None:
    """Verify a full HTTP 200 response against the saved prefix before appending."""

    with partial.open("rb") as saved:
        remaining = offset
        while remaining:
            expected = saved.read(min(DOWNLOAD_CHUNK_BYTES, remaining))
            received = response.read(len(expected))
            if received != expected:
                raise ValueError(
                    "Server ignored Range and its full response does not match the saved prefix; "
                    "the partial file was retained unchanged"
                )
            remaining -= len(expected)


def _download_once(
    url: str,
    partial: Path,
    opener: Callable[..., object],
    timeout: int,
) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > ARCHIVE_BYTES:
        raise ValueError(
            f"Partial DeepSDO archive is {offset:,} bytes, larger than {ARCHIVE_BYTES:,}; "
            "it was retained for inspection"
        )
    if offset == ARCHIVE_BYTES:
        return

    headers = {
        "User-Agent": "AstraQ-VL DeepSDO adapter",
        "Accept-Encoding": "identity",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with opener(request, timeout=timeout) as response:
        status = _response_status(response)
        if offset and status == 206:
            _validate_content_range(response, offset)
            _append_response(response, partial, offset)
        elif offset and status == 200:
            # Some simple servers ignore Range. Preserve the verified prefix rather
            # than deleting it: consume and compare the duplicate prefix, then append.
            _consume_matching_prefix(response, partial, offset)
            _append_response(response, partial, offset)
        elif not offset and status in (200, 206):
            if status == 206:
                _validate_content_range(response, 0)
            partial.touch(exist_ok=True)
            _append_response(response, partial, 0)
        else:
            raise urllib.error.HTTPError(url, status, "unexpected HTTP status", {}, None)


def download_archive(
    url: str,
    destination: Path,
    *,
    max_retries: int = DEFAULT_DOWNLOAD_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    opener: Callable[..., object] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Download the pinned archive, resuming a retained ``.part`` prefix.

    Network failures never delete the partial file. A subsequent invocation issues
    an HTTP Range request from its current size. Retries are bounded and exponential.
    """

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_archive(destination)
        print(f"Using verified archive: {destination}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    resolved_opener = opener or urllib.request.urlopen
    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            _download_once(url, partial, resolved_opener, timeout)
            validate_archive(partial)
            os.replace(partial, destination)
            print(f"Downloaded and verified: {destination}")
            return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_fn(backoff_seconds * (2 ** (attempt - 1)))
    assert last_error is not None
    raise RuntimeError(
        f"DeepSDO download failed after {max_retries} attempts; partial retained at {partial}"
    ) from last_error


def _safe_members(archive: tarfile.TarFile) -> List[tarfile.TarInfo]:
    members = archive.getmembers()
    seen: set[str] = set()
    for member in members:
        raw_name = member.name
        if "\\" in raw_name or "\x00" in raw_name or re.match(r"^[A-Za-z]:", raw_name):
            raise ValueError(f"Unsafe path in DeepSDO archive: {raw_name!r}")
        name = PurePosixPath(raw_name)
        if name.is_absolute() or not name.parts or ".." in name.parts:
            raise ValueError(f"Unsafe path in DeepSDO archive: {raw_name!r}")
        normalized = name.as_posix().rstrip("/")
        if normalized in ("", ".") or normalized in seen:
            raise ValueError(f"Duplicate or empty path in DeepSDO archive: {raw_name!r}")
        seen.add(normalized)
        if member.issym() or member.islnk():
            raise ValueError(f"Links are not allowed in DeepSDO archive: {raw_name!r}")
        if not (member.isdir() or member.isreg()):
            raise ValueError(f"Special files are not allowed in DeepSDO archive: {raw_name!r}")
        if member.size < 0:
            raise ValueError(f"Invalid member size in DeepSDO archive: {raw_name!r}")
    return members


def _extract_safely(
    archive: tarfile.TarFile, members: Sequence[tarfile.TarInfo], destination: Path
) -> None:
    for member in members:
        relative = PurePosixPath(member.name)
        target = destination.joinpath(*relative.parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Could not read DeepSDO archive member: {member.name!r}")
        with source, target.open("xb") as output:
            shutil.copyfileobj(source, output, length=DOWNLOAD_CHUNK_BYTES)


def extract_archive(archive_path: Path, destination: Path) -> None:
    marker = destination / ".deepsdo-extracted"
    if marker.is_file():
        marker_hash = marker.read_text(encoding="ascii").strip()
        if marker_hash != ARCHIVE_SHA256:
            raise ValueError(
                f"Extraction marker at {marker} contains {marker_hash!r}, expected {ARCHIVE_SHA256}"
            )
        return
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"Extraction directory is not empty and has no completion marker: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deepsdo-extract-", dir=destination.parent) as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = _safe_members(archive)
            _extract_safely(archive, members, tmp_path)
        destination.mkdir(parents=True, exist_ok=True)
        for item in tmp_path.iterdir():
            shutil.move(str(item), destination / item.name)
        marker.write_text(ARCHIVE_SHA256 + "\n", encoding="ascii")


def _record_parse_issue(
    issues: MutableSequence[dict] | None,
    path: Path,
    line_no: int,
    raw_line: str,
    issue_type: str,
    message: str,
) -> None:
    if issues is not None:
        issues.append(
            {
                "issue": issue_type,
                "line_number": line_no,
                "raw_line_sha256": sha256_text(raw_line),
                "message": message,
            }
        )


def parse_caption_file(
    path: Path,
    allow_malformed: bool = False,
    *,
    issues: MutableSequence[dict] | None = None,
) -> List[CaptionRow]:
    rows: List[CaptionRow] = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            issue_type = ""
            if "\t" not in line:
                issue_type = "missing_tab"
                message = f"{path}:{line_no}: expected image and caption separated by a tab"
            else:
                image, caption = line.split("\t", 1)
                image = image.strip()
                caption = " ".join(caption.replace("\u00a0", " ").split())
                if not image or not caption:
                    issue_type = "empty_image_or_caption"
                    message = f"{path}:{line_no}: empty image or caption"
                elif (
                    "\\" in image
                    or re.match(r"^[A-Za-z]:", image)
                    or PurePosixPath(image).name != image
                    or not image.lower().endswith(".jpg")
                ):
                    issue_type = "invalid_image_filename"
                    message = f"{path}:{line_no}: invalid image filename {image!r}"
                elif image in seen:
                    issue_type = "duplicate_image"
                    message = f"{path}:{line_no}: duplicate image {image!r}"
                else:
                    seen.add(image)
                    rows.append(CaptionRow(image=image, caption=caption, line_number=line_no))
                    continue
            _record_parse_issue(issues, path, line_no, line, issue_type, message)
            if not allow_malformed:
                raise ValueError(message)
            print(f"WARNING: {message}")
    return rows


def _nonempty_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as stream:
        return sum(1 for line in stream if line.strip())


def load_splits(
    dataset_dir: Path,
    selected: Sequence[str] = ("test",),
    *,
    verify_annotation_hashes: bool = True,
) -> Dict[str, List[CaptionRow]]:
    unknown = set(selected) - set(EXPECTED_SPLIT_COUNTS)
    if unknown:
        raise ValueError(f"Unknown DeepSDO split(s): {sorted(unknown)}")
    splits: Dict[str, List[CaptionRow]] = {}
    for split in EXPECTED_SPLIT_COUNTS:
        path = dataset_dir / f"desc_{split}.txt"
        if verify_annotation_hashes:
            digest = sha256_file(path)
            if digest != ANNOTATION_SHA256[split]:
                raise ValueError(
                    f"DeepSDO {split} annotations SHA-256 is {digest}, "
                    f"expected {ANNOTATION_SHA256[split]}"
                )
        splits[split] = parse_caption_file(path, allow_malformed=split not in selected)

    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        path = dataset_dir / f"desc_{split}.txt"
        raw_count = _nonempty_line_count(path)
        if raw_count != expected:
            raise ValueError(f"DeepSDO {split} split has {raw_count} raw rows; expected {expected}")
        valid_count = len(splits[split])
        expected_valid = EXPECTED_VALID_SPLIT_COUNTS[split]
        if valid_count != expected_valid:
            raise ValueError(
                f"DeepSDO {split} split has {valid_count} valid rows; expected {expected_valid}"
            )
        if split in selected and valid_count != expected:
            raise ValueError(
                f"DeepSDO {split} cannot be converted: {valid_count}/{expected} rows are valid"
            )

    owners: Dict[str, str] = {}
    image_dir = dataset_dir / "desc_images"
    for split, rows in splits.items():
        for row in rows:
            if row.image in owners:
                raise ValueError(
                    f"DeepSDO image {row.image!r} occurs in both {owners[row.image]} and {split}"
                )
            owners[row.image] = split
            if not (image_dir / row.image).is_file():
                message = f"Missing DeepSDO image for {split} annotation: {image_dir / row.image}"
                if split in selected:
                    raise FileNotFoundError(message)
                print(f"WARNING: {message}")
    return splits


def parse_image_metadata(filename: str) -> ImageMetadata:
    match = _IMAGE_RE.fullmatch(filename)
    if not match:
        raise ValueError(f"Unrecognized DeepSDO image filename: {filename!r}")
    try:
        timestamp = datetime.strptime(
            match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp in DeepSDO image filename: {filename!r}") from exc

    if match.group("aia"):
        channel = match.group("aia_channel")
        wavelength = int(channel)
        collapsed = "aia_304" if wavelength == 304 else "aia_other_euv"
        instrument = "AIA"
    else:
        channel = match.group("hmi_channel")
        wavelength = None
        collapsed = "hmi_continuum"
        instrument = "HMI"
    return ImageMetadata(
        timestamp_utc=timestamp.isoformat().replace("+00:00", "Z"),
        instrument=instrument,
        channel=channel,
        wavelength_angstrom=wavelength,
        collapsed_modality=collapsed,
    )


def topic_stratum_for_test_index(index: int) -> TopicStratum:
    if not 1 <= index <= EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError(f"DeepSDO test index must be in 1..102, got {index}")
    for topic in TEST_TOPIC_STRATA:
        if topic.start_index <= index <= topic.end_index:
            return topic
    raise AssertionError(f"No frozen DeepSDO topic stratum for index {index}")


def _find_missing_image_candidates(image: str, available: Sequence[str]) -> List[dict]:
    candidates: dict[str, str] = {}
    canonical = re.sub(r" \(\d+\)(?=\.jpg$)", "", image)
    for candidate in available:
        if re.sub(r" \(\d+\)(?=\.jpg$)", "", candidate) == canonical:
            candidates[candidate] = "duplicate-suffix variant of annotated filename"
    timestamp_prefix = image[:15]
    for candidate in available:
        if candidate.startswith(timestamp_prefix) and candidate != image:
            candidates.setdefault(candidate, "same observation timestamp")
    return [
        {"image": candidate, "evidence": candidates[candidate]}
        for candidate in sorted(candidates)
    ]


def audit_train_annotations(dataset_dir: Path, *, enforce_known: bool = False) -> dict:
    annotation_path = dataset_dir / "desc_train.txt"
    parser_issues: List[dict] = []
    rows = parse_caption_file(annotation_path, allow_malformed=True, issues=parser_issues)
    image_dir = dataset_dir / "desc_images"
    available = sorted(path.name for path in image_dir.iterdir() if path.is_file())
    available_set = set(available)

    anomalies: List[dict] = []
    for issue in parser_issues:
        parse_issue = issue["issue"]
        anomalies.append(
            {
                "issue": "malformed_annotation",
                "parse_issue": parse_issue,
                "line_number": issue["line_number"],
                "raw_line_sha256": issue["raw_line_sha256"],
                "annotated_image": None,
                "candidate_images": [],
                "repair_applied": False,
            }
        )
    for row in rows:
        if row.image not in available_set:
            anomalies.append(
                {
                    "issue": "missing_annotated_image",
                    "line_number": row.line_number,
                    "raw_line_sha256": sha256_text(f"{row.image}\t{row.caption}"),
                    "annotated_image": row.image,
                    "candidate_images": _find_missing_image_candidates(row.image, available),
                    "repair_applied": False,
                }
            )
    anomalies.sort(key=lambda item: item["line_number"])
    signatures = {
        (item["issue"], item["line_number"], item["annotated_image"]) for item in anomalies
    }
    if enforce_known and signatures != EXPECTED_TRAIN_ANOMALIES:
        raise ValueError(
            "DeepSDO training anomalies differ from the frozen release: "
            f"observed={sorted(signatures, key=str)!r}"
        )
    return {
        "annotation_file": "desc_train.txt",
        "valid_rows": len(rows),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "policy": "Report upstream anomalies without guessing or applying repairs.",
    }


def _metadata_datetime(row: CaptionRow) -> datetime:
    timestamp = parse_image_metadata(row.image).timestamp_utc
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def build_caption_overlap_audit(
    train_rows: Sequence[CaptionRow], test_rows: Sequence[CaptionRow]
) -> dict:
    train_by_caption: dict[str, List[CaptionRow]] = defaultdict(list)
    for row in train_rows:
        train_by_caption[normalize_caption(row.caption)].append(row)

    evidence: List[dict] = []
    for index, test_row in enumerate(test_rows, 1):
        normalized = normalize_caption(test_row.caption)
        matches = train_by_caption.get(normalized, [])
        test_time = _metadata_datetime(test_row)
        timed_matches = []
        for train_row in matches:
            train_time = _metadata_datetime(train_row)
            delta = int((train_time - test_time).total_seconds())
            timed_matches.append((abs(delta), train_row.image, delta))
        timed_matches.sort()
        nearest = timed_matches[0] if timed_matches else None
        evidence.append(
            {
                "test_id": f"deepsdo_test_{index:04d}",
                "test_image": test_row.image,
                "reference_sha256": sha256_text(test_row.caption),
                "normalized_reference_sha256": sha256_text(normalized),
                "normalized_caption_in_train": bool(matches),
                "matching_train_rows": len(matches),
                "nearest_train_image": nearest[1] if nearest else None,
                "nearest_timestamp_gap_seconds": nearest[0] if nearest else None,
                "nearest_timestamp_delta_seconds": nearest[2] if nearest else None,
            }
        )
    overlap_count = sum(item["normalized_caption_in_train"] for item in evidence)
    return {
        "normalization": CAPTION_NORMALIZATION,
        "train_valid_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_rows_with_normalized_caption_in_train": overlap_count,
        "overlap_fraction": overlap_count / len(test_rows) if test_rows else 0.0,
        "evidence": evidence,
        "interpretation": (
            "Exact normalized reference-caption reuse across official splits; nearest timestamp "
            "gaps are evidence of temporal proximity, not an image-duplicate assertion."
        ),
    }


def build_dataset_audit(
    dataset_dir: Path,
    splits: Mapping[str, Sequence[CaptionRow]],
    *,
    enforce_expected: bool = False,
) -> dict:
    train_audit = audit_train_annotations(dataset_dir, enforce_known=enforce_expected)
    overlap = build_caption_overlap_audit(splits["train"], splits["test"])
    if (
        enforce_expected
        and overlap["test_rows_with_normalized_caption_in_train"]
        != EXPECTED_NORMALIZED_TEST_TRAIN_OVERLAP
    ):
        raise ValueError(
            "DeepSDO normalized caption overlap differs from the frozen release: "
            f"{overlap['test_rows_with_normalized_caption_in_train']}/"
            f"{overlap['test_rows']}"
        )
    return {
        "dataset": "DeepSDO Description",
        "archive_sha256": ARCHIVE_SHA256,
        "train_annotation_anomalies": train_audit,
        "normalized_caption_train_test_overlap": overlap,
    }


def build_llava_records(
    rows: Iterable[CaptionRow],
    split: str,
    *,
    strict_test_count: bool = True,
) -> List[dict]:
    materialized = list(rows)
    if split == "test" and strict_test_count and len(materialized) != EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError(
            f"Frozen DeepSDO test mapping requires 102 rows, got {len(materialized)}"
        )
    records = []
    for index, row in enumerate(materialized, 1):
        metadata = parse_image_metadata(row.image)
        record = {
            "id": f"deepsdo_{split}_{index:04d}",
            "image": row.image,
            "image_id": Path(row.image).stem,
            "source_annotation_index": index,
            "record_type": "caption",
            "dataset": "DeepSDO Description",
            "split": split,
            "timestamp_utc": metadata.timestamp_utc,
            "instrument": metadata.instrument,
            "channel": metadata.channel,
            "wavelength_angstrom": metadata.wavelength_angstrom,
            "collapsed_modality": metadata.collapsed_modality,
            "reference_sha256": sha256_text(row.caption),
            "normalized_reference_sha256": sha256_text(normalize_caption(row.caption)),
            "prompt_sha256": sha256_text(CAPTION_PROMPT),
            "conversations": [
                {"from": "human", "value": f"<image>\n{CAPTION_PROMPT}"},
                {"from": "gpt", "value": row.caption},
            ],
        }
        if split == "test" and index <= EXPECTED_SPLIT_COUNTS["test"]:
            topic = topic_stratum_for_test_index(index)
            record.update(
                {
                    "topic_stratum": topic.key,
                    "topic_stratum_label": topic.label,
                    "topic_stratum_provenance": TOPIC_MAPPING_PROVENANCE,
                }
            )
        records.append(record)

    if split == "test" and strict_test_count:
        topic_counts = Counter(record["topic_stratum"] for record in records)
        channel_counts = Counter(f"{r['instrument']}/{r['channel']}" for r in records)
        modality_counts = Counter(record["collapsed_modality"] for record in records)
        if dict(topic_counts) != EXPECTED_TEST_TOPIC_COUNTS:
            raise ValueError(f"Unexpected DeepSDO test topic counts: {dict(topic_counts)!r}")
        if dict(channel_counts) != EXPECTED_TEST_CHANNEL_COUNTS:
            raise ValueError(f"Unexpected DeepSDO test channel counts: {dict(channel_counts)!r}")
        if dict(modality_counts) != EXPECTED_TEST_MODALITY_COUNTS:
            raise ValueError(f"Unexpected DeepSDO test modality counts: {dict(modality_counts)!r}")
    return records


def _split_manifest(
    records: Sequence[dict], output_path: Path, annotation_path: Path | None
) -> dict:
    topic_counts = Counter(r.get("topic_stratum") for r in records if r.get("topic_stratum"))
    channel_counts = Counter(f"{r['instrument']}/{r['channel']}" for r in records)
    modality_counts = Counter(r["collapsed_modality"] for r in records)
    reference_hashes = [r["reference_sha256"] for r in records]
    topic_mapping = [
        {"id": r["id"], "image": r["image"], "topic_stratum": r.get("topic_stratum")}
        for r in records
    ]
    return {
        "records": len(records),
        "records_file": output_path.name,
        "records_file_sha256": sha256_file(output_path),
        "annotation_file_sha256": sha256_file(annotation_path) if annotation_path else None,
        "reference_hash_algorithm": (
            "SHA-256 of UTF-8 reference text after parser whitespace normalization"
        ),
        "ordered_reference_set_sha256": sha256_text("\n".join(reference_hashes)),
        "ordered_topic_mapping_sha256": _canonical_json_sha256(topic_mapping),
        "topic_counts": dict(sorted(topic_counts.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "collapsed_modality_counts": dict(sorted(modality_counts.items())),
    }


def write_outputs(
    splits: Mapping[str, Sequence[CaptionRow]],
    selected: Sequence[str],
    output_dir: Path,
    *,
    dataset_dir: Path | None = None,
    audit: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict] = {}
    for split in selected:
        records = build_llava_records(splits[split], split)
        path = output_dir / f"{split}.json"
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        annotation_path = dataset_dir / f"desc_{split}.txt" if dataset_dir else None
        generated[split] = _split_manifest(records, path, annotation_path)
        print(f"Wrote {len(records)} {split} records to {path}")

    audit_file = None
    audit_sha256 = None
    if audit is not None:
        audit_path = output_dir / "deepsdo_audit.json"
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        audit_file = audit_path.name
        audit_sha256 = sha256_file(audit_path)

    overlap_summary = None
    anomaly_summary = None
    if audit:
        overlap = audit["normalized_caption_train_test_overlap"]
        overlap_summary = {
            key: overlap[key]
            for key in (
                "normalization",
                "train_valid_rows",
                "test_rows",
                "test_rows_with_normalized_caption_in_train",
                "overlap_fraction",
            )
        }
        train_audit = audit["train_annotation_anomalies"]
        anomaly_summary = {
            "anomaly_count": train_audit["anomaly_count"],
            "repair_applied": False,
            "audit_file": audit_file,
        }

    manifest = {
        "schema_version": 2,
        "dataset": "DeepSDO Description",
        "source_url": DATASET_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "prompt": CAPTION_PROMPT,
        "prompt_sha256": sha256_text(CAPTION_PROMPT),
        "official_split_counts": EXPECTED_SPLIT_COUNTS,
        "official_valid_row_counts": EXPECTED_VALID_SPLIT_COUNTS,
        "generated_splits": list(selected),
        "generated_split_manifests": generated,
        "topic_strata": [asdict(topic) | {"count": topic.count} for topic in TEST_TOPIC_STRATA],
        "topic_mapping_provenance": TOPIC_MAPPING_PROVENANCE,
        "normalized_caption_overlap_summary": overlap_summary,
        "train_anomaly_summary": anomaly_summary,
        "audit_file": audit_file,
        "audit_file_sha256": audit_sha256,
        "evaluation_protocol": {
            "retrieval": False,
            "few_shot": False,
            "quantization": False,
            "training_use": False,
            "reported_split": "test",
        },
        "evaluation_policy": (
            "Use the official test split only for zero-shot external caption evaluation; "
            "topic strata are descriptive analysis groups, not classification labels."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("datasets/deepsdo/raw/kasi_deepsdo_desc_dataset.tar.gz"),
    )
    parser.add_argument(
        "--download", action="store_true", help="Download from the official KASI URL."
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=DEFAULT_DOWNLOAD_RETRIES,
        help="Bounded HTTP attempts.",
    )
    parser.add_argument("--extract-dir", type=Path, default=Path("datasets/deepsdo/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/deepsdo/llava"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(EXPECTED_SPLIT_COUNTS),
        default=["test"],
        help="Splits to convert. The default intentionally converts only the official test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        download_archive(DATASET_URL, args.archive, max_retries=args.download_retries)
    digest = validate_archive(args.archive)
    extract_archive(args.archive, args.extract_dir)
    splits = load_splits(args.extract_dir, selected=args.splits)
    audit = build_dataset_audit(args.extract_dir, splits, enforce_expected=True)
    write_outputs(
        splits,
        args.splits,
        args.output_dir,
        dataset_dir=args.extract_dir,
        audit=audit,
    )
    overlap = audit["normalized_caption_train_test_overlap"]
    print(
        "DeepSDO verified: "
        f"sha256={digest}; "
        + ", ".join(f"{name}_valid={len(rows)}" for name, rows in splits.items())
        + "; normalized_test_caption_overlap="
        f"{overlap['test_rows_with_normalized_caption_in_train']}/{overlap['test_rows']}"
    )


if __name__ == "__main__":
    main()
