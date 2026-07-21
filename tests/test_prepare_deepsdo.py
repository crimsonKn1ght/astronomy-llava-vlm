from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
import urllib.error
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.prepare_deepsdo as deepsdo
from scripts.prepare_deepsdo import (
    CaptionRow,
    _safe_members,
    audit_train_annotations,
    build_caption_overlap_audit,
    build_llava_records,
    download_archive,
    parse_caption_file,
    parse_image_metadata,
    topic_stratum_for_test_index,
    write_outputs,
)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int = 200, headers: dict | None = None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def getcode(self) -> int:
        return self.status


def _patched_archive(payload: bytes):
    return mock.patch.multiple(
        deepsdo,
        ARCHIVE_BYTES=len(payload),
        ARCHIVE_SHA256=hashlib.sha256(payload).hexdigest(),
    )


def _official_shape_rows() -> list[CaptionRow]:
    channels = (
        ["HMI_Ic"] * 8
        + ["AIA_304"] * 46
        + ["AIA_171"] * 29
        + ["AIA_193"] * 13
        + ["AIA_211"] * 4
        + ["AIA_94"]
        + ["AIA_131"]
    )
    start = datetime(2020, 1, 1)
    rows = []
    for index, channel in enumerate(channels):
        stamp = (start + timedelta(hours=index)).strftime("%Y%m%d_%H%M%S")
        rows.append(CaptionRow(f"{stamp}_SDO_{channel}_512.jpg", f"Caption {index + 1}."))
    return rows


class PrepareDeepSdoTests(unittest.TestCase):
    def test_parse_caption_file_normalizes_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "desc_test.txt"
            path.write_text("solar.jpg\t A\u00a0bright   active region. \n", encoding="utf-8")
            rows = parse_caption_file(path)
            self.assertEqual(rows, [CaptionRow("solar.jpg", "A bright active region.")])
            self.assertEqual(rows[0].line_number, 1)

    def test_build_records_preserves_metadata_topics_and_reference_hashes(self) -> None:
        row = CaptionRow("20110730_030000_SDO_HMI_Ic_512.jpg", "A sunspot.")
        record = build_llava_records([row], "test", strict_test_count=False)[0]
        self.assertEqual(record["id"], "deepsdo_test_0001")
        self.assertEqual(record["topic_stratum"], "sunspots")
        self.assertEqual(record["instrument"], "HMI")
        self.assertEqual(record["channel"], "Ic")
        self.assertEqual(record["collapsed_modality"], "hmi_continuum")
        self.assertEqual(record["reference_sha256"], hashlib.sha256(b"A sunspot.").hexdigest())
        self.assertIn("<image>", record["conversations"][0]["value"])

    def test_malformed_rows_are_skipped_only_explicitly_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "desc_train.txt"
            path.write_text("caption in wrong column\t\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty image or caption"):
                parse_caption_file(path)
            issues: list[dict] = []
            self.assertEqual(parse_caption_file(path, allow_malformed=True, issues=issues), [])
            self.assertEqual(issues[0]["issue"], "empty_image_or_caption")
            self.assertEqual(issues[0]["line_number"], 1)

    def test_range_resume_uses_saved_offset_and_verifies_archive(self) -> None:
        payload = b"abcdefghij"
        with tempfile.TemporaryDirectory() as tmp, _patched_archive(payload):
            destination = Path(tmp) / "dataset.tar.gz"
            partial = destination.with_suffix(destination.suffix + ".part")
            partial.write_bytes(payload[:3])
            response = FakeResponse(
                payload[3:],
                status=206,
                headers={"Content-Range": "bytes 3-9/10"},
            )
            with mock.patch(
                "scripts.prepare_deepsdo.urllib.request.urlopen", return_value=response
            ) as mocked_urlopen:
                download_archive("https://example.invalid/data", destination, max_retries=1)
            request = mocked_urlopen.call_args.args[0]
            self.assertEqual(request.get_header("Range"), "bytes=3-")
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(partial.exists())

    def test_server_ignoring_range_keeps_and_compares_saved_prefix(self) -> None:
        payload = b"abcdefghij"
        with tempfile.TemporaryDirectory() as tmp, _patched_archive(payload):
            destination = Path(tmp) / "dataset.tar.gz"
            partial = destination.with_suffix(destination.suffix + ".part")
            partial.write_bytes(payload[:4])
            with mock.patch(
                "scripts.prepare_deepsdo.urllib.request.urlopen",
                return_value=FakeResponse(payload, status=200),
            ):
                download_archive("https://example.invalid/data", destination, max_retries=1)
            self.assertEqual(destination.read_bytes(), payload)

    def test_failed_download_retains_valid_partial_and_bounds_retries(self) -> None:
        payload = b"abcdefghij"
        with tempfile.TemporaryDirectory() as tmp, _patched_archive(payload):
            destination = Path(tmp) / "dataset.tar.gz"
            partial = destination.with_suffix(destination.suffix + ".part")
            partial.write_bytes(payload[:4])
            with mock.patch(
                "scripts.prepare_deepsdo.urllib.request.urlopen",
                side_effect=urllib.error.URLError("offline"),
            ) as mocked_urlopen:
                delays: list[float] = []
                with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
                    download_archive(
                        "https://example.invalid/data",
                        destination,
                        max_retries=2,
                        sleep_fn=delays.append,
                    )
            self.assertEqual(mocked_urlopen.call_count, 2)
            self.assertEqual(delays, [1.0])
            self.assertEqual(partial.read_bytes(), payload[:4])

    def test_safe_members_rejects_parent_traversal_backslashes_and_links(self) -> None:
        cases = ["../escape.txt", "..\\escape.txt"]
        for name in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                archive_path = Path(tmp) / "unsafe.tar"
                with tarfile.open(archive_path, "w") as archive:
                    info = tarfile.TarInfo(name)
                    info.size = 1
                    archive.addfile(info, io.BytesIO(b"x"))
                with tarfile.open(archive_path, "r") as archive:
                    with self.assertRaisesRegex(ValueError, "Unsafe path"):
                        _safe_members(archive)

        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "link.tar"
            with tarfile.open(archive_path, "w") as archive:
                link = tarfile.TarInfo("link")
                link.type = tarfile.SYMTYPE
                link.linkname = "outside"
                archive.addfile(link)
            with tarfile.open(archive_path, "r") as archive:
                with self.assertRaisesRegex(ValueError, "Links are not allowed"):
                    _safe_members(archive)

    def test_frozen_topic_boundaries_and_counts(self) -> None:
        self.assertEqual(topic_stratum_for_test_index(1).key, "sunspots")
        self.assertEqual(topic_stratum_for_test_index(8).key, "sunspots")
        self.assertEqual(topic_stratum_for_test_index(9).key, "flares")
        self.assertEqual(topic_stratum_for_test_index(89).key, "active_regions")
        self.assertEqual(topic_stratum_for_test_index(90).key, "eclipses_transits")
        self.assertEqual(topic_stratum_for_test_index(102).key, "eclipses_transits")
        counts = Counter(topic_stratum_for_test_index(i).key for i in range(1, 103))
        self.assertEqual(dict(counts), deepsdo.EXPECTED_TEST_TOPIC_COUNTS)
        with self.assertRaises(ValueError):
            topic_stratum_for_test_index(103)

    def test_filename_metadata_forms_and_invalid_names(self) -> None:
        hmi = parse_image_metadata("20110730_030000_SDO_HMI_Ic_512.jpg")
        self.assertEqual(hmi.timestamp_utc, "2011-07-30T03:00:00Z")
        self.assertEqual(hmi.wavelength_angstrom, None)
        self.assertEqual(hmi.collapsed_modality, "hmi_continuum")

        aia304 = parse_image_metadata("20110922_110408_SDO_AIA_304_512.jpg")
        self.assertEqual(aia304.instrument, "AIA")
        self.assertEqual(aia304.channel, "304")
        self.assertEqual(aia304.wavelength_angstrom, 304)
        self.assertEqual(aia304.collapsed_modality, "aia_304")

        aia171 = parse_image_metadata("20101111_231300_SDO_AIA_171_512.jpg")
        self.assertEqual(aia171.collapsed_modality, "aia_other_euv")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            parse_image_metadata("20101111_231300_SDO_AIA_999_512.jpg")
        with self.assertRaisesRegex(ValueError, "Invalid timestamp"):
            parse_image_metadata("20101340_251300_SDO_AIA_171_512.jpg")

    def test_train_anomaly_audit_records_candidates_without_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "desc_images"
            images.mkdir()
            (images / "20200101_000000_SDO_AIA_304_512 (1).jpg").write_bytes(b"image")
            (images / "20200102_000000_SDO_AIA_193_512.jpg").write_bytes(b"image")
            (root / "desc_train.txt").write_text(
                "20200101_000000_SDO_AIA_304_512.jpg\tFirst caption.\n"
                "Malformed caption in filename field\t\n"
                "20200102_000000_SDO_AIA_171_512.jpg\tSecond caption.\n",
                encoding="utf-8",
            )
            audit = audit_train_annotations(root)
            self.assertEqual(audit["anomaly_count"], 3)
            self.assertTrue(all(not item["repair_applied"] for item in audit["anomalies"]))
            self.assertEqual(
                audit["anomalies"][0]["candidate_images"][0]["evidence"],
                "duplicate-suffix variant of annotated filename",
            )
            self.assertEqual(
                audit["anomalies"][2]["candidate_images"][0]["evidence"],
                "same observation timestamp",
            )

    def test_caption_overlap_reports_nearest_timestamp_gap(self) -> None:
        train = [
            CaptionRow("20200101_000000_SDO_AIA_171_512.jpg", "Same caption."),
            CaptionRow("20200103_000000_SDO_AIA_171_512.jpg", "Same caption."),
        ]
        test = [
            CaptionRow("20200102_120000_SDO_AIA_171_512.jpg", " SAME\u00a0caption. "),
            CaptionRow("20200104_000000_SDO_AIA_171_512.jpg", "Different."),
        ]
        audit = build_caption_overlap_audit(train, test)
        self.assertEqual(audit["test_rows_with_normalized_caption_in_train"], 1)
        evidence = audit["evidence"][0]
        self.assertEqual(evidence["matching_train_rows"], 2)
        self.assertEqual(evidence["nearest_train_image"], train[1].image)
        self.assertEqual(evidence["nearest_timestamp_gap_seconds"], 12 * 60 * 60)
        self.assertEqual(evidence["nearest_timestamp_delta_seconds"], 12 * 60 * 60)
        self.assertIsNone(audit["evidence"][1]["nearest_train_image"])

    def test_manifest_freezes_counts_hashes_protocol_and_topic_mapping(self) -> None:
        rows = _official_shape_rows()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_outputs({"test": rows}, ["test"], output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            split = manifest["generated_split_manifests"]["test"]
            self.assertEqual(split["records"], 102)
            self.assertEqual(split["topic_counts"], deepsdo.EXPECTED_TEST_TOPIC_COUNTS)
            self.assertEqual(split["channel_counts"], deepsdo.EXPECTED_TEST_CHANNEL_COUNTS)
            self.assertEqual(
                split["collapsed_modality_counts"], deepsdo.EXPECTED_TEST_MODALITY_COUNTS
            )
            self.assertEqual(len(split["ordered_reference_set_sha256"]), 64)
            self.assertEqual(len(split["ordered_topic_mapping_sha256"]), 64)
            self.assertFalse(manifest["evaluation_protocol"]["retrieval"])
            self.assertFalse(manifest["evaluation_protocol"]["training_use"])


if __name__ == "__main__":
    unittest.main()
