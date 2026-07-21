from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from eval.paper.artifacts import sha256_file
from eval.paper.internal import (
    InternalDataError,
    audit_image_overlap,
    canonical_records,
    extract_frozen_test,
    phash64,
    validate_frozen_counts,
)


def row(row_id: str, image: str, prompt: str, answer: str):
    return {
        "id": row_id,
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


class InternalPaperManifestTests(unittest.TestCase):
    def test_extract_requires_exact_frozen_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = json.dumps([row("x", "x.jpg", "What?", "That")]).encode()
            archive = root / "artifact.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("test.json", payload)
            expected = __import__("hashlib").sha256(payload).hexdigest()
            target = extract_frozen_test(archive, "test.json", root / "test.json", expected)
            self.assertEqual(sha256_file(target), expected)
            with self.assertRaisesRegex(InternalDataError, "SHA-256"):
                extract_frozen_test(archive, "test.json", root / "bad.json", "0" * 64)

    def test_canonical_records_preserve_task_and_image_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            path.write_text(
                json.dumps(
                    [
                        row("astro_1", "one.jpg", "Describe it", "Caption"),
                        row("astro_1_qa0", "one.jpg", "What?", "Answer"),
                    ]
                ),
                encoding="utf-8",
            )
            records = canonical_records(path)
            self.assertEqual([item["record_type"] for item in records], ["caption", "qa"])
            self.assertEqual(records[0]["image_id"], records[1]["image_id"])
            validate_frozen_counts(
                records,
                {
                    "expected_records": 2,
                    "expected_images": 1,
                    "expected_caption_records": 1,
                    "expected_qa_records": 1,
                },
            )

    def test_phash_and_overlap_detect_identical_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (32, 32), (10, 20, 30)).save(root / "train.jpg")
            Image.new("RGB", (32, 32), (10, 20, 30)).save(root / "test.jpg")
            self.assertEqual(phash64(root / "train.jpg"), phash64(root / "test.jpg"))
            report = audit_image_overlap(
                [{"image": "train.jpg"}],
                [{"image": "test.jpg"}],
                root,
                root / "audit",
            )
            self.assertGreaterEqual(report["exact_pixel_pairs"], 1)
            self.assertGreaterEqual(report["phash_likely_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
