from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from eval.paper.assets import (
    AssetError,
    AssetRegistry,
    safe_extract_zip,
    verify_snapshot_revision,
    zip_member_sha256,
)
from eval.paper.protocol import PaperProtocol


ROOT = Path(__file__).resolve().parents[1]


class PaperAssetTests(unittest.TestCase):
    def test_snapshot_revision_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots" / ("a" * 40)
            path.mkdir(parents=True)
            self.assertEqual(verify_snapshot_revision(path, "a" * 40), "a" * 40)
            with self.assertRaisesRegex(AssetError, "expected"):
                verify_snapshot_revision(path, "b" * 40)

    def test_safe_zip_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "bad")
            with self.assertRaisesRegex(AssetError, "Unsafe"):
                safe_extract_zip(archive, Path(tmp) / "out")

    def test_safe_zip_extracts_once_with_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ok.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("checkpoint/connector.safetensors", "weights")
            out = safe_extract_zip(archive, Path(tmp) / "out")
            self.assertTrue((out / "checkpoint" / "connector.safetensors").is_file())
            self.assertEqual(safe_extract_zip(archive, out), out)

    def test_stage1_connector_can_be_anchored_to_pinned_archive_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "checkpoint.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("checkpoint/connector.safetensors", b"locked weights")
            out = safe_extract_zip(archive, Path(tmp) / "out")
            connector = out / "checkpoint" / "connector.safetensors"
            expected = zip_member_sha256(archive, "connector.safetensors")
            self.assertEqual(expected, hashlib.sha256(b"locked weights").hexdigest())
            connector.write_bytes(b"corrupt weights")
            self.assertNotEqual(expected, hashlib.sha256(connector.read_bytes()).hexdigest())

    def test_registry_rejects_stale_protocol_before_loading_weights(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")
        registry = AssetRegistry(Path("."), {}, "0" * 64, {})
        with self.assertRaisesRegex(AssetError, "protocol hash"):
            registry.validate_model("astraq_stage1", protocol)


if __name__ == "__main__":
    unittest.main()
