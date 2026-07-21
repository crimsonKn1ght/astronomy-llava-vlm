from __future__ import annotations

import copy
import unittest
from pathlib import Path

from eval.paper.protocol import PaperProtocol, ProtocolError, sha256_json, validate_protocol


ROOT = Path(__file__).resolve().parents[1]


class PaperProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")

    def test_protocol_is_deterministic_and_suite_scoped(self) -> None:
        first = self.protocol.suite_fingerprint("internal")
        second = self.protocol.suite_fingerprint("internal")
        self.assertEqual(first, second)
        self.assertNotEqual(first, self.protocol.suite_fingerprint("deepsdo"))
        self.assertEqual(set(self.protocol.selected_models("internal")), {"astraq_stage1", "astraq_stage2"})

    def test_model_fingerprint_changes_only_for_selected_model(self) -> None:
        self.assertNotEqual(
            self.protocol.model_fingerprint("deepsdo", "astraq_stage2"),
            self.protocol.model_fingerprint("deepsdo", "qwen3_vl_4b"),
        )

    def test_effective_generation_fingerprint_binds_exact_records(self) -> None:
        first = self.protocol.effective_model_fingerprint(
            "deepsdo", "astraq_stage2", "1" * 64, ROOT
        )
        second = self.protocol.effective_model_fingerprint(
            "deepsdo", "astraq_stage2", "2" * 64, ROOT
        )
        self.assertNotEqual(first, second)

    def test_analysis_changes_do_not_move_generation_outputs(self) -> None:
        changed = copy.deepcopy(self.protocol.data)
        changed["statistics"]["bootstrap_replicates"] = 1234
        changed["reporting"]["png_dpi"] = 150
        modified = PaperProtocol(self.protocol.path, changed)
        self.assertEqual(
            self.protocol.model_fingerprint("internal", "astraq_stage2"),
            modified.model_fingerprint("internal", "astraq_stage2"),
        )
        self.assertNotEqual(
            self.protocol.analysis_fingerprint("internal"),
            modified.analysis_fingerprint("internal"),
        )

    def test_later_astro_lock_configuration_does_not_rehash_completed_suites(self) -> None:
        changed = copy.deepcopy(self.protocol.data)
        changed["datasets"]["astrovlbench"]["locked_revision"] = "a" * 40
        changed["statistics"]["predeclared_comparisons"]["astrovlbench"].append(
            ["astraq_stage1", "qwen3_vl_4b"]
        )
        modified = PaperProtocol(self.protocol.path, changed)
        self.assertEqual(
            self.protocol.suite_fingerprint("internal"),
            modified.suite_fingerprint("internal"),
        )
        self.assertEqual(
            self.protocol.suite_fingerprint("deepsdo"),
            modified.suite_fingerprint("deepsdo"),
        )
        self.assertNotEqual(
            self.protocol.suite_fingerprint("astrovlbench"),
            modified.suite_fingerprint("astrovlbench"),
        )

    def test_astraq_stage2_requires_lora(self) -> None:
        arch = self.protocol.astraq_architecture("astraq_stage2")
        self.assertIn("lora", arch["language_model"])
        self.assertNotIn("lora", self.protocol.astraq_architecture("astraq_stage1")["language_model"])

    def test_astrollava_secondary_vision_tower_is_revision_locked(self) -> None:
        vision = self.protocol.data["models"]["astrollava"]["vision_encoder"]
        self.assertEqual(vision["repo_id"], "openai/clip-vit-large-patch14-336")
        self.assertRegex(vision["revision"], r"^[0-9a-f]{40}$")

    def test_retrieval_cannot_be_enabled(self) -> None:
        broken = copy.deepcopy(self.protocol.data)
        broken["study"]["retrieval"] = True
        with self.assertRaisesRegex(ProtocolError, "retrieval"):
            validate_protocol(broken)

    def test_canonical_hash_ignores_mapping_order(self) -> None:
        self.assertEqual(sha256_json({"b": 2, "a": 1}), sha256_json({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
