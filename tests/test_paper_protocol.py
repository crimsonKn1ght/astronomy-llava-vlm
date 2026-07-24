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

    def test_dedicated_astrovlbench_protocol_pins_isolated_release_contract(self) -> None:
        astro = PaperProtocol.load(ROOT / "configs" / "paper_eval_astrovlbench_v1.yaml")
        dataset = astro.data["datasets"]["astrovlbench"]
        self.assertEqual(
            dataset["locked_revision"],
            "d1708958d4d1dda45c078eb2f4d6db3e6fa96286",
        )
        self.assertEqual(dataset["expected_records"], 1995)
        self.assertEqual(
            dataset["expected_component_records"],
            {"task1": 557, "task2.first": 605, "task2.nvss": 833},
        )
        self.assertEqual(
            dataset["expected_snapshot_inventory_sha256"],
            "c282a79c42854dac4714c5e7311b468eb1a4a591f988766c4bc74830b6b47d82",
        )
        self.assertEqual(
            astro.data["runtime"]["output_root"],
            "eval_runs/paper_eval_astrovlbench_v1",
        )
        self.assertEqual(astro.data["generation"]["astrovlbench"]["max_new_tokens"], 256)
        self.assertTrue(
            astro.data["generation"]["astrovlbench"]["require_natural_termination"]
        )

    def test_completed_deepsdo_v4_suite_fingerprints_are_unchanged(self) -> None:
        v4 = PaperProtocol.load(ROOT / "configs" / "paper_eval_v4.yaml")
        self.assertEqual(
            v4.suite_fingerprint("deepsdo", "original_1024")[:16],
            "f7ccb1b883880d96",
        )
        self.assertEqual(
            v4.suite_fingerprint("deepsdo", "concise_256")[:16],
            "e5390c5615f0fca1",
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

    def test_v3_conditions_and_fifth_baseline_are_frozen(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v3.yaml")
        self.assertEqual(
            protocol.condition_ids("deepsdo"),
            ("original_512", "concise_256"),
        )
        self.assertIn("internvl3_5_4b", protocol.selected_models("deepsdo"))
        self.assertNotIn("internvl3_5_4b", protocol.selected_models("internal"))
        self.assertNotEqual(
            protocol.model_fingerprint("deepsdo", "astraq_stage2", "original_512"),
            protocol.model_fingerprint("deepsdo", "astraq_stage2", "concise_256"),
        )
        self.assertEqual(
            protocol.data["models"]["internvl3_5_4b"]["revision"],
            "dbf19f04a6a6f2fb15821cc7ae738430a3580cf5",
        )

    def test_v3_internvl_runtime_dependency_is_locked(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v3.yaml")
        packages = protocol.data["environments"]["modern_generation"]["packages"]
        self.assertEqual(packages["einops"], "0.6.1")
        requirements = (ROOT / "requirements-paper-modern.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("einops==0.6.1", requirements.splitlines())

    def test_v4_changes_only_the_original_prompt_ceiling_and_storage(self) -> None:
        v3 = PaperProtocol.load(ROOT / "configs" / "paper_eval_v3.yaml")
        v4 = PaperProtocol.load(ROOT / "configs" / "paper_eval_v4.yaml")
        self.assertEqual(v4.data["schema_version"], 4)
        self.assertEqual(v4.study_id, "astraq-vl-paper-eval-v4")
        self.assertEqual(
            v4.condition_ids("deepsdo"),
            ("original_1024", "concise_256"),
        )
        self.assertEqual(
            v4.condition_config("deepsdo", "original_1024"),
            {
                "role": "primary_continuity",
                "prompt": "Describe this solar image.",
                "max_new_tokens": 1024,
                "require_natural_termination": True,
            },
        )
        self.assertEqual(
            v4.condition_config("deepsdo", "concise_256"),
            v3.condition_config("deepsdo", "concise_256"),
        )
        self.assertEqual(
            v4.data["factuality_audit"]["condition"],
            "original_1024",
        )
        self.assertEqual(
            (
                v4.data["runtime"]["output_root"],
                v4.data["runtime"]["data_root"],
                v4.data["runtime"]["asset_root"],
            ),
            (
                "eval_runs/paper_eval_v4",
                "datasets/paper_eval_v4",
                "checkpoints/paper_eval_v4",
            ),
        )
        normalized_v4 = copy.deepcopy(v4.data)
        normalized_v4["schema_version"] = 3
        normalized_v4["study"]["id"] = v3.study_id
        normalized_v4["runtime"].update(
            {
                "output_root": v3.data["runtime"]["output_root"],
                "data_root": v3.data["runtime"]["data_root"],
                "asset_root": v3.data["runtime"]["asset_root"],
            }
        )
        original = normalized_v4["generation"]["deepsdo"]["conditions"].pop(
            "original_1024"
        )
        original["max_new_tokens"] = 512
        normalized_v4["generation"]["deepsdo"]["conditions"] = {
            "original_512": original,
            "concise_256": normalized_v4["generation"]["deepsdo"]["conditions"][
                "concise_256"
            ],
        }
        normalized_v4["factuality_audit"]["condition"] = "original_512"
        self.assertEqual(normalized_v4, v3.data)

    def test_v4_rejects_a_silent_ceiling_change(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v4.yaml")
        broken = copy.deepcopy(protocol.data)
        broken["generation"]["deepsdo"]["conditions"]["original_1024"][
            "max_new_tokens"
        ] = 2048
        with self.assertRaisesRegex(ProtocolError, "must be 1024"):
            validate_protocol(broken)


if __name__ == "__main__":
    unittest.main()
