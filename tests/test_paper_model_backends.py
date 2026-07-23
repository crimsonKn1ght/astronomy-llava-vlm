from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from eval.paper.model_backends import (
    AstroLLaVAPaperBackend,
    _termination,
    internvl_dynamic_preprocess,
)
from scripts.paper_eval_worker import (
    create_deepsdo_smoke_fixtures,
    smoke_record_plan,
    validate_generation_environment,
    validate_generation_hardware,
)


class PaperModelBackendContractTests(unittest.TestCase):
    def test_astrollava_injects_locked_local_vision_tower(self) -> None:
        captured = {}
        config = types.SimpleNamespace()

        class AutoConfig:
            @staticmethod
            def from_pretrained(path, **kwargs):
                captured["config_path"] = path
                captured["config_kwargs"] = kwargs
                return config

        model = types.SimpleNamespace(eval=lambda: None)

        def load_pretrained_model(*args, **kwargs):
            captured["loader_args"] = args
            captured["loader_kwargs"] = kwargs
            return object(), model, object(), 2048

        modules = {
            "torch": types.ModuleType("torch"),
            "transformers": types.ModuleType("transformers"),
            "llava": types.ModuleType("llava"),
            "llava.model": types.ModuleType("llava.model"),
            "llava.model.builder": types.ModuleType("llava.model.builder"),
            "llava.utils": types.ModuleType("llava.utils"),
        }
        modules["transformers"].AutoConfig = AutoConfig
        modules["llava.model.builder"].load_pretrained_model = load_pretrained_model
        modules["llava.utils"].disable_torch_init = lambda: None
        protocol = types.SimpleNamespace(
            data={"models": {"astrollava": {"conversation_mode": "llava_v1"}}}
        )
        registry = types.SimpleNamespace(
            model_path=lambda label: Path("/locked/astrollava"),
            shared_path=lambda key: Path("/locked/clip-336"),
        )

        with mock.patch.dict(sys.modules, modules):
            AstroLLaVAPaperBackend(protocol, "astrollava", registry, "cuda")

        self.assertTrue(captured["config_kwargs"]["local_files_only"])
        self.assertEqual(config.mm_vision_tower, str(Path("/locked/clip-336")))
        self.assertEqual(config.vision_tower, str(Path("/locked/clip-336")))
        self.assertIs(captured["loader_kwargs"]["config"], config)

    def test_eos_at_exact_cap_is_not_misreported_as_truncation(self) -> None:
        self.assertEqual(_termination([4, 5, 2], 2, 3), "eos")
        self.assertEqual(_termination([4, 5, 6], 2, 3), "max_new_tokens")
        self.assertEqual(_termination([4], [2, 3], 3), "model_stop")

    def test_internvl_square_image_uses_one_official_tile(self) -> None:
        from PIL import Image

        image = Image.new("RGB", (512, 512))
        tiles = internvl_dynamic_preprocess(image, image_size=448, max_num=12)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].size, (448, 448))

    def test_bf16_backend_rejects_cuda_without_bf16_support(self) -> None:
        observed = {
            "hardware": {
                "cuda_available": True,
                "gpu_capability": [8, 6],
                "bf16_supported": False,
            }
        }
        runtime = {
            "minimum_compute_capability": 8.0,
            "maximum_compute_capability_exclusive": 10.0,
        }
        with self.assertRaisesRegex(SystemExit, "BF16"):
            validate_generation_hardware(
                {"dtype": "bfloat16"}, observed, "cuda", runtime
            )

    def test_generation_environment_rejects_different_python_patch(self) -> None:
        expected = {"python": "3.11.15", "packages": {"torch": "2.8.0"}}
        observed = {"python": "3.11.14", "packages": {"torch": "2.8.0"}}
        with self.assertRaisesRegex(SystemExit, "Python 3.11.15"):
            validate_generation_environment(expected, observed)

    def test_deepsdo_synthetic_smoke_fixtures_are_separate_from_benchmark(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            records = create_deepsdo_smoke_fixtures(tmp, "Describe this solar image.")
            self.assertEqual(len(records), 3)
            self.assertTrue(all(row["not_benchmark_data"] for row in records))
            self.assertTrue(all(Path(row["image_path"]).is_file() for row in records))
            self.assertEqual(
                {row["split"] for row in records}, {"synthetic_smoke_only"}
            )

    def test_smoke_resume_keeps_fixed_cohort_instead_of_advancing(self) -> None:
        records = [
            {
                "id": "one",
                "topic_stratum": "a",
                "collapsed_modality": "x",
            },
            {
                "id": "two",
                "topic_stratum": "a",
                "collapsed_modality": "x",
            },
            {
                "id": "three",
                "topic_stratum": "b",
                "collapsed_modality": "y",
            },
            {
                "id": "four",
                "topic_stratum": "b",
                "collapsed_modality": "y",
            },
        ]
        selected, pending = smoke_record_plan(
            records,
            "deepsdo",
            2,
            {
                "one": {"status": "ok"},
                "unselected": {"status": "ok"},
            },
        )
        self.assertEqual([row["id"] for row in selected], ["one", "three"])
        self.assertEqual([row["id"] for row in pending], ["three"])

    def test_smoke_resume_treats_cached_token_cap_as_complete(self) -> None:
        records = [
            {
                "id": "capped",
                "topic_stratum": "a",
                "collapsed_modality": "x",
            }
        ]
        selected, pending = smoke_record_plan(
            records,
            "deepsdo",
            1,
            {"capped": {"status": "token_cap"}},
        )
        self.assertEqual([row["id"] for row in selected], ["capped"])
        self.assertEqual(pending, [])

    def test_astrovlbench_smoke_covers_all_eight_components_even_with_limit_five(self) -> None:
        components = (
            "task1",
            "task2.first",
            "task2.nvss",
            "task3",
            "task4",
            "task5.q1",
            "task5.q2",
            "task5.q3",
        )
        records = [
            {"id": f"{component}-first", "task_key": component}
            for component in components
        ]
        records.extend(
            {"id": f"{component}-second", "task_key": component}
            for component in components
        )
        selected, pending = smoke_record_plan(records, "astrovlbench", 5, {})
        self.assertEqual(len(selected), 8)
        self.assertEqual({row["task_key"] for row in selected}, set(components))
        self.assertEqual(selected, pending)


if __name__ == "__main__":
    unittest.main()
