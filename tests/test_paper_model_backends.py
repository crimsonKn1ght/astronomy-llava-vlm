from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from eval.paper.model_backends import AstroLLaVAPaperBackend, _termination
from scripts.paper_eval_worker import (
    create_deepsdo_smoke_fixtures,
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
        self.assertEqual(_termination([4], [2, 3], 3), "stopped")

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


if __name__ == "__main__":
    unittest.main()
