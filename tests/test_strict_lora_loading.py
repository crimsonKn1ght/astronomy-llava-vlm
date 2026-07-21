from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from training.checkpoint import load_lora_adapter


class _Tensor:
    def __init__(self, shape):
        self.shape = shape


class StrictLoraLoadingTests(unittest.TestCase):
    def _checkpoint(self, root: Path) -> Path:
        lora = root / "lora"
        lora.mkdir()
        (lora / "adapter_model.safetensors").write_bytes(b"fixture")
        (lora / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj"],
                    "bias": "none",
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_strict_loader_rejects_silently_dropped_saved_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._checkpoint(Path(tmp))
            saved = {"lora_A.weight": _Tensor((2, 2)), "embed_tokens.weight": _Tensor((3, 2))}
            peft = types.ModuleType("peft")
            peft.set_peft_model_state_dict = lambda model, state: types.SimpleNamespace(
                unexpected_keys=[]
            )
            peft.get_peft_model_state_dict = lambda *args, **kwargs: {
                "lora_A.weight": _Tensor((2, 2))
            }
            active = types.SimpleNamespace(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules={"q_proj", "v_proj"},
                bias="none",
            )
            model = types.SimpleNamespace(peft_config={"default": active})
            with mock.patch.dict(sys.modules, {"peft": peft}), mock.patch(
                "training.checkpoint.load_file", return_value=saved
            ):
                with self.assertRaisesRegex(RuntimeError, "every saved"):
                    load_lora_adapter(model, str(checkpoint), strict=True)

    def test_strict_loader_rejects_loaded_tensor_value_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._checkpoint(Path(tmp))
            saved = {"lora_A.weight": torch.ones((2, 2))}
            peft = types.ModuleType("peft")
            peft.set_peft_model_state_dict = lambda model, state: types.SimpleNamespace(
                unexpected_keys=[]
            )
            peft.get_peft_model_state_dict = lambda *args, **kwargs: {
                "lora_A.weight": torch.zeros((2, 2))
            }
            active = types.SimpleNamespace(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules={"q_proj", "v_proj"},
                bias="none",
            )
            model = types.SimpleNamespace(peft_config={"default": active})
            with mock.patch.dict(sys.modules, {"peft": peft}), mock.patch(
                "training.checkpoint.load_file", return_value=saved
            ):
                with self.assertRaisesRegex(RuntimeError, "tensor values differ"):
                    load_lora_adapter(model, str(checkpoint), strict=True)


if __name__ == "__main__":
    unittest.main()
