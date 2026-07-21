import json
import os
from typing import Optional, Any

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


def save_connector_checkpoint(
    connector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    loss: float,
    output_dir: str,
    peft_model: Optional[nn.Module] = None,
) -> str:
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    save_file(connector.state_dict(), os.path.join(checkpoint_dir, "connector.safetensors"))

    # Stage-2: also persist the LoRA adapter next to the connector. Stage-1 passes peft_model=None,
    # so its checkpoint dirs stay byte-compatible (no lora/ subdir).
    if peft_model is not None:
        peft_model.save_pretrained(os.path.join(checkpoint_dir, "lora"))

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )

    meta = {"step": step, "loss": loss}
    with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return checkpoint_dir


def load_lora_adapter(
    peft_model: nn.Module, checkpoint_path: str, *, strict: bool = False
) -> bool:
    """Load a saved LoRA adapter into an already-LoRA-wrapped model. No-op (returns False) if the
    checkpoint has no ``lora/`` subdir, so Stage-1 checkpoints load unchanged."""
    lora_dir = os.path.join(checkpoint_path, "lora")
    if not os.path.isdir(lora_dir):
        return False

    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    adapter_file = os.path.join(lora_dir, "adapter_model.safetensors")
    state_dict = load_file(adapter_file)
    result = set_peft_model_state_dict(peft_model, state_dict)
    if strict:
        config_path = os.path.join(lora_dir, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Strict LoRA loading requires {config_path}")
        with open(config_path, "r", encoding="utf-8") as stream:
            saved_config = json.load(stream)
        active_configs = getattr(peft_model, "peft_config", {})
        active = active_configs.get("default") if hasattr(active_configs, "get") else None
        if active is None:
            raise RuntimeError("PEFT model has no active 'default' adapter configuration")
        comparisons = {
            "r": (int(saved_config["r"]), int(active.r)),
            "lora_alpha": (int(saved_config["lora_alpha"]), int(active.lora_alpha)),
            "lora_dropout": (
                float(saved_config["lora_dropout"]),
                float(active.lora_dropout),
            ),
            "target_modules": (
                set(saved_config["target_modules"]),
                set(active.target_modules),
            ),
            "bias": (str(saved_config["bias"]), str(active.bias)),
        }
        mismatches = {
            key: {"saved": saved, "active": current}
            for key, (saved, current) in comparisons.items()
            if saved != current
        }
        if mismatches:
            raise RuntimeError(f"LoRA adapter configuration mismatch: {mismatches}")
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if unexpected:
            raise RuntimeError(f"LoRA adapter has unexpected keys: {unexpected[:10]}")
        roundtrip = get_peft_model_state_dict(
            peft_model,
            adapter_name="default",
            save_embedding_layers=True,
        )
        missing_saved = sorted(set(state_dict) - set(roundtrip))
        if missing_saved:
            raise RuntimeError(
                "LoRA load did not materialize every saved adapter/embedding key: "
                + ", ".join(missing_saved[:10])
            )
        shape_mismatches = [
            key
            for key, tensor in state_dict.items()
            if tuple(roundtrip[key].shape) != tuple(tensor.shape)
        ]
        if shape_mismatches:
            raise RuntimeError(
                "LoRA loaded tensor shapes differ for: " + ", ".join(shape_mismatches[:10])
            )
        value_mismatches = []
        for key, saved_tensor in state_dict.items():
            loaded_tensor = roundtrip[key]
            if not isinstance(saved_tensor, torch.Tensor) or not isinstance(
                loaded_tensor, torch.Tensor
            ):
                continue
            if saved_tensor.numel() == 0:
                continue
            positions = sorted({0, saved_tensor.numel() // 2, saved_tensor.numel() - 1})
            saved_sample = (
                saved_tensor.detach().reshape(-1)[positions].to(dtype=torch.float32, device="cpu")
            )
            loaded_sample = (
                loaded_tensor.detach().reshape(-1)[positions].to(dtype=torch.float32, device="cpu")
            )
            if not torch.allclose(saved_sample, loaded_sample, rtol=1e-4, atol=1e-6):
                value_mismatches.append(key)
        if value_mismatches:
            raise RuntimeError(
                "LoRA loaded tensor values differ from the saved checkpoint for: "
                + ", ".join(value_mismatches[:10])
            )
    return True


def load_connector_checkpoint(
    connector: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> int:
    connector_path = os.path.join(checkpoint_path, "connector.safetensors")
    state_dict = load_file(connector_path)
    connector.load_state_dict(state_dict)

    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if (optimizer is not None or scheduler is not None) and os.path.exists(
        training_state_path
    ):
        training_state = torch.load(training_state_path, weights_only=True)
        if optimizer is not None:
            optimizer.load_state_dict(training_state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(training_state["scheduler"])

    meta_path = os.path.join(checkpoint_path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta.get("step", 0)

    return 0
