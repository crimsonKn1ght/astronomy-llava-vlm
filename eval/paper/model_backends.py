"""Isolated, deterministic generation backends for the four paper models."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from decode_utils import split_assistant_response, strip_prompt_if_echoed, trim_completion_ids

from .assets import AssetRegistry
from .protocol import PaperProtocol


@dataclass
class GenerationOutput:
    response: str
    raw_response: str
    generated_token_ids: list[int]
    prompt_token_count: int
    termination_reason: str
    rendered_prompt: str
    template_source: str
    template_sha256: str
    extra: Dict[str, Any] = field(default_factory=dict)


def _termination(token_ids: Sequence[int], eos_ids: int | Sequence[int] | None, cap: int) -> str:
    eos = {int(eos_ids)} if isinstance(eos_ids, int) else {int(item) for item in (eos_ids or [])}
    if token_ids and int(token_ids[-1]) in eos:
        return "eos"
    if len(token_ids) >= cap:
        return "max_new_tokens"
    return "stopped"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AstraQPaperBackend:
    def __init__(
        self,
        protocol: PaperProtocol,
        model_label: str,
        registry: AssetRegistry,
        device: str,
    ) -> None:
        import torch

        from inference import load_vlm

        self.torch = torch
        self.device = device
        self.model_label = model_label
        self.model_config = protocol.astraq_architecture(model_label)
        self.model_config["vision_encoder"]["model_name"] = str(
            registry.shared_path("base_models.vision_encoder")
        )
        self.model_config["language_model"]["model_name"] = str(
            registry.shared_path("base_models.language_model")
        )
        checkpoint = registry.checkpoint_path(model_label)
        self.model = load_vlm(
            None,
            str(checkpoint),
            device,
            config=self.model_config,
            strict_lora=protocol.data["models"][model_label].get("stage") == 2,
        )

    def generate(self, record: Mapping[str, Any], max_new_tokens: int) -> GenerationOutput:
        from data.image_processing import load_and_process_image
        from vlm_model.utils import IMAGE_TOKEN

        prompt = str(record["prompt"])
        template_contract = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{IMAGE_TOKEN}\n{TASK_PROMPT}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        rendered = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{IMAGE_TOKEN}\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer = self.model.tokenizer
        tokenizer.padding_side = "left"
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        pixels = load_and_process_image(str(record["image_path"]), self.model.image_processor)
        pixels = pixels.unsqueeze(0).to(self.device)
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "eos_token_id": eos_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        context = (
            self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with self.torch.inference_mode(), context:
            output = self.model.generate(
                input_ids=input_ids,
                images=pixels,
                attention_mask=attention_mask,
                **kwargs,
            )
        token_ids = [int(value) for value in output[0].detach().cpu().tolist()]
        raw = tokenizer.decode(token_ids, skip_special_tokens=False)
        response = split_assistant_response(raw)
        return GenerationOutput(
            response=response.strip(),
            raw_response=raw,
            generated_token_ids=token_ids,
            prompt_token_count=int(input_ids.shape[-1]),
            termination_reason=_termination(token_ids, eos_id, max_new_tokens),
            rendered_prompt=rendered,
            template_source="astraq_chatml_v1",
            template_sha256=_text_sha256(template_contract),
        )


class Qwen3VLPaperBackend:
    def __init__(
        self,
        protocol: PaperProtocol,
        model_label: str,
        registry: AssetRegistry,
        device: str,
    ) -> None:
        import torch
        import transformers

        self.torch = torch
        self.device = device
        path = registry.model_path(model_label)
        cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if cls is None:
            cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if cls is None:
            raise RuntimeError("This Transformers build does not support Qwen3-VL")
        dtype = torch.bfloat16
        kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if device.startswith("cuda"):
            kwargs["device_map"] = "auto"
        self.model = cls.from_pretrained(str(path), **kwargs)
        if not device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()
        self.processor = transformers.AutoProcessor.from_pretrained(str(path))
        self.template_sha256 = _text_sha256(str(self.processor.chat_template or ""))

    @staticmethod
    def messages(image_path: str | Path, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Path(image_path).resolve().as_uri()},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def generate(self, record: Mapping[str, Any], max_new_tokens: int) -> GenerationOutput:
        from qwen_vl_utils import process_vision_info

        messages = self.messages(record["image_path"], str(record["prompt"]))
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_device = getattr(self.model, "device", self.device)
        inputs = inputs.to(model_device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        full_ids = [int(value) for value in output[0].detach().cpu().tolist()]
        input_list = [int(value) for value in inputs.input_ids[0].detach().cpu().tolist()]
        completion = trim_completion_ids(input_list, full_ids)
        raw = self.processor.batch_decode(
            [full_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        response = self.processor.batch_decode(
            [completion], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        return GenerationOutput(
            response=response,
            raw_response=raw,
            generated_token_ids=[int(item) for item in completion],
            prompt_token_count=len(input_list),
            termination_reason=_termination(completion, eos, max_new_tokens),
            rendered_prompt=rendered,
            template_source="qwen3_processor_apply_chat_template",
            template_sha256=self.template_sha256,
        )


class AstroLLaVAPaperBackend:
    def __init__(
        self,
        protocol: PaperProtocol,
        model_label: str,
        registry: AssetRegistry,
        device: str,
    ) -> None:
        import torch
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from transformers import AutoConfig

        self.torch = torch
        self.device = device
        self.conv_mode = str(protocol.data["models"][model_label]["conversation_mode"])
        path = registry.model_path(model_label)
        vision_path = registry.shared_path("models.astrollava.vision_encoder")
        locked_config = AutoConfig.from_pretrained(str(path), local_files_only=True)
        locked_config.mm_vision_tower = str(vision_path)
        locked_config.vision_tower = str(vision_path)
        disable_torch_init()
        # The HF cache snapshot basename is only a commit SHA. The pinned
        # official loader gates vision initialization on "llava" appearing in
        # model_name, so provide the locked repository identity explicitly.
        model_name = "AstroLLaVA"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            str(path),
            None,
            model_name,
            load_8bit=False,
            load_4bit=False,
            device_map="auto" if device.startswith("cuda") else "none",
            device=device,
            use_flash_attn=False,
            config=locked_config,
        )
        self.model.eval()

    def generate(self, record: Mapping[str, Any], max_new_tokens: int) -> GenerationOutput:
        from PIL import Image
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        image = Image.open(record["image_path"]).convert("RGB")
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            image_token = DEFAULT_IMAGE_TOKEN
        conv = conv_templates[self.conv_mode].copy()
        template_contract = json.dumps(
            {
                "system": conv.system,
                "roles": list(conv.roles),
                "sep_style": str(conv.sep_style),
                "sep": conv.sep,
                "sep2": conv.sep2,
                "version": getattr(conv, "version", None),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        conv.append_message(conv.roles[0], image_token + "\n" + str(record["prompt"]))
        conv.append_message(conv.roles[1], None)
        rendered = conv.get_prompt()
        input_ids = tokenizer_image_token(
            rendered, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        images = process_images([image], self.image_processor, self.model.config).to(
            self.model.device, dtype=self.torch.float16
        )
        kwargs = {
            "images": images,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        with self.torch.inference_mode():
            try:
                output = self.model.generate(input_ids, image_sizes=[image.size], **kwargs)
            except TypeError:
                output = self.model.generate(input_ids, **kwargs)
        output_list = [int(value) for value in output[0].detach().cpu().tolist()]
        input_list = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        completion = strip_prompt_if_echoed(input_list, output_list)
        raw = self.tokenizer.decode(output_list, skip_special_tokens=False)
        response = self.tokenizer.batch_decode([completion], skip_special_tokens=True)[0].strip()
        stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if stop and response.endswith(stop):
            response = response[: -len(stop)].strip()
        return GenerationOutput(
            response=response,
            raw_response=raw,
            generated_token_ids=[int(item) for item in completion],
            prompt_token_count=len(input_list),
            termination_reason=_termination(completion, self.tokenizer.eos_token_id, max_new_tokens),
            rendered_prompt=rendered,
            template_source=f"astrollava_official:{self.conv_mode}",
            template_sha256=_text_sha256(template_contract),
        )


def create_backend(
    protocol: PaperProtocol,
    model_label: str,
    registry: AssetRegistry,
    device: str,
) -> Any:
    backend = protocol.data["models"][model_label]["backend"]
    if backend == "astraq":
        return AstraQPaperBackend(protocol, model_label, registry, device)
    if backend == "qwen3_vl":
        return Qwen3VLPaperBackend(protocol, model_label, registry, device)
    if backend == "astrollava":
        return AstroLLaVAPaperBackend(protocol, model_label, registry, device)
    raise ValueError(f"Unsupported paper backend: {backend}")
