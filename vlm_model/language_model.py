from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils import freeze_module, IMAGE_TOKEN


class LanguageModel(nn.Module):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [IMAGE_TOKEN]}
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        freeze_module(self.model)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.LongTensor:
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
