from __future__ import annotations

import unittest
from unittest.mock import patch


class ModelRevisionPropagationTests(unittest.TestCase):
    def test_vision_encoder_passes_revision_to_model_and_processor(self) -> None:
        with patch("vlm_model.vision_encoder.CLIPVisionModel.from_pretrained") as model, patch(
            "vlm_model.vision_encoder.CLIPImageProcessor.from_pretrained"
        ) as processor, patch("vlm_model.vision_encoder.freeze_module"):
            model.return_value.config.hidden_size = 1
            model.return_value.config.image_size = 2
            model.return_value.config.patch_size = 1
            from vlm_model.vision_encoder import VisionEncoder

            VisionEncoder("clip", revision="a" * 40)
            model.assert_called_once_with("clip", revision="a" * 40)
            processor.assert_called_once_with("clip", revision="a" * 40)

    def test_language_model_passes_revision_to_model_and_tokenizer(self) -> None:
        with patch("vlm_model.language_model.AutoModelForCausalLM.from_pretrained") as model, patch(
            "vlm_model.language_model.AutoTokenizer.from_pretrained"
        ) as tokenizer, patch("vlm_model.language_model.freeze_module"):
            fake_tokenizer = tokenizer.return_value
            fake_tokenizer.get_vocab.return_value = {"<image>": 1}
            fake_tokenizer.pad_token = "pad"
            from vlm_model.language_model import LanguageModel

            LanguageModel("lm", revision="b" * 40)
            self.assertEqual(model.call_args.kwargs["revision"], "b" * 40)
            self.assertEqual(tokenizer.call_args.kwargs["revision"], "b" * 40)


if __name__ == "__main__":
    unittest.main()
