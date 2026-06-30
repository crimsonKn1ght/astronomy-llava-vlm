import argparse
import logging
import os

import yaml
import torch
from accelerate import Accelerator

from vlm_model.vlm import VLMForCausalLM
from vlm_model.utils import count_trainable_parameters, count_total_parameters
from data.dataset import LLaVAPretrainDataset
from training.trainer import VLMTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train VLM Stage 1 - Feature Alignment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_stage1.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    use_bf16 = train_cfg.get("bf16", True)

    accelerator = Accelerator(
        mixed_precision="bf16" if use_bf16 else "no",
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 32),
    )

    if accelerator.is_main_process:
        os.makedirs(train_cfg.get("output_dir", "./checkpoints"), exist_ok=True)

    logger.info("Building model...")
    model = VLMForCausalLM(config)

    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    logger.info(f"Trainable parameters: {trainable:,} ({trainable / total:.4%} of {total:,})")

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  [TRAINABLE] {name}: {param.shape}")

    data_cfg = config.get("data", {})
    logger.info("Building dataset...")
    dataset = LLaVAPretrainDataset(
        data_path=data_cfg["train_data_path"],
        image_dir=data_cfg["image_dir"],
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        image_token_id=model.image_token_id,
        max_length=data_cfg.get("max_length", 2048),
    )
    logger.info(f"Dataset size: {len(dataset)} samples")

    trainer = VLMTrainer(
        model=model,
        train_dataset=dataset,
        config=config,
        accelerator=accelerator,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Done.")


if __name__ == "__main__":
    main()
