---
license: cc-by-sa-4.0
base_model:
  - Qwen/Qwen2.5-1.5B-Instruct
  - openai/clip-vit-large-patch14
datasets:
  - UniverseTBD/AstroLLaVA_convos
language:
  - en
pipeline_tag: image-text-to-text
tags:
  - vision-language-model
  - llava
  - astronomy
  - multimodal
  - image-captioning
  - connector
---

# AstroLLaVA Stage-1 (connector alignment)

A LLaVA-style vision–language connector that lets **Qwen2.5-1.5B-Instruct** describe astronomy
images encoded by **CLIP ViT-L/14**. Only the connector (~3.9M params) is trained; both backbones
stay frozen. This is the **Stage-1 feature-alignment** stage, trained on
[`UniverseTBD/AstroLLaVA_convos`](https://huggingface.co/datasets/UniverseTBD/AstroLLaVA_convos).

> ⚠️ This repo ships the **connector checkpoint only** (`connector.safetensors`, ~16 MB). It is
> **not** a standalone `transformers` model — it needs the custom VLM code from the
> [astronomy-vlm](https://github.com/crimsonKn1ght/astronomy-vlm) repo plus the two base models
> (auto-downloaded from the Hub) to run.

## Architecture

```
image ─► CLIP ViT-L/14 (frozen) ─► MLP connector (TRAINED) ─► Qwen2.5-1.5B-Instruct (frozen) ─► text
                                    1024 → 1536 → 1536
```

- **Vision:** `openai/clip-vit-large-patch14`, penultimate layer patch features (frozen)
- **Connector:** 2-layer MLP with GELU, 1024→1536→1536 (the only trained weights)
- **LLM:** `Qwen/Qwen2.5-1.5B-Instruct` (frozen)
- **Trainable / total:** 3,935,232 / 1,850,414,592 (0.21%)

## Training

| | |
|---|---|
| Data | `UniverseTBD/AstroLLaVA_convos` → 164,924 records (29,596 captions + 135,328 QA) |
| Image prep | long side ≤ 384 px, JPEG |
| Objective | next-token cross-entropy on answer tokens only (connector-only) |
| Effective batch | 128 (per-device 8 × grad-accum 16) |
| LR / schedule | 1e-3, cosine with 3% warmup |
| Precision | bf16 (autocast) |
| Max length | 512 (+256 image tokens) |
| Hardware | 1× RTX 6000 Ada (48 GB), ~26 samples/s, ~38 GB VRAM |
| Loss | ~2.1 → ~1.6 |

`checkpoint-1300` ≈ one full epoch and is the recommended checkpoint.

## Usage

```bash
# 1. get the code
git clone https://github.com/crimsonKn1ght/astronomy-vlm && cd astronomy-vlm
pip install -r requirements.txt

# 2. download + unzip a checkpoint from this repo
hf download <your-username>/astrollava-stage1 astrollava-stage1-checkpoint-1300.zip --local-dir .
unzip astrollava-stage1-checkpoint-1300.zip -d ckpt

# 3. caption an image (CLIP + Qwen auto-download on first run)
python inference.py \
  --config ckpt/pretrain_astrollava.yaml \
  --checkpoint ckpt/checkpoint-1300 \
  --image your_astro_image.jpg \
  --prompt "Describe this astronomical image." \
  --temperature 0
```

`predictions.jsonl` (included in the zips) holds example outputs with their ground-truth captions.

## Capabilities & limitations

**What it does well** — it grounds on *coarse visual structure*: a blue sphere → Earth, a cluster
of bright stars → star cluster, a ring around a star → supernova remnant, an oval all-sky map →
all-sky map. Outputs vary appropriately with image content and read in proper caption style.

**What it doesn't** — it **hallucinates fine details** (exact object names, telescopes, distances,
wavelengths), filling specifics from the frozen LLM's language prior rather than the pixels. This
is the expected Stage-1 ceiling: the connector supplies a coarse visual category and the frozen LLM
improvises the rest. For factual specificity, a **Stage-2 fine-tune** (unfreezing the LLM, e.g. via
LoRA, on the QA pairs) is the next step — more Stage-1 epochs do not fix it.

`predictions.jsonl` are example outputs on **training** images — there is no held-out test split,
so treat them as qualitative samples, not a benchmark.

## Reproduction

See `REPRODUCE.md` inside each zip for the exact code commit, build command, and package versions.

```
build:  python scripts/build_astrollava_trainset.py --include-qa --max-image-size 384
train:  python train.py --config configs/pretrain_astrollava.yaml
```

## License & attribution

- **Weights:** `cc-by-sa-4.0`, inherited from the training data.
- **Training data:** [`UniverseTBD/AstroLLaVA_convos`](https://huggingface.co/datasets/UniverseTBD/AstroLLaVA_convos)
  (CC-BY-SA-4.0); imagery from NASA APOD, ESO, and NASA/ESA Hubble.
- **Base models:** Qwen2.5-1.5B-Instruct (Apache-2.0), CLIP ViT-L/14 (OpenAI, MIT).
- Built on the AstroLLaVA work ([arXiv:2504.08583](https://arxiv.org/abs/2504.08583)).
