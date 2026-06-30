# Vision-Language Model (VLM) - Stage 1 Feature Alignment

A minimal, efficient implementation of a Vision-Language Model following the LLaVA architecture. This project trains only a lightweight MLP connector to align frozen CLIP vision features with a frozen LLM, achieving effective multimodal understanding with minimal computational cost.

## Overview

This implementation bridges a frozen CLIP vision encoder (`openai/clip-vit-large-patch14`) and a frozen instruction-tuned LLM (`Qwen/Qwen2.5-1.5B-Instruct`) using a simple 2-layer MLP connector. Only the connector (~3.9M parameters) is trained on image-caption pairs; both the vision encoder and LLM remain frozen throughout training.

**Key Design Principles:**
- **Simplicity**: No Q-Former, no cross-attention — just a linear projection with GELU
- **Efficiency**: Frozen models save memory; only train the connector
- **Proven approach**: Follows LLaVA Stage 1 alignment, a well-validated recipe

> A connector trained with this codebase on astronomy image–text data is released at
> [`grKnight/astrollava-stage1`](https://huggingface.co/grKnight/astrollava-stage1).
> See [Trained Model: AstroLLaVA Stage-1](#trained-model-astrollava-stage-1-astronomy) for the
> dataset, training, testing, and download details.

## Architecture

```
Image (3, 224, 224)
    ↓
[CLIP ViT-L/14 — Frozen] → 256 patch tokens (B, 256, 1024)
    ↓
[MLP Connector — Trainable] → (B, 256, 1536)
    ↓ (concatenated with)
Text Embeddings (B, T, 1536) from LLM embedding table
    ↓
[Qwen2.5-1.5B-Instruct — Frozen] → Loss
```

**Training objective**: Next-token prediction on image-text pairs. Visual tokens are masked out of the loss; only the caption tokens contribute to gradient updates on the connector.

## Installation

```bash
# Clone or navigate to the project directory
cd vlm

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.1.0
- CUDA 11.8+ (for GPU training)
- 12GB+ GPU memory (tested on RTX 3060, A100)

## Quick Start

### 1. Prepare Data

Download the LLaVA-Pretrain dataset or use your own image-text pairs in LLaVA format:

```json
[
  {
    "id": "...",
    "image": "path/to/image.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe this image."},
      {"from": "gpt", "value": "A detailed caption here."}
    ]
  },
  ...
]
```

### 2. Configure Training

Edit `configs/pretrain_stage1.yaml`:

```yaml
data:
  train_data_path: "/path/to/blip_laion_cc_sbu_558k.json"  # or your dataset
  image_dir: "/path/to/images"
  max_length: 2048

training:
  output_dir: "./checkpoints/pretrain-stage1"
  per_device_batch_size: 8
  gradient_accumulation_steps: 32  # effective batch = 256
  num_epochs: 1
```

### 3. Train

```bash
# Single GPU
python train.py --config configs/pretrain_stage1.yaml

# Multi-GPU with accelerate
accelerate launch train.py --config configs/pretrain_stage1.yaml
```

Training logs appear in stdout. Checkpoints are saved every 500 steps to `./checkpoints/pretrain-stage1/`.

### 4. Inference

```bash
python inference.py \
  --config configs/pretrain_stage1.yaml \
  --checkpoint ./checkpoints/pretrain-stage1/checkpoint-500 \
  --image /path/to/image.jpg \
  --prompt "What is in this image?"
```

## Project Structure

```
vlm/
├── train.py                        # Training entry point
├── inference.py                    # Inference script
├── requirements.txt                # Dependencies
├── configs/
│   └── pretrain_stage1.yaml        # Hyperparameter config
├── vlm_model/
│   ├── utils.py                    # Constants, helper functions
│   ├── connector.py                # MLP projection layer (trainable)
│   ├── vision_encoder.py           # CLIP ViT-L/14 wrapper
│   ├── language_model.py           # Qwen2.5-1.5B-Instruct wrapper
│   └── vlm.py                      # Composite VLM model
├── data/
│   ├── image_processing.py         # CLIP image transforms
│   ├── conversation.py             # Conversation tokenization + label masking
│   ├── dataset.py                  # LLaVAPretrainDataset
│   └── collator.py                 # Batch collation
└── training/
    ├── lr_scheduler.py             # Cosine warmup scheduler
    ├── checkpoint.py               # Checkpoint save/load
    └── trainer.py                  # Training loop
```

## Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2e-3 | High (connector only), follows LLaVA |
| Weight Decay | 0.0 | No regularization needed for small connector |
| Batch Size | 256 (effective) | Via 8 × 32 (batch × accumulation) |
| Warmup | 3% of steps | Short warmup; model components are pre-trained |
| Schedule | Cosine decay | Smooth convergence to near-zero lr |
| Precision | bf16 | Mixed precision for efficiency |
| Epochs | 1 | Single pass over 558K samples (if using LLaVA-Pretrain) |

## Performance & Memory

**Trainable Parameters**: ~3.9M (only the MLP connector)
**Frozen Parameters**: ~1.8B (CLIP + LLM)
**GPU Memory**: ~6–8 GB (with batch_size=8, bf16 precision)

Expected training time on a single RTX 3060 (12GB) for 558K samples:
- ~3 days at batch_size=8 with gradient_accumulation=32

## What Gets Trained

Only the connector's weights are updated:
- `model.connector.mlp[0].weight` — (1536, 1024)
- `model.connector.mlp[0].bias` — (1536,)
- `model.connector.mlp[2].weight` — (1536, 1536)
- `model.connector.mlp[2].bias` — (1536,)

Both the vision encoder (`vision_encoder.model`) and LLM (`language_model.model`) are frozen via `requires_grad=False`.

## Data Format Details

The dataset expects a JSON file with the LLaVA-Pretrain structure:
- Each entry must have `"image"` (relative path to image), `"conversations"` (list of turns)
- Conversations are `[{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]`
- The `<image>` placeholder in the human message is replaced with visual token embeddings during training
- Labels are masked (`-100`) for all tokens except the assistant's response

## Inference Details

During inference:
1. The image is preprocessed via CLIP's image processor (224×224 resize, normalization)
2. CLIP encodes it to 256 patch tokens (1024-dim each)
3. The connector projects to 1536-dim (LLM embedding space)
4. Text prompt is tokenized and embedded via the LLM's embedding layer
5. Visual and text embeddings are concatenated and fed to the LLM
6. Autoregressive generation proceeds with `model.generate(...)`

The `<image>` token in the prompt is automatically replaced with visual embeddings; it does not appear in the final token sequence.

## Trained Model: AstroLLaVA Stage-1 (Astronomy)

A connector trained with this codebase on real astronomy image–text data is released on the
Hugging Face Hub:

**https://huggingface.co/grKnight/astrollava-stage1**

The release contains two archives. Each bundles the checkpoint(s) together with the training
config, the example predictions (`predictions.jsonl`), and a `REPRODUCE.md` provenance file:

| Archive | Contents |
|---------|----------|
| `astrollava-stage1-checkpoint-1300.zip` | `checkpoint-1300` (≈ one epoch; the recommended checkpoint) |
| `astrollava-stage1-all-checkpoints.zip` | every saved checkpoint (steps 100–1300) |

Each checkpoint is the connector only (`connector.safetensors`, ~16 MB) plus optimizer state. It
is **not** a standalone `transformers` model — it requires this repository's code and the two base
models (downloaded from the Hub) to run.

### Dataset

Training used [`UniverseTBD/AstroLLaVA_convos`](https://huggingface.co/datasets/UniverseTBD/AstroLLaVA_convos)
(CC-BY-SA-4.0): ~29.8k astronomy images from NASA APOD, ESO, and the NASA/ESA Hubble Space
Telescope, each with a human-written caption and GPT-4-generated question–answer turns.
`scripts/build_astrollava_trainset.py` streams the dataset and converts it to the LLaVA JSON format
this repo expects:

```bash
python scripts/build_astrollava_trainset.py \
  --output-dir datasets/astrollava_llava --split train \
  --include-qa --max-image-size 384
```

- Caption records use the human-written caption; `--include-qa` additionally emits one single-turn
  record per QA pair (the tokenizer keeps only the last turn, so conversations are flattened).
- `--max-image-size 384` caps the long edge (CLIP uses 224×224 regardless) and images are
  re-encoded as JPEG to keep the extracted set small (~1–2 GB instead of tens of GB as PNG).
- Oversized frames (some Hubble images exceed 100 megapixels) and the occasional unreadable image
  are skipped rather than aborting the run.

The resulting set was **164,924 records** (29,596 captions + 135,328 QA) over ~29.7k images;
41 rows were skipped as unreadable.

### Training

Config: `configs/pretrain_astrollava.yaml`.

```bash
python train.py --config configs/pretrain_astrollava.yaml
```

| Setting | Value |
|---------|-------|
| Trainable / total params | 3,935,232 / 1,850,414,592 (0.21%) |
| Effective batch | 128 (per-device 8 × grad-accum 16) |
| Learning rate / schedule | 1e-3, cosine, 3% warmup |
| Max length | 512 (+256 image tokens) |
| Precision | bf16 |
| Hardware | 1× RTX 6000 Ada (48 GB) |
| Throughput / memory | ~26 samples/s, ~38 GB VRAM |
| Loss | ~2.1 → ~1.6 |

The connector is checkpointed every 100 steps. `checkpoint-1300` corresponds to roughly one full
epoch and is the released default; the loss is largely flat after the first ~100 steps, as expected
for Stage-1 alignment. A RunPod workflow for the full setup → build → train sequence is documented
in `RUNPOD.md` (`scripts/runpod_setup.sh`, `scripts/runpod_train.sh`).

### Testing

Two paths were used to inspect the trained connector:

```bash
# single image
python inference.py \
  --config configs/pretrain_astrollava.yaml \
  --checkpoint checkpoints/astrollava-stage1/checkpoint-1300 \
  --image datasets/astrollava_llava/images/astrollava_train_0.jpg \
  --prompt "Describe this astronomical image." --temperature 0

# batched: one model load over a sample, writing predictions.jsonl
python scripts/batch_inference.py \
  --config configs/pretrain_astrollava.yaml \
  --checkpoint checkpoints/astrollava-stage1/checkpoint-1300 \
  --image-dir datasets/astrollava_llava/images \
  --train-json datasets/astrollava_llava/train.json \
  --num-samples 200 --temperature 0 --output predictions.jsonl
```

`batch_inference.py` loads the model once and captions a sample of images, attaching each image's
ground-truth caption for side-by-side comparison; the resulting `predictions.jsonl` is bundled into
both release archives. Generation runs under bf16 autocast to match training.

**Observed behavior.** The connector grounds reliably on coarse visual structure — across the
sampled images it distinguishes planetary nebulae, galaxies, galaxy clusters, deep fields,
planetary surfaces, and all-sky maps, and several outputs name the correct object class or catalog
(for example NGC 6302 as a planetary nebula, an Abell galaxy cluster, a Martian crater with
water-related features). Fine details are less reliable: catalog numbers, instrument/telescope,
dates, and distances are frequently supplied from the language model's prior rather than the image,
and common caption phrasing (ESO/Hubble attributions) recurs. This is the expected Stage-1 ceiling —
the connector supplies a coarse visual category and the frozen LLM improvises the specifics.
Improving factual specificity is a Stage-2 task (unfreezing the LLM), not more Stage-1 training.

> **Note on evaluation.** No held-out test split was created — all records went into `train.json`
> and the `predictions.jsonl` images were sampled from that same set. The example outputs therefore
> reflect fit on seen data and may include some memorization/leakage; treat them as qualitative
> samples, not a benchmark.

### Reproduction

Each archive includes a `REPRODUCE.md` pinning the code commit, base models, dataset build command,
training command, and package versions (`torch 2.8.0+cu128`, `transformers 5.12.1`). The build and
train commands above reproduce the run. Note that `num_epochs` in the config is 3, whereas the
released `checkpoint-1300` is the ≈1-epoch checkpoint.

## Medical RAG Layer (Retrieval-Augmented Grounding)

On top of the frozen VLM, this repo includes an **inference-time retrieval-augmented
generation (RAG) pipeline** for reducing hallucination in medical visual question
answering. It implements the research methodology in three phases:

1. **Phase 1 — Baseline hallucination characterization.** Evaluate the frozen VLM on
   **VQA-RAD** / **PathVQA** with **exact-match accuracy** *and* **NLI-based
   factual-consistency** (`eval/`).
2. **Phase 2 — Retrieval + ablation.** Build a corpus of image-report pairs, index dense
   clinical-SBERT report vectors and pooled CLIP visual embeddings in **FAISS** plus a
   **BM25** sparse index, retrieve top-k pairs and **prepend** them as a structured context
   block. Ablate **dense-visual / sparse-BM25 / hybrid (RRF + cross-encoder rerank)**
   (`retrieval/`, `corpus/`, `eval/ablation.py`).
3. **Phase 3 — Refinements (scaffolded).** Query decomposition, adaptive context windows,
   modality-aware weighting are wired as inert, config-gated hooks (`ragcore/phase3_stubs.py`).

**Key property:** grounding is entirely prompt-level. The retrieved references are inserted
*after* the `<image>` token and *before* the question, so `prepare_inputs_embeds` is
unchanged — **no retraining, no model/tokenizer/connector edits**. Model ids are
config-driven, so a domain-specialized medical VLM / encoder swaps in by editing
`configs/rag_eval.yaml` only.

### New packages

```
retrieval/   BaseRetriever seam + dense_visual / sparse_bm25 / hybrid + encoders, faiss, bm25, reranker
corpus/      CorpusLoader interface: iu_xray, roco (open), mimic_cxr (credentialed stub), synthetic; build_index
eval/        VQA-RAD/PathVQA loaders, exact-match, NLI consistency, runner, ablation driver
ragcore/     context formatting (prepend), checkpoint-optional model loader, Phase-3 stubs
rag_inference.py   retrieve -> format -> prepend -> model.generate
configs/rag_eval.yaml
```

### Install the extra dependencies

```bash
pip install faiss-cpu sentence-transformers rank-bm25
# (NLI factual-consistency reuses the existing `transformers` pipeline — no extra dep.)
```

### Usage

```bash
# 1. Build a tiny synthetic index (no downloads, no PHI) to smoke-test the wiring
python -m corpus.build_index --config configs/rag_eval.yaml --synthetic --max_pairs 10

# 2. Phase-1 no-retrieval baseline on a few VQA-RAD samples
python -m eval.ablation --config configs/rag_eval.yaml --modes no_retrieval --num_samples 5

# 3. Full ablation + comparison table (./rag_results/comparison.md)
python -m eval.ablation --config configs/rag_eval.yaml \
    --modes no_retrieval dense_visual sparse_bm25 hybrid --num_samples 5

# 4. Single grounded answer
python rag_inference.py --config configs/rag_eval.yaml \
    --image path/to/image.png --question "Is there cardiomegaly?" --mode hybrid
```

To use a real open corpus, set `corpus.name: roco` (verified: `eltorio/ROCO-radiology`) in
the config and re-run `corpus.build_index`. `iu_xray` has no canonical HF mirror — set
`corpus.hf_id` to one you've verified. `mimic_cxr` is a credentialed stub (PhysioNet access).
The model loader runs with an **untrained connector** when `model.checkpoint` is null — fine
for exercising retrieval and prompt formatting; set it to a trained connector dir for
meaningful generations.

### Running the real experiment on a GPU

A CUDA box (e.g. RTX 3060) is recommended; the first run downloads the encoder/LLM/NLI
weights (~7 GB total) and caches them. Datasets and the corpus **stream and are capped**, so
only what's needed downloads.

```bash
pip install -r requirements.txt          # includes faiss-cpu, sentence-transformers, rank-bm25

# One command: build the ROCO index, then run the full 4-way ablation on VQA-RAD.
python run_rag.py --config configs/rag_eval_gpu.yaml

# Reuse an already-built index and just re-run the eval:
python run_rag.py --config configs/rag_eval_gpu.yaml --skip-index
```

Results land in `./rag_results/` (`results_<mode>.json` per mode + `comparison.md`).
Tune `configs/rag_eval_gpu.yaml`: `corpus.max_pairs` (index size), `eval.num_samples`
(toward the full ~451 VQA-RAD test set for final numbers), `eval.benchmark` (`path_vqa`),
and `eval.modes`.

> **For meaningful numbers you need a trained VLM.** With `model.checkpoint: null` the
> connector is random and the answers (hence EM/NLI) are not meaningful — the run still
> validates the full pipeline. Train the connector first (`python train.py ...`) and set
> `model.checkpoint`, or swap `model.config` to a fully-trained medical VLM.

### Porting to another domain (astronomy example)

The pipeline is domain-agnostic; an astronomy variant ships as a worked example. The core
(`retrieval/`, `eval/runner.py`, `eval/ablation.py`, the prepend logic, Phase-3 hooks) is
reused **unchanged** — only domain-specific pieces were added:

- **Corpus:** `corpus/galaxy_zoo.py` — Galaxy10 DECaLS (`matthieulel/galaxy10_decals`,
  verified); morphology class labels are turned into descriptive "reports".
- **Benchmark:** `eval/astro_benchmarks.py` — VQA generated from the held-out test split
  (open "what type" + balanced closed yes/no), returned as the same `VQASample`.
- **Prompt wording:** now config-driven. `ragcore/context_format.py` reads an optional
  `prompt:` block (`system`, `references_header`, `references_footer`); medical text remains
  the default, so existing configs are untouched.
- **Config:** `configs/rag_eval_astro.yaml` (galaxy corpus/benchmark, general SBERT,
  astronomy prompt).

```bash
python run_rag.py --config configs/rag_eval_astro.yaml          # GPU
python run_rag.py --config configs/rag_eval_astro.yaml --synthetic --num_samples 5   # CPU smoke
```

What a *production* astronomy port still needs (documented, not done): swapping the CLIP
vision tower for an astronomy model (e.g. AstroCLIP) — which is not a drop-in `CLIPVisionModel`
and **requires retraining the connector** — plus FITS/dynamic-range image handling in
`data/image_processing.py`. Standard CLIP on RGB galaxy cutouts is adequate only for the
prototype.

### Known limitations
- **CLIP 224×224** downsamples fine medical detail (caps both VLM and dense-visual quality);
  `retrieval.visual_encoder_id` is swappable (e.g. BiomedCLIP).
- **NLI domain mismatch:** the default `roberta-large-mnli` under-handles clinical
  negation/hedging; swap `eval.nli_model_id` for a MedNLI/SciNLI checkpoint and read NLI as
  relative-across-modes.
- **FAISS `IndexFlatIP`** is exact and correct at sample scale; swap to IVF/HNSW in
  `retrieval/faiss_index.py` for MIMIC-scale corpora.

## Next Steps

This implementation covers **Stage 1: Feature Alignment**. Future extensions might include:

1. **Stage 2: Full Model Tuning** — Unfreeze LLM layers and fine-tune on instruction-following data
2. **Cross-Attention Connector** — Replace MLP with a learned cross-attention mechanism for better spatial reasoning
3. **Higher-Resolution Images** — Support variable image resolutions and dynamic patching
4. **Multi-Image Support** — Handle multiple images per prompt
5. **Evaluation** — Add benchmarks (VQA, captioning, visual reasoning tasks)

## Troubleshooting

**Out of Memory (OOM)**
- Reduce `per_device_batch_size` (try 4 or 2)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `bf16` mixed precision (already default)

**Slow Data Loading**
- Increase `dataloader_num_workers` (try 8 or 16)
- Pre-extract CLIP features to disk to bypass image I/O

**NaN Loss**
- Check label masking: visual token positions should have `labels = -100`
- Verify `attention_mask` has correct shape and no all-zero rows
- Reduce learning rate if instability persists

## References

- **LLaVA**: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- **CLIP**: [Learning Transferable Models for Compositional Vision](https://arxiv.org/abs/2103.14030)
- **Qwen**: [Qwen2.5 Technical Report](https://qwenlm.github.io/blog/qwen2.5/)

## License

MIT License. See LICENSE file (if present) for details.

## Contributing

Contributions are welcome. Please:
1. Test your changes with a small dataset subset
2. Verify trainable params count and gradient flow
3. Include clear commit messages

## Questions?

For issues or questions about the implementation, check the inline comments in `vlm_model/vlm.py` (especially `prepare_inputs_embeds()`) and `training/trainer.py` for detailed logic.
