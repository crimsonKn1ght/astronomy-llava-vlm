# Paper evaluation v2

This is the definitive, reproducible evaluation workflow for the AstraQ-VL paper. It evaluates the frozen Stage 1 and Stage 2 checkpoints internally, then compares them with AstroLLaVA and Qwen3-VL-4B on external astronomy data. It does **not** train or fine-tune any model, and a missing or mismatched checkpoint is a hard error rather than a request to train one.

The complete protocol is frozen in `configs/paper_eval_v2.yaml`. Model, dataset, tokenizer, processor, scorer, and AstroLLaVA code revisions are pinned there; checkpoint files are verified by SHA-256 before inference. Prompt text, the model-native rendered chat template, token IDs, raw and cleaned generations, termination reason, latency, errors, and protocol hashes are retained so scoring and reporting can be repeated without using a GPU.

## Scope

| Suite | Models | Required canonical predictions |
|---|---|---:|
| Internal frozen held-out set | AstraQ-VL Stage 1, AstraQ-VL Stage 2 | 3,271 per model |
| DeepSDO official test split | Stage 1, Stage 2, AstroLLaVA, Qwen3-VL-4B | 102 per model |
| AstroVLBench | The same four models | One per locked expected ID per model |

The internal set contains 591 images, 586 caption records, and 2,685 QA records. AstroLLaVA is deliberately omitted from the internal comparison because its training-data lineage overlaps the AstroLLaVA-derived dataset. It is used only on the external suites.

Retrieval is not part of this study. Stage 1 and Stage 2 were trained without RAG, and the paper protocol fixes `retrieval: false`, `few_shot: false`, and `quantization: false`. The separate RAG code in this repository is an optional inference-time research component; the paper runner never imports it, builds a retrieval index, or adds retrieved context.

## RunPod requirements

The default target is one nominal 24 GB NVIDIA Ampere, Ada, or Hopper GPU reporting at least 22,000 MiB VRAM, 32 GiB system RAM, and 80 GiB free persistent storage. The frozen legacy AstroLLaVA environment supports compute capability `>=8.0` and `<10.0`; Blackwell is therefore not accepted by this protocol. Clone the evaluation branch into persistent storage and run from the repository root. A clean Git worktree is required for a final run.

The wrapper bootstraps pinned `uv==0.11.30` and a managed CPython `3.11.15`, so the pod image does not need to provide Python 3.11 or the `venv` module. A lightweight Git/disk/RAM/GPU check runs before either CUDA environment is installed. Environment cache markers bind the Python, pip, Torch, torchvision, CUDA wheel index, and requirements content, and installed versions are revalidated before a marker is trusted.

The wrapper creates and reuses two isolated environments:

- `.paper_eval_venvs/modern` for AstraQ-VL, Qwen3-VL, scoring, and reporting.
- `.paper_eval_venvs/astrollava` for the legacy AstroLLaVA backend.

It also checks out AstroLLaVA source revision `697cfbf11fbe16ce326dbbdab06bd9d93ccba3e9` under `.paper_eval_runtime/` and verifies the installed package's direct Git revision. AstroLLaVA's secondary `openai/clip-vit-large-patch14-336` vision tower is downloaded independently at pinned revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` and its local snapshot path is injected into the locked model configuration, preventing an unpinned runtime download. Stage 1, Stage 2, and Qwen3 run in BF16; AstroLLaVA runs in FP16 with its pinned `llava_v1` conversation mode. Quantization and FlashAttention are not required. COCO METEOR requires Java; the wrapper installs an OpenJDK 17 headless runtime when Java is absent and the pod permits `apt-get`, otherwise the scorer fails explicitly before paper outputs are accepted.

## One-command internal and DeepSDO run

```bash
bash scripts/runpod/run_paper_eval.sh \
  --suites internal,deepsdo \
  --models all \
  --resume
```

With no positional command, the wrapper runs `all`: preflight checks, dataset preparation and audits, asset download and verification, stratified smoke tests, sequential inference, completeness validation, scoring with confidence intervals, paper tables/figures, and packaging. Models run sequentially to fit the supported nominal-24-GB GPU class.

The first internal preparation reconstructs the frozen image-level split from the pinned `UniverseTBD/AstroLLaVA_convos` snapshot and may take substantial time and disk space. It stops rather than evaluating if the reconstructed `test.json` does not match SHA-256 `c3765a252e18ed53d51378c1356970232b6c6e104e475ec46f529b5829daa4d5`.

## Staged execution

The same workflow can be run stage by stage. Keep `--suites`, `--models`, and any custom root arguments identical across commands.

```bash
# Environment, Git, and hardware checks
bash scripts/runpod/run_paper_eval.sh preflight \
  --suites internal,deepsdo --models all

# Freeze and audit canonical dataset records
bash scripts/runpod/run_paper_eval.sh prepare \
  --suites internal,deepsdo --models all

# Download pinned snapshots/checkpoints and verify revisions/hashes
bash scripts/runpod/run_paper_eval.sh download \
  --suites internal,deepsdo --models all

# Run a stratified subset through every selected backend
bash scripts/runpod/run_paper_eval.sh smoke \
  --suites internal,deepsdo --models all --resume

# Complete all missing generations
bash scripts/runpod/run_paper_eval.sh run \
  --suites internal,deepsdo --models all --resume

# GPU-free rescoring, bootstrap intervals, tables, and figures
bash scripts/runpod/run_paper_eval.sh analyze \
  --suites internal,deepsdo --models all

# GPU-free checksums and private/public archives
bash scripts/runpod/run_paper_eval.sh package \
  --suites internal,deepsdo --models all
```

Use `--dry-run` to inspect the planned paths and subprocesses. `--allow-dirty`, `--skip-hardware-check`, and `--diagnostic-allow-partial` are diagnostic flags, not settings for a definitive paper run. In particular, partial analysis is not acceptable evidence for the paper.

## Resume and failure recovery

The attempts log is append-only. A row becomes canonical only when its record fingerprint and effective model-generation fingerprint match and it has a non-empty, leak-free response. Missing, errored, empty, or decode-leaking rows remain pending; a later invocation with `--resume` retries only those rows. Successful real benchmark smoke rows are retained and reused in the full run. DeepSDO also runs three deterministic, clearly marked synthetic solar-like images through each backend; those outputs are stored under `smoke_fixtures/` and never enter the official 102-row prediction store or any score.

After correcting an interrupted download, out-of-space condition, transient Hub error, or generation error, rerun the same command:

```bash
bash scripts/runpod/run_paper_eval.sh \
  --suites internal,deepsdo --models all --resume
```

Do not delete `attempts.jsonl` to resume, and do not combine files from different fingerprint directories. Canonical reuse is bound to an effective generation fingerprint containing the suite generation contract, model/checkpoint contract, exact canonical-records hash, and generation implementation hash; resume also rejects a different locked software environment. A changed prompt, template policy, model revision, checkpoint, records file, or generation code therefore creates a different output path and cannot silently reuse an incompatible row. Analysis-only changes, such as a scorer or plotting change, alter the analysis/report fingerprint but deliberately reuse the retained generations, so they do not require another GPU run. By default each technical failure is attempted twice; increase `--max-attempts` only when a transient backend problem warrants it.

AstroVLBench label-parser failures are different from technical generation failures: they are retained as incorrect model answers and included in the invalid-response rate, not retried or dropped.

## AstroVLBench after approval

AstroVLBench is gated. After the dataset owner approves access, create a Hugging Face token with permission to read the dataset and accept any access conditions on the dataset page. First resolve and lock the exact snapshot:

```bash
HF_TOKEN=... bash scripts/runpod/run_paper_eval.sh --lock-astrovlbench
```

The lock records the resolved commit, every required file hash, and the official guided-prompt hashes. The adapter refuses to infer without a complete lock. It also reports, rather than hides, any discrepancy between the locked files and counts stated in the public summary.

Then run the independently hashed suite:

```bash
HF_TOKEN=... bash scripts/runpod/run_paper_eval.sh \
  --suites astrovlbench \
  --models all \
  --resume
```

The protocol evaluates image modality only, zero-shot, with all five official tasks and the official guided prompts. FIRST and NVSS are kept separate in Task 2; Q1, Q2, and Q3 are kept separate in Task 5. Task 3 receives no numerical table or added redshift text. The expected count is generated from the locked release rather than hard-coded because the public summary has an apparent FIRST/total-count inconsistency.

## Dataset safeguards and caveats

### Internal held-out set

- The existing 2% image-level test split is frozen; captions and QA turns for an image remain in the same split.
- The exact required totals are 3,271 records, 591 images, 586 captions, and 2,685 QA examples.
- Preparation reports exact-file, decoded-pixel, and perceptual-hash train/test overlap. pHash distance `<=4` is flagged as a likely duplicate and `<=8` as a sensitivity candidate. Inspection does not trigger an automatic resplit.
- A completed leakage audit is reused on `--resume` only when the pinned train/test manifests, image file state, thresholds, and audit implementation fingerprint still match; otherwise it is recomputed.
- The lineage output separates captions, QA, documented GPT-4-generated answers, and rows whose provenance cannot be established. Foundation-model pretraining overlap cannot be exhaustively audited.
- Internal caption and QA results are reported separately. Stage 2 minus Stage 1 intervals cluster all records belonging to the same image.

### DeepSDO

- DeepSDO is a public, evaluation-only solar caption dataset. Only its official 102-image test split is evaluated; no DeepSDO example is used for training, prompt tuning, or checkpoint selection.
- The archive must be exactly 36,555,441 bytes with SHA-256 `508382874f62510add0ce925a35fe51d58e0673b79ac559ebcfcae84adfd139e`.
- The fixed zero-shot prompt is `Describe this solar image.` and each model generates at most 128 new tokens.
- DeepSDO is caption generation, not nine-class classification. The nine `topic_stratum` values are derived descriptive groups, so topic accuracy and macro-F1 are not computed. Groups with fewer than 10 items are marked exploratory.
- Channel metadata are derived from filenames: 8 HMI-continuum, 46 AIA 304 Angstrom, and 48 other AIA-EUV items. Topic and wavelength are strongly confounded.
- One hundred of 102 normalized test captions also occur in the training annotations, frequently for nearby observations. This is disclosed as a split limitation even though the evaluated VLMs are zero-shot on DeepSDO.
- Three known upstream annotation anomalies occur on the training side and are recorded without guessed repairs; they do not affect the official test conversion.
- The supervised DeepSDO M2 result is not treated as an equivalent baseline for these zero-shot systems.

See [DeepSDO external caption benchmark](deepsdo_external_eval.md) for source, citation, credit, and redistribution details.

### AstroVLBench

- Access approval and `HF_TOKEN` are required. Gated source files, images, annotations, and labels are not placed in the public archive.
- Results are separated by task, survey, spectral question, and class. The project-defined five-task average is clearly distinguished from an official leaderboard score.
- Bootstrap sampling clusters repeated survey views and spectral questions by their underlying source object.

## Metrics and generated paper material

Internal captions use CIDEr, METEOR, ROUGE-L, BLEU-1 through BLEU-4, and SBERT cosine. Internal QA uses token-F1, exact match, ROUGE-L, and SBERT cosine. Caption metrics and QA ROUGE-L use the pinned `pycocoevalcap==1.2` COCO-caption pipeline with its PTB tokenizer; METEOR runs the package's pinned Java scorer. Corpus BLEU is recomputed for every bootstrap replicate from the retained PTB-tokenized strings rather than averaged from per-item BLEU values. NLI contradiction and reference-supported-specificity values are supplementary proxies, not proof of hallucination.

DeepSDO predeclares CIDEr as primary and reports BLEU-1 through BLEU-4, METEOR, and ROUGE-L overall and by the supported descriptive strata. AstroVLBench reports accuracy, macro-F1, balanced accuracy, per-class recall, confusion matrices, and invalid-response rate. Confidence intervals use 10,000 paired bootstrap replicates with seed 42 and the suite-specific clustering rules in the protocol.

Every final run writes Markdown, LaTeX, JSON, and CSV tables; PDF/SVG and 300-DPI PNG figures; per-sample results; absolute and paired-difference confidence intervals; manifests; and a cautious, number-filled `paper_results.md`.

## Output layout

Default paths are:

```text
datasets/paper_eval_v2/
  internal/                 frozen/rebuilt records, lineage, leakage audits
  deepsdo/                  resumable archive, extracted private data, canonical records
  astrovlbench/             gated snapshot lock, adapter report, canonical records
checkpoints/paper_eval_v2/
  asset_manifest.json       verified model/checkpoint locations and hashes
eval_runs/paper_eval_v2/
  preflight.json
  internal/<suite-generation-hash>/
    astraq_stage1/<effective-generation-hash>/
      attempts.jsonl, predictions.jsonl, run_manifest.json
    astraq_stage2/<effective-generation-hash>/
  deepsdo/<suite-generation-hash>/
    astraq_stage1/<effective-generation-hash>/
    astraq_stage2/<effective-generation-hash>/
    astrollava/<effective-generation-hash>/
    qwen3_vl_4b/<effective-generation-hash>/
  astrovlbench/<suite-generation-hash>/
    <model>/<effective-generation-hash>/
  reports/<analysis-run-hash>/  tables, figures, per-sample scores, paper_results.md
  audit_inputs/              protocol, environment, split/leakage/lineage evidence, reproduction commands
  bundles/                   archives, bundle_manifest.json
```

The suite hash covers only generation-affecting suite settings. The effective model hash additionally binds the exact canonical-records file and the relevant generation implementation. The analysis-run hash binds the selected suite set, suite-specific analysis contracts, canonical-record hashes, locked audit/adapter evidence, and analysis implementation, so a later AstroVLBench-only analysis cannot overwrite an internal/DeepSDO report. `CHECKSUMS.sha256` is generated for the report and staged audit trees and inside each archive. The protocol, suite, model, environment, records, prompts, predictions, and analysis inputs are independently hashed in their manifests.

## Private and public bundles

Packaging creates:

- `astraq-vl-paper-eval-v2-private.tar.gz`, containing the evaluation evidence under the output root, including full predictions/references and staged copies of the frozen protocol, requirement locks, preflight manifest, canonical record manifests, split/leakage/lineage audits, exact reproduction commands, and their checksums. It does not add model weights, raw dataset archives, or extracted image directories, which live outside the output root.
- `astraq-vl-paper-eval-v2-public-redacted.tar.gz`, excluding the staged dataset/source-evidence tree and raw/gated files, and removing references, source identifiers, image paths, gated benchmark IDs/hashes, annotations, and other sensitive fields from structured outputs. It retains shareable aggregate tables, figures, manifests, and redacted model-generation evidence.

Packaging refuses to run before successful analysis, verifies the current report's protocol fingerprint, stages only the current suite-generation directories and current report, and fails closed if a structured file cannot be safely redacted. Older protocol directories under the same output root are not mixed into either archive. The public archive is the paper-sharing artifact. The private archive is for auditing and should not be redistributed without checking every source dataset's terms. Both archive hashes are recorded in `bundles/bundle_manifest.json`.

## Acceptance gates

A final internal/DeepSDO run is complete only when both internal models have exactly 3,271 canonical rows and all four DeepSDO models have exactly 102. An AstroVLBench run requires every ID in the locked manifest exactly once for every model. There must be no missing, duplicate, technical-error, empty, or decode-leaking canonical rows. Parser-invalid AstroVLBench answers remain represented and count as failures.

After those gates pass, `analyze` and `package` can be rerun as often as needed without model loading or GPU inference.
