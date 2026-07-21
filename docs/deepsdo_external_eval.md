# DeepSDO external caption benchmark

DeepSDO Description is a public KASI release containing 1,051 human-captioned Solar Dynamics Observatory images. AstraQ-VL uses only its official 102-image test split as an out-of-distribution, zero-shot caption benchmark. DeepSDO is not used for training, prompt tuning, checkpoint selection, retrieval, or few-shot examples.

## Access, provenance, and redistribution

- Official page: <https://sdo.kasi.re.kr/dataset_deepsdo_description.aspx>
- Official archive: <http://swds.kasi.re.kr/sdo/kasi_deepsdo_desc_dataset.tar.gz>
- Verified archive size: 36,555,441 bytes
- SHA-256: `508382874f62510add0ce925a35fe51d58e0673b79ac559ebcfcae84adfd139e`
- Citation: Baek et al. (2024), *Deep learning-based solar image captioning*, *Advances in Space Research*, 73(6), 3270-3281.
- Required image credit: “Courtesy of NASA/SDO and the AIA, EVE, and HMI science teams.”

KASI serves the archive without an account or approval step, so it is publicly downloadable. KASI also states that the annotations and images belong to KDC/KASI. The raw archive and extracted images therefore remain under the Git-ignored `datasets/` tree and are excluded from the shareable evaluation bundle.

## Recommended paper workflow

DeepSDO is integrated into the frozen paper protocol. On a RunPod pod, run it with the internal evaluation:

```bash
bash scripts/runpod/run_paper_eval.sh \
  --suites internal,deepsdo \
  --models all \
  --resume
```

To run DeepSDO alone:

```bash
bash scripts/runpod/run_paper_eval.sh \
  --suites deepsdo \
  --models all \
  --resume
```

This prepares the data, downloads and verifies the four frozen model backends, runs smoke tests and full inference sequentially, validates all 102 rows per model, computes confidence intervals, produces paper tables/figures, and creates audit/shareable bundles. The four models are AstraQ-VL Stage 1, AstraQ-VL Stage 2, AstroLLaVA, and Qwen3-VL-4B. AstroLLaVA's CLIP-336 vision tower is separately revision-locked and loaded from the verified local snapshot. The runner never trains a missing checkpoint and never invokes the repository's RAG path.

See [Paper evaluation v2](paper_evaluation_v2.md) for staged commands, resume behavior, output layout, and acceptance gates.

## Standalone preparation and audit

The preparation stage uses an HTTP Range-resumable download with retry/backoff, checks both archive size and SHA-256, rejects unsafe archive paths, and converts only the official test split:

```bash
python scripts/prepare_deepsdo.py --download
```

If the archive was downloaded manually to the expected location, omit `--download`. The full paper orchestrator invokes this preparation automatically and writes canonical records under `datasets/paper_eval_v2/deepsdo/`.

The release contains 847 training, 102 validation, and 102 test annotations. Three known anomalies are confined to the training annotations: one malformed row and two annotation filenames without exact image-name matches. The adapter records these without inventing repairs; the official test split is complete and unaffected.

## Frozen generation and scoring protocol

- Prompt: `Describe this solar image.`
- Decoding: greedy, `do_sample=false`, one beam, seed 42
- Limit: 128 new tokens
- RAG/few-shot/quantization: disabled
- Primary metric: CIDEr
- Secondary metrics: BLEU-1 through BLEU-4, METEOR, and ROUGE-L
- Metric implementation: pinned `pycocoevalcap==1.2` COCO-caption scorers after COCO PTB tokenization
- METEOR runtime: Java (the RunPod wrapper installs OpenJDK 17 when permitted and otherwise fails explicitly)
- Uncertainty: paired 10,000-replicate bootstrap confidence intervals, seed 42

Corpus BLEU is recomputed within every bootstrap replicate rather than formed by averaging per-image BLEU. Every private prediction preserves its source ID, image and prompt hashes, reference hash, rendered native chat-template hash, raw and cleaned text, generated token IDs, token count, termination reason, latency, model/checkpoint revisions, and effective generation fingerprint. This makes all scoring and subgroup analysis repeatable offline without another GPU run.

## Descriptive strata, not class labels

DeepSDO is a caption-generation benchmark, not a nine-class classification dataset. The adapter derives the following `topic_stratum` groups only to summarize caption metrics:

| Derived topic stratum | n |
|---|---:|
| Sunspots | 8 |
| Flares | 18 |
| Prominences | 9 |
| Prominence eruptions | 14 |
| Coronal holes | 11 |
| Coronal loops | 12 |
| Filaments | 6 |
| Active regions | 11 |
| Eclipses/transits | 13 |

Topic accuracy and macro-F1 are not meaningful and are not calculated. Topic rows with `n<10` are marked exploratory. Per-topic summaries use CIDEr, METEOR, and ROUGE-L rather than headline BLEU values.

Instrument/channel metadata are derived separately from filenames:

| Channel grouping | n |
|---|---:|
| HMI continuum | 8 |
| AIA 304 Å | 46 |
| Other AIA EUV channels | 48 |

Individual AIA channels are retained in supplementary outputs. Topic and wavelength are strongly confounded, so category differences must not be interpreted as independent causal effects.

## Split and comparison limitations

One hundred of the 102 normalized test captions also appear in the training annotations, often for nearby observations. The paper output reports this upstream split limitation. It does not imply that the four evaluated models trained on DeepSDO: the suite remains zero-shot and the training/validation examples are not used by the runner.

DeepSDO provides a single reference caption per test image, which limits automatic metrics’ coverage of valid alternative descriptions. Absent details should not automatically be called hallucinations. The DeepSDO-trained M2 result published with the dataset is supervised on a different protocol and is not treated as a directly comparable zero-shot baseline.

DeepSDO results remain separate from AstroVLBench and are not folded into one cross-benchmark headline score.

## Audit and redistribution bundles

The private audit archive retains references plus staged canonical records, protocol/environment manifests, DeepSDO conversion reports, exact reproduction commands, and checksums. Raw archives, model weights, and extracted images remain outside the evaluation output root and are not added to that archive. The public paper archive excludes the staged dataset evidence and redacts references, source identifiers, paths, hashes tied to source records, and other non-redistributable fields while retaining aggregate tables, figures, manifests, and redacted model outputs. Packaging runs only after analysis succeeds and fails closed if a structured result cannot be safely redacted.
