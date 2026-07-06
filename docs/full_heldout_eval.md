# Full held-out evaluation

This workflow scores every held-out AstroLLaVA record, not only one caption per image.
For the released split that means 3,271 caption+QA records over 591 unseen images.

## Branches

- Stage 1: `releases/stg1-v2.0-full-heldout`
- Stage 2: `releases/stg2-v1.0-full-heldout`

These are evaluation-release branches. They do not define new model weights.

## Dry run

```bash
python scripts/run_full_heldout_eval.py --stage stage1 --dry-run --no-train-if-missing
python scripts/run_full_heldout_eval.py --stage stage2 --dry-run --no-train-if-missing
```

To validate record extraction without loading a model:

```bash
python scripts/generate_heldout_records.py \
  --records-json datasets/astrollava_llava/test.json \
  --image-dir datasets/astrollava_llava/images \
  --num-samples 5 \
  --dry-run
```

## Full Stage-1 run

Download or place the Stage-1 checkpoints under `checkpoints/astrollava-stage1/`:

- `checkpoint-1300`
- `checkpoint-2500`
- `checkpoint-3789`

Then run:

```bash
python scripts/run_full_heldout_eval.py \
  --stage stage1 \
  --num-samples 0 \
  --resume \
  --package
```

If a checkpoint is missing, the script trains with `configs/pretrain_astrollava.yaml`
and then re-checks the expected checkpoint names. Use `--no-train-if-missing` for an
eval-only run that fails fast instead.

## Full Stage-2 run

Download or place the Stage-2 checkpoint under `checkpoints/astrollava-stage2/checkpoint-2526`.
The checkpoint must include both `connector.safetensors` and `lora/adapter_model.safetensors`.

```bash
python scripts/run_full_heldout_eval.py \
  --stage stage2 \
  --num-samples 0 \
  --resume \
  --package
```

If the Stage-2 checkpoint is missing, the script trains with
`configs/finetune_astrollava_stage2.yaml` and then re-checks `checkpoint-2526`.

## Outputs

Outputs are written under `eval_runs/full_heldout/`:

- `stage1_ep1/`, `stage1_ep2/`, `stage1_ep3/`, or `stage2/`
- `predictions_full_heldout.jsonl`
- `metrics_full_heldout.json`
- `metrics_full_heldout.per_sample.jsonl`
- `comparison/full_heldout_comparison.{json,csv,md}`
- `astrollava-stage1-full-heldout-eval-v1.zip` or `astrollava-stage2-full-heldout-eval-v1.zip`

The metrics are produced by `scripts/score_predictions.py`: ROUGE-L, token-F1,
exact match, specificity hallucination, NLI consistency, contradiction rate, and
SBERT cosine. Summaries include `overall`, `splits.caption`, and `splits.qa`.
