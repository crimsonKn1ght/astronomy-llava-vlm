# AstroVLBench evaluation v1

This protocol evaluates the image-only, zero-shot guided-prompt portion of the
pinned gated AstroVLBench release. It is isolated from the completed DeepSDO v4
study and never trains, tunes, or retrieves examples for any model.

## Dataset preparation

Accept the dataset access conditions and either download it explicitly:

```bash
hf download XiaomanZhang/AstroVLBench \
  --repo-type dataset \
  --revision d1708958d4d1dda45c078eb2f4d6db3e6fa96286 \
  --local-dir datasets/paper_eval_astrovlbench_v1/astrovlbench/snapshot

bash scripts/runpod/run_paper_eval.sh prepare \
  --protocol configs/paper_eval_astrovlbench_v1.yaml \
  --suites astrovlbench --models all \
  --astrovlbench-snapshot datasets/paper_eval_astrovlbench_v1/astrovlbench/snapshot
```

or let the wrapper download the same immutable revision:

```bash
HF_TOKEN=... bash scripts/runpod/run_paper_eval.sh prepare \
  --protocol configs/paper_eval_astrovlbench_v1.yaml \
  --suites astrovlbench --models all --lock-astrovlbench
```

Preparation hashes the complete raw snapshot, statically extracts the official
prompts, writes a derived correction overlay, and verifies the raw inventory
again. The upstream snapshot is never edited. The FIRST overlay retains 605
readable images and records 228 exclusions: 227 missing files and one zero-byte
PNG. NVSS images are never substituted for missing FIRST observations.

The canonical component counts are:

| Component | Records |
|---|---:|
| Task 1 | 557 |
| Task 2 FIRST | 605 |
| Task 2 NVSS | 833 |
| Task 3 | 168 |
| Task 4 | 142 |
| Task 5 Q1 | 700 |
| Task 5 Q2 | 500 |
| Task 5 Q3 | 400 |
| Total | 3,905 |

## Generation and analysis

```bash
bash scripts/runpod/run_paper_eval.sh download --protocol configs/paper_eval_astrovlbench_v1.yaml --suites astrovlbench --models all
bash scripts/runpod/run_paper_eval.sh smoke   --protocol configs/paper_eval_astrovlbench_v1.yaml --suites astrovlbench --models all --resume
bash scripts/runpod/run_paper_eval.sh run     --protocol configs/paper_eval_astrovlbench_v1.yaml --suites astrovlbench --models all --resume
bash scripts/runpod/run_paper_eval.sh analyze --protocol configs/paper_eval_astrovlbench_v1.yaml --suites astrovlbench --models all
bash scripts/runpod/run_paper_eval.sh package --protocol configs/paper_eval_astrovlbench_v1.yaml --suites astrovlbench --models all
```

Smoke selects one record from each of the eight task components. A token-cap
row proves only that the backend ran; it remains a protocol failure and must not
appear in the definitive run. Each model must finish with exactly 3,905
successful rows and no missing, failed, extra, duplicate, or capped rows.

Official system and user prompt text are extracted without importing upstream
code and combined with the frozen `official-system-plus-user-v1` separator.
Task 3 receives no numeric redshift or catalog photometry. Invalid or ambiguous
model labels are retained as incorrect answers rather than retried or discarded.
