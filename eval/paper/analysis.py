"""Offline-only scoring, clustered confidence intervals, and paper artifacts."""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

from .artifacts import (
    ArtifactError,
    checksum_tree,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .bootstrap import bootstrap_ci, paired_bootstrap_ci, percentile
from .metrics import (
    classification_metrics,
    coco_bleu_compatible_scores,
    exact_match,
    optional_caption_metrics,
    rouge_l,
    token_f1,
    valid_response_rate,
)
from .protocol import PaperProtocol, sha256_json
from .reporting import (
    cautious_comparison,
    plot_estimates,
    plot_heatmap,
    write_table_bundle,
)


class AnalysisError(RuntimeError):
    pass


def _unit_key(suite: str, condition_id: str | None) -> str:
    return suite if condition_id is None else f"{suite}:{condition_id}"


def _records_path(data_root: str | Path, suite: str, condition_id: str | None) -> Path:
    root = Path(data_root) / suite
    if condition_id is None:
        return root / "records.jsonl"
    return root / "conditions" / condition_id / "records.jsonl"


def analysis_evidence_hashes(
    data_root: str | Path, suites: Sequence[str]
) -> Dict[str, str]:
    root = Path(data_root)
    files: set[Path] = set()
    if "internal" in suites:
        internal = root / "internal"
        for name in ("manifest.json", "split_manifest.csv", "data_lineage.json"):
            if (internal / name).is_file():
                files.add(internal / name)
        leakage = internal / "leakage_audit"
        if leakage.is_dir():
            files.update(path for path in leakage.rglob("*") if path.is_file())
    if "deepsdo" in suites:
        llava = root / "deepsdo" / "llava"
        if llava.is_dir():
            files.update(path for path in llava.glob("*audit*.json") if path.is_file())
    if "astrovlbench" in suites:
        astro = root / "astrovlbench"
        for name in ("astrovlbench.lock.json", "adapter_report.json"):
            if (astro / name).is_file():
                files.add(astro / name)
    return {
        path.resolve().relative_to(root.resolve()).as_posix(): sha256_file(path)
        for path in sorted(files, key=lambda item: item.as_posix())
    }


def analysis_run_fingerprint(
    protocol: PaperProtocol,
    suites: Sequence[str],
    record_hashes: Mapping[str, str],
    data_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    conditions: Mapping[str, Sequence[str | None]] | None = None,
) -> str:
    ordered_suites = tuple(sorted(str(suite) for suite in suites))
    condition_map = conditions or {
        suite: list(protocol.condition_ids(suite)) for suite in ordered_suites
    }
    expected_units = {
        _unit_key(suite, condition_id)
        for suite in ordered_suites
        for condition_id in condition_map[suite]
    }
    if expected_units != set(record_hashes):
        raise AnalysisError("Analysis record hashes do not match the selected suite/condition set")
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    implementation_paths = {
        "eval/paper/analysis.py",
        "eval/paper/artifacts.py",
        "eval/paper/bootstrap.py",
        "eval/paper/metrics.py",
        "eval/paper/reporting.py",
    }
    if "internal" in ordered_suites:
        implementation_paths.update({"eval/metrics_nli.py", "scripts/score_predictions.py"})
    if "astrovlbench" in ordered_suites:
        implementation_paths.add("eval/paper/astrovlbench.py")
    implementation = {}
    for relative in sorted(implementation_paths):
        path = root / relative
        if not path.is_file():
            raise AnalysisError(f"Analysis implementation file is missing: {path}")
        implementation[relative] = sha256_file(path)
    return sha256_json(
        {
            "schema_version": protocol.data["schema_version"],
            "study_id": protocol.study_id,
            "suites": list(ordered_suites),
            "suite_analysis_hashes": {
                _unit_key(suite, condition_id): protocol.analysis_fingerprint(
                    suite, condition_id
                )
                for suite in ordered_suites
                for condition_id in condition_map[suite]
            },
            "record_hashes": {
                key: record_hashes[key] for key in sorted(record_hashes)
            },
            "evidence_hashes": analysis_evidence_hashes(data_root, ordered_suites),
            "implementation": implementation,
        }
    )


def report_output_dir(
    protocol: PaperProtocol,
    suites: Sequence[str],
    output_root: str | Path,
    data_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    conditions: Mapping[str, Sequence[str | None]] | None = None,
) -> Path:
    condition_map = conditions or {
        suite: list(protocol.condition_ids(suite)) for suite in suites
    }
    hashes = {}
    for suite in suites:
        for condition_id in condition_map[suite]:
            records_path = _records_path(data_root, suite, condition_id)
            if not records_path.is_file():
                raise AnalysisError(f"Missing canonical records for {suite}: {records_path}")
            hashes[_unit_key(str(suite), condition_id)] = sha256_file(records_path)
    fingerprint = analysis_run_fingerprint(
        protocol,
        suites,
        hashes,
        data_root,
        repo_root=repo_root,
        conditions=condition_map,
    )
    return Path(output_root) / "reports" / fingerprint[:16]


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def _model_predictions(
    protocol: PaperProtocol,
    suite: str,
    output_root: Path,
    model_label: str,
    expected_records: Sequence[Mapping[str, Any]],
    records_sha256: str,
    *,
    paper_mode: bool,
    condition_id: str | None = None,
) -> list[Dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[2]
    model_root = protocol.model_output_dir(
        suite, model_label, records_sha256, repo_root, output_root, condition_id
    )
    path = model_root / "predictions.jsonl"
    manifest_path = model_root / "run_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise AnalysisError(f"Missing completed {suite}/{model_label} prediction artifacts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = protocol.effective_model_fingerprint(
        suite, model_label, records_sha256, repo_root, condition_id
    )
    if manifest.get("model_protocol_hash") != expected_hash:
        raise AnalysisError(f"{suite}/{model_label} manifest protocol hash mismatch")
    if manifest.get("records_file_sha256") != records_sha256:
        raise AnalysisError(f"{suite}/{model_label} canonical records hash mismatch")
    if manifest.get("condition_id") != condition_id:
        raise AnalysisError(f"{suite}/{model_label} condition ID mismatch")
    if manifest.get("predictions_sha256") != sha256_file(path):
        raise AnalysisError(f"{suite}/{model_label} predictions checksum mismatch")
    if paper_mode and not (manifest.get("completion") or {}).get("complete"):
        raise AnalysisError(f"{suite}/{model_label} is not technically complete")
    rows = read_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        row_id = str(row.get("id") or "")
        if row_id in by_id:
            duplicates.append(row_id)
        by_id[row_id] = row
    expected_ids = [str(row["id"]) for row in expected_records]
    missing = [row_id for row_id in expected_ids if row_id not in by_id]
    extra = sorted(set(by_id) - set(expected_ids))
    if paper_mode and (missing or extra or duplicates):
        raise AnalysisError(
            f"{suite}/{model_label} ID validation failed: missing={len(missing)} "
            f"extra={len(extra)} duplicates={len(duplicates)}"
        )
    aligned: list[Dict[str, Any]] = []
    for record in expected_records:
        row_id = str(record["id"])
        prediction = by_id.get(row_id)
        if prediction is None:
            aligned.append(
                {
                    **dict(record),
                    "model_label": model_label,
                    "response": "",
                    "status": "missing",
                    "error": {"type": "MissingPrediction"},
                }
            )
        else:
            aligned.append({**dict(record), **prediction})
    return aligned


def _score_device() -> str:
    configured = os.environ.get("PAPER_SCORE_DEVICE")
    if configured:
        return configured
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _attach_coco_tokenization(
    rows: Sequence[MutableMapping[str, Any]], optional: Mapping[str, Mapping[str, Any]]
) -> bool:
    prepared = optional.get("_coco_tokenization") or {}
    if prepared.get("status") != "ok":
        return False
    ground_truth = prepared["ground_truth"]
    results = prepared["results"]
    for index, row in enumerate(rows):
        key: Any = index if index in results else str(index)
        row["coco_tokenized_prediction"] = str(results[key][0])
        row["coco_tokenized_references"] = [str(value) for value in ground_truth[key]]
    return True


def score_caption_rows(
    rows: Sequence[Mapping[str, Any]],
    protocol: PaperProtocol,
    *,
    paper_mode: bool,
    include_sbert: bool = True,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    scored: list[Dict[str, Any]] = []
    predictions: list[str] = []
    references: list[str] = []
    for row in rows:
        response = str(row.get("response") or "") if row.get("status") == "ok" else ""
        reference = str(row.get("reference") or "")
        predictions.append(response)
        references.append(reference)
        scored.append(
            {
                **dict(row),
                "prediction": response,
                "rouge_l": rouge_l(response, reference),
                "token_f1": token_f1(response, reference),
                "exact_match": exact_match(response, reference),
            }
        )
    requested = ["bleu", "cider", "meteor", "rouge"] + (["sbert"] if include_sbert else [])
    optional = optional_caption_metrics(
        predictions,
        references,
        metrics=requested,
        paper_mode=paper_mode,
        pins=protocol.data.get("metric_packages", {}),
        sbert_model=protocol.data["scorers"]["sbert"]["repo_id"],
        sbert_revision=protocol.data["scorers"]["sbert"]["revision"],
        sbert_device=_score_device(),
    )
    has_coco_tokens = _attach_coco_tokenization(scored, optional)
    for metric_name in ("cider", "meteor", "rouge", "sbert"):
        result = optional.get(metric_name)
        if result and result.get("status") == "ok":
            key = {"sbert": "sbert", "rouge": "rouge_l"}.get(metric_name, metric_name)
            for row, value in zip(scored, result["per_item"]):
                row[key] = float(value)
    bleu_predictions = (
        [str(row["coco_tokenized_prediction"]) for row in scored]
        if has_coco_tokens else predictions
    )
    bleu_references: list[str | Sequence[str]] = (
        [list(row["coco_tokenized_references"]) for row in scored]
        if has_coco_tokens else references
    )
    compatible_bleu = coco_bleu_compatible_scores(bleu_predictions, bleu_references)
    package_bleu = optional.get("bleu", {})
    if package_bleu.get("status") == "ok":
        bleu = {key: float(value) for key, value in package_bleu["value"].items()}
        for key, value in compatible_bleu.items():
            if not math.isclose(float(bleu[key]), float(value), rel_tol=1e-9, abs_tol=1e-12):
                raise AnalysisError(
                    f"Pinned COCO {key}={bleu[key]} disagrees with the bootstrap-compatible "
                    f"implementation ({value})"
                )
        for key, values in package_bleu["per_item"].items():
            for row, value in zip(scored, values):
                row[key] = float(value)
    else:
        # Diagnostic mode remains usable without optional dependencies. Paper
        # mode cannot reach this branch because the pinned scorer is mandatory.
        bleu = compatible_bleu
    aggregate: Dict[str, Any] = {
        "n": len(scored),
        **valid_response_rate(scored, response_key="prediction", leak_key="leak_flags"),
        "rouge_l": optional.get("rouge", {}).get("value")
        if optional.get("rouge", {}).get("status") == "ok"
        else _mean(row["rouge_l"] for row in scored),
        "token_f1": _mean(row["token_f1"] for row in scored),
        **bleu,
        "cider": optional.get("cider", {}).get("value"),
        "meteor": optional.get("meteor", {}).get("value"),
        "sbert": optional.get("sbert", {}).get("value"),
        "token_cap_rate": _mean(float(bool(row.get("token_cap_hit"))) for row in scored),
        "optional_scorer_details": {
            name: {key: value for key, value in result.items() if key != "per_item"}
            for name, result in optional.items()
            if not name.startswith("_")
        },
    }
    return scored, aggregate


def score_qa_rows(
    rows: Sequence[Mapping[str, Any]],
    protocol: PaperProtocol,
    *,
    paper_mode: bool,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    scored: list[Dict[str, Any]] = []
    predictions: list[str] = []
    references: list[str] = []
    for row in rows:
        response = str(row.get("response") or "") if row.get("status") == "ok" else ""
        reference = str(row.get("reference") or "")
        predictions.append(response)
        references.append(reference)
        scored.append(
            {
                **dict(row),
                "prediction": response,
                "token_f1": token_f1(response, reference),
                "exact_match": exact_match(response, reference),
                "rouge_l": rouge_l(response, reference),
            }
        )
    semantic_results = optional_caption_metrics(
        predictions,
        references,
        metrics=["rouge", "sbert"],
        paper_mode=paper_mode,
        pins=protocol.data.get("metric_packages", {}),
        sbert_model=protocol.data["scorers"]["sbert"]["repo_id"],
        sbert_revision=protocol.data["scorers"]["sbert"]["revision"],
        sbert_device=_score_device(),
    )
    _attach_coco_tokenization(scored, semantic_results)
    rouge_result = semantic_results["rouge"]
    if rouge_result.get("status") == "ok":
        for row, value in zip(scored, rouge_result["per_item"]):
            row["rouge_l"] = float(value)
    semantic = semantic_results["sbert"]
    if semantic.get("status") == "ok":
        for row, value in zip(scored, semantic["per_item"]):
            row["sbert"] = float(value)
    aggregate = {
        "n": len(scored),
        **valid_response_rate(scored, response_key="prediction", leak_key="leak_flags"),
        "token_f1": _mean(row["token_f1"] for row in scored),
        "exact_match": _mean(row["exact_match"] for row in scored),
        "rouge_l": rouge_result.get("value")
        if rouge_result.get("status") == "ok"
        else _mean(row["rouge_l"] for row in scored),
        "sbert": semantic.get("value"),
        "token_cap_rate": _mean(float(bool(row.get("token_cap_hit"))) for row in scored),
    }
    return scored, aggregate


def add_internal_supplementary_metrics(
    rows: list[Dict[str, Any]], protocol: PaperProtocol, *, paper_mode: bool
) -> Dict[str, Any]:
    """Add NLI and specificity proxies over valid outputs only."""

    valid = [row for row in rows if row.get("status") == "ok" and str(row.get("prediction") or "").strip()]
    result: Dict[str, Any] = {"n_valid": len(valid), "n": len(rows)}
    try:
        from scripts.score_predictions import specificity_row

        for row in valid:
            row.update(specificity_row(str(row["prediction"]), str(row["reference"])))
        result["specificity_hallucination_proxy"] = _mean(
            float(row.get("has_specificity_hallucination", False)) for row in valid
        )
        result["unsupported_specifics_per_record_proxy"] = _mean(
            float(row.get("unsupported_specific_count", 0)) for row in valid
        )
    except Exception as exc:
        if paper_mode:
            raise AnalysisError(f"Specificity proxy scoring failed: {exc}") from exc
        result["specificity_error"] = repr(exc)
    try:
        from eval.metrics_nli import NLIScorer

        cfg = protocol.data["scorers"]["nli"]
        scorer = NLIScorer(cfg["repo_id"], _score_device(), revision=cfg["revision"])
        for row in valid:
            value = scorer.consistency(str(row["reference"]), str(row["prediction"]))
            row["nli"] = float(value)
            row["contradiction_proxy"] = float(value < 0)
        result["nli_consistency"] = _mean(row["nli"] for row in valid)
        result["contradiction_rate_proxy"] = _mean(row["contradiction_proxy"] for row in valid)
    except Exception as exc:
        if paper_mode:
            raise AnalysisError(f"Pinned NLI scoring failed: {exc}") from exc
        result["nli_error"] = repr(exc)
    return result


def _statistic(metric: str) -> Callable[[Sequence[Mapping[str, Any]]], float]:
    if metric.startswith("bleu_"):
        order = int(metric.rsplit("_", 1)[1])
        def bleu_statistic(sample: Sequence[Mapping[str, Any]]) -> float:
            usable = [row for row in sample if not row.get("_bootstrap_placeholder")]
            if not usable:
                return 0.0
            return coco_bleu_compatible_scores(
                [
                    str(row.get("coco_tokenized_prediction") or row.get("prediction") or "")
                    for row in usable
                ],
                [
                    row.get("coco_tokenized_references")
                    or str(row.get("reference") or "")
                    for row in usable
                ],
                max_order=order,
            )[metric]

        return bleu_statistic
    return lambda sample: _mean(
        float(row.get(metric, 0.0) or 0.0)
        for row in sample
        if not row.get("_bootstrap_placeholder")
    )


def _with_cluster_universe(
    rows: Sequence[Mapping[str, Any]],
    cluster_key: str | None,
    cluster_universe: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    materialized = list(rows)
    if not cluster_key or cluster_universe is None:
        return materialized
    present = {str(row[cluster_key]) for row in materialized}
    for cluster in cluster_universe:
        cluster_id = str(cluster)
        if cluster_id not in present:
            materialized.append(
                {
                    "id": f"__bootstrap_placeholder__:{cluster_id}",
                    cluster_key: cluster_id,
                    "_bootstrap_placeholder": True,
                }
            )
    return materialized


def metric_intervals(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    protocol: PaperProtocol,
    *,
    cluster_key: str | None,
    model_label: str,
    suite: str,
    task: str,
    cluster_universe: Sequence[str] | None = None,
) -> list[Dict[str, Any]]:
    settings = protocol.data["statistics"]
    result = []
    bootstrap_rows = _with_cluster_universe(rows, cluster_key, cluster_universe)
    for metric in metrics:
        if not rows or (metric not in rows[0] and not metric.startswith("bleu_")):
            continue
        interval = bootstrap_ci(
            bootstrap_rows,
            _statistic(metric),
            n_resamples=int(settings["bootstrap_replicates"]),
            seed=int(settings["seed"]),
            confidence=float(settings["confidence_level"]),
            resampling_unit="cluster" if cluster_key else "record",
            cluster_key=cluster_key,
        )
        result.append(
            {
                "suite": suite,
                "task": task,
                "model": model_label,
                "metric": metric,
                **interval.as_dict(),
            }
        )
    return result


def paired_metric_intervals(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    protocol: PaperProtocol,
    *,
    cluster_key: str | None,
    left_label: str,
    right_label: str,
    suite: str,
    task: str,
    cluster_universe: Sequence[str] | None = None,
) -> list[Dict[str, Any]]:
    right_by_id = {str(row["id"]): row for row in right}
    left_aligned = [row for row in left if str(row["id"]) in right_by_id]
    right_aligned = [right_by_id[str(row["id"])] for row in left_aligned]
    if len(left_aligned) != len(left) or len(right_aligned) != len(right):
        raise AnalysisError(f"Cannot pair {left_label} and {right_label}: record IDs differ")
    left_aligned = _with_cluster_universe(left_aligned, cluster_key, cluster_universe)
    right_aligned = _with_cluster_universe(right_aligned, cluster_key, cluster_universe)
    settings = protocol.data["statistics"]
    result = []
    for metric in metrics:
        if not left_aligned or (metric not in left_aligned[0] and not metric.startswith("bleu_")):
            continue
        stat = _statistic(metric)
        interval = paired_bootstrap_ci(
            left_aligned,
            right_aligned,
            lambda a, b, fn=stat: fn(a) - fn(b),
            n_resamples=int(settings["bootstrap_replicates"]),
            seed=int(settings["seed"]),
            confidence=float(settings["confidence_level"]),
            resampling_unit="cluster" if cluster_key else "record",
            cluster_key=cluster_key,
        )
        result.append(
            {
                "suite": suite,
                "task": task,
                "target": left_label,
                "baseline": right_label,
                "metric": metric,
                **interval.as_dict(),
            }
        )
    return result


def _wide_row(model: str, task: str, aggregate: Mapping[str, Any], metrics: Sequence[str]) -> Dict[str, Any]:
    return {
        "model": model,
        "task": task,
        "n": aggregate.get("n"),
        "valid_response_rate": aggregate.get("valid_response_rate"),
        "token_cap_rate": aggregate.get("token_cap_rate"),
        **{metric: aggregate.get(metric) for metric in metrics},
    }


def analyze_internal(
    protocol: PaperProtocol,
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
    report_root: Path,
    records_sha256: str,
    *,
    paper_mode: bool,
) -> Dict[str, Any]:
    models = protocol.selected_models("internal")
    scored_by_model_task: Dict[str, Dict[str, list[Dict[str, Any]]]] = defaultdict(dict)
    summary_rows: list[Dict[str, Any]] = []
    ci_rows: list[Dict[str, Any]] = []
    all_per_sample: list[Dict[str, Any]] = []
    internal_image_universe = sorted({str(row["image_id"]) for row in records})
    caption_metrics = ["cider", "meteor", "rouge_l", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "sbert"]
    qa_metrics = ["token_f1", "exact_match", "rouge_l", "sbert"]
    supplementary: Dict[str, Any] = {}
    for model_label in models:
        predictions = _model_predictions(protocol, "internal", output_root, model_label, records, records_sha256, paper_mode=paper_mode)
        caption_raw = [row for row in predictions if row["record_type"] == "caption"]
        qa_raw = [row for row in predictions if row["record_type"] == "qa"]
        captions, caption_summary = score_caption_rows(caption_raw, protocol, paper_mode=paper_mode)
        qa, qa_summary = score_qa_rows(qa_raw, protocol, paper_mode=paper_mode)
        supplementary[model_label] = {
            "caption": add_internal_supplementary_metrics(captions, protocol, paper_mode=paper_mode),
            "qa": add_internal_supplementary_metrics(qa, protocol, paper_mode=paper_mode),
        }
        scored_by_model_task[model_label] = {"caption": captions, "qa": qa}
        all_per_sample.extend({**row, "analysis_task": "caption"} for row in captions)
        all_per_sample.extend({**row, "analysis_task": "qa"} for row in qa)
        summary_rows.append(_wide_row(model_label, "caption", caption_summary, caption_metrics))
        summary_rows.append(_wide_row(model_label, "qa", qa_summary, qa_metrics))
        ci_rows.extend(metric_intervals(captions, caption_metrics, protocol, cluster_key="image_id", model_label=model_label, suite="internal", task="caption", cluster_universe=internal_image_universe))
        ci_rows.extend(metric_intervals(qa, qa_metrics, protocol, cluster_key="image_id", model_label=model_label, suite="internal", task="qa_micro", cluster_universe=internal_image_universe))
        combined = [*captions, *qa]
        supplementary[model_label]["mixed_caption_qa"] = {
            "n": len(combined),
            "n_images": len(internal_image_universe),
            **valid_response_rate(combined, response_key="prediction", leak_key="leak_flags"),
            "rouge_l": _mean(float(row.get("rouge_l", 0) or 0) for row in combined),
            "sbert": _mean(float(row.get("sbert", 0) or 0) for row in combined),
            "token_cap_rate": _mean(float(bool(row.get("token_cap_hit"))) for row in combined),
            "reporting_role": "supplementary_only",
        }
        # Supplemental image-macro QA estimate: aggregate record metrics per image first.
        by_image: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in qa:
            by_image[str(row["image_id"])].append(row)
        macro_rows = [
            {"image_id": image, **{metric: _mean(float(row.get(metric, 0) or 0) for row in group) for metric in qa_metrics}}
            for image, group in by_image.items()
        ]
        ci_rows.extend(metric_intervals(macro_rows, qa_metrics, protocol, cluster_key=None, model_label=model_label, suite="internal", task="qa_image_macro"))
    delta_rows: list[Dict[str, Any]] = []
    for target, baseline in protocol.data["statistics"]["predeclared_comparisons"]["internal"]:
        for task, metrics in (("caption", caption_metrics), ("qa", qa_metrics)):
            delta_rows.extend(
                paired_metric_intervals(
                    scored_by_model_task[target][task],
                    scored_by_model_task[baseline][task],
                    metrics,
                    protocol,
                    cluster_key="image_id",
                    left_label=target,
                    right_label=baseline,
                    suite="internal",
                    task=task,
                    cluster_universe=internal_image_universe,
                )
            )
    suite_root = report_root / "internal"
    write_table_bundle(suite_root / "internal_results", summary_rows, caption="Internal caption and QA results", label="tab:internal-results", note="Caption and QA records are reported separately; mixed-task aggregates are not headline results.")
    write_table_bundle(suite_root / "internal_absolute_ci", ci_rows, caption="Image-cluster bootstrap confidence intervals", label="tab:internal-ci")
    write_table_bundle(suite_root / "internal_paired_differences", delta_rows, caption="Paired Stage 2 minus Stage 1 confidence intervals", label="tab:internal-deltas")
    write_jsonl_atomic(suite_root / "internal_per_sample.jsonl", all_per_sample)
    write_json_atomic(suite_root / "internal_supplementary_proxies.json", supplementary)
    write_table_bundle(
        suite_root / "internal_mixed_supplementary",
        [
            {"model": model, **values["mixed_caption_qa"]}
            for model, values in supplementary.items()
        ],
        caption="Supplementary mixed caption and QA summary",
        label="tab:internal-mixed-supplementary",
        note="This heterogeneous aggregate is supplementary only; caption and QA tables are the primary internal reporting.",
    )
    for task, metric in (("caption", "cider"), ("qa", "token_f1")):
        plot_rows = [
            {"label": row["model"], "estimate": row["estimate"], "lower": row["lower"], "upper": row["upper"]}
            for row in ci_rows if row["task"] == task and row["metric"] == metric
        ]
        if plot_rows:
            plot_estimates(plot_rows, suite_root / f"internal_{task}_{metric}", title=f"Internal {task}: {metric}", ylabel=metric, paper_mode=paper_mode)
    return {"summary": summary_rows, "absolute_ci": ci_rows, "paired_ci": delta_rows, "scored": scored_by_model_task}


def _subgroup_table(
    scored: Mapping[str, Sequence[Mapping[str, Any]]], group_key: str
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for model, model_rows in scored.items():
        groups: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in model_rows:
            groups[str(row.get(group_key) or "unknown")].append(row)
        for group, values in sorted(groups.items()):
            rows.append(
                {
                    "model": model,
                    group_key: group,
                    "n": len(values),
                    "small_sample": len(values) < 10,
                    "cider": _mean(float(row.get("cider", 0) or 0) for row in values),
                    "meteor": _mean(float(row.get("meteor", 0) or 0) for row in values),
                    "rouge_l": _mean(float(row.get("rouge_l", 0) or 0) for row in values),
                }
            )
    return rows


def _apply_holm_adjustment(rows: Sequence[MutableMapping[str, Any]]) -> None:
    """Add Holm-adjusted p-values to predeclared comparisons, per metric."""

    grouped: Dict[str, list[MutableMapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("headline_comparison") and row.get("two_sided_p_value") is not None:
            grouped[str(row.get("metric"))].append(row)
        else:
            row["holm_adjusted_p_value"] = None
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: float(row["two_sided_p_value"]))
        running = 0.0
        total = len(ordered)
        for index, row in enumerate(ordered):
            adjusted = min(
                1.0,
                (total - index) * float(row["two_sided_p_value"]),
            )
            running = max(running, adjusted)
            row["holm_adjusted_p_value"] = running


def _deepsdo_completion_diagnostics(
    predictions_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    condition_id: str | None,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for model_label, predictions in predictions_by_model.items():
        responses = [str(row.get("response") or "").strip() for row in predictions]
        token_counts = [int(row.get("completion_token_count") or 0) for row in predictions]
        word_counts = [len(response.split()) for response in responses]
        reference_words = sum(
            len(str(row.get("reference") or "").split()) for row in predictions
        )
        normalized = [re.sub(r"\s+", " ", response.casefold()).strip() for response in responses]
        repeated = sum(count - 1 for count in Counter(normalized).values() if count > 1)
        termination = Counter(
            "model_stop" if row.get("termination_reason") == "stopped" else str(row.get("termination_reason") or "unknown")
            for row in predictions
        )
        unfinished = sum(
            bool(row.get("token_cap_hit"))
            or bool(response and not re.search(r"[.!?\]\)]$", response))
            for row, response in zip(predictions, responses)
        )
        rows.append(
            {
                "condition_id": condition_id,
                "model": model_label,
                "n": len(predictions),
                "eos_count": termination.get("eos", 0),
                "model_stop_count": termination.get("model_stop", 0),
                "max_new_tokens_count": termination.get("max_new_tokens", 0),
                "unknown_stop_count": termination.get("unknown", 0),
                "token_cap_hits": sum(bool(row.get("token_cap_hit")) for row in predictions),
                "empty_outputs": sum(not response for response in responses),
                "malformed_outputs": sum(not isinstance(row.get("response"), str) for row in predictions),
                "repeated_outputs_beyond_first": repeated,
                "possible_unfinished_outputs": unfinished,
                "completion_tokens_min": min(token_counts) if token_counts else 0,
                "completion_tokens_median": percentile(token_counts, 0.5) if token_counts else 0,
                "completion_tokens_p95": percentile(token_counts, 0.95) if token_counts else 0,
                "completion_tokens_max": max(token_counts) if token_counts else 0,
                "words_median": percentile(word_counts, 0.5) if word_counts else 0,
                "words_p95": percentile(word_counts, 0.95) if word_counts else 0,
                "candidate_reference_word_ratio": (
                    sum(word_counts) / reference_words if reference_words else None
                ),
                "peak_gpu_memory_allocated_bytes": max(
                    (int(row.get("peak_gpu_memory_allocated_bytes") or 0) for row in predictions),
                    default=0,
                ),
                "peak_gpu_memory_reserved_bytes": max(
                    (int(row.get("peak_gpu_memory_reserved_bytes") or 0) for row in predictions),
                    default=0,
                ),
            }
        )
    return rows


def analyze_deepsdo(
    protocol: PaperProtocol,
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
    report_root: Path,
    records_sha256: str,
    *,
    paper_mode: bool,
    condition_id: str | None = None,
) -> Dict[str, Any]:
    scored: Dict[str, list[Dict[str, Any]]] = {}
    aggregates: Dict[str, Dict[str, Any]] = {}
    summary_rows: list[Dict[str, Any]] = []
    ci_rows: list[Dict[str, Any]] = []
    metrics = ["cider", "meteor", "rouge_l", "bleu_1", "bleu_2", "bleu_3", "bleu_4"]
    all_per_sample: list[Dict[str, Any]] = []
    predictions_by_model: Dict[str, list[Dict[str, Any]]] = {}
    for model_label in protocol.selected_models("deepsdo"):
        predictions = _model_predictions(
            protocol,
            "deepsdo",
            output_root,
            model_label,
            records,
            records_sha256,
            paper_mode=paper_mode,
            condition_id=condition_id,
        )
        predictions_by_model[model_label] = predictions
        rows, aggregate = score_caption_rows(predictions, protocol, paper_mode=paper_mode, include_sbert=False)
        scored[model_label] = rows
        aggregates[model_label] = aggregate
        summary_rows.append(
            {
                "condition_id": condition_id,
                **_wide_row(model_label, "caption", aggregate, metrics),
            }
        )
        ci_rows.extend(
            {
                "condition_id": condition_id,
                **row,
            }
            for row in metric_intervals(
                rows,
                metrics,
                protocol,
                cluster_key=None,
                model_label=model_label,
                suite="deepsdo",
                task="caption",
            )
        )
        all_per_sample.extend(
            {"condition_id": condition_id, **row} for row in rows
        )
    delta_rows: list[Dict[str, Any]] = []
    declared_pairs = [
        tuple(pair)
        for pair in protocol.data["statistics"]["predeclared_comparisons"]["deepsdo"]
    ]
    all_pairs = list(declared_pairs)
    seen_pairs = {frozenset(pair) for pair in declared_pairs}
    labels = list(scored)
    for index, target in enumerate(labels):
        for baseline in labels[index + 1 :]:
            if frozenset((target, baseline)) not in seen_pairs:
                all_pairs.append((target, baseline))
                seen_pairs.add(frozenset((target, baseline)))
    for target, baseline in all_pairs:
        pair_rows = paired_metric_intervals(
                scored[target], scored[baseline], metrics, protocol,
                cluster_key=None, left_label=target, right_label=baseline,
                suite="deepsdo", task="caption",
            )
        is_headline = (target, baseline) in declared_pairs
        delta_rows.extend(
            {
                "condition_id": condition_id,
                **row,
                "headline_comparison": is_headline,
            }
            for row in pair_rows
        )
    _apply_holm_adjustment(delta_rows)
    topic_rows = _subgroup_table(scored, "topic_stratum")
    channel_rows = _subgroup_table(scored, "channel")
    modality_rows = _subgroup_table(scored, "collapsed_modality")
    for table in (topic_rows, channel_rows, modality_rows):
        for row in table:
            row["condition_id"] = condition_id
    diagnostics = _deepsdo_completion_diagnostics(
        predictions_by_model, condition_id
    )
    suite_root = report_root / "deepsdo"
    if condition_id is not None:
        suite_root = suite_root / condition_id
    precision = ".8g"
    note = (
        "CIDEr is retained for continuity and reported with significant digits. "
        "DeepSDO has one reference per image, so near-floor CIDEr values are sensitive "
        "to candidate/reference length mismatch and must not be interpreted as practical "
        "superiority. The published supervised M2 result is not an equivalent zero-shot baseline."
    )
    write_table_bundle(suite_root / "deepsdo_results", summary_rows, caption="Zero-shot DeepSDO caption results", label="tab:deepsdo-results", note=note, float_format=precision)
    write_table_bundle(suite_root / "deepsdo_absolute_ci", ci_rows, caption="DeepSDO paired-item bootstrap confidence intervals", label="tab:deepsdo-ci", float_format=precision)
    write_table_bundle(suite_root / "deepsdo_paired_differences", delta_rows, caption="DeepSDO paired model differences with Holm-adjusted p-values", label="tab:deepsdo-deltas", float_format=precision)
    write_table_bundle(suite_root / "deepsdo_completion_diagnostics", diagnostics, caption="DeepSDO generation completion and length diagnostics", label="tab:deepsdo-completion", float_format=precision)
    write_table_bundle(suite_root / "deepsdo_topics", topic_rows, caption="DeepSDO descriptive topic strata", label="tab:deepsdo-topics", note="Topics are derived descriptive strata, not official class labels; n<10 rows are exploratory, and topic is confounded with wavelength.", float_format=precision)
    write_table_bundle(suite_root / "deepsdo_channels", channel_rows, caption="DeepSDO channel-stratified results", label="tab:deepsdo-channels", float_format=precision)
    write_table_bundle(suite_root / "deepsdo_modalities", modality_rows, caption="DeepSDO collapsed modality results", label="tab:deepsdo-modalities", float_format=precision)
    write_jsonl_atomic(suite_root / "deepsdo_per_sample.jsonl", all_per_sample)
    cider_plot = [
        {"label": row["model"], "estimate": row["estimate"], "lower": row["lower"], "upper": row["upper"]}
        for row in ci_rows if row["metric"] == "cider"
    ]
    if cider_plot:
        plot_estimates(cider_plot, suite_root / "deepsdo_cider_overall", title="DeepSDO CIDEr", ylabel="CIDEr", paper_mode=paper_mode)
    delta_plot = [
        {"label": f"{row['target']} - {row['baseline']}", "estimate": row["estimate"], "lower": row["lower"], "upper": row["upper"]}
        for row in delta_rows if row["metric"] == "cider"
    ]
    if delta_plot:
        plot_estimates(delta_plot, suite_root / "deepsdo_cider_differences", title="DeepSDO paired CIDEr differences", ylabel="CIDEr difference", paper_mode=paper_mode)
    topics = sorted({row["topic_stratum"] for row in topic_rows})
    models = list(scored)
    for metric in ("cider", "meteor", "rouge_l"):
        lookup = {(row["model"], row["topic_stratum"]): float(row[metric]) for row in topic_rows}
        matrix = [[lookup[(model, topic)] for model in models] for topic in topics]
        if matrix:
            plot_heatmap(matrix, topics, models, suite_root / f"deepsdo_topic_{metric}", title=f"DeepSDO topic {metric}", colorbar_label=metric, paper_mode=paper_mode)
    return {
        "condition_id": condition_id,
        "summary": summary_rows,
        "absolute_ci": ci_rows,
        "paired_ci": delta_rows,
        "completion_diagnostics": diagnostics,
        "topics": topic_rows,
        "scored": scored,
    }


def _astro_top_level(task_key: str) -> str:
    return str(task_key).split(".", 1)[0]


def _astro_component_scores(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    by_task: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_key"])].append(row)
    for task_key, group in by_task.items():
        labels = list(group[0]["allowed_labels"])
        scores[task_key] = float(
            classification_metrics(
                [row["reference_label"] for row in group],
                [row.get("prediction") for row in group],
                labels=labels,
            )["macro_f1"]
        )
    return scores


def _astro_stratified_resample(
    rows: Sequence[Mapping[str, Any]], rng: random.Random
) -> list[Mapping[str, Any]]:
    by_top: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_top[_astro_top_level(str(row["task_key"]))].append(row)
    sample: list[Mapping[str, Any]] = []
    for top_level in sorted(by_top):
        members: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        ordered_clusters: list[str] = []
        for row in by_top[top_level]:
            cluster = str(row["source_object_id"])
            if cluster not in members:
                ordered_clusters.append(cluster)
            members[cluster].append(row)
        for _ in ordered_clusters:
            selected = ordered_clusters[rng.randrange(len(ordered_clusters))]
            sample.extend(members[selected])
    return sample


def _astro_hierarchy_values(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    from .astrovlbench import hierarchical_project_aggregate

    component_scores = _astro_component_scores(rows)
    hierarchy = hierarchical_project_aggregate(
        component_scores,
        included_components=component_scores,
    )
    return {
        **{key: float(value) for key, value in hierarchy["top_level_task_scores"].items()},
        "project_summary": float(hierarchy["project_macro_average"]),
    }


def _astro_hierarchy_intervals(
    rows: Sequence[Mapping[str, Any]], protocol: PaperProtocol, model_label: str
) -> list[Dict[str, Any]]:
    settings = protocol.data["statistics"]
    n_resamples = int(settings["bootstrap_replicates"])
    seed = int(settings["seed"])
    confidence = float(settings["confidence_level"])
    estimate = _astro_hierarchy_values(rows)
    replicates: Dict[str, list[float]] = {key: [] for key in estimate}
    rng = random.Random(seed)
    for _ in range(n_resamples):
        values = _astro_hierarchy_values(_astro_stratified_resample(rows, rng))
        for key in replicates:
            replicates[key].append(values[key])
    alpha = 1.0 - confidence
    return [
        {
            "suite": "astrovlbench",
            "model": model_label,
            "task_key": f"hierarchy.{key}",
            "metric": "macro_f1",
            "estimate": value,
            "lower": percentile(replicates[key], alpha / 2.0),
            "upper": percentile(replicates[key], 1.0 - alpha / 2.0),
            "confidence": confidence,
            "n_resamples": n_resamples,
            "seed": seed,
            "resampling_unit": "stratified_source_object_cluster",
            "n_items": len(rows),
            "n_clusters": len({str(row["source_object_id"]) for row in rows}),
            "method": "percentile",
        }
        for key, value in estimate.items()
    ]


def _astro_paired_hierarchy_interval(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    protocol: PaperProtocol,
    target: str,
    baseline: str,
) -> Dict[str, Any]:
    right_by_id = {str(row["id"]): row for row in right}
    if set(right_by_id) != {str(row["id"]) for row in left}:
        raise AnalysisError(f"Cannot pair Astro hierarchy for {target} and {baseline}")
    paired_rows = [
        {**dict(row), "_paired_right": right_by_id[str(row["id"])]}
        for row in left
    ]
    settings = protocol.data["statistics"]
    n_resamples = int(settings["bootstrap_replicates"])
    seed = int(settings["seed"])
    confidence = float(settings["confidence_level"])

    def difference(sample: Sequence[Mapping[str, Any]]) -> float:
        left_sample = [{key: value for key, value in row.items() if key != "_paired_right"} for row in sample]
        right_sample = [row["_paired_right"] for row in sample]
        return (
            _astro_hierarchy_values(left_sample)["project_summary"]
            - _astro_hierarchy_values(right_sample)["project_summary"]
        )

    estimate = difference(paired_rows)
    rng = random.Random(seed)
    replicates = [
        difference(_astro_stratified_resample(paired_rows, rng))
        for _ in range(n_resamples)
    ]
    alpha = 1.0 - confidence
    return {
        "suite": "astrovlbench",
        "task_key": "hierarchy.project_summary",
        "target": target,
        "baseline": baseline,
        "metric": "macro_f1",
        "estimate": estimate,
        "lower": percentile(replicates, alpha / 2.0),
        "upper": percentile(replicates, 1.0 - alpha / 2.0),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "resampling_unit": "paired_stratified_source_object_cluster",
        "n_items": len(left),
        "n_clusters": len({str(row["source_object_id"]) for row in left}),
        "method": "percentile",
    }


def analyze_astrovlbench(
    protocol: PaperProtocol,
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
    report_root: Path,
    records_sha256: str,
    *,
    paper_mode: bool,
) -> Dict[str, Any]:
    from .astrovlbench import hierarchical_project_aggregate, parse_label

    task_keys = sorted({str(row["task_key"]) for row in records})
    predictions_by_model: Dict[str, list[Dict[str, Any]]] = {}
    summary_rows: list[Dict[str, Any]] = []
    recall_rows: list[Dict[str, Any]] = []
    ci_rows: list[Dict[str, Any]] = []
    confusion: Dict[str, Any] = {}
    hierarchy_rows: list[Dict[str, Any]] = []
    for model_label in protocol.selected_models("astrovlbench"):
        raw = _model_predictions(protocol, "astrovlbench", output_root, model_label, records, records_sha256, paper_mode=paper_mode)
        parsed: list[Dict[str, Any]] = []
        for row in raw:
            result = parse_label(row, str(row.get("response") or ""))
            parsed.append({**row, "prediction": result.label, "parser": result.to_dict()})
        predictions_by_model[model_label] = parsed
        component_macro: Dict[str, float] = {}
        for task_key in task_keys:
            group = [row for row in parsed if row["task_key"] == task_key]
            labels = list(group[0]["allowed_labels"])
            metrics = classification_metrics(
                [row["reference_label"] for row in group],
                [row["prediction"] for row in group],
                labels=labels,
            )
            component_macro[task_key] = float(metrics["macro_f1"])
            summary_rows.append(
                {
                    "model": model_label,
                    "task_key": task_key,
                    "n": metrics["n"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "invalid_response_rate": metrics["invalid_response_rate"],
                }
            )
            for label, value in metrics["per_class_recall"].items():
                recall_rows.append({"model": model_label, "task_key": task_key, "class": label, "recall": value, "support": metrics["support"][label]})
            confusion[f"{model_label}:{task_key}"] = {
                "labels": metrics["labels"],
                "prediction_labels": metrics["prediction_labels"],
                "matrix": metrics["confusion_matrix"],
            }
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                interval = bootstrap_ci(
                    group,
                    lambda sample, key=metric, allowed=labels: float(
                        classification_metrics(
                            [row["reference_label"] for row in sample],
                            [row["prediction"] for row in sample],
                            labels=allowed,
                        )[key]
                    ),
                    n_resamples=int(protocol.data["statistics"]["bootstrap_replicates"]),
                    seed=int(protocol.data["statistics"]["seed"]),
                    confidence=float(protocol.data["statistics"]["confidence_level"]),
                    resampling_unit="cluster",
                    cluster_key="source_object_id",
                )
                ci_rows.append({"suite": "astrovlbench", "model": model_label, "task_key": task_key, "metric": metric, **interval.as_dict()})
        hierarchy = hierarchical_project_aggregate(
            component_macro,
            included_components=component_macro,
        )
        hierarchy_intervals = _astro_hierarchy_intervals(parsed, protocol, model_label)
        ci_rows.extend(hierarchy_intervals)
        project_ci = next(
            row for row in hierarchy_intervals
            if row["task_key"] == "hierarchy.project_summary"
        )
        hierarchy_rows.append(
            {
                "model": model_label,
                "metric": "macro_f1",
                **hierarchy["top_level_task_scores"],
                "project_macro_average": hierarchy["project_macro_average"],
                "project_ci_lower": project_ci["lower"],
                "project_ci_upper": project_ci["upper"],
            }
        )
    delta_rows: list[Dict[str, Any]] = []
    for target, baseline in protocol.data["statistics"]["predeclared_comparisons"]["astrovlbench"]:
        for task_key in task_keys:
            left = [row for row in predictions_by_model[target] if row["task_key"] == task_key]
            right_map = {row["id"]: row for row in predictions_by_model[baseline] if row["task_key"] == task_key}
            right = [right_map[row["id"]] for row in left]
            labels = list(left[0]["allowed_labels"])
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                interval = paired_bootstrap_ci(
                    left,
                    right,
                    lambda a, b, key=metric, allowed=labels: float(
                        classification_metrics([r["reference_label"] for r in a], [r["prediction"] for r in a], labels=allowed)[key]
                    ) - float(
                        classification_metrics([r["reference_label"] for r in b], [r["prediction"] for r in b], labels=allowed)[key]
                    ),
                    n_resamples=int(protocol.data["statistics"]["bootstrap_replicates"]),
                    seed=int(protocol.data["statistics"]["seed"]),
                    confidence=float(protocol.data["statistics"]["confidence_level"]),
                    resampling_unit="cluster",
                    cluster_key="source_object_id",
                )
                delta_rows.append({"suite": "astrovlbench", "task_key": task_key, "target": target, "baseline": baseline, "metric": metric, **interval.as_dict()})
        delta_rows.append(
            _astro_paired_hierarchy_interval(
                predictions_by_model[target],
                predictions_by_model[baseline],
                protocol,
                target,
                baseline,
            )
        )
    suite_root = report_root / "astrovlbench"
    write_table_bundle(suite_root / "astrovlbench_tasks", summary_rows, caption="AstroVLBench task results", label="tab:astrovlbench-tasks")
    write_table_bundle(suite_root / "astrovlbench_hierarchy", hierarchy_rows, caption="Project-defined hierarchical AstroVLBench summary", label="tab:astrovlbench-hierarchy", note="This transparent hierarchy is project-defined, not an official benchmark leaderboard score.")
    write_table_bundle(suite_root / "astrovlbench_class_recall", recall_rows, caption="AstroVLBench per-class recall", label="tab:astrovlbench-recall")
    write_table_bundle(suite_root / "astrovlbench_absolute_ci", ci_rows, caption="Clustered AstroVLBench confidence intervals", label="tab:astrovlbench-ci")
    write_table_bundle(suite_root / "astrovlbench_paired_differences", delta_rows, caption="Paired AstroVLBench model differences", label="tab:astrovlbench-deltas")
    write_json_atomic(suite_root / "astrovlbench_confusion_matrices.json", confusion)
    write_jsonl_atomic(suite_root / "astrovlbench_per_sample.jsonl", [row for rows in predictions_by_model.values() for row in rows])
    models = list(predictions_by_model)
    lookup = {(row["model"], row["task_key"]): float(row["macro_f1"]) for row in summary_rows}
    matrix = [[lookup[(model, task)] for model in models] for task in task_keys]
    if matrix:
        plot_heatmap(matrix, task_keys, models, suite_root / "astrovlbench_task_macro_f1", title="AstroVLBench task macro-F1", colorbar_label="Macro-F1", paper_mode=paper_mode)
    for key, values in confusion.items():
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", key).replace(":", "__")
        plot_heatmap(
            values["matrix"],
            values["labels"],
            values["prediction_labels"],
            suite_root / "confusion_matrices" / safe_name,
            title=f"AstroVLBench confusion: {key}",
            colorbar_label="Count",
            paper_mode=paper_mode,
        )
    return {"summary": summary_rows, "hierarchy": hierarchy_rows, "absolute_ci": ci_rows, "paired_ci": delta_rows}


def _model_manifest(protocol: PaperProtocol, suites: Sequence[str]) -> list[Dict[str, Any]]:
    used = {label for suite in suites for label in protocol.selected_models(suite)}
    rows = []
    for label, model in protocol.data["models"].items():
        if label not in used:
            continue
        rows.append(
            {
                "label": label,
                "paper_label": model["paper_label"],
                "backend": model["backend"],
                "repo_id": model["repo_id"],
                "revision": model["revision"],
                "dtype": model["dtype"],
                "environment": model["environment"],
                "checkpoint_sha256": model.get("checkpoint_sha256"),
                "connector_sha256": model.get("connector_sha256"),
                "lora_sha256": model.get("lora_sha256"),
                "external_code_revision": model.get("code_revision"),
                "auxiliary_vision_repo": (model.get("vision_encoder") or {}).get(
                    "repo_id"
                ),
                "auxiliary_vision_revision": (model.get("vision_encoder") or {}).get(
                    "revision"
                ),
                "external_zero_shot": any(suite != "internal" for suite in set(model["suites"]) & set(suites)),
                "retrieval": False,
            }
        )
    return rows


def _dataset_manifest(
    protocol: PaperProtocol,
    suites: Sequence[str],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    conditions: Mapping[str, Sequence[str | None]],
) -> list[Dict[str, Any]]:
    rows = []
    for suite in suites:
        cfg = protocol.data["datasets"][suite]
        for condition_id in conditions[suite]:
            rows.append(
                {
                    "suite": suite,
                    "condition_id": condition_id,
                    "paper_label": cfg["paper_label"],
                    "n_records": len(records[suite]),
                    "n_images_or_objects": len({str(row.get("source_object_id")) for row in records[suite]}),
                    "evaluation_only": cfg.get("evaluation_only"),
                    "rag_used": False,
                    "protocol_hash": protocol.suite_fingerprint(suite, condition_id),
                }
            )
    return rows


def _split_manifest_rows(
    suites: Sequence[str], records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for suite in suites:
        key = "task_key" if suite == "astrovlbench" else "record_type"
        counts = Counter(str(row.get(key) or "unknown") for row in records[suite])
        for group, count in sorted(counts.items()):
            rows.append(
                {
                    "suite": suite,
                    "split": "test",
                    "grouping": key,
                    "group": group,
                    "records": count,
                    "source_objects": len(
                        {
                            str(row.get("source_object_id"))
                            for row in records[suite]
                            if str(row.get(key) or "unknown") == group
                        }
                    ),
                }
            )
    return rows


def _training_lineage_rows(protocol: PaperProtocol, suites: Sequence[str]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for label in {model for suite in suites for model in protocol.selected_models(suite)}:
        if label in {"astraq_stage1", "astraq_stage2"}:
            lineage = {
                "training_dataset": "UniverseTBD/AstroLLaVA_convos",
                "training_dataset_revision": protocol.data["datasets"]["internal"]["source_revision"],
                "caption_lineage": "source-documented human-written captions",
                "qa_lineage": "source/build-documented GPT-4-generated conversations",
                "external_benchmark_used_for_training": False,
            }
        elif label == "astrollava":
            lineage = {
                "training_dataset": "AstroLLaVA training corpus (includes the internal source corpus)",
                "training_dataset_revision": "not fully recoverable per row from released weights",
                "caption_lineage": "source project documentation",
                "qa_lineage": "GPT-4-generated answers documented by source project",
                "external_benchmark_used_for_training": "unknown",
            }
        elif label == "qwen3_vl_4b":
            lineage = {
                "training_dataset": "Qwen3-VL foundation/instruction data",
                "training_dataset_revision": "not publicly auditable per example",
                "caption_lineage": "not established per row",
                "qa_lineage": "not established per row",
                "external_benchmark_used_for_training": "unknown",
            }
        else:
            lineage = {
                "training_dataset": "InternVL3.5 foundation/instruction data",
                "training_dataset_revision": "not publicly auditable per example",
                "caption_lineage": "not established per row",
                "qa_lineage": "not established per row",
                "external_benchmark_used_for_training": "unknown",
            }
        rows.append(
            {
                "model": label,
                **lineage,
                "rag_used_in_project_training": False,
                "rag_used_in_evaluation": False,
            }
        )
    return sorted(rows, key=lambda row: row["model"])


def _leakage_summary_rows(data_root: Path, suites: Sequence[str]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if "internal" in suites:
        image_path = data_root / "internal" / "leakage_audit" / "image_overlap_report.json"
        text_path = data_root / "internal" / "leakage_audit" / "text_overlap_report.json"
        if image_path.is_file():
            report = json.loads(image_path.read_text(encoding="utf-8"))
            for audit, key in (
                ("exact-byte train/test image pairs", "exact_byte_pairs"),
                ("decoded-pixel train/test image pairs", "exact_pixel_pairs"),
                ("pHash distance <=4 candidates", "phash_likely_pairs"),
                ("pHash distance 5--8 sensitivity candidates", "phash_sensitivity_pairs"),
            ):
                rows.append({"suite": "internal", "audit": audit, "count": report.get(key)})
        if text_path.is_file():
            report = json.loads(text_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "suite": "internal",
                    "audit": "exact normalized reference matches across train/test",
                    "count": report.get("exact_normalized_reference_matches"),
                }
            )
    if "deepsdo" in suites:
        manifest_candidates = list((data_root / "deepsdo" / "llava").glob("*audit*.json"))
        overlap_count: int | None = None
        for path in manifest_candidates:
            value = json.loads(path.read_text(encoding="utf-8"))
            overlap = value.get("normalized_caption_train_test_overlap") or value.get("dataset_audit", {}).get("normalized_caption_train_test_overlap")
            if overlap:
                overlap_count = int(overlap["test_rows_with_normalized_caption_in_train"])
                break
        rows.append(
            {
                "suite": "deepsdo",
                "audit": "normalized test captions also present in training",
                "count": 100 if overlap_count is None else overlap_count,
                "denominator": 102,
                "interpretation": "often nearby observations; not an image-duplicate claim",
            }
        )
    return rows


def _paper_narrative(protocol: PaperProtocol, results: Mapping[str, Any]) -> str:
    lines = [
        "# AstraQ-VL paper evaluation results",
        "",
        "All comparisons below are conditional on the frozen datasets, prompts, decoding settings, and model revisions. RAG was not used in training or evaluation.",
        "",
    ]
    for suite, suite_value in results.items():
        condition_values = (
            suite_value.get("conditions", {})
            if isinstance(suite_value, Mapping) and "conditions" in suite_value
            else {None: suite_value}
        )
        for condition_id, value in condition_values.items():
            heading = suite if condition_id is None else f"{suite} — {condition_id}"
            lines.extend([f"## {heading}", ""])
            if condition_id is not None:
                config = protocol.condition_config(suite, condition_id)
                lines.append(
                    f"Prompt: `{config['prompt']}`; maximum new tokens: "
                    f"{config['max_new_tokens']}. Conditions are scored and interpreted separately."
                )
                lines.append("")
            summaries = value.get("summary", [])
            deltas = value.get("paired_ci", [])
            for delta in deltas:
                if delta.get("headline_comparison") is False:
                    continue
                target = delta.get("target")
                baseline = delta.get("baseline")
                metric = delta.get("metric")
                task = delta.get("task") or delta.get("task_key") or "overall"
                left = next((row for row in summaries if row.get("model") == target and (row.get("task") == task or row.get("task_key") == task)), None)
                right = next((row for row in summaries if row.get("model") == baseline and (row.get("task") == task or row.get("task_key") == task)), None)
                if left and right and left.get(metric) is not None and right.get(metric) is not None:
                    lines.append(
                        "- "
                        + cautious_comparison(
                            str(target), float(left[metric]), str(baseline), float(right[metric]),
                            metric=str(metric), difference_ci=(float(delta["lower"]), float(delta["upper"])),
                            scope=f"the frozen {heading} {task} evaluation",
                            precision=6,
                        )
                    )
            lines.append("")
            if suite == "deepsdo":
                lines.append(
                    "DeepSDO topic groups are derived descriptive strata rather than class labels; topic and wavelength are confounded, and the single reference per image limits semantic coverage. CIDEr is retained for continuity, but near-floor values and their exact absolute differences must be interpreted alongside caption-length diagnostics. The supervised published M2 model is not treated as an equivalent zero-shot baseline."
                )
                lines.append("")
            if suite == "astrovlbench":
                lines.append(
                    "The project summary is a transparent macro-average over the predeclared hierarchy and is not presented as an official AstroVLBench leaderboard score. Parser-invalid answers are retained as failures."
                )
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def analyze_study(
    protocol: PaperProtocol,
    suites: Sequence[str],
    output_root: Path,
    *,
    paper_mode: bool = True,
    data_root: Path | None = None,
    conditions: Mapping[str, Sequence[str | None]] | None = None,
) -> Path:
    data_root = Path(data_root or protocol.data["runtime"]["data_root"])
    condition_map = conditions or {
        suite: list(protocol.condition_ids(suite)) for suite in suites
    }
    records: Dict[str, list[Dict[str, Any]]] = {}
    record_hashes: Dict[str, str] = {}
    results: Dict[str, Any] = {}
    condition_records: Dict[tuple[str, str | None], list[Dict[str, Any]]] = {}
    for suite in suites:
        for condition_id in condition_map[suite]:
            path = _records_path(data_root, suite, condition_id)
            if not path.is_file():
                raise AnalysisError(f"Missing canonical records for {suite}: {path}")
            unit_records = read_jsonl(path)
            condition_records[(suite, condition_id)] = unit_records
            records.setdefault(suite, unit_records)
            record_hashes[_unit_key(suite, condition_id)] = sha256_file(path)
    analysis_hash = analysis_run_fingerprint(
        protocol, suites, record_hashes, data_root, conditions=condition_map
    )
    report_root = output_root / "reports" / analysis_hash[:16]
    report_root.mkdir(parents=True, exist_ok=True)
    for suite in suites:
        if suite == "internal":
            results[suite] = analyze_internal(protocol, records[suite], output_root, report_root, record_hashes[_unit_key(suite, None)], paper_mode=paper_mode)
        elif suite == "deepsdo":
            if condition_map[suite] == [None]:
                results[suite] = analyze_deepsdo(
                    protocol,
                    condition_records[(suite, None)],
                    output_root,
                    report_root,
                    record_hashes[_unit_key(suite, None)],
                    paper_mode=paper_mode,
                )
            else:
                results[suite] = {"conditions": {}}
                for condition_id in condition_map[suite]:
                    results[suite]["conditions"][str(condition_id)] = analyze_deepsdo(
                        protocol,
                        condition_records[(suite, condition_id)],
                        output_root,
                        report_root,
                        record_hashes[_unit_key(suite, condition_id)],
                        paper_mode=paper_mode,
                        condition_id=condition_id,
                    )
        else:
            results[suite] = analyze_astrovlbench(protocol, records[suite], output_root, report_root, record_hashes[_unit_key(suite, None)], paper_mode=paper_mode)
    model_rows = _model_manifest(protocol, suites)
    dataset_rows = _dataset_manifest(protocol, suites, records, condition_map)
    split_rows = _split_manifest_rows(suites, records)
    lineage_rows = _training_lineage_rows(protocol, suites)
    leakage_rows = _leakage_summary_rows(data_root, suites)
    write_table_bundle(report_root / "model_manifest", model_rows, caption="Frozen model manifest", label="tab:model-manifest")
    write_table_bundle(report_root / "dataset_protocol", dataset_rows, caption="Dataset and protocol manifest", label="tab:dataset-protocol")
    write_table_bundle(report_root / "split_manifest", split_rows, caption="Frozen evaluation split composition", label="tab:split-manifest")
    write_table_bundle(report_root / "training_data_lineage", lineage_rows, caption="Model and training-data lineage", label="tab:training-lineage", note="Unknown entries are reported as unknown; foundation-model pretraining overlap cannot be exhaustively audited from released weights.")
    if leakage_rows:
        write_table_bundle(report_root / "leakage_audit", leakage_rows, caption="Dataset leakage and overlap audits", label="tab:leakage-audit", note="Candidate overlaps are reported without silently changing the frozen splits.")
    (report_root / "paper_results.md").write_text(_paper_narrative(protocol, results), encoding="utf-8")
    # Do not serialize the in-memory per-sample maps embedded for pairwise computation.
    def without_scored(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: without_scored(item) for key, item in value.items() if key != "scored"}
        if isinstance(value, list):
            return [without_scored(item) for item in value]
        return value

    public_results = without_scored(results)
    write_json_atomic(report_root / "results.json", public_results)
    manifest = {
        "study_id": protocol.study_id,
        "protocol_sha256": protocol.fingerprint,
        "analysis_run_sha256": analysis_hash,
        "suite_analysis_hashes": {
            _unit_key(suite, condition_id): protocol.analysis_fingerprint(suite, condition_id)
            for suite in suites
            for condition_id in condition_map[suite]
        },
        "records_file_sha256": record_hashes,
        "analysis_evidence_sha256": analysis_evidence_hashes(data_root, suites),
        "suites": list(suites),
        "conditions": {suite: list(values) for suite, values in condition_map.items()},
        "retrieval": False,
        "artifacts": checksum_tree(report_root, exclude={"results_manifest.json"}),
    }
    write_json_atomic(report_root / "results_manifest.json", manifest)
    return report_root
