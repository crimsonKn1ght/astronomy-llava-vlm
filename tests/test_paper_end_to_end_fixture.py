from __future__ import annotations

import copy
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval.paper.analysis import analyze_study
from eval.paper.artifacts import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from eval.paper.astrovlbench import ALLOWED_LABELS
from eval.paper.protocol import PaperProtocol
from scripts.run_paper_eval import package


ROOT = Path(__file__).resolve().parents[1]


def _fake_caption_scores(rows, protocol, *, paper_mode, include_sbert=True):
    scored = []
    for row in rows:
        value = {
            **dict(row),
            "prediction": str(row.get("response") or ""),
            "cider": 1.0,
            "meteor": 1.0,
            "rouge_l": 1.0,
            "bleu_1": 1.0,
            "bleu_2": 1.0,
            "bleu_3": 1.0,
            "bleu_4": 1.0,
            "sbert": 1.0,
            "coco_tokenized_prediction": str(row.get("response") or ""),
            "coco_tokenized_references": [str(row.get("reference") or "")],
        }
        scored.append(value)
    aggregate = {
        "n": len(scored),
        "valid_response_rate": 1.0,
        "max_token_rate": 0.0,
        "token_cap_rate": 0.0,
        "cider": 1.0,
        "meteor": 1.0,
        "rouge_l": 1.0,
        "bleu_1": 1.0,
        "bleu_2": 1.0,
        "bleu_3": 1.0,
        "bleu_4": 1.0,
        "sbert": 1.0 if include_sbert else None,
    }
    return scored, aggregate


def _fake_qa_scores(rows, protocol, *, paper_mode):
    scored = [
        {
            **dict(row),
            "prediction": str(row.get("response") or ""),
            "token_f1": 1.0,
            "exact_match": 1.0,
            "rouge_l": 1.0,
            "sbert": 1.0,
        }
        for row in rows
    ]
    return scored, {
        "n": len(scored),
        "valid_response_rate": 1.0,
        "token_cap_rate": 0.0,
        "token_f1": 1.0,
        "exact_match": 1.0,
        "rouge_l": 1.0,
        "sbert": 1.0,
    }


def _fake_plot(*args, **kwargs):
    if len(args) >= 4 and isinstance(args[0], list) and isinstance(args[1], list):
        stem = Path(args[3])
    else:
        stem = Path(args[1])
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{suffix}")
        path.write_bytes(f"fixture-{suffix}".encode("ascii"))
        outputs[suffix] = path
    return outputs


class TinyPaperEndToEndFixtureTests(unittest.TestCase):
    def _protocol(self) -> PaperProtocol:
        source = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")
        data = copy.deepcopy(source.data)
        data["statistics"]["bootstrap_replicates"] = 5
        return PaperProtocol(source.path, data)

    @staticmethod
    def _records() -> dict[str, list[dict]]:
        internal = [
            {
                "id": "internal-caption",
                "dataset": "internal",
                "split": "test",
                "record_type": "caption",
                "image_id": "internal-image",
                "source_object_id": "internal-image",
                "image_path": "/fixture/internal.png",
                "prompt": "Describe this image.",
                "reference": "A spiral galaxy.",
            },
            {
                "id": "internal-qa",
                "dataset": "internal",
                "split": "test",
                "record_type": "qa",
                "image_id": "internal-image",
                "source_object_id": "internal-image",
                "image_path": "/fixture/internal.png",
                "prompt": "What is shown?",
                "reference": "A spiral galaxy.",
            },
        ]
        deepsdo = [
            {
                "id": "deepsdo-caption",
                "dataset": "deepsdo",
                "split": "test",
                "record_type": "caption",
                "image_id": "solar-image",
                "source_object_id": "solar-image",
                "image_path": "/fixture/solar.png",
                "prompt": "Describe this solar image.",
                "reference": "A sunspot is visible.",
                "topic_stratum": "Sunspots",
                "channel": "HMI continuum",
                "collapsed_modality": "HMI continuum",
            }
        ]
        astro = []
        for task_key in (
            "task1",
            "task2.first",
            "task2.nvss",
            "task3",
            "task4",
            "task5.q1",
            "task5.q2",
            "task5.q3",
        ):
            top = task_key.split(".", 1)[0]
            cluster = "radio-source" if top == "task2" else "spectral-source" if top == "task5" else task_key
            labels = list(ALLOWED_LABELS[task_key])
            astro.append(
                {
                    "id": f"astro-{task_key}",
                    "dataset": "astrovlbench",
                    "split": "test",
                    "record_type": "classification",
                    "task_key": task_key,
                    "source_object_id": cluster,
                    "image_id": f"{task_key}.png",
                    "image_path": f"/fixture/{task_key}.png",
                    "prompt": "Use the official guided prompt.",
                    "reference": labels[0],
                    "reference_label": labels[0],
                    "allowed_labels": labels,
                }
            )
        return {"internal": internal, "deepsdo": deepsdo, "astrovlbench": astro}

    @staticmethod
    def _write_completed_predictions(
        protocol: PaperProtocol,
        output_root: Path,
        suite: str,
        records_path: Path,
        records: list[dict],
    ) -> None:
        records_hash = sha256_file(records_path)
        for model_label, model in protocol.selected_models(suite).items():
            directory = protocol.model_output_dir(
                suite, model_label, records_hash, ROOT, output_root
            )
            predictions = []
            for record in records:
                response = (
                    record["reference_label"]
                    if suite == "astrovlbench"
                    else record["reference"]
                )
                predictions.append(
                    {
                        **record,
                        "model_label": model_label,
                        "model_revision": model["revision"],
                        "backend": model["backend"],
                        "response": response,
                        "raw_response": response,
                        "status": "ok",
                        "leak_flags": [],
                        "token_cap_hit": False,
                        "termination_reason": "eos",
                    }
                )
            predictions_path = directory / "predictions.jsonl"
            write_jsonl_atomic(predictions_path, predictions)
            write_json_atomic(
                directory / "run_manifest.json",
                {
                    "model_protocol_hash": protocol.effective_model_fingerprint(
                        suite, model_label, records_hash, ROOT
                    ),
                    "records_file_sha256": records_hash,
                    "predictions_sha256": sha256_file(predictions_path),
                    "completion": {"complete": True, "expected": len(records)},
                },
            )

    def test_tiny_fixture_produces_tables_figures_manifests_and_bundles(self) -> None:
        protocol = self._protocol()
        records = self._records()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "datasets"
            output_root = root / "outputs"
            for suite, suite_records in records.items():
                records_path = data_root / suite / "records.jsonl"
                write_jsonl_atomic(records_path, suite_records)
                self._write_completed_predictions(
                    protocol, output_root, suite, records_path, suite_records
                )

            with mock.patch(
                "eval.paper.analysis.score_caption_rows", side_effect=_fake_caption_scores
            ), mock.patch(
                "eval.paper.analysis.score_qa_rows", side_effect=_fake_qa_scores
            ), mock.patch(
                "eval.paper.analysis.add_internal_supplementary_metrics",
                side_effect=lambda rows, protocol, paper_mode: {
                    "n": len(rows),
                    "n_valid": len(rows),
                },
            ), mock.patch(
                "eval.paper.analysis.plot_estimates", side_effect=_fake_plot
            ), mock.patch(
                "eval.paper.analysis.plot_heatmap", side_effect=_fake_plot
            ):
                report = analyze_study(
                    protocol,
                    ("internal", "deepsdo", "astrovlbench"),
                    output_root,
                    paper_mode=True,
                    data_root=data_root,
                )

            for suffix in ("json", "csv", "md", "tex"):
                self.assertTrue((report / f"model_manifest.{suffix}").is_file())
            for suffix in ("pdf", "svg", "png"):
                self.assertTrue(
                    (report / "deepsdo" / f"deepsdo_cider_overall.{suffix}").is_file()
                )
                self.assertTrue(
                    (report / "astrovlbench" / f"astrovlbench_task_macro_f1.{suffix}").is_file()
                )
            self.assertTrue((report / "paper_results.md").is_file())
            self.assertTrue((report / "results_manifest.json").is_file())

            old_report = output_root / "reports" / "old-protocol" / "must-not-package.txt"
            old_report.parent.mkdir(parents=True)
            old_report.write_text("stale evidence", encoding="utf-8")
            old_generation = (
                protocol.output_dir("deepsdo", output_root)
                / "astraq_stage1"
                / "old-effective-hash"
                / "must-not-package-generation.txt"
            )
            old_generation.parent.mkdir(parents=True)
            old_generation.write_text("stale generation", encoding="utf-8")
            private, public = package(
                protocol,
                report,
                SimpleNamespace(
                    output_root=str(output_root),
                    data_root=str(data_root),
                    asset_root=str(root / "assets"),
                    hf_cache=str(root / "hf-cache"),
                    suites="internal,deepsdo,astrovlbench",
                    models="all",
                    dry_run=False,
                ),
            )
            self.assertGreater(private.stat().st_size, 0)
            self.assertGreater(public.stat().st_size, 0)
            with tarfile.open(private, "r:gz") as archive:
                self.assertFalse(
                    any(name.endswith("must-not-package.txt") for name in archive.getnames())
                )
                self.assertFalse(
                    any(
                        name.endswith("must-not-package-generation.txt")
                        for name in archive.getnames()
                    )
                )


if __name__ == "__main__":
    unittest.main()
