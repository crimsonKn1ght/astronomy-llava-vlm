from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.paper.astrovlbench import (
    AstroVLBenchError,
    LockValidationError,
    create_lock_manifest,
    discover_records,
    extract_official_guided_prompts,
    hierarchical_project_aggregate,
    materialize_locked_records,
    parse_label,
    parse_label_response,
    read_lock_manifest,
    resolve_and_lock_snapshot,
    validate_lock_manifest,
    write_lock_manifest,
)
from eval.paper.records import validate_unique_records


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture image: " + path.as_posix().encode("ascii"))


def _build_snapshot(root: Path) -> None:
    prompt_sources = {
        1: 'PROMPTS = {"guided": "Classify the optical source as AGN or Galaxy."}\n',
        2: 'PROMPTS = {"guided": "Classify the radio morphology as FRI or FRII."}\n',
        3: 'GUIDED_PROMPT = "Classify this SED plot as Type-1 AGN, Type-2 AGN, or Galaxy."\n',
        4: 'PROMPTS = {"guided": "Classify the light-curve plot."}\n',
        5: (
            'PROMPTS = {\n'
            '    "Q1": {"guided": "Are both H-alpha and H-beta present?"},\n'
            '    "Q2": {"guided": "Is this a broad-line AGN?"},\n'
            '    "Q3": {"guided": "Give the BPT class."},\n'
            '}\n'
        ),
    }
    for task, source in prompt_sources.items():
        path = root / "code" / f"task{task}" / "llm.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    task1 = root / "data" / "Task1_QSOHost"
    _image(task1 / "images" / "agn.jpg")
    _image(task1 / "images" / "galaxy.jpg")
    _write_csv(
        task1 / "image_labels.csv",
        ["image_path", "label", "source_id"],
        [
            {"image_path": "images/agn.jpg", "label": "AGN", "source_id": "opt-1"},
            {"image_path": "images/galaxy.jpg", "label": "Galaxy", "source_id": "opt-2"},
        ],
    )

    first = root / "data" / "Task2_RadioMorph" / "MiraBest_F"
    _image(first / "images" / "FRI" / "first-source-001.png")
    (first / "metadata.jsonl").write_text(
        json.dumps(
            {"filename": "first-source-001.png", "label": "FR I", "source_id": "source-001"}
        )
        + "\n",
        encoding="utf-8",
    )
    nvss = root / "data" / "Task2_RadioMorph" / "MiraBest_N"
    _image(nvss / "images" / "FRI" / "nvss-source-001.png")
    _image(nvss / "images" / "FRII" / "nvss-source-002.png")
    (nvss / "metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "filename": "nvss-source-001.png",
                        "label": "FRI",
                        "source_id": "source-001",
                    }
                ),
                json.dumps(
                    {
                        "filename": "nvss-source-002.png",
                        "label": "FRII",
                        "source_id": "source-002",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    task3 = root / "data" / "Task3_SED"
    sed_rows = []
    for index, label in enumerate(("Type-1 AGN", "Type-2 AGN", "Galaxy"), 1):
        name = f"sed-{index}.png"
        _image(task3 / "images" / name)
        sed_rows.append(
            {
                "image": name,
                "source_type": label,
                "source_id": f"sed-{index}",
                "redshift": 0.1 * index,
                "g_mag": 20 + index,
            }
        )
    _write_csv(
        task3 / "nirsed_v2_catalog.csv",
        ["image", "source_type", "source_id", "redshift", "g_mag"],
        sed_rows,
    )

    task4 = root / "data" / "Task4_LightCurve"
    task4_rows = []
    for name, label in (("lc-agn.png", "AGN"), ("lc-tde.png", "TDE")):
        _image(task4 / "figures" / label / name)
        task4_rows.append({"figure": name, "class": label, "object_id": name[:-4]})
    _write_csv(task4 / "manifest.csv", ["figure", "class", "object_id"], task4_rows)

    task5 = root / "data" / "Task5_SpecType"
    task5_rows = []
    for group in ("A", "B", "C1", "C2", "C3", "C4", "D"):
        source_id = f"spec-{group.casefold()}"
        filename = f"{source_id}.png"
        _image(task5 / "figures" / f"Group_{group}" / filename)
        task5_rows.append({"source_id": source_id, "group": f"Group_{group}", "image": filename})
    _write_csv(
        task5 / "ASIB_v1_selection_with_snr.csv",
        ["source_id", "group", "image"],
        task5_rows,
    )


class AstroVLBenchAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.snapshot = self.base / "snapshot"
        _build_snapshot(self.snapshot)
        self.lock_path = self.base / "astrovlbench.lock.json"
        self.manifest = create_lock_manifest(
            self.snapshot,
            repo_id="fixture/AstroVLBench",
            requested_revision="fixture",
            commit_sha="a" * 40,
        )
        write_lock_manifest(self.manifest, self.lock_path)

    def test_lock_roundtrip_hashes_every_file_and_prompts(self) -> None:
        loaded = read_lock_manifest(self.lock_path)
        validated = validate_lock_manifest(self.snapshot, loaded)
        actual_files = [path for path in self.snapshot.rglob("*") if path.is_file()]
        self.assertEqual(len(validated["files"]), len(actual_files))
        self.assertEqual(
            {entry["path"] for entry in validated["official_prompt_sources"]},
            {f"code/task{index}/llm.py" for index in range(1, 6)},
        )
        self.assertIn("Classify the optical", validated["official_guided_prompts"]["task1"]["default"])
        self.assertEqual(set(validated["official_guided_prompts"]["task5"]), {"q1", "q2", "q3"})

    def test_lock_detects_tampering_and_unexpected_files(self) -> None:
        image = self.snapshot / "data" / "Task1_QSOHost" / "images" / "agn.jpg"
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(LockValidationError, "Size mismatch|SHA-256 mismatch"):
            validate_lock_manifest(self.snapshot, self.manifest)

        _build_snapshot(self.snapshot)
        extra = self.snapshot / "data" / "unexpected.txt"
        extra.write_text("new upstream content", encoding="utf-8")
        with self.assertRaisesRegex(LockValidationError, "unexpected"):
            validate_lock_manifest(self.snapshot, self.manifest)

    def test_remote_lock_requires_token_before_optional_hub_import(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(AstroVLBenchError, "HF_TOKEN"):
                resolve_and_lock_snapshot(self.base / "download", self.base / "lock.json")

    def test_extracts_static_prompts_without_importing_release_code(self) -> None:
        prompts = extract_official_guided_prompts(self.snapshot)
        self.assertEqual(prompts["task5"]["q1"], "Are both H-alpha and H-beta present?")
        self.assertEqual(prompts["task3"]["default"], "Classify this SED plot as Type-1 AGN, Type-2 AGN, or Galaxy.")

    def test_discovers_all_documented_tasks_and_reports_fixture_discrepancies(self) -> None:
        result = discover_records(self.snapshot, self.lock_path)
        self.assertEqual(
            result.expected_counts,
            {
                "task1": 2,
                "task2.first": 1,
                "task2.nvss": 2,
                "task3": 3,
                "task4": 2,
                "task5.q1": 7,
                "task5.q2": 5,
                "task5.q3": 4,
            },
        )
        self.assertEqual(len(result.records), 26)
        self.assertEqual(len(result.discrepancies), 9)
        self.assertTrue(any("paper Methods" in item for item in result.discrepancies))
        self.assertEqual(result.source_audit["task2_matched_source_objects"], 1)
        self.assertEqual(result.source_audit["task5_unique_source_objects"], 7)

        task3 = [record for record in result.records if record.task == "task3"]
        self.assertEqual(len(task3), 3)
        for record in task3:
            serialized = json.dumps(record.to_dict()).casefold()
            self.assertNotIn("redshift", serialized)
            self.assertNotIn("g_mag", serialized)
            self.assertEqual(record.metadata["modality"], "image")

        serialized_records, report = materialize_locked_records(self.lock_path)
        self.assertEqual(len(serialized_records), 26)
        self.assertEqual(report["expected_counts_from_locked_snapshot"]["task5.q3"], 4)
        canonical = serialized_records[0]
        self.assertEqual(canonical["id"], canonical["sample_id"])
        self.assertEqual(canonical["dataset"], "astrovlbench")
        self.assertEqual(canonical["split"], "test")
        self.assertEqual(canonical["record_type"], "classification")
        self.assertEqual(canonical["reference"], canonical["reference_label"])
        self.assertEqual(canonical["image"], canonical["image_relpath"])
        self.assertEqual(canonical["image_id"], canonical["image_relpath"])
        self.assertEqual(len(canonical["prompt_sha256"]), 64)
        self.assertEqual(len(canonical["reference_sha256"]), 64)
        self.assertEqual(len(canonical["image_sha256"]), 64)
        self.assertEqual(canonical["record_index"], 1)
        validate_unique_records(serialized_records)

    def test_task2_and_task5_share_source_ids_for_clustered_bootstrap(self) -> None:
        records = discover_records(self.snapshot, self.manifest).records
        radio_001 = [record for record in records if record.source_object_id == "radio_source_001"]
        self.assertEqual({record.task_key for record in radio_001}, {"task2.first", "task2.nvss"})
        spectrum_c1 = [record for record in records if record.source_object_id == "spectrum_spec_c1"]
        self.assertEqual({record.task_key for record in spectrum_c1}, {"task5.q1", "task5.q2", "task5.q3"})
        self.assertEqual({record.reference_label for record in spectrum_c1}, {"Yes", "No", "Star-Forming"})

    def test_strict_label_parser_retains_invalid_and_ambiguous_outputs(self) -> None:
        parsed = parse_label_response("Answer: FR II", "task2.first")
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.label, "FRII")

        structured = parse_label_response(
            '{"answer": "AGN", "reason": "I compared AGN and Galaxy morphology."}',
            "task1",
        )
        self.assertTrue(structured.valid)
        self.assertEqual(structured.label, "AGN")
        self.assertEqual(structured.reason, "json_answer")

        via_record = parse_label(
            {"task": "task1", "subtask": None},
            '{"answer": "Galaxy", "reason": "resolved host"}',
        )
        self.assertTrue(via_record.valid)
        self.assertEqual(via_record.label, "Galaxy")

        negated = parse_label_response("Answer: not a BLAGN", "task5.q2")
        self.assertTrue(negated.valid)
        self.assertEqual(negated.label, "No")

        ambiguous = parse_label_response(
            "The source could be Type-1 AGN or Type-2 AGN.", "task3"
        )
        self.assertFalse(ambiguous.valid)
        self.assertEqual(ambiguous.reason, "multiple_allowed_labels")
        self.assertIn("could be", ambiguous.raw_response)

        invalid = parse_label_response("Insufficient information", "task4")
        self.assertFalse(invalid.valid)
        self.assertIsNone(invalid.label)
        self.assertEqual(invalid.reason, "no_allowed_label")

    def test_hierarchical_project_aggregation_weights_top_level_tasks_equally(self) -> None:
        scores = {
            "task1": 0.5,
            "task2.first": 0.4,
            "task2.nvss": 0.8,
            "task3": 0.5,
            "task4": 0.5,
            "task5.q1": 0.6,
            "task5.q2": 0.7,
            "task5.q3": 0.8,
        }
        aggregate = hierarchical_project_aggregate(scores)
        self.assertAlmostEqual(aggregate["top_level_task_scores"]["task2"], 0.6)
        self.assertAlmostEqual(aggregate["top_level_task_scores"]["task5"], 0.7)
        self.assertAlmostEqual(aggregate["project_macro_average"], 0.56)


if __name__ == "__main__":
    unittest.main()
