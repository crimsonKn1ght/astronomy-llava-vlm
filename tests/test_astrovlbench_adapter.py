from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from eval.paper.astrovlbench import (
    AstroVLBenchError,
    LockValidationError,
    create_lock_manifest,
    discover_records,
    extract_official_guided_prompts,
    hierarchical_project_aggregate,
    lock_local_snapshot,
    materialize_locked_records,
    parse_label,
    parse_label_response,
    read_lock_manifest,
    resolve_and_lock_snapshot,
    sha256_file,
    validate_lock_manifest,
    validate_protocol_release_contract,
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
    Image.new("RGB", (2, 2), color=(80, 120, 160)).save(path)


def _build_snapshot(root: Path) -> None:
    prompt_sources = {
        1: (
            'SYSTEM_PROMPT_GUIDED = "Classify the optical source as AGN or Galaxy."\n'
            'USER_TEXT = "Label this optical image."\n'
        ),
        2: (
            'def build_prompt_guided(survey: str) -> str:\n'
            '    return f"Classify the {survey} radio morphology as FRI or FRII."\n'
            'USER_TEXT = "Label this radio image."\n'
        ),
        3: (
            'SYSTEM_PROMPT_IMAGE = "Classify this SED plot. {REDSHIFT_BLOCK}"\n'
            'def build_image_prompt(redshift_mode, redshift, redshift_err, prompt_type="guided"):\n'
            '    if redshift_mode == "with":\n'
            '        redshift_block = f"Redshift: {redshift}"\n'
            '        redshift_instruction = "Use redshift."\n'
            '    else:\n'
            '        redshift_block = "Redshift: not provided."\n'
            '        redshift_instruction = "Do not assume redshift; use the observed SED."\n'
            '    return SYSTEM_PROMPT_IMAGE.format(REDSHIFT_BLOCK=redshift_block)\n'
            'USER_TEXT = "Label this SED image."\n'
        ),
        4: (
            'SYSTEM_PROMPT_IMAGE = "Classify the light-curve plot."\n'
            'USER_TEXT_IMAGE = "Label this light curve."\n'
        ),
        5: (
            'SYSTEM_PROMPT_Q1 = "Are both H-alpha and H-beta present?"\n'
            'SYSTEM_PROMPT_Q2 = "Is this a broad-line AGN?"\n'
            'SYSTEM_PROMPT_Q3 = "Give the BPT class."\n'
            'USER_TEXT_Q1 = "Answer Q1."\n'
            'USER_TEXT_Q2 = "Answer Q2."\n'
            'USER_TEXT_Q3 = "Answer Q3."\n'
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
            {
                "filename": "images/FRI/first-source-001.png",
                "label": "FR I",
                "source_id": "source-001",
            }
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
        prefix = {"Type-1 AGN": "Type1AGN", "Type-2 AGN": "Type2AGN", "Galaxy": "Galaxy"}[label]
        name = f"{prefix}_{index}.png"
        _image(task3 / "images" / name)
        sed_rows.append(
            {
                "targetid": str(index),
                "class": label,
                "redshift": 0.1 * index,
                "g_mag": 20 + index,
            }
        )
    _write_csv(
        task3 / "nirsed_v2_catalog.csv",
        ["targetid", "class", "redshift", "g_mag"],
        sed_rows,
    )

    task4 = root / "data" / "Task4_LightCurve"
    task4_rows = []
    for name, label in (("lc-agn.png", "AGN"), ("lc-tde.png", "TDE")):
        _image(task4 / "figures" / label / name)
        task4_rows.append(
            {"fig_path": f"figures/{label}/{name}", "class": label, "object_id": name[:-4]}
        )
    _write_csv(task4 / "manifest.csv", ["fig_path", "class", "object_id"], task4_rows)

    task5 = root / "data" / "Task5_SpecType"
    task5_rows = []
    for group in ("A", "B", "C1", "C2", "C3", "C4", "D"):
        source_id = f"spec-{group.casefold()}"
        filename = f"spectrum_{source_id}.png"
        _image(task5 / "figures" / f"Group_{group}_Fixture" / filename)
        task5_rows.append({"TARGETID": source_id, "SUB_GROUP": group})
    _write_csv(
        task5 / "ASIB_v1_selection_with_snr.csv",
        ["TARGETID", "SUB_GROUP"],
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
        self.assertIn("Official user instruction", validated["official_guided_prompts"]["task1"]["default"])

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

    def test_local_snapshot_lock_uses_external_pointer_without_copying_source(self) -> None:
        bundle = self.base / "prepared" / "astrovlbench"
        lock = lock_local_snapshot(
            self.snapshot,
            bundle,
            repo_id="fixture/AstroVLBench",
            revision="c" * 40,
        )
        manifest = read_lock_manifest(lock)
        self.assertEqual(Path(manifest["snapshot_path"]), self.snapshot.resolve())
        self.assertFalse((bundle / "snapshot").exists())
        validate_lock_manifest(self.snapshot, lock)

    def test_protocol_release_contract_rejects_a_different_inventory(self) -> None:
        config = {
            "source_repo": self.manifest["repo_id"],
            "locked_revision": self.manifest["commit_sha"],
            "expected_snapshot_files": len(self.manifest["files"]),
            "expected_snapshot_bytes": sum(
                int(entry["size"]) for entry in self.manifest["files"]
            ),
            "expected_snapshot_inventory_sha256": self.manifest[
                "snapshot_inventory_sha256"
            ],
        }
        validate_protocol_release_contract(self.manifest, config)
        changed = dict(config)
        changed["expected_snapshot_inventory_sha256"] = "0" * 64
        with self.assertRaisesRegex(LockValidationError, "protocol-pinned release"):
            validate_protocol_release_contract(self.manifest, changed)

    def test_extracts_static_prompts_without_importing_release_code(self) -> None:
        prompts = extract_official_guided_prompts(self.snapshot)
        self.assertIn("Are both H-alpha and H-beta present?", prompts["task5"]["q1"])
        self.assertIn("Redshift: not provided.", prompts["task3"]["default"])
        self.assertIn("Label this SED image.", prompts["task3"]["default"])

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
            self.assertNotIn("g_mag", serialized)
            self.assertEqual(record.metadata["modality"], "image")
            self.assertEqual(record.metadata["redshift_mode"], "without")
            self.assertNotIn('"redshift": 0.', serialized)

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

    def test_first_overlay_excludes_missing_and_zero_byte_rows_without_mutating_source(self) -> None:
        metadata = (
            self.snapshot
            / "data"
            / "Task2_RadioMorph"
            / "MiraBest_F"
            / "metadata.jsonl"
        )
        with metadata.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "filename": "images/FRII/missing.png",
                        "label": "FRII",
                        "source_split": "test",
                    }
                )
                + "\n"
            )
            stream.write(
                json.dumps(
                    {
                        "filename": "images/FRII/empty.png",
                        "label": "FRII",
                        "source_split": "test",
                    }
                )
                + "\n"
            )
        empty = metadata.parent / "images" / "FRII" / "empty.png"
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty.write_bytes(b"")
        self.manifest = create_lock_manifest(
            self.snapshot,
            repo_id="fixture/AstroVLBench",
            requested_revision="fixture",
            commit_sha="b" * 40,
        )
        write_lock_manifest(self.manifest, self.lock_path)
        before = {
            path.relative_to(self.snapshot).as_posix(): sha256_file(path)
            for path in self.snapshot.rglob("*")
            if path.is_file()
        }

        records, report = materialize_locked_records(self.lock_path)

        after = {
            path.relative_to(self.snapshot).as_posix(): sha256_file(path)
            for path in self.snapshot.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertEqual(len([row for row in records if row["task_key"] == "task2.first"]), 1)
        first = report["repair_report"]["first_metadata"]
        self.assertEqual(first["raw_rows"], 3)
        self.assertEqual(first["valid_rows"], 1)
        self.assertEqual(
            first["excluded_reason_counts"],
            {
                "missing_in_pinned_snapshot": 1,
                "upstream_zero_byte_image": 1,
            },
        )
        exclusions = [
            json.loads(line)
            for line in (
                self.base
                / "overlay"
                / "data"
                / "Task2_RadioMorph"
                / "MiraBest_F"
                / "exclusions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual({row["filename"] for row in exclusions}, {
            "images/FRII/missing.png",
            "images/FRII/empty.png",
        })

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
