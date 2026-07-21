from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from eval.paper.artifacts import (
    ArtifactError,
    PredictionStore,
    build_attempt,
    create_bundle,
    read_jsonl,
    redact_value,
)


class PaperArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "id": "one",
                "prompt": "Question one?",
                "reference": "Answer one",
                "image_sha256": "a" * 64,
            },
            {
                "id": "two",
                "prompt": "Question two?",
                "reference": "Answer two",
                "image_sha256": "b" * 64,
            },
        ]
        self.model_hash = "c" * 64

    def attempt(self, record, response="ok", error=None):
        return build_attempt(
            record=record,
            model_label="model",
            model_revision="d" * 40,
            backend="fixture",
            run_id="run",
            suite_protocol_hash="e" * 64,
            model_protocol_hash=self.model_hash,
            response=response,
            error=error,
        )

    def test_resume_only_accepts_clean_fingerprint_matched_successes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionStore(tmp, self.records, self.model_hash)
            store.append(self.attempt(self.records[0], response="", error="OOM"))
            self.assertEqual([row["id"] for row in store.pending_records()], ["one", "two"])
            store.append(self.attempt(self.records[0], response="final answer"))
            self.assertEqual([row["id"] for row in store.pending_records()], ["two"])
            with self.assertRaises(ArtifactError):
                store.finalize()
            store.append(self.attempt(self.records[1], response="second answer"))
            report = store.finalize()
            self.assertTrue(report.complete)
            self.assertEqual([row["id"] for row in read_jsonl(store.predictions_path)], ["one", "two"])

    def test_protocol_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionStore(tmp, self.records, self.model_hash)
            row = self.attempt(self.records[0])
            row["model_protocol_hash"] = "f" * 64
            with self.assertRaisesRegex(ArtifactError, "protocol"):
                store.append(row)

    def test_truncated_last_attempt_is_ignored_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionStore(tmp, self.records, self.model_hash)
            store.attempts_path.write_text('{"id":"broken"', encoding="utf-8")
            self.assertEqual(store.attempts(), [])

    def test_public_bundle_removes_references_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            (root / "predictions.jsonl").write_text(
                json.dumps({"id": "x", "reference": "owned text", "response": "model text"}) + "\n",
                encoding="utf-8",
            )
            (root / "image.jpg").write_bytes(b"image")
            bundle = create_bundle(root, Path(tmp) / "public.tar.gz", public=True)
            with tarfile.open(bundle, "r:gz") as archive:
                names = archive.getnames()
                self.assertFalse(any(name.endswith("image.jpg") for name in names))
                pred_name = next(name for name in names if name.endswith("predictions.jsonl"))
                payload = archive.extractfile(pred_name).read().decode("utf-8")
                self.assertNotIn("owned text", payload)
                self.assertIn("model text", payload)

    def test_recursive_redaction_covers_schema_variants_and_keeps_model_outputs(self) -> None:
        value = {
            "id": "sample-1",
            "model_label": "astraq_stage2",
            "predicted_label": "model-output-class",
            "response": "model output",
            "reference": "licensed caption",
            "Reference-Label": "AGN",
            "allowed_labels": ["AGN", "Galaxy"],
            "gold_answer": "secret",
            "target_label": "secret-class",
            "annotation": {"text": "source annotation"},
            "raw_annotation": "raw source annotation",
            "image": "source.png",
            "image_path": "/private/source.png",
            "image_relpath": "images/source.png",
            "source_object_id": "gated-object-17",
            "source_record": {"class": "secret"},
            "locked_files": ["task1/private.csv"],
            "conversations": [
                {"from": "human", "value": "Describe this image."},
                {"from": "gpt", "value": "licensed nested caption"},
            ],
            "reference_sha256": "a" * 64,
            "image_sha256": "b" * 64,
            "nested": [{"label": "hidden", "metric": 0.75}],
        }

        redacted = redact_value(value)

        self.assertEqual(redacted["model_label"], "astraq_stage2")
        self.assertEqual(redacted["predicted_label"], "model-output-class")
        self.assertEqual(redacted["response"], "model output")
        self.assertNotIn("reference_sha256", redacted)
        self.assertNotIn("image_sha256", redacted)
        self.assertEqual(redacted["nested"], [{"metric": 0.75}])
        serialized = json.dumps(redacted)
        for secret in (
            "licensed caption",
            "AGN",
            "Galaxy",
            "secret-class",
            "source annotation",
            "/private/source.png",
            "gated-object-17",
            "task1/private.csv",
            "licensed nested caption",
        ):
            self.assertNotIn(secret, serialized)

    def test_deepsdo_llava_source_is_private_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            llava = root / "audit_inputs" / "datasets" / "deepsdo" / "llava"
            llava.mkdir(parents=True)
            (llava / "test.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "deep-secret-1",
                            "conversations": [
                                {"from": "human", "value": "<image> Describe it."},
                                {"from": "gpt", "value": "licensed DeepSDO caption"},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            reports = root / "reports" / "deepsdo"
            reports.mkdir(parents=True)
            (reports / "summary.json").write_text('{"cider": 1.0}', encoding="utf-8")

            private = create_bundle(root, Path(tmp) / "private.tar.gz", public=False)
            public = create_bundle(root, Path(tmp) / "public.tar.gz", public=True)

            with tarfile.open(private, "r:gz") as archive:
                private_name = next(name for name in archive.getnames() if name.endswith("llava/test.json"))
                private_payload = archive.extractfile(private_name).read().decode("utf-8")
                self.assertIn("licensed DeepSDO caption", private_payload)
            with tarfile.open(public, "r:gz") as archive:
                public_names = archive.getnames()
                self.assertFalse(any(name.endswith("llava/test.json") for name in public_names))
                public_payload = "\n".join(
                    archive.extractfile(name).read().decode("utf-8", errors="ignore")
                    for name in public_names
                    if archive.getmember(name).isfile()
                )
                self.assertNotIn("licensed DeepSDO caption", public_payload)

    def test_gated_astro_rows_drop_reversible_ids_fingerprints_and_error_paths(self) -> None:
        redacted = redact_value(
            {
                "dataset": "astrovlbench",
                "id": "astrovlbench_task5_q1_secret-source",
                "sample_id": "secret-source",
                "image_id": "data/Task5/secret.fits",
                "reference_sha256": "a" * 64,
                "record_fingerprint": "b" * 64,
                "metadata": {
                    "asib_group": "C4",
                    "cluster_id": "spectrum_secret-source",
                    "source_row": "17",
                    "survey": "FIRST",
                },
                "response": "model rationale",
                "error": {
                    "type": "FileNotFoundError",
                    "message": "/gated/secret-source.png",
                    "traceback": "private traceback",
                },
            }
        )
        self.assertEqual(redacted["response"], "model rationale")
        self.assertEqual(redacted["error_type"], "FileNotFoundError")
        self.assertEqual(redacted["metadata"], {"survey": "FIRST"})
        for key in ("id", "sample_id", "image_id", "reference_sha256", "record_fingerprint", "error"):
            self.assertNotIn(key, redacted)
        self.assertNotIn("secret-source", json.dumps(redacted))

    def test_public_astro_bundle_removes_gated_metadata_from_jsonl_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            reports = root / "reports" / "astrovlbench"
            reports.mkdir(parents=True)
            gated_row = {
                "dataset": "astrovlbench",
                "id": "astro-secret-1",
                "sample_id": "astro-secret-1",
                "source_object_id": "spectrum-secret-1",
                "metadata": {
                    "asib_group": "C3",
                    "cluster_id": "spectrum-secret-1",
                    "survey": "NVSS",
                },
                "response": "model rationale retained",
            }
            (reports / "per_sample.jsonl").write_text(
                json.dumps(gated_row) + "\n", encoding="utf-8"
            )
            (reports / "per_sample.csv").write_text(
                "id,sample_id,asib_group,cluster_id,survey,response\n"
                "astro-secret-1,astro-secret-1,C3,spectrum-secret-1,NVSS,model rationale retained\n",
                encoding="utf-8",
            )

            bundle = create_bundle(root, Path(tmp) / "public.tar.gz", public=True)

            with tarfile.open(bundle, "r:gz") as archive:
                jsonl_name = next(name for name in archive.getnames() if name.endswith("per_sample.jsonl"))
                csv_name = next(name for name in archive.getnames() if name.endswith("per_sample.csv"))
                jsonl_payload = archive.extractfile(jsonl_name).read().decode("utf-8")
                csv_payload = archive.extractfile(csv_name).read().decode("utf-8")
            for payload in (jsonl_payload, csv_payload):
                self.assertNotIn("astro-secret-1", payload)
                self.assertNotIn("spectrum-secret-1", payload)
                self.assertNotIn("C3", payload)
                self.assertIn("model rationale retained", payload)
                self.assertIn("NVSS", payload)

    def test_public_bundle_redacts_csv_and_retains_generated_report_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            reports = root / "reports" / "deepsdo"
            reports.mkdir(parents=True)
            (reports / "per_sample.csv").write_text(
                "id,model_label,reference_label,answer,source_path,response,score\n"
                "x,stage2,AGN,owned,/gated/x.png,model-output,0.8\n",
                encoding="utf-8",
            )
            (root / "raw_source.png").write_bytes(b"raw image")
            (root / "source_annotations.txt").write_text(
                "gated annotation evidence", encoding="utf-8"
            )
            (reports / "figure.png").write_bytes(b"generated paper figure")

            bundle = create_bundle(root, Path(tmp) / "public.tar.gz", public=True)

            with tarfile.open(bundle, "r:gz") as archive:
                names = archive.getnames()
                self.assertFalse(any(name.endswith("raw_source.png") for name in names))
                self.assertFalse(any(name.endswith("source_annotations.txt") for name in names))
                self.assertTrue(any(name.endswith("reports/deepsdo/figure.png") for name in names))
                csv_name = next(name for name in names if name.endswith("per_sample.csv"))
                payload = archive.extractfile(csv_name).read().decode("utf-8")
                self.assertEqual(
                    payload,
                    "id,model_label,response,score\nx,stage2,model-output,0.8\n",
                )

    def test_bundle_directories_are_never_recursively_packaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            (root / "reports").mkdir(parents=True)
            (root / "reports" / "result.json").write_text('{"score": 1}', encoding="utf-8")
            nested = root / "reports" / "bundles"
            nested.mkdir()
            (nested / "old-private.tar.gz").write_bytes(b"previous archive")
            top = root / "bundles"
            top.mkdir()
            (top / "old-public.tar.gz").write_bytes(b"previous archive")

            bundle = create_bundle(root, top / "new-private.tar.gz", public=False)

            with tarfile.open(bundle, "r:gz") as archive:
                names = archive.getnames()
            self.assertFalse(any("bundles" in Path(name).parts for name in names))
            self.assertTrue(any(name.endswith("reports/result.json") for name in names))

    def test_public_bundle_fails_closed_on_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            (root / "predictions.json").write_text(
                '{"reference": "must never be copied"', encoding="utf-8"
            )
            target = Path(tmp) / "public.tar.gz"

            with self.assertRaisesRegex(ArtifactError, "safely redact"):
                create_bundle(root, target, public=True)
            self.assertFalse(target.exists())

    def test_bundle_rejects_symlinked_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            outside = Path(tmp) / "outside.txt"
            outside.write_text("outside data", encoding="utf-8")
            link = root / "linked.txt"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("This platform does not permit symlink creation")

            with self.assertRaisesRegex(ArtifactError, "symlinked"):
                create_bundle(root, Path(tmp) / "private.tar.gz", public=False)


if __name__ == "__main__":
    unittest.main()
