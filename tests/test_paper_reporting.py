from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from eval.paper.reporting import (
    cautious_comparison,
    escape_latex,
    escape_markdown,
    redact_for_sharing,
    save_figure,
    write_table_bundle,
)


class ReportingTests(TestCase):
    def test_table_bundle_writes_all_formats_and_escapes(self) -> None:
        rows = [
            {"model": "A&B_1", "score": 0.5, "note": "left|right\nnext"},
            {"model": "B", "score": 0.25, "note": None},
        ]
        with TemporaryDirectory() as temporary_directory:
            paths = write_table_bundle(
                Path(temporary_directory) / "scores",
                rows,
                columns=["model", "score", "note"],
                caption="Model results",
                label="tab:model_results",
            )
            self.assertEqual(set(paths), {"json", "csv", "markdown", "latex"})
            self.assertTrue(all(path.exists() for path in paths.values()))

            self.assertEqual(json.loads(paths["json"].read_text(encoding="utf-8")), rows)
            with paths["csv"].open(encoding="utf-8", newline="") as handle:
                parsed = list(csv.DictReader(handle))
            self.assertEqual(parsed[0]["model"], "A&B_1")
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn(r"left\|right<br>next", markdown)
            latex = paths["latex"].read_text(encoding="utf-8")
            self.assertIn(r"A\&B\_1", latex)
            self.assertIn(r"\begin{tabular}{lrr}", latex)

    def test_escape_helpers(self) -> None:
        self.assertEqual(escape_markdown("a|b\nc"), r"a\|b<br>c")
        self.assertEqual(escape_latex("a_b & 50%"), r"a\_b \& 50\%")

    def test_cautious_narrative_when_interval_crosses_zero(self) -> None:
        narrative = cautious_comparison(
            "Stage 2",
            0.61,
            "Stage 1",
            0.59,
            metric="CIDEr",
            difference_ci=(-0.01, 0.05),
            scope="the DeepSDO test split",
        )
        self.assertIn("point estimate favored Stage 2", narrative)
        self.assertIn("includes zero", narrative)
        self.assertIn("specific to the evaluated data and protocol", narrative)
        self.assertNotIn("objectively", narrative.casefold())

    def test_cautious_narrative_when_interval_excludes_zero(self) -> None:
        narrative = cautious_comparison(
            "Stage 2",
            0.71,
            "Stage 1",
            0.55,
            metric="ROUGE-L",
            difference_ci=(0.04, 0.25),
        )
        self.assertIn("excludes zero", narrative)
        self.assertNotIn("significant", narrative.casefold())

    def test_recursive_redaction_removes_sources_but_preserves_hashes_and_predictions(self) -> None:
        data = {
            "sample_id": "deep-1",
            "reference": "copyrighted caption",
            "reference_hash": "abc123",
            "prediction": "model output",
            "nested": {
                "raw_annotation": {"caption": "source"},
                "score": 0.8,
            },
            "items": [{"image_path": "/private/image.jpg", "metric": 1.0}],
        }
        redacted = redact_for_sharing(data)
        self.assertNotIn("reference", redacted)
        self.assertEqual(redacted["reference_hash"], "abc123")
        self.assertEqual(redacted["prediction"], "model output")
        self.assertEqual(redacted["nested"], {"score": 0.8})
        self.assertEqual(redacted["items"], [{"metric": 1.0}])

    def test_save_figure_requests_svg_png_and_pdf(self) -> None:
        class FakeFigure:
            def __init__(self) -> None:
                self.calls = []

            def savefig(self, path, **kwargs) -> None:
                self.calls.append((Path(path), kwargs))
                Path(path).write_bytes(b"figure")

        figure = FakeFigure()
        with TemporaryDirectory() as temporary_directory:
            outputs = save_figure(figure, Path(temporary_directory) / "figure")
            self.assertEqual(set(outputs), {"svg", "png", "pdf"})
            self.assertTrue(all(path.exists() for path in outputs.values()))
            self.assertEqual([call[1]["dpi"] for call in figure.calls], [300, 300, 300])


if __name__ == "__main__":
    main()
