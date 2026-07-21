from __future__ import annotations

import importlib.util
import math
import subprocess
import unittest
from unittest import TestCase, main, mock

from eval.paper.metrics import (
    MetricDependencyError,
    classification_metrics,
    coco_bleu_compatible_scores,
    coco_cider_scores,
    corpus_bleu,
    corpus_bleu_scores,
    exact_match,
    normalize_text,
    optional_caption_metrics,
    response_is_valid,
    rouge_l,
    sentence_bleu,
    token_f1,
    valid_response_rate,
)


class TextMetricTests(TestCase):
    def test_normalized_exact_match(self) -> None:
        self.assertEqual(normalize_text("  Solar-flare! "), "solar flare")
        self.assertEqual(exact_match("A solar flare.", ["a solar flare", "coronal hole"]), 1.0)
        self.assertEqual(exact_match("flare present", "flare"), 0.0)

    def test_token_f1_uses_multiset_overlap(self) -> None:
        self.assertAlmostEqual(token_f1("red red blue", "red blue blue"), 2.0 / 3.0)
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("flare", "sunspot"), 0.0)

    def test_rouge_l(self) -> None:
        self.assertAlmostEqual(rouge_l("a b c", "a x c"), 2.0 / 3.0)
        self.assertEqual(rouge_l("", ""), 1.0)
        self.assertEqual(rouge_l("abc", "xyz"), 0.0)

    def test_bleu_exact_and_clipped_unigram(self) -> None:
        predictions = ["the active region is bright", "a dark coronal hole"]
        references = [["the active region is bright"], ["a dark coronal hole"]]
        scores = corpus_bleu_scores(predictions, references)
        self.assertEqual(scores, {"bleu_1": 1.0, "bleu_2": 1.0, "bleu_3": 1.0, "bleu_4": 1.0})
        self.assertAlmostEqual(corpus_bleu(["cat sat"], ["cat slept"], max_order=1), 0.5)

    def test_bleu_brevity_penalty_and_sentence_smoothing(self) -> None:
        score = corpus_bleu(["a b"], ["a b c d"], max_order=1)
        self.assertAlmostEqual(score, math.exp(-1.0))
        self.assertGreater(sentence_bleu("a x", "a b", max_order=2, smooth=True), 0.0)
        self.assertEqual(sentence_bleu("a", "a", max_order=4, smooth=True), 0.0)

    def test_bleu_rejects_misaligned_or_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            corpus_bleu([], [])
        with self.assertRaises(ValueError):
            corpus_bleu(["one"], [])

    def test_coco_compatible_bleu_uses_whitespace_and_historical_smoothing(self) -> None:
        scores = coco_bleu_compatible_scores(["cat sat"], ["cat slept"])
        self.assertAlmostEqual(scores["bleu_1"], 0.5)
        self.assertGreater(scores["bleu_2"], 0.0)
        self.assertLess(scores["bleu_2"], 1e-6)
        self.assertLess(
            coco_bleu_compatible_scores(["Sun."], ["sun"])["bleu_1"],
            1e-6,
        )


class ValidityAndClassificationTests(TestCase):
    def test_validity_contract_and_denominator(self) -> None:
        rows = [
            {"prediction": "usable", "status": "ok"},
            {"prediction": "", "status": "ok"},
            {"prediction": "text", "error": "out of memory"},
            {"prediction": "text", "leak_flag": True},
        ]
        self.assertTrue(response_is_valid("answer", status="complete"))
        self.assertFalse(response_is_valid("answer", status="failed"))
        self.assertEqual(
            valid_response_rate(rows),
            {"valid_response_rate": 0.25, "n_valid": 1, "n_invalid": 3, "n": 4},
        )

    def test_classification_keeps_invalid_predictions_as_failures(self) -> None:
        result = classification_metrics(
            ["a", "a", "b", "b"],
            ["a", None, "a", "b"],
            labels=["a", "b"],
        )
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], (0.5 + 2.0 / 3.0) / 2.0)
        self.assertEqual(result["invalid_response_rate"], 0.25)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 1], [1, 1, 0]])
        self.assertEqual(result["prediction_labels"], ["a", "b", "__invalid__"])

    def test_unknown_predictions_are_invalid_and_unknown_truth_is_error(self) -> None:
        result = classification_metrics([0, 1], [2, 1], labels=[0, 1])
        self.assertEqual(result["n_invalid"], 1)
        with self.assertRaises(ValueError):
            classification_metrics([2], [1], labels=[0, 1])


class OptionalMetricAdapterTests(TestCase):
    @mock.patch("eval.paper.metrics._installed_version", return_value=None)
    def test_missing_optional_dependency_is_explicit_in_paper_mode(self, _version: mock.Mock) -> None:
        with self.assertRaises(MetricDependencyError):
            coco_cider_scores(["prediction"], ["reference"], paper_mode=True)

    @mock.patch("eval.paper.metrics._installed_version", return_value=None)
    def test_missing_optional_dependency_is_structured_in_exploration_mode(self, _version: mock.Mock) -> None:
        result = coco_cider_scores(["prediction"], ["reference"], paper_mode=False)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["value"])

    @unittest.skipUnless(
        importlib.util.find_spec("pycocoevalcap") is not None,
        "pinned optional COCO scorer is not installed",
    )
    def test_pinned_coco_caption_golden_fixture(self) -> None:
        predictions = [
            "A bright solar flare erupts from the active region.",
            "Dark coronal hole near the southern pole.",
            "A filament crosses the solar disk.",
        ]
        references = [
            "A bright flare erupts from an active region on the Sun.",
            "A dark coronal hole is visible near the south pole.",
            "A long filament extends across the solar disk.",
        ]
        real_popen = subprocess.Popen

        def portable_heap(command, *args, **kwargs):
            command = list(command)
            if "-Xmx2G" in command:
                command[command.index("-Xmx2G")] = "-Xmx512m"
            return real_popen(command, *args, **kwargs)

        with mock.patch("eval.paper.metrics.subprocess.Popen", side_effect=portable_heap):
            result = optional_caption_metrics(
                predictions,
                references,
                metrics=["bleu", "cider", "meteor", "rouge"],
                paper_mode=True,
                pins={"pycocoevalcap": "1.2"},
            )
        expected_bleu = [
            0.6282699850093113,
            0.4652917009107046,
            0.3090929448829108,
            3.585377008170886e-05,
        ]
        for order, expected in enumerate(expected_bleu, 1):
            self.assertAlmostEqual(result["bleu"]["value"][f"bleu_{order}"], expected, places=10)
        self.assertAlmostEqual(result["cider"]["value"], 2.9051410454269586, places=10)
        self.assertAlmostEqual(result["meteor"]["value"], 0.3656883036880605, places=10)
        self.assertAlmostEqual(result["rouge"]["value"], 0.689353275206188, places=10)


if __name__ == "__main__":
    main()
