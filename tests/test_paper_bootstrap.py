from __future__ import annotations

import random
from unittest import TestCase, main

from eval.paper.bootstrap import (
    bootstrap_ci,
    paired_bootstrap_ci,
    percentile,
    resample_indices,
)
from eval.paper.metrics import corpus_bleu


class BootstrapTests(TestCase):
    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(percentile([7.0], 0.95), 7.0)

    def test_default_is_ten_thousand_seed_42(self) -> None:
        result = bootstrap_ci([3.0])
        self.assertEqual(result.estimate, 3.0)
        self.assertEqual(result.lower, 3.0)
        self.assertEqual(result.upper, 3.0)
        self.assertEqual(result.n_resamples, 10_000)
        self.assertEqual(result.seed, 42)

    def test_record_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_ci([1.0, 2.0, 5.0, 8.0], n_resamples=250, seed=17)
        second = bootstrap_ci([1.0, 2.0, 5.0, 8.0], n_resamples=250, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first.resampling_unit, "record")

    def test_cluster_sampling_carries_all_cluster_records(self) -> None:
        cluster_ids = ["image-a", "image-a", "image-b"]
        indices = resample_indices(
            3,
            random.Random(9),
            resampling_unit="cluster",
            cluster_ids=cluster_ids,
        )
        self.assertEqual(indices.count(0), indices.count(1))
        interval = bootstrap_ci(
            [1.0, 3.0, 9.0],
            n_resamples=200,
            resampling_unit="cluster",
            cluster_ids=cluster_ids,
        )
        self.assertEqual(interval.n_clusters, 2)
        self.assertEqual(interval.n_items, 3)

    def test_cluster_key_supports_mapping_rows(self) -> None:
        rows = [
            {"image_id": "x", "score": 1.0},
            {"image_id": "x", "score": 3.0},
            {"image_id": "y", "score": 5.0},
        ]
        interval = bootstrap_ci(
            rows,
            lambda sample: sum(row["score"] for row in sample) / len(sample),
            n_resamples=100,
            resampling_unit="cluster",
            cluster_key="image_id",
        )
        self.assertEqual(interval.estimate, 3.0)
        self.assertEqual(interval.n_clusters, 2)

    def test_paired_interval_preserves_pairs(self) -> None:
        result = paired_bootstrap_ci(
            [2.0, 4.0, 8.0],
            [1.0, 3.0, 7.0],
            n_resamples=300,
        )
        self.assertAlmostEqual(result.estimate, 1.0)
        self.assertAlmostEqual(result.lower, 1.0)
        self.assertAlmostEqual(result.upper, 1.0)

    def test_callback_recomputes_corpus_bleu_each_replicate(self) -> None:
        left = [
            {"prediction": "a bright active region", "reference": "a bright active region", "image": "x"},
            {"prediction": "a dark coronal hole", "reference": "a dark coronal hole", "image": "y"},
        ]
        right = [
            {"prediction": "unrelated words appear here", "reference": "a bright active region", "image": "x"},
            {"prediction": "another wrong generated caption", "reference": "a dark coronal hole", "image": "y"},
        ]
        calls = 0

        def bleu_difference(left_sample, right_sample):
            nonlocal calls
            calls += 1
            left_score = corpus_bleu(
                [row["prediction"] for row in left_sample],
                [row["reference"] for row in left_sample],
                max_order=2,
            )
            right_score = corpus_bleu(
                [row["prediction"] for row in right_sample],
                [row["reference"] for row in right_sample],
                max_order=2,
            )
            return left_score - right_score

        result = paired_bootstrap_ci(
            left,
            right,
            bleu_difference,
            n_resamples=50,
            resampling_unit="cluster",
            cluster_key="image",
        )
        self.assertEqual(calls, 51)
        self.assertEqual(result.estimate, 1.0)
        self.assertEqual(result.n_clusters, 2)

    def test_invalid_inputs_fail_early(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_ci([], n_resamples=10)
        with self.assertRaises(ValueError):
            bootstrap_ci([1.0], n_resamples=0)
        with self.assertRaises(ValueError):
            bootstrap_ci([1.0], resampling_unit="cluster")
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1.0], [1.0, 2.0], n_resamples=10)


if __name__ == "__main__":
    main()
