"""Deterministic percentile bootstrap utilities for paper evaluation.

Callbacks receive the complete resampled records, so non-additive corpus metrics
(for example corpus BLEU) are recomputed for every replicate rather than averaged
from per-record approximations.  Cluster mode samples source/image identifiers and
then carries every record belonging to each selected cluster.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Generic, Literal, TypeVar


T = TypeVar("T")
U = TypeVar("U")
ResamplingUnit = Literal["record", "cluster"]


@dataclass(frozen=True)
class BootstrapInterval:
    """A scalar estimate and deterministic percentile confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    seed: int
    resampling_unit: ResamplingUnit
    n_items: int
    n_clusters: int | None = None
    method: str = "percentile"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def mean_statistic(values: Sequence[float]) -> float:
    """Arithmetic mean suitable as the default record-level statistic."""

    if not values:
        raise ValueError("cannot calculate a mean for an empty sample")
    return sum(float(value) for value in values) / len(values)


def _validate_options(n_items: int, n_resamples: int, confidence: float, unit: str) -> None:
    if n_items <= 0:
        raise ValueError("at least one item is required")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if unit not in {"record", "cluster"}:
        raise ValueError("resampling_unit must be 'record' or 'cluster'")


def _finite_scalar(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must return a scalar number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context} returned a non-finite value: {result!r}")
    return result


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolation percentile (the common R-7/NumPy default rule)."""

    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def _cluster_ids_from_key(
    rows: Sequence[T],
    cluster_key: str | Callable[[T], Hashable],
) -> list[Hashable]:
    if callable(cluster_key):
        return [cluster_key(row) for row in rows]
    ids: list[Hashable] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("a string cluster_key requires mapping rows")
        if cluster_key not in row:
            raise KeyError(f"row {index} has no cluster key {cluster_key!r}")
        value = row[cluster_key]
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"cluster ID at row {index} is not hashable") from exc
        ids.append(value)
    return ids


def _resolve_cluster_ids(
    rows: Sequence[T],
    *,
    resampling_unit: ResamplingUnit,
    cluster_ids: Sequence[Hashable] | None,
    cluster_key: str | Callable[[T], Hashable] | None,
) -> list[Hashable] | None:
    if cluster_ids is not None and cluster_key is not None:
        raise ValueError("provide cluster_ids or cluster_key, not both")
    if resampling_unit == "record":
        if cluster_ids is not None or cluster_key is not None:
            raise ValueError("cluster IDs are only valid when resampling_unit='cluster'")
        return None
    resolved = list(cluster_ids) if cluster_ids is not None else None
    if resolved is None and cluster_key is not None:
        resolved = _cluster_ids_from_key(rows, cluster_key)
    if resolved is None:
        raise ValueError("cluster resampling requires cluster_ids or cluster_key")
    if len(resolved) != len(rows):
        raise ValueError("cluster_ids and rows must have equal length")
    if any(value is None for value in resolved):
        raise ValueError("cluster IDs cannot be None")
    for index, value in enumerate(resolved):
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"cluster ID at row {index} is not hashable") from exc
    return resolved


def _cluster_members(cluster_ids: Sequence[Hashable]) -> tuple[list[Hashable], dict[Hashable, list[int]]]:
    members: dict[Hashable, list[int]] = defaultdict(list)
    ordered_ids: list[Hashable] = []
    for index, cluster_id in enumerate(cluster_ids):
        if cluster_id not in members:
            ordered_ids.append(cluster_id)
        members[cluster_id].append(index)
    return ordered_ids, dict(members)


def resample_indices(
    n_items: int,
    rng: random.Random,
    *,
    resampling_unit: ResamplingUnit = "record",
    cluster_ids: Sequence[Hashable] | None = None,
) -> list[int]:
    """Draw one bootstrap sample as source indices."""

    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if resampling_unit == "record":
        if cluster_ids is not None:
            raise ValueError("cluster_ids require resampling_unit='cluster'")
        return [rng.randrange(n_items) for _ in range(n_items)]
    if resampling_unit != "cluster":
        raise ValueError("resampling_unit must be 'record' or 'cluster'")
    if cluster_ids is None or len(cluster_ids) != n_items:
        raise ValueError("cluster mode requires one cluster ID per item")
    ordered_ids, members = _cluster_members(cluster_ids)
    selected = [ordered_ids[rng.randrange(len(ordered_ids))] for _ in ordered_ids]
    return [index for cluster_id in selected for index in members[cluster_id]]


def _interval(
    estimate: float,
    replicates: Sequence[float],
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
    resampling_unit: ResamplingUnit,
    n_items: int,
    n_clusters: int | None,
) -> BootstrapInterval:
    alpha = 1.0 - confidence
    return BootstrapInterval(
        estimate=estimate,
        lower=percentile(replicates, alpha / 2.0),
        upper=percentile(replicates, 1.0 - alpha / 2.0),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        resampling_unit=resampling_unit,
        n_items=n_items,
        n_clusters=n_clusters,
    )


def bootstrap_ci(
    rows: Sequence[T],
    statistic: Callable[[Sequence[T]], float] | None = None,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
    resampling_unit: ResamplingUnit = "record",
    cluster_ids: Sequence[Hashable] | None = None,
    cluster_key: str | Callable[[T], Hashable] | None = None,
) -> BootstrapInterval:
    """Absolute percentile interval for a scalar statistic.

    When ``statistic`` is omitted, rows must be numeric and their arithmetic
    mean is used.  Supply a callback to recompute corpus BLEU/CIDEr or any other
    non-additive metric from each complete resample.
    """

    rows = list(rows)
    _validate_options(len(rows), n_resamples, confidence, resampling_unit)
    resolved_clusters = _resolve_cluster_ids(
        rows,
        resampling_unit=resampling_unit,
        cluster_ids=cluster_ids,
        cluster_key=cluster_key,
    )
    if statistic is None:
        statistic = mean_statistic  # type: ignore[assignment]

    estimate = _finite_scalar(statistic(rows), context="statistic")
    rng = random.Random(seed)
    replicates: list[float] = []
    for replicate_index in range(n_resamples):
        indices = resample_indices(
            len(rows),
            rng,
            resampling_unit=resampling_unit,
            cluster_ids=resolved_clusters,
        )
        sample = [rows[index] for index in indices]
        replicates.append(
            _finite_scalar(statistic(sample), context=f"statistic at bootstrap replicate {replicate_index}")
        )

    return _interval(
        estimate,
        replicates,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        resampling_unit=resampling_unit,
        n_items=len(rows),
        n_clusters=len(set(resolved_clusters)) if resolved_clusters is not None else None,
    )


def paired_bootstrap_ci(
    left_rows: Sequence[T],
    right_rows: Sequence[U],
    statistic: Callable[[Sequence[T], Sequence[U]], float] | None = None,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
    resampling_unit: ResamplingUnit = "record",
    cluster_ids: Sequence[Hashable] | None = None,
    cluster_key: str | Callable[[Any], Hashable] | None = None,
) -> BootstrapInterval:
    """Paired interval using identical sampled positions for both systems.

    The default statistic is ``mean(left) - mean(right)``.  A custom callback
    may recompute corpus metrics for each side and return their difference.
    Pairing is positional; callers should align and validate expected IDs first.
    """

    left = list(left_rows)
    right = list(right_rows)
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    _validate_options(len(left), n_resamples, confidence, resampling_unit)
    resolved_clusters = _resolve_cluster_ids(
        left,
        resampling_unit=resampling_unit,
        cluster_ids=cluster_ids,
        cluster_key=cluster_key,
    )
    if resampling_unit == "cluster" and cluster_key is not None and cluster_ids is None:
        right_clusters = _cluster_ids_from_key(right, cluster_key)
        if right_clusters != resolved_clusters:
            raise ValueError("paired rows do not have identical cluster IDs in the same order")

    if statistic is None:
        def default_statistic(left_values: Sequence[Any], right_values: Sequence[Any]) -> float:
            return mean_statistic(left_values) - mean_statistic(right_values)

        statistic = default_statistic

    estimate = _finite_scalar(statistic(left, right), context="paired statistic")
    rng = random.Random(seed)
    replicates: list[float] = []
    for replicate_index in range(n_resamples):
        indices = resample_indices(
            len(left),
            rng,
            resampling_unit=resampling_unit,
            cluster_ids=resolved_clusters,
        )
        left_sample = [left[index] for index in indices]
        right_sample = [right[index] for index in indices]
        replicates.append(
            _finite_scalar(
                statistic(left_sample, right_sample),
                context=f"paired statistic at bootstrap replicate {replicate_index}",
            )
        )

    return _interval(
        estimate,
        replicates,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        resampling_unit=resampling_unit,
        n_items=len(left),
        n_clusters=len(set(resolved_clusters)) if resolved_clusters is not None else None,
    )


def bootstrap_metric_map(
    rows: Sequence[T],
    statistics: Mapping[str, Callable[[Sequence[T]], float]],
    **bootstrap_options: Any,
) -> dict[str, BootstrapInterval]:
    """Convenience wrapper for named absolute statistics."""

    if not statistics:
        raise ValueError("at least one statistic is required")
    return {
        name: bootstrap_ci(rows, statistic, **bootstrap_options)
        for name, statistic in statistics.items()
    }


def paired_bootstrap_metric_map(
    left_rows: Sequence[T],
    right_rows: Sequence[U],
    statistics: Mapping[str, Callable[[Sequence[T], Sequence[U]], float]],
    **bootstrap_options: Any,
) -> dict[str, BootstrapInterval]:
    """Convenience wrapper for named paired statistics."""

    if not statistics:
        raise ValueError("at least one statistic is required")
    return {
        name: paired_bootstrap_ci(left_rows, right_rows, statistic, **bootstrap_options)
        for name, statistic in statistics.items()
    }


# Descriptive alias used by orchestration code.
absolute_bootstrap_ci = bootstrap_ci


__all__ = [
    "BootstrapInterval",
    "absolute_bootstrap_ci",
    "bootstrap_ci",
    "bootstrap_metric_map",
    "mean_statistic",
    "paired_bootstrap_ci",
    "paired_bootstrap_metric_map",
    "percentile",
    "resample_indices",
]
