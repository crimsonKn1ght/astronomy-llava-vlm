"""Dependency-light metrics used by the paper evaluation pipeline.

The functions in this module deliberately accept plain strings, sequences and
mappings.  They do not depend on the paper protocol/configuration objects, which
makes it possible to re-score saved generations without importing model code.

Optional reference implementations (COCO caption scorers and Sentence-BERT) are
loaded only when their adapter is called.  In ``paper_mode`` an unavailable or
incorrectly pinned dependency is a hard error; exploratory callers may instead
receive a structured ``unavailable`` result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
import math
from pathlib import Path
import queue
import re
import subprocess
import threading
from typing import Any, Callable


_TOKEN_RE = re.compile(r"\w+(?:['\N{RIGHT SINGLE QUOTATION MARK}]\w+)?", re.UNICODE)
INVALID_PREDICTION = "__invalid__"


class MetricDependencyError(RuntimeError):
    """Raised when a pinned optional scorer cannot be used reproducibly."""


def tokenize(text: Any) -> list[str]:
    """Return stable, lowercase word tokens for a scalar value."""

    if text is None:
        return []
    return _TOKEN_RE.findall(str(text).casefold())


def normalize_text(text: Any) -> str:
    """Normalize text for exact-match comparisons."""

    return " ".join(tokenize(text))


def _references(references: str | Sequence[str]) -> list[str]:
    if isinstance(references, str):
        return [references]
    refs = [str(reference) for reference in references]
    if not refs:
        raise ValueError("at least one reference is required")
    return refs


def exact_match(prediction: Any, references: str | Sequence[str]) -> float:
    """Normalized exact match against one or more references (0.0 or 1.0)."""

    normalized = normalize_text(prediction)
    return float(any(normalized == normalize_text(reference) for reference in _references(references)))


def token_f1(prediction: Any, references: str | Sequence[str]) -> float:
    """Maximum multiset token F1 against one or more references."""

    prediction_tokens = tokenize(prediction)

    def score(reference: str) -> float:
        reference_tokens = tokenize(reference)
        if not prediction_tokens and not reference_tokens:
            return 1.0
        if not prediction_tokens or not reference_tokens:
            return 0.0
        overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        return 2.0 * precision * recall / (precision + recall)

    return max(score(reference) for reference in _references(references))


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """LCS length using O(min(m, n)) memory."""

    if len(left) < len(right):
        short, long = left, right
    else:
        short, long = right, left
    previous = [0] * (len(short) + 1)
    for long_token in long:
        current = [0]
        for index, short_token in enumerate(short, start=1):
            if long_token == short_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: Any, references: str | Sequence[str], *, beta: float = 1.0) -> float:
    """Maximum ROUGE-L F-score against one or more references."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    prediction_tokens = tokenize(prediction)

    def score(reference: str) -> float:
        reference_tokens = tokenize(reference)
        if not prediction_tokens and not reference_tokens:
            return 1.0
        if not prediction_tokens or not reference_tokens:
            return 0.0
        lcs = _lcs_length(prediction_tokens, reference_tokens)
        precision = lcs / len(prediction_tokens)
        recall = lcs / len(reference_tokens)
        denominator = recall + (beta * beta * precision)
        if denominator == 0:
            return 0.0
        return (1.0 + beta * beta) * precision * recall / denominator

    return max(score(reference) for reference in _references(references))


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    if order <= 0:
        raise ValueError("ngram order must be positive")
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _closest_reference_length(prediction_length: int, references: Sequence[Sequence[str]]) -> int:
    return min((len(reference) for reference in references), key=lambda length: (abs(length - prediction_length), length))


def _bleu_from_counts(
    matches: Sequence[int],
    possible: Sequence[int],
    prediction_length: int,
    reference_length: int,
    *,
    smooth: bool,
) -> float:
    if prediction_length == 0:
        return 0.0
    precisions: list[float] = []
    for matched, total in zip(matches, possible):
        if total == 0:
            return 0.0
        if smooth:
            precisions.append((matched + 1.0) / (total + 1.0))
        else:
            if matched == 0:
                return 0.0
            precisions.append(matched / total)
    geo_mean = math.exp(sum(math.log(value) for value in precisions) / len(precisions))
    brevity_penalty = 1.0 if prediction_length > reference_length else math.exp(
        1.0 - (reference_length / prediction_length)
    )
    return brevity_penalty * geo_mean


def corpus_bleu(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    max_order: int = 4,
    smooth: bool = False,
) -> float:
    """Compute cumulative modified-precision corpus BLEU.

    ``references[i]`` may be either one reference or a sequence of references.
    The implementation follows the standard clipped-count and closest-reference
    length definitions and intentionally returns a 0--1 value.
    """

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        raise ValueError("at least one prediction is required")
    if max_order <= 0:
        raise ValueError("max_order must be positive")

    matches = [0] * max_order
    possible = [0] * max_order
    prediction_length = 0
    reference_length = 0

    for prediction, reference_values in zip(predictions, references):
        prediction_tokens = tokenize(prediction)
        reference_tokens = [tokenize(reference) for reference in _references(reference_values)]
        prediction_length += len(prediction_tokens)
        reference_length += _closest_reference_length(len(prediction_tokens), reference_tokens)

        for order in range(1, max_order + 1):
            prediction_ngrams = _ngrams(prediction_tokens, order)
            maximum_reference_counts: Counter[tuple[str, ...]] = Counter()
            for tokenized_reference in reference_tokens:
                reference_ngrams = _ngrams(tokenized_reference, order)
                for ngram, count in reference_ngrams.items():
                    maximum_reference_counts[ngram] = max(maximum_reference_counts[ngram], count)
            matches[order - 1] += sum((prediction_ngrams & maximum_reference_counts).values())
            possible[order - 1] += sum(prediction_ngrams.values())

    return _bleu_from_counts(
        matches,
        possible,
        prediction_length,
        reference_length,
        smooth=smooth,
    )


def corpus_bleu_scores(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    max_order: int = 4,
    smooth: bool = False,
) -> dict[str, float]:
    """Return cumulative BLEU-1 through BLEU-``max_order``."""

    return {
        f"bleu_{order}": corpus_bleu(
            predictions,
            references,
            max_order=order,
            smooth=smooth,
        )
        for order in range(1, max_order + 1)
    }


def coco_bleu_compatible_scores(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    max_order: int = 4,
) -> dict[str, float]:
    """Recompute the corpus BLEU used by ``pycocoevalcap`` efficiently.

    COCO's ``BleuScorer`` uses whitespace tokenization, closest-reference
    lengths, and its historical ``tiny``/``small`` numerical smoothing.  This
    dependency-free equivalent is used inside 10,000-replicate bootstraps,
    where constructing the package scorer on every replicate would be
    needlessly expensive.  Paper-mode aggregate scoring still calls the pinned
    package and cross-checks its result against this implementation.
    """

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        raise ValueError("at least one prediction is required")
    if max_order <= 0:
        raise ValueError("max_order must be positive")

    total_test_length = 0
    total_reference_length = 0
    guesses = [0] * max_order
    correct = [0] * max_order
    for prediction, reference_values in zip(predictions, references):
        prediction_tokens = str(prediction).split()
        reference_tokens = [str(reference).split() for reference in _references(reference_values)]
        test_length = len(prediction_tokens)
        reference_length = min(
            (len(tokens) for tokens in reference_tokens),
            key=lambda length: (abs(length - test_length), length),
        )
        total_test_length += test_length
        total_reference_length += reference_length

        maximum_reference_counts: Counter[tuple[str, ...]] = Counter()
        for tokens in reference_tokens:
            for order in range(1, max_order + 1):
                for ngram, count in _ngrams(tokens, order).items():
                    maximum_reference_counts[ngram] = max(
                        maximum_reference_counts[ngram], count
                    )
        for order in range(1, max_order + 1):
            prediction_counts = _ngrams(prediction_tokens, order)
            guesses[order - 1] += max(0, test_length - order + 1)
            correct[order - 1] += sum(
                min(count, maximum_reference_counts.get(ngram, 0))
                for ngram, count in prediction_counts.items()
            )

    tiny = 1e-15
    small = 1e-9
    cumulative = 1.0
    scores: dict[str, float] = {}
    ratio = (total_test_length + tiny) / (total_reference_length + small)
    brevity_penalty = math.exp(1.0 - 1.0 / ratio) if ratio < 1.0 else 1.0
    for order in range(1, max_order + 1):
        cumulative *= (correct[order - 1] + tiny) / (guesses[order - 1] + small)
        scores[f"bleu_{order}"] = cumulative ** (1.0 / order) * brevity_penalty
    return scores


def sentence_bleu(
    prediction: str,
    references: str | Sequence[str],
    *,
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    """Sentence BLEU, smoothed by default because per-item counts are sparse."""

    return corpus_bleu([prediction], [references], max_order=max_order, smooth=smooth)


_INVALID_STATUSES = {"error", "failed", "invalid", "empty", "leak", "missing", "technical_failure"}


def response_is_valid(
    response: Any,
    *,
    status: Any = None,
    error: Any = None,
    leak_flag: Any = False,
) -> bool:
    """Return whether a generation is technically valid for scoring."""

    if error not in (None, "", False):
        return False
    if bool(leak_flag):
        return False
    if status is not None and str(status).strip().casefold() in _INVALID_STATUSES:
        return False
    return bool(str(response).strip()) if response is not None else False


def valid_response_rate(
    rows: Sequence[Mapping[str, Any]],
    *,
    response_key: str = "prediction",
    status_key: str = "status",
    error_key: str = "error",
    leak_key: str = "leak_flag",
) -> dict[str, int | float]:
    """Return valid/invalid counts and rate without dropping any expected row."""

    valid = sum(
        response_is_valid(
            row.get(response_key),
            status=row.get(status_key),
            error=row.get(error_key),
            leak_flag=row.get(leak_key, False),
        )
        for row in rows
    )
    total = len(rows)
    return {
        "valid_response_rate": valid / total if total else 0.0,
        "n_valid": valid,
        "n_invalid": total - valid,
        "n": total,
    }


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    invalid_label: str = INVALID_PREDICTION,
) -> dict[str, Any]:
    """Classification metrics with invalid predictions retained as failures.

    Macro-F1 and balanced accuracy are averaged over the true/declared classes.
    A blank, ``None``, or out-of-vocabulary prediction is placed in the explicit
    invalid column of the returned confusion matrix.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if labels is None:
        class_labels = sorted(set(y_true), key=lambda value: str(value))
    else:
        class_labels = list(labels)
    if not class_labels:
        raise ValueError("at least one class label is required")
    if len(set(class_labels)) != len(class_labels):
        raise ValueError("labels must be unique")
    unknown_truth = [value for value in y_true if value not in class_labels]
    if unknown_truth:
        raise ValueError(f"true labels outside declared labels: {unknown_truth[:3]!r}")

    label_to_index = {label: index for index, label in enumerate(class_labels)}
    matrix = [[0 for _ in range(len(class_labels) + 1)] for _ in class_labels]
    normalized_predictions: list[Any] = []
    invalid_count = 0
    correct = 0

    for truth, prediction in zip(y_true, y_pred):
        is_blank = prediction is None or (isinstance(prediction, str) and not prediction.strip())
        if is_blank or prediction not in label_to_index:
            normalized = invalid_label
            prediction_index = len(class_labels)
            invalid_count += 1
        else:
            normalized = prediction
            prediction_index = label_to_index[prediction]
        normalized_predictions.append(normalized)
        matrix[label_to_index[truth]][prediction_index] += 1
        correct += int(normalized == truth)

    recalls: dict[str, float] = {}
    f1_scores: dict[str, float] = {}
    supports: dict[str, int] = {}
    for class_index, label in enumerate(class_labels):
        support = sum(matrix[class_index])
        true_positive = matrix[class_index][class_index]
        predicted_as_class = sum(row[class_index] for row in matrix)
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_as_class if predicted_as_class else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        key = str(label)
        recalls[key] = recall
        f1_scores[key] = f1
        supports[key] = support

    count = len(y_true)
    return {
        "accuracy": correct / count if count else 0.0,
        "macro_f1": sum(f1_scores.values()) / len(class_labels),
        "balanced_accuracy": sum(recalls.values()) / len(class_labels),
        "per_class_recall": recalls,
        "per_class_f1": f1_scores,
        "support": supports,
        "invalid_response_rate": invalid_count / count if count else 0.0,
        "n_invalid": invalid_count,
        "n_valid": count - invalid_count,
        "n": count,
        "labels": [str(label) for label in class_labels],
        "prediction_labels": [str(label) for label in class_labels] + [invalid_label],
        "confusion_matrix": matrix,
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _check_pin(distribution: str, expected_version: str | None, *, paper_mode: bool) -> str | None:
    installed = _installed_version(distribution)
    if installed is None:
        if paper_mode:
            raise MetricDependencyError(f"required metric package {distribution!r} is not installed")
        return None
    if expected_version is not None and installed != expected_version:
        message = f"{distribution}=={expected_version} is required, found {installed}"
        if paper_mode:
            raise MetricDependencyError(message)
        return None
    return installed


def _coco_records(
    predictions: Sequence[str], references: Sequence[str | Sequence[str]]
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        raise ValueError("at least one prediction is required")
    # The scorer classes themselves consume ``image_id -> list[str]``.  The
    # ``[{caption: ...}]`` representation belongs to COCO's JSON loader and is
    # not accepted by Cider/Bleu/Meteor.compute_score directly.
    ground_truth = {
        index: _references(reference_values)
        for index, reference_values in enumerate(references)
    }
    results = {index: [str(prediction)] for index, prediction in enumerate(predictions)}
    return ground_truth, results


def prepare_coco_caption_inputs(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    paper_mode: bool = True,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """PTB-tokenize captions exactly as the COCO-caption evaluator does."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        raise ValueError("at least one prediction is required")
    installed = _check_pin("pycocoevalcap", expected_version, paper_mode=paper_mode)
    if installed is None:
        return {"status": "unavailable", "package_version": None}
    raw_ground_truth = {
        index: [{"caption": reference} for reference in _references(reference_values)]
        for index, reference_values in enumerate(references)
    }
    raw_results = {
        index: [{"caption": str(prediction)}]
        for index, prediction in enumerate(predictions)
    }
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore[import-not-found]

        tokenizer = PTBTokenizer()
        ground_truth = tokenizer.tokenize(raw_ground_truth)
        results = tokenizer.tokenize(raw_results)
    except Exception as exc:  # pragma: no cover - depends on optional Java/runtime
        if paper_mode:
            raise MetricDependencyError(f"COCO PTB tokenizer failed: {exc}") from exc
        return {
            "status": "unavailable",
            "package_version": installed,
            "error": str(exc),
        }
    return {
        "status": "ok",
        "package_version": installed,
        "ground_truth": ground_truth,
        "results": results,
    }


def _prepared_coco_records(
    prepared: Mapping[str, Any] | None,
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    paper_mode: bool,
    expected_version: str | None,
) -> tuple[dict[int, list[str]], dict[int, list[str]], str | None] | None:
    value = dict(prepared) if prepared is not None else prepare_coco_caption_inputs(
        predictions,
        references,
        paper_mode=paper_mode,
        expected_version=expected_version,
    )
    if value.get("status") != "ok":
        return None
    return value["ground_truth"], value["results"], value.get("package_version")


def _as_float_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [float(value) for value in values]


def coco_cider_scores(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    paper_mode: bool = True,
    expected_version: str | None = None,
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """CIDEr from the optional COCO caption scorer."""

    records = _prepared_coco_records(
        prepared, predictions, references,
        paper_mode=paper_mode, expected_version=expected_version,
    )
    if records is None:
        return {"status": "unavailable", "metric": "cider", "value": None, "per_item": None}
    ground_truth, results, installed = records
    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore[import-not-found]

        score, per_item = Cider().compute_score(ground_truth, results)
    except Exception as exc:  # pragma: no cover - depends on optional scorer/runtime
        if paper_mode:
            raise MetricDependencyError(f"COCO CIDEr scorer failed: {exc}") from exc
        return {"status": "unavailable", "metric": "cider", "value": None, "per_item": None, "error": str(exc)}
    return {
        "status": "ok",
        "metric": "cider",
        "value": float(score),
        "per_item": _as_float_list(per_item),
        "package_version": installed,
    }


def coco_meteor_scores(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    paper_mode: bool = True,
    expected_version: str | None = None,
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """METEOR from the optional COCO caption scorer (and its Java runtime)."""

    records = _prepared_coco_records(
        prepared, predictions, references,
        paper_mode=paper_mode, expected_version=expected_version,
    )
    if records is None:
        return {"status": "unavailable", "metric": "meteor", "value": None, "per_item": None}
    ground_truth, results, installed = records
    process: subprocess.Popen[bytes] | None = None
    try:
        import pycocoevalcap.meteor.meteor as meteor_module  # type: ignore[import-not-found]

        jar = Path(meteor_module.__file__).with_name("meteor-1.5.jar")
        if not jar.is_file():
            raise FileNotFoundError(jar)
        process = subprocess.Popen(
            [
                "java", "-Xmx2G", "-jar", str(jar),
                "-", "-", "-stdio", "-l", "en", "-norm",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None and process.stdout is not None

        def read_line() -> str:
            output: queue.Queue[bytes] = queue.Queue(maxsize=1)
            thread = threading.Thread(
                target=lambda: output.put(process.stdout.readline()), daemon=True
            )
            thread.start()
            try:
                line = output.get(timeout=120.0)
            except queue.Empty as exc:
                process.kill()
                raise TimeoutError("METEOR Java scorer timed out") from exc
            if not line:
                stderr = b""
                if process.stderr is not None:
                    stderr = process.stderr.read()
                raise RuntimeError(
                    "METEOR Java scorer ended unexpectedly: "
                    + stderr.decode("utf-8", errors="replace").strip()
                )
            return line.decode("utf-8").strip()

        statistics: list[str] = []
        for key in ground_truth:
            hypothesis = str(results[key][0]).replace("|||", "").replace("  ", " ")
            score_line = " ||| ".join(
                ("SCORE", " ||| ".join(ground_truth[key]), hypothesis)
            )
            process.stdin.write((score_line + "\n").encode("utf-8"))
            process.stdin.flush()
            statistics.append(read_line())
        process.stdin.write(("EVAL ||| " + " ||| ".join(statistics) + "\n").encode("utf-8"))
        process.stdin.flush()
        per_item = [float(read_line()) for _ in ground_truth]
        score = float(read_line())
    except Exception as exc:  # pragma: no cover - depends on optional scorer/Java
        if paper_mode:
            raise MetricDependencyError(f"COCO METEOR scorer failed: {exc}") from exc
        return {"status": "unavailable", "metric": "meteor", "value": None, "per_item": None, "error": str(exc)}
    finally:
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    return {
        "status": "ok",
        "metric": "meteor",
        "value": float(score),
        "per_item": _as_float_list(per_item),
        "package_version": installed,
    }


def coco_bleu_scores(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    paper_mode: bool = True,
    expected_version: str | None = None,
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """BLEU-1..4 from the optional COCO caption scorer."""

    records = _prepared_coco_records(
        prepared, predictions, references,
        paper_mode=paper_mode, expected_version=expected_version,
    )
    if records is None:
        return {"status": "unavailable", "metric": "bleu", "value": None, "per_item": None}
    ground_truth, results, installed = records
    try:
        from pycocoevalcap.bleu.bleu import Bleu  # type: ignore[import-not-found]

        values, per_item = Bleu(4).compute_score(ground_truth, results)
    except Exception as exc:  # pragma: no cover - depends on optional scorer/runtime
        if paper_mode:
            raise MetricDependencyError(f"COCO BLEU scorer failed: {exc}") from exc
        return {"status": "unavailable", "metric": "bleu", "value": None, "per_item": None, "error": str(exc)}
    value_list = _as_float_list(values)
    per_order = [_as_float_list(order_values) for order_values in per_item]
    return {
        "status": "ok",
        "metric": "bleu",
        "value": {f"bleu_{index + 1}": value for index, value in enumerate(value_list)},
        "per_item": {f"bleu_{index + 1}": values for index, values in enumerate(per_order)},
        "package_version": installed,
    }


def coco_rouge_scores(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    paper_mode: bool = True,
    expected_version: str | None = None,
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """COCO-caption ROUGE-L after the standard PTB tokenization step."""

    records = _prepared_coco_records(
        prepared, predictions, references,
        paper_mode=paper_mode, expected_version=expected_version,
    )
    if records is None:
        return {"status": "unavailable", "metric": "rouge_l", "value": None, "per_item": None}
    ground_truth, results, installed = records
    try:
        from pycocoevalcap.rouge.rouge import Rouge  # type: ignore[import-not-found]

        score, per_item = Rouge().compute_score(ground_truth, results)
    except Exception as exc:  # pragma: no cover - depends on optional scorer/runtime
        if paper_mode:
            raise MetricDependencyError(f"COCO ROUGE-L scorer failed: {exc}") from exc
        return {
            "status": "unavailable",
            "metric": "rouge_l",
            "value": None,
            "per_item": None,
            "error": str(exc),
        }
    return {
        "status": "ok",
        "metric": "rouge_l",
        "value": float(score),
        "per_item": _as_float_list(per_item),
        "package_version": installed,
    }


def sbert_cosine_scores(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    model_name: str,
    revision: str,
    paper_mode: bool = True,
    expected_version: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Maximum-reference Sentence-BERT cosine similarity per item."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        raise ValueError("at least one prediction is required")
    installed = _check_pin("sentence-transformers", expected_version, paper_mode=paper_mode)
    if installed is None:
        return {"status": "unavailable", "metric": "sbert_cosine", "value": None, "per_item": None}
    if not model_name or not revision:
        raise ValueError("model_name and immutable revision are required for SBERT scoring")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        model = SentenceTransformer(model_name, revision=revision, device=device)
        per_item: list[float] = []
        for prediction, reference_values in zip(predictions, references):
            texts = [str(prediction)] + _references(reference_values)
            embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            prediction_embedding = embeddings[0]
            similarities = [
                float(sum(float(left) * float(right) for left, right in zip(prediction_embedding, embedding)))
                for embedding in embeddings[1:]
            ]
            per_item.append(max(similarities))
    except Exception as exc:  # pragma: no cover - depends on optional model/runtime
        if paper_mode:
            raise MetricDependencyError(f"Sentence-BERT scorer failed: {exc}") from exc
        return {
            "status": "unavailable",
            "metric": "sbert_cosine",
            "value": None,
            "per_item": None,
            "error": str(exc),
        }
    return {
        "status": "ok",
        "metric": "sbert_cosine",
        "value": sum(per_item) / len(per_item) if per_item else 0.0,
        "per_item": per_item,
        "package_version": installed,
        "model_name": model_name,
        "model_revision": revision,
    }


def optional_caption_metrics(
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
    *,
    metrics: Sequence[str] = ("cider", "meteor", "sbert"),
    paper_mode: bool = True,
    pins: Mapping[str, str] | None = None,
    sbert_model: str | None = None,
    sbert_revision: str | None = None,
    sbert_device: str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Run requested optional scorers through one generic adapter."""

    pins = dict(pins or {})
    coco_metrics = {"bleu", "cider", "meteor", "rouge"}
    prepared_coco = None
    if coco_metrics.intersection(metrics):
        prepared_coco = prepare_coco_caption_inputs(
            predictions,
            references,
            paper_mode=paper_mode,
            expected_version=pins.get("pycocoevalcap"),
        )
    adapters: dict[str, Callable[[], dict[str, Any]]] = {
        "bleu": lambda: coco_bleu_scores(
            predictions,
            references,
            paper_mode=paper_mode,
            expected_version=pins.get("pycocoevalcap"),
            prepared=prepared_coco,
        ),
        "cider": lambda: coco_cider_scores(
            predictions,
            references,
            paper_mode=paper_mode,
            expected_version=pins.get("pycocoevalcap"),
            prepared=prepared_coco,
        ),
        "meteor": lambda: coco_meteor_scores(
            predictions,
            references,
            paper_mode=paper_mode,
            expected_version=pins.get("pycocoevalcap"),
            prepared=prepared_coco,
        ),
        "rouge": lambda: coco_rouge_scores(
            predictions,
            references,
            paper_mode=paper_mode,
            expected_version=pins.get("pycocoevalcap"),
            prepared=prepared_coco,
        ),
        "sbert": lambda: sbert_cosine_scores(
            predictions,
            references,
            model_name=sbert_model or "",
            revision=sbert_revision or "",
            paper_mode=paper_mode,
            expected_version=pins.get("sentence-transformers"),
            device=sbert_device,
        ),
    }
    unknown = sorted(set(metrics) - set(adapters))
    if unknown:
        raise ValueError(f"unknown optional caption metrics: {', '.join(unknown)}")
    results = {metric: adapters[metric]() for metric in metrics}
    if prepared_coco is not None:
        results["_coco_tokenization"] = prepared_coco
    return results


__all__ = [
    "INVALID_PREDICTION",
    "MetricDependencyError",
    "classification_metrics",
    "coco_bleu_compatible_scores",
    "coco_bleu_scores",
    "coco_cider_scores",
    "coco_meteor_scores",
    "coco_rouge_scores",
    "corpus_bleu",
    "corpus_bleu_scores",
    "exact_match",
    "normalize_text",
    "optional_caption_metrics",
    "prepare_coco_caption_inputs",
    "response_is_valid",
    "rouge_l",
    "sbert_cosine_scores",
    "sentence_bleu",
    "token_f1",
    "tokenize",
    "valid_response_rate",
]
