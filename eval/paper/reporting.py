"""Generic paper-table, narrative, plotting and redaction helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, is_dataclass
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


class ReportingDependencyError(RuntimeError):
    """Raised when a paper artifact cannot be produced reproducibly."""


DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "reference",
        "references",
        "reference_caption",
        "raw_reference",
        "answer",
        "answers",
        "gold",
        "gold_answer",
        "ground_truth",
        "target",
        "targets",
        "annotation",
        "annotations",
        "raw_annotation",
        "dataset_row",
        "source_record",
        "image",
        "image_path",
        "image_bytes",
        "image_data",
        "raw_image",
        "source_path",
        "raw_prompt",
        "dataset_prompt",
    }
)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value, key=str)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _atomic_text_write(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def write_json(path: str | Path, data: Any, *, indent: int = 2, sort_keys: bool = True) -> Path:
    """Write deterministic UTF-8 JSON atomically."""

    payload = json.dumps(
        data,
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
        default=_json_default,
        allow_nan=False,
    )
    return _atomic_text_write(path, payload + "\n")


def _columns(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        result = list(columns)
        if len(set(result)) != len(result):
            raise ValueError("columns must be unique")
        return result
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                result.append(key)
    if not result:
        raise ValueError("columns are required when rows are empty")
    return result


def _cell(value: Any, *, float_format: str | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("table values cannot contain NaN or infinity")
        return format(value, float_format) if float_format else repr(value)
    if isinstance(value, (dict, list, tuple, set)) or is_dataclass(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default, allow_nan=False)
    return str(value)


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    float_format: str | None = None,
) -> Path:
    """Write RFC-4180-style CSV with stable column ordering."""

    rows = list(rows)
    fieldnames = _columns(rows, columns)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _cell(row.get(column), float_format=float_format) for column in fieldnames})
    return _atomic_text_write(path, stream.getvalue())


def escape_markdown(value: Any) -> str:
    """Escape a scalar for a GitHub/CommonMark table cell."""

    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def write_markdown_table(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    column_labels: Mapping[str, str] | None = None,
    float_format: str | None = ".4f",
    title: str | None = None,
    note: str | None = None,
) -> Path:
    """Write a compact Markdown table with escaped cells."""

    rows = list(rows)
    fieldnames = _columns(rows, columns)
    labels = dict(column_labels or {})
    lines: list[str] = []
    if title:
        lines.extend([f"## {title}", ""])
    lines.append("| " + " | ".join(escape_markdown(labels.get(column, column)) for column in fieldnames) + " |")
    lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                escape_markdown(_cell(row.get(column), float_format=float_format)) for column in fieldnames
            )
            + " |"
        )
    if note:
        lines.extend(["", escape_markdown(note)])
    return _atomic_text_write(path, "\n".join(lines) + "\n")


_LATEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def escape_latex(value: Any) -> str:
    """Escape arbitrary text for a LaTeX table cell."""

    return "".join(_LATEX_ESCAPES.get(character, character) for character in str(value)).replace("\r\n", " ").replace("\n", " ")


def write_latex_table(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    column_labels: Mapping[str, str] | None = None,
    float_format: str | None = ".4f",
    caption: str | None = None,
    label: str | None = None,
    alignment: str | None = None,
    note: str | None = None,
) -> Path:
    """Write standalone table markup without requiring ``booktabs``."""

    rows = list(rows)
    fieldnames = _columns(rows, columns)
    labels = dict(column_labels or {})
    alignment = alignment or ("l" + "r" * (len(fieldnames) - 1))
    if len(alignment) != len(fieldnames) or any(character not in "lcr" for character in alignment):
        raise ValueError("alignment must contain one of l/c/r for every column")

    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{escape_latex(caption)}}}")
    if label:
        lines.append(rf"\label{{{escape_latex(label)}}}")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\hline",
            " & ".join(escape_latex(labels.get(column, column)) for column in fieldnames) + r" \\",
            r"\hline",
        ]
    )
    for row in rows:
        cells = [escape_latex(_cell(row.get(column), float_format=float_format)) for column in fieldnames]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    if note:
        lines.append(rf"\par\smallskip\footnotesize {escape_latex(note)}")
    lines.append(r"\end{table}")
    return _atomic_text_write(path, "\n".join(lines) + "\n")


def write_table_bundle(
    output_stem: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    column_labels: Mapping[str, str] | None = None,
    float_format: str | None = ".4f",
    caption: str | None = None,
    label: str | None = None,
    note: str | None = None,
) -> dict[str, Path]:
    """Write the same rows as JSON, CSV, Markdown and LaTeX."""

    stem = Path(output_stem)
    rows = list(rows)
    return {
        "json": write_json(stem.with_suffix(".json"), rows),
        "csv": write_csv(stem.with_suffix(".csv"), rows, columns=columns, float_format=float_format),
        "markdown": write_markdown_table(
            stem.with_suffix(".md"),
            rows,
            columns=columns,
            column_labels=column_labels,
            float_format=float_format,
            title=caption,
            note=note,
        ),
        "latex": write_latex_table(
            stem.with_suffix(".tex"),
            rows,
            columns=columns,
            column_labels=column_labels,
            float_format=float_format,
            caption=caption,
            label=label,
            note=note,
        ),
    }


def cautious_comparison(
    model_a: str,
    score_a: float,
    model_b: str,
    score_b: float,
    *,
    metric: str,
    difference_ci: tuple[float, float] | None = None,
    higher_is_better: bool = True,
    scope: str = "this evaluation",
    precision: int = 3,
) -> str:
    """Produce a bounded comparison that does not claim universal superiority."""

    if precision < 0:
        raise ValueError("precision must be non-negative")
    for name, value in (("score_a", score_a), ("score_b", score_b)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    difference = float(score_a) - float(score_b)
    favors_a = difference > 0 if higher_is_better else difference < 0
    favors_b = difference < 0 if higher_is_better else difference > 0
    if favors_a:
        opening = f"Within {scope}, the {metric} point estimate favored {model_a} over {model_b}"
    elif favors_b:
        opening = f"Within {scope}, the {metric} point estimate favored {model_b} over {model_a}"
    else:
        opening = f"Within {scope}, {model_a} and {model_b} had the same {metric} point estimate"
    values = (
        f"{model_a}={score_a:.{precision}f}, {model_b}={score_b:.{precision}f}, "
        f"difference ({model_a}-{model_b})={difference:.{precision}f}"
    )
    if difference_ci is None:
        return f"{opening} ({values}). This descriptive result is specific to the evaluated data and protocol."
    lower, upper = (float(difference_ci[0]), float(difference_ci[1]))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("difference_ci must be a finite ordered pair")
    interval = f"95% CI [{lower:.{precision}f}, {upper:.{precision}f}]"
    if lower <= 0.0 <= upper:
        conclusion = "The interval includes zero, so these data do not establish a clear difference."
    else:
        conclusion = "The interval excludes zero under the prespecified paired resampling analysis."
    return f"{opening} ({values}; {interval}). {conclusion} The comparison is specific to the evaluated data and protocol."


def redact_for_sharing(
    data: Any,
    *,
    sensitive_keys: Sequence[str] | None = None,
) -> Any:
    """Recursively remove references, raw annotations and image-bearing fields.

    Matching is exact and case-insensitive, so hashes such as ``reference_hash``
    remain available for auditability.  Callers may extend or replace the key set
    for dataset-specific licensing requirements.
    """

    keys = {key.casefold() for key in (sensitive_keys or DEFAULT_SENSITIVE_KEYS)}

    def redact(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: redact(child)
                for key, child in value.items()
                if str(key).casefold() not in keys
            }
        if isinstance(value, list):
            return [redact(child) for child in value]
        if isinstance(value, tuple):
            return tuple(redact(child) for child in value)
        return value

    return redact(data)


def _load_pyplot(*, paper_mode: bool):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as pyplot
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        if paper_mode:
            raise ReportingDependencyError(f"matplotlib is required for paper figures: {exc}") from exc
        return None
    return pyplot


def save_figure(
    figure: Any,
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("svg", "png", "pdf"),
    dpi: int = 300,
) -> dict[str, Path]:
    """Save a Matplotlib figure in all requested paper formats."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    normalized_formats = [value.casefold().lstrip(".") for value in formats]
    if not normalized_formats or any(value not in {"svg", "png", "pdf"} for value in normalized_formats):
        raise ValueError("formats must be drawn from svg, png and pdf")
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for output_format in normalized_formats:
        path = stem.with_suffix(f".{output_format}")
        figure.savefig(path, format=output_format, dpi=dpi, bbox_inches="tight")
        outputs[output_format] = path
    return outputs


def plot_estimates(
    rows: Sequence[Mapping[str, Any]],
    output_stem: str | Path,
    *,
    label_key: str = "label",
    estimate_key: str = "estimate",
    lower_key: str = "lower",
    upper_key: str = "upper",
    title: str | None = None,
    ylabel: str | None = None,
    formats: Sequence[str] = ("svg", "png", "pdf"),
    dpi: int = 300,
    paper_mode: bool = True,
) -> dict[str, Path]:
    """Create a bar plot with asymmetric confidence intervals."""

    pyplot = _load_pyplot(paper_mode=paper_mode)
    if pyplot is None:
        return {}
    rows = list(rows)
    if not rows:
        raise ValueError("at least one estimate row is required")
    labels = [str(row[label_key]) for row in rows]
    estimates = [float(row[estimate_key]) for row in rows]
    lower_errors = [estimate - float(row[lower_key]) for estimate, row in zip(estimates, rows)]
    upper_errors = [float(row[upper_key]) - estimate for estimate, row in zip(estimates, rows)]
    if any(value < 0 for value in lower_errors + upper_errors):
        raise ValueError("confidence bounds must surround each estimate")
    figure, axis = pyplot.subplots(figsize=(max(5.0, len(rows) * 1.15), 4.0))
    positions = list(range(len(rows)))
    axis.bar(positions, estimates, yerr=[lower_errors, upper_errors], capsize=4, color="#4472C4")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    if title:
        axis.set_title(title)
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    try:
        return save_figure(figure, output_stem, formats=formats, dpi=dpi)
    finally:
        pyplot.close(figure)


def plot_heatmap(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    output_stem: str | Path,
    *,
    title: str | None = None,
    colorbar_label: str | None = None,
    formats: Sequence[str] = ("svg", "png", "pdf"),
    dpi: int = 300,
    paper_mode: bool = True,
) -> dict[str, Path]:
    """Create a compact annotated heatmap for subgroup/task results."""

    pyplot = _load_pyplot(paper_mode=paper_mode)
    if pyplot is None:
        return {}
    values = [[float(value) for value in row] for row in matrix]
    if len(values) != len(row_labels) or any(len(row) != len(column_labels) for row in values):
        raise ValueError("matrix dimensions must match row and column labels")
    if not values or not column_labels:
        raise ValueError("heatmap cannot be empty")
    figure, axis = pyplot.subplots(
        figsize=(max(5.0, len(column_labels) * 1.0), max(3.5, len(row_labels) * 0.55))
    )
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(column_labels)), column_labels, rotation=35, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    if title:
        axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color="white")
    try:
        return save_figure(figure, output_stem, formats=formats, dpi=dpi)
    finally:
        pyplot.close(figure)


__all__ = [
    "DEFAULT_SENSITIVE_KEYS",
    "ReportingDependencyError",
    "cautious_comparison",
    "escape_latex",
    "escape_markdown",
    "plot_estimates",
    "plot_heatmap",
    "redact_for_sharing",
    "save_figure",
    "write_csv",
    "write_json",
    "write_latex_table",
    "write_markdown_table",
    "write_table_bundle",
]
