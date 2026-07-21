"""Run one frozen model/suite pair and write strict append-only prediction evidence."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.paper.artifacts import (  # noqa: E402
    PredictionStore,
    build_attempt,
    environment_manifest,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from eval.paper.assets import AssetRegistry  # noqa: E402
from eval.paper.model_backends import create_backend  # noqa: E402
from eval.paper.protocol import PaperProtocol, sha256_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def generation_environment(environment_spec: Mapping[str, Any], device: str) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in environment_spec["packages"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    hardware: dict[str, Any] = {"requested_device": device}
    try:
        import torch

        hardware["cuda_runtime"] = torch.version.cuda
        hardware["cuda_available"] = bool(torch.cuda.is_available())
        hardware["cuda_device_count"] = int(torch.cuda.device_count())
        hardware["bf16_supported"] = bool(
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            index = torch.cuda.current_device()
            hardware.update(
                {
                    "gpu_name": torch.cuda.get_device_name(index),
                    "gpu_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    except Exception as exc:
        hardware["torch_probe_error"] = repr(exc)
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "packages": versions,
        "hardware": hardware,
    }


def validate_generation_environment(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    observed_python = str(observed["python"])
    if observed_python != str(expected["python"]):
        raise SystemExit(
            f"Generation requires Python {expected['python']}, found {observed_python}"
        )
    mismatches: list[str] = []
    for package, expected_version in expected["packages"].items():
        actual = str(observed["packages"].get(package, "missing"))
        expected_text = str(expected_version)
        if actual != expected_text and not actual.startswith(expected_text + "+"):
            mismatches.append(f"{package}=={expected_text} (found {actual})")
    if mismatches:
        raise SystemExit("Generation environment lock mismatch: " + "; ".join(mismatches))


def validate_generation_hardware(
    model: Mapping[str, Any],
    observed: Mapping[str, Any],
    device: str,
    runtime_requirements: Mapping[str, Any],
) -> None:
    if not device.startswith("cuda"):
        return
    hardware = observed.get("hardware") or {}
    if not hardware.get("cuda_available"):
        raise SystemExit("CUDA inference was requested but torch.cuda.is_available() is false")
    capability = hardware.get("gpu_capability")
    if not isinstance(capability, list) or len(capability) < 2:
        raise SystemExit("Could not determine the CUDA device compute capability")
    numeric_capability = float(f"{int(capability[0])}.{int(capability[1])}")
    minimum = float(runtime_requirements["minimum_compute_capability"])
    maximum = float(runtime_requirements["maximum_compute_capability_exclusive"])
    if numeric_capability < minimum or numeric_capability >= maximum:
        raise SystemExit(
            f"The frozen environments support compute capability >={minimum:.1f} "
            f"and <{maximum:.1f} "
            f"(Ampere, Ada, or Hopper); found {numeric_capability:.1f}"
        )
    if str(model.get("dtype")) == "bfloat16" and not hardware.get("bf16_supported"):
        raise SystemExit("This model requires CUDA BF16 support, but torch reports none")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def max_tokens(protocol: PaperProtocol, suite: str, record: Mapping[str, Any]) -> int:
    settings = protocol.data["generation"][suite]
    if suite == "internal":
        key = "caption_max_new_tokens" if record.get("record_type") == "caption" else "qa_max_new_tokens"
        return int(settings[key])
    return int(settings["max_new_tokens"])


def validate_locked_output_path(
    protocol: PaperProtocol,
    suite: str,
    model_label: str,
    records_hash: str,
    output: str | Path,
    *,
    repo_root: str | Path = ROOT,
) -> Path:
    resolved_output = Path(output).resolve()
    try:
        output_root = resolved_output.parents[3]
    except IndexError as exc:
        raise SystemExit(
            f"Output directory is too shallow to match the locked run layout: {output}"
        ) from exc
    expected = protocol.model_output_dir(
        suite,
        model_label,
        records_hash,
        repo_root,
        output_root,
    ).resolve()
    if resolved_output != expected:
        raise SystemExit(
            "Output directory does not match the locked generation fingerprint: "
            f"expected {expected}"
        )
    return expected


def create_deepsdo_smoke_fixtures(
    output_dir: str | Path, prompt: str
) -> list[dict[str, Any]]:
    """Create deterministic, explicitly non-benchmark solar-like backend fixtures."""

    from PIL import Image, ImageDraw

    root = Path(output_dir) / "smoke_fixtures"
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index in range(3):
        image = Image.new("RGB", (224, 224), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 20, 204, 204), fill=(225, 105 + index * 25, 24))
        if index == 0:
            draw.ellipse((82, 88, 105, 111), fill=(35, 20, 15))
            draw.ellipse((126, 122, 143, 139), fill=(50, 25, 18))
        elif index == 1:
            draw.arc((45, 42, 188, 180), start=205, end=335, fill=(255, 245, 180), width=7)
        else:
            draw.ellipse((35, 35, 190, 190), outline=(255, 220, 115), width=8)
            draw.rectangle((105, 18, 118, 206), fill=(15, 15, 15))
        path = root / f"synthetic_solar_{index + 1}.png"
        image.save(path, format="PNG", optimize=False)
        records.append(
            {
                "id": f"deepsdo-synthetic-smoke-{index + 1}",
                "dataset": "deepsdo_synthetic_smoke_fixture",
                "split": "synthetic_smoke_only",
                "record_type": "caption",
                "image": path.name,
                "image_id": f"synthetic-solar-{index + 1}",
                "source_object_id": f"synthetic-solar-{index + 1}",
                "image_path": str(path.resolve()),
                "prompt": prompt,
                "reference": "",
                "not_benchmark_data": True,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/paper_eval_v2.yaml")
    parser.add_argument("--suite", required=True, choices=["internal", "deepsdo", "astrovlbench"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a task-stratified subset into the real store, then leave it resumable.",
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--diagnostic-allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    protocol = PaperProtocol.load(args.protocol)
    model = protocol.selected_models(args.suite).get(args.model)
    if model is None:
        raise SystemExit(f"{args.model!r} is not enabled for {args.suite!r}")
    records = read_jsonl(args.records)
    if not records:
        raise SystemExit("No canonical evaluation records were loaded")
    configured_count = protocol.data["datasets"][args.suite].get("expected_records")
    if configured_count is not None and len(records) != int(configured_count):
        raise SystemExit(
            f"{args.suite} records file contains {len(records)} rows; expected {configured_count}"
        )
    output = Path(args.output_dir)
    records_hash = sha256_file(args.records)
    base_model_hash = protocol.model_fingerprint(args.suite, args.model)
    model_hash = protocol.effective_model_fingerprint(
        args.suite, args.model, records_hash, ROOT
    )
    suite_hash = protocol.suite_fingerprint(args.suite)
    analysis_hash = protocol.analysis_fingerprint(args.suite)
    environment_spec = protocol.data["environments"][model["environment"]]
    runtime_environment = generation_environment(environment_spec, args.device)
    validate_generation_environment(environment_spec, runtime_environment)
    validate_generation_hardware(
        model, runtime_environment, args.device, protocol.data["runtime"]
    )
    runtime_environment_hash = sha256_json(runtime_environment)
    software_environment_hash = sha256_json(
        {
            "python": runtime_environment["python"],
            "packages": runtime_environment["packages"],
        }
    )
    seed = int(protocol.data["generation"]["common"]["seed"])
    seed_everything(seed)
    expected_output = validate_locked_output_path(
        protocol,
        args.suite,
        args.model,
        records_hash,
        output,
    )
    if output.exists() and any(output.iterdir()) and not args.resume and not args.overwrite:
        raise SystemExit(f"{output} already contains a run; pass --resume or --overwrite")
    if args.overwrite and output.exists():
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"{protocol.study_id}-{args.suite}-{args.model}-{model_hash[:16]}"
    store = PredictionStore(output, records, model_hash)
    initial_manifest = {
        "schema_version": protocol.data["schema_version"],
        "run_id": run_id,
        "study_id": protocol.study_id,
        "suite": args.suite,
        "model_label": args.model,
        "model_revision": model["revision"],
        "backend": model["backend"],
        "suite_protocol_hash": suite_hash,
        "suite_analysis_hash": analysis_hash,
        "model_generation_config_hash": base_model_hash,
        "model_protocol_hash": model_hash,
        "protocol_file_sha256": sha256_file(args.protocol),
        "records_file_sha256": records_hash,
        "generation_implementation": protocol.generation_implementation_payload(args.model, ROOT),
        "generation_environment": runtime_environment,
        "generation_environment_sha256": runtime_environment_hash,
        "generation_software_environment_sha256": software_environment_hash,
        "seed": seed,
        "do_sample": False,
        "num_beams": 1,
        "retrieval": False,
        "started_at_utc": utc_now(),
        "command": sys.argv,
        "environment": environment_manifest(Path(__file__).resolve().parents[1]),
    }
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("model_protocol_hash") != model_hash:
            raise SystemExit("Cannot resume: run manifest model protocol hash differs")
        if previous.get("records_file_sha256") != records_hash:
            raise SystemExit("Cannot resume: canonical records file hash differs")
        if previous.get("generation_software_environment_sha256") != software_environment_hash:
            raise SystemExit("Cannot resume: locked generation software environment differs")
        initial_manifest["started_at_utc"] = previous.get("started_at_utc")
        initial_manifest["resumed_at_utc"] = utc_now()
        history = list(previous.get("resume_history") or [])
        history.append(
            {
                "resumed_at_utc": initial_manifest["resumed_at_utc"],
                "previous_finished_at_utc": previous.get("finished_at_utc"),
                "previous_completion": previous.get("completion"),
                "previous_command": previous.get("command"),
                "generation_environment_sha256": previous.get("generation_environment_sha256"),
                "generation_environment": previous.get("generation_environment"),
            }
        )
        initial_manifest["resume_history"] = history
    write_json_atomic(manifest_path, initial_manifest)
    pending = store.pending_records() if args.resume else records
    if args.smoke:
        selected = []
        seen_groups = set()
        required_groups = set()
        if args.suite == "internal":
            required_groups = {record.get("record_type") for record in pending}
        elif args.suite == "astrovlbench":
            required_groups = {record.get("task_key") for record in pending}
        for record in pending:
            if args.suite == "internal":
                group = record.get("record_type")
            elif args.suite == "astrovlbench":
                group = record.get("task_key")
            else:
                group = (record.get("topic_stratum"), record.get("collapsed_modality"))
            if group not in seen_groups:
                selected.append(record)
                seen_groups.add(group)
            if required_groups and seen_groups == required_groups:
                break
            if not required_groups and len(selected) >= (args.limit or 5):
                break
        pending = selected
    elif args.limit:
        pending = pending[: args.limit]
    print(
        f"{args.suite}/{args.model}: expected={len(records)} completed={len(records)-len(pending)} "
        f"pending={len(pending)} protocol={model_hash[:16]}"
    )
    if args.dry_run:
        for record in pending[:5]:
            print(json.dumps({"id": record["id"], "image_path": record["image_path"], "max_new_tokens": max_tokens(protocol, args.suite, record)}))
        return
    if not pending:
        report = store.finalize(allow_partial=args.diagnostic_allow_partial or args.smoke)
        final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_manifest.update(
            {
                "finished_at_utc": utc_now(),
                "completion": report.as_dict(),
                "attempts_sha256": sha256_file(store.attempts_path),
                "predictions_sha256": sha256_file(store.predictions_path),
            }
        )
        write_json_atomic(manifest_path, final_manifest)
        print(json.dumps(report.as_dict(), indent=2))
        return
    registry = AssetRegistry.load(args.asset_manifest)
    registry.validate_model(args.model, protocol)
    backend = create_backend(protocol, args.model, registry, args.device)
    component_hashes = {
        key: model[key]
        for key in ("checkpoint_sha256", "connector_sha256", "lora_sha256")
        if model.get(key)
    }
    model_evidence = {
        "checkpoint_component_hashes": component_hashes,
        "model_code_revision": model.get("code_revision"),
        "auxiliary_model_revisions": {
            key: value.get("revision")
            for key, value in model.items()
            if isinstance(value, Mapping) and value.get("repo_id") and value.get("revision")
        },
        "generation_environment_name": model["environment"],
    }
    if args.smoke and args.suite == "deepsdo":
        synthetic_results: list[dict[str, Any]] = []
        synthetic_records = create_deepsdo_smoke_fixtures(
            output, str(protocol.data["datasets"]["deepsdo"]["prompt"])
        )
        for fixture in synthetic_records:
            started = time.perf_counter()
            try:
                generated = backend.generate(
                    fixture, int(protocol.data["generation"]["deepsdo"]["max_new_tokens"])
                )
                smoke_row = build_attempt(
                    record=fixture,
                    model_label=args.model,
                    model_revision=model["revision"],
                    backend=model["backend"],
                    run_id=run_id,
                    suite_protocol_hash=suite_hash,
                    model_protocol_hash=model_hash,
                    response=generated.response,
                    raw_response=generated.raw_response,
                    generated_token_ids=generated.generated_token_ids,
                    prompt_token_count=generated.prompt_token_count,
                    termination_reason=generated.termination_reason,
                    rendered_prompt=generated.rendered_prompt,
                    latency_seconds=time.perf_counter() - started,
                    extra={
                        "template_source": generated.template_source,
                        "template_sha256": generated.template_sha256,
                        "smoke_fixture": True,
                        **model_evidence,
                    },
                )
            except Exception as exc:
                smoke_row = build_attempt(
                    record=fixture,
                    model_label=args.model,
                    model_revision=model["revision"],
                    backend=model["backend"],
                    run_id=run_id,
                    suite_protocol_hash=suite_hash,
                    model_protocol_hash=model_hash,
                    latency_seconds=time.perf_counter() - started,
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    extra={"smoke_fixture": True, **model_evidence},
                )
            synthetic_results.append(smoke_row)
        write_json_atomic(
            output / "smoke_fixtures" / "results.json",
            {
                "not_benchmark_data": True,
                "canonical_score_rows_added": 0,
                "results": synthetic_results,
            },
        )
        failed_synthetic = [
            row["id"] for row in synthetic_results if row.get("status") != "ok"
        ]
        if failed_synthetic:
            raise SystemExit(
                "Synthetic DeepSDO backend smoke failed for: "
                + ", ".join(failed_synthetic)
            )
    for position, record in enumerate(pending, 1):
        for attempt_number in range(1, max(1, args.max_attempts) + 1):
            started = time.perf_counter()
            try:
                image_path = Path(record["image_path"])
                if not image_path.is_file():
                    raise FileNotFoundError(str(image_path))
                cap = max_tokens(protocol, args.suite, record)
                generated = backend.generate(record, cap)
                row = build_attempt(
                    record=record,
                    model_label=args.model,
                    model_revision=model["revision"],
                    backend=model["backend"],
                    run_id=run_id,
                    suite_protocol_hash=suite_hash,
                    model_protocol_hash=model_hash,
                    response=generated.response,
                    raw_response=generated.raw_response,
                    generated_token_ids=generated.generated_token_ids,
                    prompt_token_count=generated.prompt_token_count,
                    termination_reason=generated.termination_reason,
                    rendered_prompt=generated.rendered_prompt,
                    latency_seconds=time.perf_counter() - started,
                    extra={
                        "template_source": generated.template_source,
                        "template_sha256": generated.template_sha256,
                        "token_cap": cap,
                        "token_cap_hit": generated.termination_reason == "max_new_tokens",
                        "attempt_number": attempt_number,
                        **model_evidence,
                        **generated.extra,
                    },
                )
            except Exception as exc:  # preserve evidence and continue/retry
                row = build_attempt(
                    record=record,
                    model_label=args.model,
                    model_revision=model["revision"],
                    backend=model["backend"],
                    run_id=run_id,
                    suite_protocol_hash=suite_hash,
                    model_protocol_hash=model_hash,
                    latency_seconds=time.perf_counter() - started,
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    extra={"attempt_number": attempt_number, **model_evidence},
                )
            store.append(row)
            print(
                f"[{position}/{len(pending)}] {record['id']} attempt={attempt_number} status={row['status']}",
                flush=True,
            )
            if row["status"] == "ok":
                break
    report = store.finalize(allow_partial=args.diagnostic_allow_partial or args.smoke)
    if args.smoke:
        successful = store.successes()
        failed_smoke = [str(record["id"]) for record in pending if str(record["id"]) not in successful]
        if failed_smoke:
            raise SystemExit(f"Smoke generation failed for: {', '.join(failed_smoke)}")
    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_manifest.update(
        {
            "finished_at_utc": utc_now(),
            "completion": report.as_dict(),
            "attempts_sha256": sha256_file(store.attempts_path),
            "predictions_sha256": sha256_file(store.predictions_path),
        }
    )
    write_json_atomic(manifest_path, final_manifest)
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
