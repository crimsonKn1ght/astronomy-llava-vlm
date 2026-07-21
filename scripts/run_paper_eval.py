"""One-command, resumable AstraQ-VL paper evaluation orchestrator.

This runner never trains.  It prepares frozen datasets, verifies/downloads pinned
assets, executes one isolated model process at a time, validates predictions,
scores them, creates paper outputs, and emits private/public bundles.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.paper.artifacts import (  # noqa: E402
    ArtifactError,
    create_bundle,
    environment_manifest,
    git_state,
    read_jsonl,
    sha256_file,
    write_checksum_file,
    write_json_atomic,
)
from eval.paper.assets import materialize_model_assets  # noqa: E402
from eval.paper.internal import (  # noqa: E402
    audit_image_overlap,
    audit_text_overlap,
    canonical_records as internal_records,
    extract_frozen_test,
    validate_frozen_counts,
    write_split_outputs,
)
from eval.paper.protocol import (  # noqa: E402
    PaperProtocol,
    ProtocolError,
    SUPPORTED_SUITES,
    parse_csv_selection,
    sha256_json,
)
from eval.paper.records import canonicalize_llava, write_records  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "paper_eval_worker.py"


def _internal_image_state(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    paths = {
        Path(str(record["image_path"])).resolve()
        for record in [*train_records, *test_records]
    }
    return [
        {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]


def run(command: Sequence[str], *, dry_run: bool, env: Mapping[str, str] | None = None) -> None:
    print("$ " + subprocess.list2cmdline([str(item) for item in command]), flush=True)
    if not dry_run:
        subprocess.run([str(item) for item in command], cwd=ROOT, check=True, env=dict(env or os.environ))


def _available_disk_gib(path: Path) -> float:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / (1024**3)


def _ram_gib() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return None


def _gpu_inventory() -> list[Dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    inventory: list[Dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.rsplit(",", 2)]
        if len(parts) != 3:
            continue
        try:
            inventory.append(
                {
                    "name": parts[0],
                    "memory_mib": int(float(parts[1])),
                    "compute_capability": float(parts[2]),
                }
            )
        except ValueError:
            continue
    return inventory


def preflight(protocol: PaperProtocol, args: argparse.Namespace) -> Dict[str, Any]:
    runtime = protocol.data["runtime"]
    state = git_state(ROOT)
    if state.get("dirty") and not args.allow_dirty and not args.dry_run:
        raise SystemExit(
            "Paper mode requires a clean Git worktree. Commit/stash your changes or pass "
            "--allow-dirty for a diagnostic run; the dirty patch hash will be recorded."
        )
    disk = _available_disk_gib(Path(args.asset_root))
    ram = _ram_gib()
    gpu = _gpu_inventory()
    report = {
        "protocol_sha256": protocol.fingerprint,
        "git": state,
        "disk_free_gib": round(disk, 2),
        "ram_gib": None if ram is None else round(ram, 2),
        "gpus": gpu,
        "retrieval": False,
        "training_allowed": False,
    }
    if not args.skip_hardware_check and not args.dry_run:
        if disk < float(runtime["minimum_disk_gib"]):
            raise SystemExit(
                f"Only {disk:.1f} GiB free; paper evaluation requires at least "
                f"{runtime['minimum_disk_gib']} GiB before downloads."
            )
        if ram is not None and ram < float(runtime["minimum_ram_gib"]):
            raise SystemExit(
                f"Only {ram:.1f} GiB RAM; expected at least {runtime['minimum_ram_gib']} GiB."
            )
        inference_command = args.command in {"all", "preflight", "smoke", "run"}
        if inference_command:
            if not gpu:
                raise SystemExit(
                    "No NVIDIA GPU detected; pass --skip-hardware-check only for CPU diagnostics."
                )
            minimum_memory = int(runtime["minimum_gpu_memory_mib"])
            minimum_capability = float(runtime["minimum_compute_capability"])
            maximum_capability = float(runtime["maximum_compute_capability_exclusive"])
            eligible = [
                item
                for item in gpu
                if int(item["memory_mib"]) >= minimum_memory
                and minimum_capability <= float(item["compute_capability"]) < maximum_capability
            ]
            if not eligible:
                raise SystemExit(
                    "No GPU satisfies the frozen runtime gate: at least "
                    f"{minimum_memory} MiB and compute capability >="
                    f"{minimum_capability:.1f}, <{maximum_capability:.1f}. Observed: {gpu}"
                )
    return report


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def prepare_internal(protocol: PaperProtocol, args: argparse.Namespace) -> Path:
    cfg = protocol.data["datasets"]["internal"]
    root = Path(args.data_root).resolve() / "internal"
    frozen = root / "frozen" / "test.json"
    rebuilt = root / "rebuilt"
    records_path = root / "records.jsonl"
    if args.dry_run:
        print(f"PLAN prepare internal dataset -> {root}")
        return records_path
    extract_frozen_test(
        ROOT / cfg["frozen_artifact"],
        cfg["frozen_member"],
        frozen,
        cfg["test_json_sha256"],
    )
    rebuilt_test = rebuilt / "test.json"
    if not rebuilt_test.is_file():
        builder = cfg["builder"]
        command = [
            sys.executable,
            "scripts/build_astrollava_trainset.py",
            "--hf-id",
            cfg["source_repo"],
            "--revision",
            cfg["source_revision"],
            "--output-dir",
            str(rebuilt),
            "--include-qa",
            "--max-image-size",
            str(builder["max_image_size"]),
            "--test-fraction",
            str(builder["test_fraction"]),
            "--seed",
            str(builder["seed"]),
        ]
        # The builder deterministically reuses already materialized images. If
        # a previous attempt stopped after train.json but before test.json,
        # overwrite only the manifests and continue from the image cache.
        if (rebuilt / "train.json").exists():
            command.append("--overwrite")
        run(command, dry_run=False)
    observed_hash = sha256_file(rebuilt_test)
    if observed_hash != cfg["test_json_sha256"]:
        raise SystemExit(
            f"Reconstructed internal test.json hash is {observed_hash}; expected "
            f"{cfg['test_json_sha256']}. The runner will not evaluate a changed split."
        )
    images = rebuilt / "images"
    test_records = internal_records(rebuilt_test, images, require_images=True)
    validate_frozen_counts(test_records, cfg)
    write_split_outputs(
        test_records,
        root,
        cfg["source_revision"],
        cfg["test_json_sha256"],
        _git_commit(),
        cfg["builder"],
    )
    train_records = internal_records(rebuilt / "train.json", images, require_images=True)
    audit_root = root / "leakage_audit"
    train_json_sha256 = sha256_file(rebuilt / "train.json")
    audit_fingerprint = sha256_json(
        {
            "source_revision": cfg["source_revision"],
            "train_json_sha256": train_json_sha256,
            "test_json_sha256": observed_hash,
            "image_state": _internal_image_state(train_records, test_records),
            "phash_likely_threshold": 4,
            "phash_sensitivity_threshold": 8,
            "implementation_sha256": sha256_file(ROOT / "eval" / "paper" / "internal.py"),
        }
    )
    audit_manifest_path = audit_root / "audit_manifest.json"
    image_report_path = audit_root / "image_overlap_report.json"
    text_report_path = audit_root / "text_overlap_report.json"
    cached_manifest: Dict[str, Any] = {}
    if audit_manifest_path.is_file():
        try:
            cached_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached_manifest = {}
    if (
        cached_manifest.get("audit_fingerprint") == audit_fingerprint
        and image_report_path.is_file()
        and text_report_path.is_file()
        and all(
            (audit_root / name).is_file()
            for name in (
                "image_overlap_candidates.jsonl",
                "train_image_index.jsonl",
                "test_image_index.jsonl",
            )
        )
    ):
        image_report = json.loads(image_report_path.read_text(encoding="utf-8"))
        text_report = json.loads(text_report_path.read_text(encoding="utf-8"))
        print(f"REUSE verified internal leakage audit {audit_fingerprint[:16]}")
    else:
        image_report = audit_image_overlap(train_records, test_records, images, audit_root)
        text_report = audit_text_overlap(train_records, test_records)
        write_json_atomic(text_report_path, text_report)
        write_json_atomic(
            audit_manifest_path,
            {
                "audit_fingerprint": audit_fingerprint,
                "source_revision": cfg["source_revision"],
                "train_json_sha256": train_json_sha256,
                "test_json_sha256": observed_hash,
                "train_records": len(train_records),
                "test_records": len(test_records),
                "image_report": image_report,
                "text_exact_match_count": text_report[
                    "exact_normalized_reference_matches"
                ],
                "foundation_pretraining_overlap_auditable": False,
            },
        )
    return records_path


def prepare_deepsdo(protocol: PaperProtocol, args: argparse.Namespace) -> Path:
    cfg = protocol.data["datasets"]["deepsdo"]
    root = Path(args.data_root).resolve() / "deepsdo"
    archive = root / "raw" / "kasi_deepsdo_desc_dataset.tar.gz"
    extracted = root / "extracted"
    llava = root / "llava"
    records_path = root / "records.jsonl"
    if args.dry_run:
        print(f"PLAN download/verify DeepSDO {cfg['archive_sha256']} -> {root}")
        return records_path
    command = [
        sys.executable,
        "scripts/prepare_deepsdo.py",
        "--archive",
        str(archive),
        "--download",
        "--extract-dir",
        str(extracted),
        "--output-dir",
        str(llava),
        "--splits",
        "test",
    ]
    run(command, dry_run=False)
    records = canonicalize_llava(llava / "test.json", extracted / "desc_images", "deepsdo")
    if len(records) != int(cfg["expected_records"]):
        raise SystemExit(f"DeepSDO prepared {len(records)} records; expected {cfg['expected_records']}")
    write_records(records_path, records)
    return records_path


def lock_astrovlbench(protocol: PaperProtocol, args: argparse.Namespace) -> Path:
    from eval.paper import astrovlbench

    root = Path(args.data_root).resolve() / "astrovlbench"
    if args.dry_run:
        print(f"PLAN lock gated AstroVLBench snapshot -> {root}")
        return root / "astrovlbench.lock.json"
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to lock the gated AstroVLBench snapshot")
    # The adapter owns revision discovery and content/prompt hashes.  No inference occurs before lock.
    return astrovlbench.lock_huggingface_snapshot(
        repo_id=protocol.data["datasets"]["astrovlbench"]["source_repo"],
        output_dir=root,
        token=token,
        cache_dir=Path(args.hf_cache),
    )


def prepare_astrovlbench(protocol: PaperProtocol, args: argparse.Namespace) -> Path:
    from eval.paper import astrovlbench

    root = Path(args.data_root).resolve() / "astrovlbench"
    records_path = root / "records.jsonl"
    lock_path = root / "astrovlbench.lock.json"
    if args.dry_run:
        print(f"PLAN validate AstroVLBench lock and materialize all five tasks -> {records_path}")
        return records_path
    if not lock_path.is_file():
        raise SystemExit(
            "AstroVLBench is not locked. Run with --lock-astrovlbench after access is approved."
        )
    records, report = astrovlbench.materialize_locked_records(lock_path)
    write_records(records_path, records)
    write_json_atomic(root / "adapter_report.json", report)
    return records_path


def prepare_suites(
    protocol: PaperProtocol, suites: Sequence[str], args: argparse.Namespace
) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for suite in suites:
        if suite == "internal":
            result[suite] = prepare_internal(protocol, args)
        elif suite == "deepsdo":
            result[suite] = prepare_deepsdo(protocol, args)
        else:
            result[suite] = prepare_astrovlbench(protocol, args)
    return result


def models_for_suites(
    protocol: PaperProtocol, suites: Sequence[str], requested: str
) -> Dict[str, list[str]]:
    requested_labels = None
    if requested != "all":
        requested_labels = set(parse_csv_selection(requested, protocol.data["models"]))
    result: Dict[str, list[str]] = {}
    for suite in suites:
        labels = list(protocol.selected_models(suite))
        if requested_labels is not None:
            labels = [label for label in labels if label in requested_labels]
        if not labels:
            raise ProtocolError(f"No requested models are enabled for suite {suite}")
        result[suite] = labels
    return result


def download_assets(
    protocol: PaperProtocol,
    suite_models: Mapping[str, Sequence[str]],
    args: argparse.Namespace,
) -> Path:
    labels = sorted({label for models in suite_models.values() for label in models})
    return materialize_model_assets(
        protocol,
        labels,
        args.asset_root,
        args.hf_cache,
        dry_run=args.dry_run,
        suites=suite_models.keys(),
    )


def _worker_python(model: Mapping[str, Any]) -> str:
    if model["backend"] == "astrollava":
        return os.environ.get("PAPER_ASTROLLAVA_PYTHON", sys.executable)
    return os.environ.get("PAPER_MODERN_PYTHON", sys.executable)


def run_models(
    protocol: PaperProtocol,
    records: Mapping[str, Path],
    suite_models: Mapping[str, Sequence[str]],
    asset_manifest: Path,
    args: argparse.Namespace,
) -> None:
    for suite, labels in suite_models.items():
        suite_root = protocol.output_dir(suite, args.output_root)
        records_path = Path(records[suite])
        records_hash = sha256_file(records_path) if records_path.is_file() else None
        for label in labels:
            model = protocol.data["models"][label]
            output = (
                protocol.model_output_dir(
                    suite, label, str(records_hash), ROOT, args.output_root
                )
                if records_hash
                else suite_root / label / "locked-after-dataset-prepare"
            )
            base = [
                _worker_python(model),
                str(WORKER),
                "--protocol",
                str(protocol.path),
                "--suite",
                suite,
                "--model",
                label,
                "--records",
                str(records[suite]),
                "--asset-manifest",
                str(asset_manifest),
                "--output-dir",
                str(output),
                "--device",
                args.device,
                "--max-attempts",
                str(args.max_attempts),
            ]
            if args.resume:
                base.append("--resume")
            if args.diagnostic_allow_partial:
                base.append("--diagnostic-allow-partial")
            if args.dry_run:
                print(f"PLAN smoke then full inference: {suite}/{label} -> {output}")
                continue
            if not args.skip_smoke:
                smoke = [*base, "--smoke", "--limit", str(args.smoke_samples), "--diagnostic-allow-partial"]
                if "--resume" not in smoke and output.exists() and any(output.iterdir()):
                    smoke.append("--resume")
                run(smoke, dry_run=False)
                if "--resume" not in base:
                    base.append("--resume")
            run(base, dry_run=False)


def analyze(protocol: PaperProtocol, suites: Sequence[str], args: argparse.Namespace) -> Path:
    from eval.paper.analysis import analyze_study

    if args.dry_run:
        path = Path(args.output_root) / "reports"
        print(f"PLAN score, bootstrap, and render paper outputs -> {path}")
        return path
    return analyze_study(
        protocol,
        suites,
        Path(args.output_root),
        paper_mode=not args.diagnostic_allow_partial,
        data_root=Path(args.data_root),
    )


def stage_audit_inputs(protocol: PaperProtocol, args: argparse.Namespace) -> Path:
    """Collect non-weight protocol/data evidence beside generated results."""

    output_root = Path(args.output_root).resolve()
    audit_root = output_root / "audit_inputs"
    if audit_root.exists():
        if audit_root.parent != output_root:
            raise ArtifactError(f"Refusing to replace unexpected audit path: {audit_root}")
        shutil.rmtree(audit_root)
    audit_root.mkdir(parents=True)

    def copy(source: Path, relative: str) -> None:
        if source.is_file():
            destination = audit_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    copy(protocol.path, "protocol/configs/paper_eval_v2.yaml")
    for name in (
        "requirements-paper-modern.txt",
        "requirements-paper-astrollava.txt",
        "requirements-astrollava-reference.txt",
    ):
        copy(ROOT / name, f"protocol/{name}")
    copy(ROOT / "docs" / "paper_evaluation_v2.md", "protocol/paper_evaluation_v2.md")
    copy(Path(args.asset_root) / "asset_manifest.json", "assets/asset_manifest.json")
    copy(output_root / "preflight.json", "environment/preflight.json")

    data_root = Path(args.data_root).resolve()
    internal = data_root / "internal"
    for name in ("records.jsonl", "manifest.json", "split_manifest.csv", "data_lineage.json"):
        copy(internal / name, f"datasets/internal/{name}")
    leakage = internal / "leakage_audit"
    if leakage.is_dir():
        for source in leakage.rglob("*"):
            if source.is_file():
                copy(source, f"datasets/internal/leakage_audit/{source.relative_to(leakage).as_posix()}")

    deepsdo = data_root / "deepsdo"
    for name in ("records.jsonl",):
        copy(deepsdo / name, f"datasets/deepsdo/{name}")
    for source in (deepsdo / "llava").glob("*") if (deepsdo / "llava").is_dir() else ():
        if source.is_file() and source.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".txt"}:
            copy(source, f"datasets/deepsdo/llava/{source.name}")

    astro = data_root / "astrovlbench"
    for name in ("astrovlbench.lock.json", "adapter_report.json", "records.jsonl"):
        copy(astro / name, f"datasets/astrovlbench/{name}")

    selected_suites = args.suites
    selected_models = args.models
    commands = [
        "# Exact recovery/reproduction commands",
        "",
        "```bash",
        "bash scripts/runpod/run_paper_eval.sh \\",
        f"  --suites {shlex.quote(str(selected_suites))} \\",
        f"  --models {shlex.quote(str(selected_models))} \\",
        f"  --output-root {shlex.quote(str(args.output_root))} \\",
        f"  --data-root {shlex.quote(str(args.data_root))} \\",
        f"  --asset-root {shlex.quote(str(args.asset_root))} \\",
        f"  --hf-cache {shlex.quote(str(args.hf_cache))} \\",
        "  --resume",
        "```",
    ]
    if "astrovlbench" in selected_suites:
        commands.extend(
            [
                "",
                "AstroVLBench must first be locked from the approved gated snapshot:",
                "",
                "```bash",
                "HF_TOKEN=... bash scripts/runpod/run_paper_eval.sh --lock-astrovlbench",
                "```",
            ]
        )
    (audit_root / "REPRODUCE.md").write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_checksum_file(audit_root)
    return audit_root


def package(protocol: PaperProtocol, reports: Path, args: argparse.Namespace) -> list[Path]:
    bundle_root = Path(args.output_root) / "bundles"
    if args.dry_run:
        print(f"PLAN private/public redacted bundles -> {bundle_root}")
        return []
    from eval.paper.analysis import report_output_dir

    selected_suites = parse_csv_selection(args.suites, SUPPORTED_SUITES)
    report_root = report_output_dir(
        protocol,
        selected_suites,
        args.output_root,
        args.data_root,
        repo_root=ROOT,
    )
    manifest_path = report_root / "results_manifest.json"
    if not report_root.is_dir() or not manifest_path.is_file():
        raise SystemExit(
            "Current-protocol paper results are missing; run the analyze command successfully "
            f"before packaging: {manifest_path}"
        )
    report_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report_manifest.get("protocol_sha256") != protocol.fingerprint:
        raise SystemExit("Paper results manifest does not match the current protocol")
    if report_manifest.get("analysis_run_sha256", "")[:16] != report_root.name:
        raise SystemExit("Paper results analysis-run hash does not match its directory")
    if set(report_manifest.get("suites") or []) != set(selected_suites):
        raise SystemExit(
            "Paper results suite set does not match --suites; rerun analyze with the same selection"
        )
    stage_audit_inputs(protocol, args)
    write_checksum_file(report_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root).resolve()
    with tempfile.TemporaryDirectory(
        prefix="paper-eval-current-", dir=output_root.parent
    ) as temporary:
        current = Path(temporary) / protocol.study_id
        current.mkdir()

        def copy_tree(source: Path, destination: Path) -> None:
            if not source.is_dir():
                raise SystemExit(f"Required package evidence is missing: {source}")
            shutil.copytree(source, destination, symlinks=True)

        for suite in selected_suites:
            records_path = Path(args.data_root) / suite / "records.jsonl"
            if not records_path.is_file():
                raise SystemExit(f"Canonical records are missing for packaging: {records_path}")
            records_hash = sha256_file(records_path)
            suite_root = protocol.output_dir(suite, output_root)
            for model_label in protocol.selected_models(suite):
                model_root = protocol.model_output_dir(
                    suite,
                    model_label,
                    records_hash,
                    ROOT,
                    output_root,
                )
                copy_tree(
                    model_root,
                    current
                    / suite
                    / suite_root.name
                    / model_label
                    / model_root.name,
                )
        copy_tree(report_root, current / "reports" / report_root.name)
        copy_tree(output_root / "audit_inputs", current / "audit_inputs")
        preflight_path = output_root / "preflight.json"
        if preflight_path.is_file():
            shutil.copy2(preflight_path, current / "preflight.json")
        private = create_bundle(
            current,
            bundle_root / f"{protocol.study_id}-private.tar.gz",
            public=False,
        )
        public = create_bundle(
            current,
            bundle_root / f"{protocol.study_id}-public-redacted.tar.gz",
            public=True,
        )
    write_json_atomic(
        bundle_root / "bundle_manifest.json",
        {
            "private": {"path": str(private), "sha256": sha256_file(private)},
            "public": {"path": str(public), "sha256": sha256_file(public)},
        },
    )
    return [private, public]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "preflight", "prepare", "download", "smoke", "run", "analyze", "package"],
    )
    parser.add_argument("--protocol", default="configs/paper_eval_v2.yaml")
    parser.add_argument("--suites", default="internal,deepsdo")
    parser.add_argument("--models", default="all")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--hf-cache", default=os.environ.get("HF_HOME", "hf_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-hardware-check", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--diagnostic-allow-partial", action="store_true")
    parser.add_argument("--lock-astrovlbench", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = PaperProtocol.load(ROOT / args.protocol if not Path(args.protocol).is_absolute() else args.protocol)
    args.output_root = args.output_root or protocol.data["runtime"]["output_root"]
    args.data_root = args.data_root or protocol.data["runtime"]["data_root"]
    args.asset_root = args.asset_root or protocol.data["runtime"]["asset_root"]
    suites = parse_csv_selection(args.suites, SUPPORTED_SUITES)
    if args.lock_astrovlbench:
        lock = lock_astrovlbench(protocol, args)
        print(f"AstroVLBench lock: {lock}")
        if args.command == "all" and args.suites == "internal,deepsdo":
            return
    preflight_report = preflight(protocol, args)
    preflight_path = Path(args.output_root) / "preflight.json"
    if not args.dry_run:
        write_json_atomic(preflight_path, preflight_report)
    else:
        print(json.dumps(preflight_report, indent=2))
    if args.command == "preflight":
        return
    suite_models = models_for_suites(protocol, suites, args.models)
    records: Dict[str, Path] = {
        suite: Path(args.data_root) / suite / "records.jsonl" for suite in suites
    }
    if args.command in {"all", "prepare"}:
        records = prepare_suites(protocol, suites, args)
        if args.command == "prepare":
            return
    elif not args.dry_run:
        missing = [str(path) for path in records.values() if not path.is_file()]
        if missing:
            raise SystemExit("Prepared records are missing; run prepare first: " + ", ".join(missing))
    asset_manifest = Path(args.asset_root) / "asset_manifest.json"
    if args.command in {"all", "download"}:
        asset_manifest = download_assets(protocol, suite_models, args)
        if args.command == "download":
            return
    elif not args.dry_run and not asset_manifest.is_file():
        raise SystemExit("Asset manifest is missing; run download first")
    if args.command in {"all", "smoke", "run"}:
        original_skip = args.skip_smoke
        if args.command == "smoke":
            args.skip_smoke = False
            # Run_models normally follows smoke with full generation; invoke workers directly by
            # marking the command through a temporary dry/full choice is clearer for users.
            for suite, labels in suite_models.items():
                suite_root = protocol.output_dir(suite, args.output_root)
                records_path = Path(records[suite])
                records_hash = sha256_file(records_path) if records_path.is_file() else None
                for label in labels:
                    model = protocol.data["models"][label]
                    output = (
                        protocol.model_output_dir(
                            suite, label, str(records_hash), ROOT, args.output_root
                        )
                        if records_hash
                        else suite_root / label / "locked-after-dataset-prepare"
                    )
                    cmd = [
                        _worker_python(model), str(WORKER), "--protocol", str(protocol.path),
                        "--suite", suite, "--model", label, "--records", str(records[suite]),
                        "--asset-manifest", str(asset_manifest), "--output-dir", str(output),
                        "--device", args.device, "--smoke", "--limit", str(args.smoke_samples),
                        "--diagnostic-allow-partial", "--max-attempts", str(args.max_attempts),
                    ]
                    if args.resume or output.exists():
                        cmd.append("--resume")
                    run(cmd, dry_run=args.dry_run)
            return
        run_models(protocol, records, suite_models, asset_manifest, args)
        args.skip_smoke = original_skip
        if args.command == "run":
            return
    reports = Path(args.output_root) / "reports"
    if args.command in {"all", "analyze"}:
        reports = analyze(protocol, suites, args)
        if args.command == "analyze":
            return
    if args.command in {"all", "package"}:
        bundles = package(protocol, reports, args)
        if bundles:
            print("Bundles:")
            for path in bundles:
                print(f"  {path}")


def _print_recovery_command() -> None:
    arguments = list(sys.argv[1:])
    if "--resume" not in arguments:
        arguments.append("--resume")
    print(
        "Recovery command:\n  "
        + shlex.join([sys.executable, str(Path(__file__).resolve()), *arguments]),
        file=sys.stderr,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            _print_recovery_command()
        raise
    except Exception:
        _print_recovery_command()
        raise
