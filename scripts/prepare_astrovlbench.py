"""Prepare the pinned AstroVLBench release for paper evaluation.

The raw Hugging Face snapshot is immutable.  This command either adopts an
already downloaded directory or downloads the revision frozen in the protocol,
then writes a lock, correction overlay, exclusion ledger, audit reports, and
canonical records below the protocol's data root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.paper import astrovlbench  # noqa: E402
from eval.paper.protocol import PaperProtocol  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="configs/paper_eval_astrovlbench_v1.yaml",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--snapshot-dir",
        type=Path,
        help="Existing `hf download` directory to validate and adopt without modifying.",
    )
    source.add_argument(
        "--download",
        action="store_true",
        help="Download the protocol-pinned revision using the configured token.",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--hf-cache", type=Path, default=Path(os.environ.get("HF_HOME", "hf_cache")))
    parser.add_argument("--token-env", default="HF_TOKEN")
    return parser


def prepare(args: argparse.Namespace) -> dict[str, object]:
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol = PaperProtocol.load(protocol_path)
    dataset = protocol.data["datasets"]["astrovlbench"]
    repo_id = str(dataset["source_repo"])
    revision = str(dataset.get("locked_revision") or "")
    if revision != astrovlbench.PINNED_RELEASE_REVISION:
        raise SystemExit(
            "AstroVLBench protocol must pin release "
            f"{astrovlbench.PINNED_RELEASE_REVISION}; observed {revision!r}"
        )
    data_root = Path(args.data_root or protocol.data["runtime"]["data_root"]).resolve()
    bundle = data_root / "astrovlbench"
    if args.download:
        token = os.environ.get(args.token_env)
        if not token:
            raise SystemExit(
                f"{args.token_env} is required to download the gated AstroVLBench dataset"
            )
        lock_path = astrovlbench.lock_huggingface_snapshot(
            repo_id=repo_id,
            output_dir=bundle,
            token=token,
            cache_dir=Path(args.hf_cache),
            revision=revision,
        )
        source_mode = "pinned_download"
    else:
        lock_path = astrovlbench.lock_local_snapshot(
            Path(args.snapshot_dir),
            bundle,
            repo_id=repo_id,
            revision=revision,
        )
        source_mode = "pre_downloaded_snapshot"

    astrovlbench.validate_protocol_release_contract(lock_path, dataset)
    records, report = astrovlbench.materialize_locked_records(lock_path)
    expected_total = int(dataset["expected_records"])
    expected_components = {
        str(key): int(value)
        for key, value in dict(dataset["expected_component_records"]).items()
    }
    observed_components = {
        str(key): int(value)
        for key, value in dict(report["expected_counts_from_locked_snapshot"]).items()
    }
    if len(records) != expected_total or observed_components != expected_components:
        raise SystemExit(
            "Prepared AstroVLBench counts do not match the protocol: "
            f"total={len(records)}/{expected_total}, "
            f"components={observed_components}/{expected_components}"
        )
    lock = astrovlbench.read_lock_manifest(lock_path)
    derived = dict(lock["derived_artifacts"])
    return {
        "source_mode": source_mode,
        "repo_id": repo_id,
        "revision": revision,
        "bundle": str(bundle),
        "lock": str(lock_path),
        "records": expected_total,
        "component_records": observed_components,
        "first_exclusions": report["repair_report"]["first_metadata"]["excluded_rows"],
        "raw_snapshot_unchanged": report["raw_snapshot_unchanged"],
        "records_sha256": derived["records"]["sha256"],
        "adapter_report_sha256": derived["adapter_report"]["sha256"],
        "repair_report_sha256": derived["repair_report"]["sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(prepare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except astrovlbench.AstroVLBenchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
