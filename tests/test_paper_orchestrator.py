from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval.paper.protocol import PaperProtocol
from scripts.paper_eval_worker import validate_locked_output_path
from scripts.run_paper_eval import preflight


ROOT = Path(__file__).resolve().parents[1]


class PaperOrchestratorDryRunTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "RunPod wrapper executes under POSIX bash")
    def test_runpod_wrapper_help_is_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            env = dict(os.environ)
            env["PAPER_EVAL_VENV_ROOT"] = str(temporary / "venvs")
            env["PAPER_EVAL_RUNTIME_ROOT"] = str(temporary / "runtime")
            completed = subprocess.run(
                ["bash", str(ROOT / "scripts" / "runpod" / "run_paper_eval.sh"), "--help"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Usage:", completed.stdout)
            self.assertFalse((temporary / "venvs").exists())
            self.assertFalse((temporary / "runtime").exists())

    @unittest.skipIf(os.name == "nt", "RunPod wrapper executes under POSIX bash")
    def test_runpod_wrapper_dry_run_does_not_bootstrap_environments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            env = dict(os.environ)
            env["PAPER_EVAL_VENV_ROOT"] = str(temporary / "venvs")
            env["PAPER_EVAL_RUNTIME_ROOT"] = str(temporary / "runtime")
            completed = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts" / "runpod" / "run_paper_eval.sh"),
                    "all",
                    "--suites",
                    "internal,deepsdo",
                    "--models",
                    "astraq_stage1",
                    "--dry-run",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("PLAN prepare internal dataset", completed.stdout)
            self.assertFalse((temporary / "venvs").exists())
            self.assertFalse((temporary / "runtime").exists())

    @unittest.skipIf(os.name == "nt", "RunPod shell syntax is checked on POSIX hosts")
    def test_runpod_wrapper_has_valid_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / "runpod" / "run_paper_eval.sh")],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_nominal_24gb_ampere_gpu_passes_realistic_preflight_gate(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")
        args = SimpleNamespace(
            allow_dirty=False,
            dry_run=False,
            skip_hardware_check=False,
            command="all",
            asset_root=ROOT,
        )
        with mock.patch("scripts.run_paper_eval.git_state", return_value={"dirty": False}), \
            mock.patch("scripts.run_paper_eval._available_disk_gib", return_value=100.0), \
            mock.patch("scripts.run_paper_eval._ram_gib", return_value=64.0), \
            mock.patch(
                "scripts.run_paper_eval._gpu_inventory",
                return_value=[
                    {"name": "NVIDIA L4", "memory_mib": 23034, "compute_capability": 8.9}
                ],
            ):
            report = preflight(protocol, args)
        self.assertEqual(report["gpus"][0]["name"], "NVIDIA L4")

    def test_preflight_rejects_pre_ampere_or_undersized_gpu(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")
        args = SimpleNamespace(
            allow_dirty=False,
            dry_run=False,
            skip_hardware_check=False,
            command="run",
            asset_root=ROOT,
        )
        observed = [
            {"name": "legacy", "memory_mib": 24000, "compute_capability": 7.5},
            {"name": "small", "memory_mib": 21999, "compute_capability": 8.6},
        ]
        with mock.patch("scripts.run_paper_eval.git_state", return_value={"dirty": False}), \
            mock.patch("scripts.run_paper_eval._available_disk_gib", return_value=100.0), \
            mock.patch("scripts.run_paper_eval._ram_gib", return_value=64.0), \
            mock.patch("scripts.run_paper_eval._gpu_inventory", return_value=observed):
            with self.assertRaisesRegex(SystemExit, "No GPU satisfies"):
                preflight(protocol, args)

    def test_worker_rejects_non_fingerprinted_overwrite_target(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")
        records_hash = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            expected = protocol.model_output_dir(
                "deepsdo", "astraq_stage2", records_hash, ROOT, run_root
            )
            self.assertEqual(
                validate_locked_output_path(
                    protocol,
                    "deepsdo",
                    "astraq_stage2",
                    records_hash,
                    expected,
                ),
                expected.resolve(),
            )
            with self.assertRaisesRegex(SystemExit, "locked generation fingerprint"):
                validate_locked_output_path(
                    protocol,
                    "deepsdo",
                    "astraq_stage2",
                    records_hash,
                    Path(tmp) / "unrelated" / "directory" / "must-not-delete",
                )

    def test_all_dry_run_plans_complete_pipeline_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            output_root = temporary / "outputs"
            data_root = temporary / "datasets"
            asset_root = temporary / "assets"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_paper_eval.py"),
                "all",
                "--dry-run",
                "--skip-hardware-check",
                "--allow-dirty",
                "--suites",
                "internal,deepsdo",
                "--models",
                "astraq_stage1",
                "--output-root",
                str(output_root),
                "--data-root",
                str(data_root),
                "--asset-root",
                str(asset_root),
            ]

            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for message in (
                "PLAN prepare internal dataset",
                "PLAN download/verify DeepSDO",
                "PLAN smoke then full inference: internal/astraq_stage1",
                "PLAN smoke then full inference: deepsdo/original_1024/astraq_stage1",
                "PLAN smoke then full inference: deepsdo/concise_256/astraq_stage1",
                "PLAN score, bootstrap, and render paper outputs",
                "PLAN private/public redacted bundles",
            ):
                self.assertIn(message, completed.stdout)

            # A true dry run must not prepare data, download/write an asset
            # manifest, generate predictions, reports, or bundles.
            manifest_path = asset_root / "asset_manifest.json"
            self.assertFalse(manifest_path.exists())
            self.assertFalse(asset_root.exists())
            self.assertFalse(data_root.exists())
            self.assertFalse(output_root.exists())

    def test_astrovlbench_dry_run_accepts_pre_downloaded_snapshot(self) -> None:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_paper_eval.py"),
            "prepare",
            "--protocol",
            "configs/paper_eval_astrovlbench_v1.yaml",
            "--dry-run",
            "--skip-hardware-check",
            "--allow-dirty",
            "--suites",
            "astrovlbench",
            "--models",
            "all",
            "--astrovlbench-snapshot",
            "/workspace/downloaded/AstroVLBench",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("PLAN lock gated AstroVLBench snapshot", completed.stdout)
        self.assertIn(
            "materialize selected components (task1,task2.first,task2.nvss)",
            completed.stdout,
        )

    def test_runpod_wrapper_contains_recovery_hardening(self) -> None:
        wrapper = (ROOT / "scripts" / "runpod" / "run_paper_eval.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("venv --clear --seed", wrapper)
        self.assertIn("HF_HUB_ENABLE_HF_TRANSFER", wrapper)
        self.assertIn("index.lock", wrapper)
        self.assertIn("configs/paper_eval_v4.yaml", wrapper)
        self.assertIn("--astrovlbench-snapshot", wrapper)


if __name__ == "__main__":
    unittest.main()
