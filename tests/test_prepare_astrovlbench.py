from __future__ import annotations

import contextlib
import io
import unittest

from scripts.prepare_astrovlbench import _parser


class PrepareAstroVLBenchCliTests(unittest.TestCase):
    def test_cli_supports_pinned_download_or_existing_snapshot(self) -> None:
        parser = _parser()
        downloaded = parser.parse_args(["--download"])
        self.assertTrue(downloaded.download)
        self.assertIsNone(downloaded.snapshot_dir)

        local = parser.parse_args(["--snapshot-dir", "/workspace/AstroVLBench"])
        self.assertFalse(local.download)
        self.assertEqual(local.snapshot_dir.as_posix(), "/workspace/AstroVLBench")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--download",
                        "--snapshot-dir",
                        "/workspace/AstroVLBench",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
