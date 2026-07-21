"""Reproducible, paper-grade evaluation infrastructure for AstraQ-VL.

The package is deliberately import-light.  GPU libraries and optional metric
packages are imported only by the worker that needs them, so protocol checks,
artifact validation, and report generation can run on a CPU-only machine.
"""

SCHEMA_VERSION = 2
