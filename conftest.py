"""Root-level pytest configuration.

This file is discovered by pytest before any test packages are imported,
so it's the right place to set ``JAX_PLATFORMS`` — JAX reads that
environment variable at first import and selects the backend; setting
it from a per-package conftest is too late, since pytest's collection
machinery has already imported ``jax`` by then.

Tests use the CPU backend because the synthetic problems are small
(R ≤ 5, N ≤ 100) and dominated by GPU launch overhead — CPU runs
~25× faster end-to-end on this hardware.  Production fits done by the
driver scripts under ``scripts/`` are unaffected.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
