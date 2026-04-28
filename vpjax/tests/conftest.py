"""Test configuration.

Force JAX to use the CPU backend for the test suite.  The synthetic
test problems are small (R ≤ 5, N ≤ 100) and dominated by GPU launch
overhead on CUDA — CPU runs ~25× faster end-to-end on this hardware.
The downstream production fits (whole-brain, R ≈ 85) still benefit from
the GPU and are run via the driver scripts under ``scripts/``.

Setting ``JAX_PLATFORMS`` before any test module imports JAX is the
simplest reliable way to do this — pytest loads ``conftest.py`` before
collecting tests, so the variable is in place by the time
``import vpjax`` triggers ``import jax``.
"""

"""Per-package conftest — kept for test discovery.

JAX backend selection is handled by the root-level ``conftest.py``
because that file runs before pytest imports ``jax``.
"""
