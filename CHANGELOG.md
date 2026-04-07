# Changelog

All notable changes to vpjax are documented in this file.

## [Unreleased]

### Changed
- Require Python >= 3.11 (JAX 0.9.2 minimum).
- Pin JAX >= 0.9.2; remove separate jaxlib dependency (bundled in jax 0.9+).

### Fixed
- JAX CUDA dependency for aarch64 DGX Spark: require jax >= 0.5 for GPU.
- JAX CUDA dependency: platform marker and remove version cap.

### Added
- GPU optional dependency for JAX CUDA.

## [0.1.0] -- 2025

Initial release of vpjax: Virtual Physiology in JAX.

### Added

#### Stochastic Models
- SDE Balloon model for noise-driven hemodynamic fluctuations.
- Fokker-Planck solver for Balloon state probability density evolution.
- Stochastic sleep transition model.

#### Validation
- NVC model comparison framework (awaiting hippy-feat preprocessed BOLD).
- Sleep EEG-fMRI validation pipeline (ds003768).
- Vasomotion prediction, cardiac ECG extraction, and all-runs validation.

#### Brainstem
- Brainstem package: mICA nuclei identification from MELODIC.

#### Sleep
- Improved N3 deep sleep model with 5 physiological additions.
- Sleep package: state-dependent NVC, vasomotion, glymphatic coupling.

#### Cardiac
- Cardiac package: heart-brain coupling models (vagal, baroreceptor, pulsatility).
- Upgraded baroreceptor (Pulse-style) and added SIMULA glymphatic model.

#### Vascular
- Angiography module: TOF to subject-specific model parameters.

#### Core
- Presets (3T/7T parameter bundles) and integration tests.
- VASO and QSM subpackages.
- Phase 2: Advanced physiological models (Riera, CMRO2, qBOLD, layers).
- Phase 0-1: Balloon-Windkessel ODE and observation functions.
- Cortical layer analysis, QSM pipeline, and VASO modules.
- qBOLD module and CMRO2 hierarchy with Bulte references.

### References
- Riera JJ et al. (2006/2007). Nonlinear local electrovascular coupling. HBM.
- Buxton RB et al. (1998). Dynamics of blood flow and oxygenation changes. MRM.
- Friston KJ et al. (2000). Nonlinear responses in fMRI. NeuroImage.
- Lu H, Ge Y (2008). TRUST MRI. MRM.
- He X, Yablonskiy DA (2007). Quantitative BOLD. MRM.
