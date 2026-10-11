# Changelog

All notable changes to vpjax are documented in this file.

## [Unreleased]

### Changed
- Require Python >= 3.11 (JAX 0.9.2 minimum).
- Pin JAX >= 0.9.2; remove separate jaxlib dependency (bundled in jax 0.9+).

### Fixed
- `residual_diagnostics`: the lag-1 innovation autocorrelation used raw
  rather than mean-removed products, so a channel with an offset could
  report a value above 1. Now a proper autocorrelation.
- JAX CUDA dependency for aarch64 DGX Spark: require jax >= 0.5 for GPU.
- JAX CUDA dependency: platform marker and remove version cap.

### Added

- `vpjax/validation/natview.py`: loader for the NKI NATVIEW simultaneous
  EEG-fMRI dataset (Telesford et al. 2023; FCP-INDI, CC BY 4.0) putting
  EEG, pupil area and respiration-belt streams on one clock through the
  scanner triggers each of them recorded — a linear fit through the
  shared trigger trains, refused if the residual exceeds the stream's
  timestamp quantisation. Pupil is binned with a validity mask (blinks,
  eye closure) rather than filled; respiration is reduced to a breathing
  amplitude envelope. `scripts/validate_natview_statespace.py` fits
  BOLD-only and any combination of the three channels on the same run.
- `eeg_fmri_statespace` generalised to any number of auxiliary channels
  (`param_names(channels)`), each with its own loading, nuisance state
  and noise; loading signs chosen greedily (k+1 fits, not 2^k). Sparse
  channels enter through the presence mask.
- `scripts/ds003768_rois.py`: visual-cortex and motor-strip BOLD series
  in native EPI space via FSL (mean EPI → T1w → MNI, Harvard-Oxford
  max-prob labels), for regional rather than global pairing with the
  occipital alpha envelope. `validate_ds003768_statespace.py` takes
  `--roi-file/--roi`.
- `eeg_artifacts.load_brainvision` falls back to an EEGLAB `.set` beside
  the BrainVision header, taking triggers from the `.vmrk`.

- `vpjax/validation/eeg_artifacts.py`: in-scanner EEG cleaning as a
  transparent baseline — gradient average-artifact subtraction locked to
  the volume trigger (Allen et al. 2000), R-peak detection and
  ballistocardiogram AAS (Allen et al. 1998), anti-aliased downsampling,
  and binned Hilbert band envelopes. Refuses recordings whose trigger
  spacing jitters, since AAS without a scanner-locked clock smears the
  template. `clean_rest_run` returns the envelope and the volume-trigger
  times on one clock so BOLD volumes are stamped without an offset guess.
- `vpjax/validation/eeg_fmri_statespace.py`: fits the drive-plus-Balloon
  model to a recorded run with the observation noise variances and the
  EEG loading (sign chosen by likelihood) estimated rather than supplied,
  reporting innovation whiteness and identifiability for a BOLD-only and
  a BOLD+EEG arm. The EEG channel gets its own fast OU nuisance state by
  default: on ds003768 a single shared state is rejected by whiteness
  (BOLD innovation lag-1 0.89), the six-state model is calibrated on
  both channels. `scripts/validate_ds003768_statespace.py` runs it on an
  OpenNeuro ds003768 rest run.
- `statespace_recovery`: a shape-only `drug_shaped_input` (unit-peak
  two-exponential; every time scale supplied by the caller), a `beta`
  gain on the drive input estimated alongside the other parameters, and
  `fit_names` on `fit_simulation` to hold any subset at truth — the
  "known from a separate session" counterfactual. Exogenous inputs reach
  the filter through `ll_filter(inputs=...)`, sampled on the grid.
- `scripts/simultaneity_case.py`: multi-seed version of the question
  "does identifiability need simultaneous EEG?": an envelope-SNR sweep,
  the fixed-drive-statistics counterfactual, and the identifiability of a
  drug-contrast gain with and without the fast channel.
- `pharmacokinetics/` subpackage: PK/PD layer above the existing
  hemodynamic and neural models.
  - Linear mamillary compartment models (1-3 compartments, optional
    first-order absorption depot) solved in closed form via the matrix
    exponential of the augmented generator, so the solution is exact
    and carries no solver tolerance into the parameter estimates.
  - Saturable (Michaelis-Menten) elimination integrated with Diffrax
    `Kvaerno5` for the nonlinear case.
  - Dose events as known-time discontinuities: `lax.scan` over
    inter-dose intervals with boluses applied at the boundaries, rather
    than Diffrax's state-triggered event system. Dose times are static,
    dose amounts stay differentiable.
  - Effect-site compartment, Hill/Emax occupancy, and regional
    weighting by receptor density -- the join to vbjax neural masses
    and the vpjax vascular models.
  - Combined additive-plus-proportional residual error model.
  - Single-subject estimation by Optimistix Levenberg-Marquardt on log
    parameters; population (NLME) model in NumPyro with non-centred
    random effects and an LKJ prior on their correlation.
  - Parameterisation, error model and closed-form solutions follow the
    NONMEM/Pumas conventions so that fits can be cross-checked against
    an established estimator.
- `statespace/` subpackage: continuous-discrete state-space estimation
  for multimodal recordings, where one latent physiological state is
  observed by several sensors at different rates.
  - Local-linearization filter and RTS smoother. Mean by the LL step
    from an augmented matrix exponential (exact as the Jacobian
    approaches singularity, unlike the ridge in
    `integrators/local_linearization.py`); process noise by Van Loan's
    identity rather than quadrature.
  - Observations assimilated as sequential scalar updates. Identical to
    the joint update for diagonal R, but masking and the
    per-observation log-likelihood then decompose exactly, so an absent
    channel costs nothing and needs no special case.
  - `multirate.build_observation_grid` merges asynchronous streams onto
    a union-of-timestamps grid with per-channel presence masks. Nothing
    is interpolated; `max_dt` subdivides long intervals to keep the
    linearization fresh between volumes.
  - `estimation.fit_statespace` maximises the innovation likelihood
    through the filter recursion by autodiff, with multi-start (the
    kappa-tau ridge in the Balloon model traps a single start) and
    best-so-far solver wrapping.
  - `estimation.parameter_uncertainty` gives relative standard errors
    and an identifiability verdict from the observed Fisher
    information. This catches what innovation whiteness cannot: a model
    that whitens its residuals with badly wrong parameters because two
    of them trade off.
  - `estimation.residual_diagnostics` reports whiteness per channel,
    with lag-1 computed over consecutive *present* samples of each
    channel rather than adjacent grid rows.
- `validation/statespace_recovery.py` and
  `scripts/validate_statespace_recovery.py`: recovery of known
  parameters from a simulated EEG-fMRI run, and a BOLD-only vs
  BOLD+EEG contrast. At resting-state amplitude over 300 s, all four
  parameters recover to within 7% from both modalities, while from 151
  BOLD volumes alone the model is not identifiable (relative standard
  errors of 17 and 35 on the drive parameters) despite its innovations
  being white. The measured amplitude limit of the first-order filter
  is tabulated in the module docstring.
- `slow` pytest marker for the state-space fitting tests.
- `nlme` optional dependency group (numpyro, optax).
- `examples/pkpd_drive.py`: simulates dose to regional drive from an
  explicit JSON of PK/PD assumptions and records them in the output.
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
