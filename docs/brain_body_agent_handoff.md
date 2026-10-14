# Brain/body work: agent handoff

Work in `/Users/mhough/Workspace/vpjax`, repository
`https://github.com/m9h/vpjax`, branch `pkpd-statespace`.
The Step 0/1/1b baseline is commit `6115228`; subsequent periodic-perturbation
work is in the commit containing this handoff. Use the repository `.venv`:
system Python lacks the required numerical dependencies.

## First finding to carry forward

The reported 479% SE for `g_bb` was a units error in the scratch probe.
`parameter_uncertainty` returns log-parameter SE, already approximately relative.
The historical 0.2394 is approximately 24%, not 479%. `fit_topology` preserves
its existing `standard_error` meaning and now adds explicit scale metadata,
absolute delta-method errors, optimizer convergence, Hessian eigenvalues,
positive-definiteness and parameter correlations.

The new `init` mapping is in physical units, overrides only fitted names and
rejects unknown/nonfinite/nonpositive values. With `restarts=0`, each requested
start runs alone; no random restart can hide start dependence.

## Current result

The canonical 30-fit experiment returns to the same solution from all tested
starts: gain and time-constant within-recording relative ranges are under
0.012%, versus gain CVs of 20.3%/24.0% across seeds at 32/64 s. The original
479% discrepancy is resolved by the log-SE units correction; initialization
inheritance is not supported by this test.

28/30 fits report convergence. At 64 s, seed 4 has `q_z` near zero and an
effectively singular nuisance Hessian (one tiny negative eigenvalue). Treat
pseudo-inverse SEs there as diagnostics. Product empirical CVs 9.4%/11.0%
are lower, but covariance-based product log SEs (~0.239/0.204 medians among
PD cells) do not establish the stronger claim that only the product is
determined. See the numerical result section in `periodic_perturbation.md`.

## Reproduce the initialization check

From the repository root:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m scripts.brain_body_initialization --periods 32 64 --seeds 0 1 2 3 4 --output docs/results/brain_body_initialization.json
```

Add `--resume` to reuse completed cells; saved settings and precision must match.
The experiment uses a 480 s efferent recording, fast+HR channels, six free
parameters, starts at 0.25/1/4 times true `g_bb` and the inverse times true
`tau_a`. The starting product is fixed but every parameter remains freely fitted.
JSON checkpoints contain all settings, truth, initial values, estimates,
likelihood, diagnostics, correlated product SE and elapsed time. These are
simulation results conditional on the model and known input, not empirical
human gain estimates.

The preliminary `brain_body_initialization_float32_64s.json` is a 64 s period
float32 run. `brain_body_initialization_float32_partial.json`
is the interrupted float32 two-period experiment, retained because some fits
reported nonconvergence despite agreeing in their estimates. The canonical
`brain_body_initialization.json` uses float64. Check its provenance and
completion count (30 fits).

A 1000-step retry of the seed-4, 32 s, factor-0.25 fit still reports
nonconvergence (`brain_body_initialization_retry.json`). It returns the same
endpoint as the converged factor-4 fit. Increasing the step limit alone does
not resolve that stopping flag; keep it visible when interpreting precision.

## Verification

`JAX_PLATFORMS=cpu .venv/bin/python -m pytest vpjax/tests/test_brain_body.py -q`:
54 passed. `... -m pytest vpjax/tests/test_statespace.py -q`: 43 passed.
Named-start propagation, SE unit conversion and periodic schedules have direct
regression coverage. `git diff --check` and script compilation also pass.

## Other saved material

- `docs/periodic_perturbation.md`: corrected asymmetry, SE units and period
  heuristic. The historical literature discussion was inherited; this numerical
  investigation is not an independent audit of those clinical claims.
- `scripts/brain_body_arm_probe.py`, `brain_body_period_sweep.py`,
  `brain_body_dcgain.py`: promoted scratch scripts; invoke with `python -m
  scripts.<name>` from this repo, using `.venv` and CPU.
- `docs/results/brain_body_*_legacy.txt`: original aggregate outputs retained
  with their invalid SE ratios labelled. They lack convergence/PD diagnostics.
- `docs/brain_body_system_models.md` and existing results JSONs: Steps 0/1/1b.

## Remaining work

1. Step 2: sweep duration, period and effect size for **each** drug-free arm.
   Define usable precision before choosing minutes. A receiver filter corner
   `period≈2*pi*tau` is a heuristic, not a universal optimal period. Historical
   probes support 32–64 s brain-first and 8–16 s body-first candidates.
2. Hold absolute measurement noise fixed in a design-comparison sensitivity
   analysis. Current `simulate` scales noise by each design's clean excursion,
   holding SNR fixed. Paced breathing and the modeled slow autonomic drive
   also need an explicit physical mapping before making protocol claims.
3. Five seeds assess start dependence here, not SE calibration or population
   precision. The three-channel/BOLD observation configuration needs separate
   duration sizing if it is the intended measurement set.
4. `neural_inputs` wiring in `eeg_fmri_statespace` remains partial. E/I dependency
   decision is untouched; `vbjax` is not installed by this work.
5. DMT data request remains a draft; no message has been sent.
6. Recorded-data test bed (2026-10-13): the recorded-run model
   (`eeg_fmri_statespace.py`, not `brain_body.py`) has now run on six NATVIEW
   resting subjects with six arms each (`docs/results/natview_rest_sub01-06.csv`;
   per-subject JSON under `~/Workspace/data/vpjax_results/natview/`, sub-01 as
   `_v4`, sub-02..06 as `_v3`). Respiration as a *vascular input* is supported in
   5/6 (LR 16–42 for 2 df, loading 18–28% SE, negative in 4, lag 4.5–36 s;
   sub-01, the original one-subject table, is the null exception); respiration
   as an *observation of the neural drive* is degenerate in 6/6 (noise floor,
   non-white innovations). That is a real-data vote for respiration on the
   vasculature's input side, i.e. the measurement/afferent cells, not the
   common-drive cell. The alpha loading sign is subject-dependent on NATVIEW
   (negative sub-01, positive sub-02/03) unlike ds003768 (16/18 negative).
   `brain_body.py` itself has still never been fitted to a recording; the NIMH
   deep-breath sessions under `~/Workspace/data/nimh_bolt/` are the intended
   first case (needs a loader) and that build awaits the user's go. QC: NATVIEW
   sub-04's ECG is unusable (R-peak detector at its RR floor), so its EEG arms
   are not evidence; sub-02's 44 bpm is genuine.
7. E/I architecture decided (2026-10-13, user chose option 4): gnf-jax is the
   latent E/I block, wrapped in `vpjax/neural/ei_block.py` (extra `ei`;
   installed in `.venv` from `git+https://github.com/m9h/gnf-jax.git`, its own
   28 tests pass under jax 0.11). `docs/ei_autonomic_coupling_plan.md` records
   the decision and what remains: compose with the Balloon in
   `eeg_fmri_statespace.py`, and supply a graph. On graphs: every local dataset
   (NATVIEW, ds003768, NIMH/Bolt) ships a T1w and none ships diffusion, so a
   subject-specific *tractography* connectome is not available; the options
   are per-subject geometric eigenmodes from the FreeSurfer cortical surface
   (T1w suffices; the pinned FreeSurfer container on this Mac can run it) or a
   group template connectome parcellated to the ROI set. Aqil's own repository
   ships no connectome, only per-subject fitted parameters from HCP DTI.

Longer topology confusion matrices were reported to take about 75 minutes with
three channels and 6.5 hours with BOLD+HR on this laptop. The new initialization
probe is much smaller (seconds per fit; several minutes for 30 cells).
