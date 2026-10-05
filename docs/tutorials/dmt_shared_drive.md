# Joint autonomic observations: first inference example

Use neurojax's `examples/dmt_observations.py` to export baseline-standardized
observations from [D'Amelio's DMT release](https://doi.org/10.5281/zenodo.19893951).
vpjax consumes that file without a neurojax dependency. The dataset has no fMRI.

The initial model is deliberately reduced:

`y_k(t) = loading_k * x(t) + measurement_noise_k`

`x(t + dt) = x(t) + process_noise`, with variance `process_sd² * dt`.

It performs scalar Kalman filtering and Rauch–Tung–Striebel smoothing on the
union of sensor timestamps and the requested output grid. Missing observations
are skipped, with increased conditional uncertainty. No interpolation is used
to fabricate observations. Selected sensors require fixed signed loadings and
explicit positive noise SDs. The prior mean is zero, with variance `prior_sd²`.

Example **illustrative assumptions**, to be assessed on held-out data:

```json
{
  "loadings": {"heart_rate": 1.0, "smna": 1.0, "rvt": 1.0},
  "noise_sd": {"heart_rate": 1.0, "smna": 1.0, "rvt": 1.0},
  "process_sd": 0.1,
  "prior_sd": 1.0
}
```

Save to `assumptions.json`, then from the vpjax checkout:

```bash
PYTHONPATH=. python examples/dmt_shared_drive.py observations.npz assumptions.json posterior.npz
```

The posterior includes `time_s`, `mean`, `sd`, and JSON metadata containing
the assumptions and source observation provenance. Numerical uncertainty is
conditional on the assumed loadings/noise; it excludes parameter, preprocessing
and model uncertainty. The latent variable is an uncalibrated shared drive,
not an identified sympathetic, vagal or receptor-occupancy state. Positive
RVT loading is a hypothesis, not a physiological requirement. Compare alternate
signs/loadings and sensor subsets on held-out sessions. Do not count overlapping
samples as independent participants.

EEG can be included as another baseline-standardized scalar observation with
an explicitly chosen signed loading and noise. Inclusion asserts a shared-drive
relationship and does not establish a causal EEG–autonomic mechanism.

This example does not yet implement sensor response delays, respiratory gas
exchange, a coupled pressure-generating circulation, Diamond/SimCVR, drug PK/PD,
or neural-to-BOLD coupling. Those are subsequent model extensions. It leaves
the existing baroreflex, vagal and NVC solvers unchanged. A future physiological
model can consume the same timestamped observations and compare its held-out
predictions with this reduced baseline and the authors' PCA arousal index.
