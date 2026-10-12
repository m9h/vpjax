# Periodic perturbation: design heuristic and numerical checks

## Where this came from

Brainclinics released *Heart-Brain Connect Live*, which computes heart-brain
coupling (HBC) in real time during any rTMS protocol with an on/off duty cycle,
from a Polar H10 chest strap. The underlying method is in Dijkstra, Frandsen,
van Dijk, Duecker, Taylor, Sack, Arns & Siddiqi, *Nature Mental Health* 2(7),
809-817 (2024), "Probing prefrontal-sgACC connectivity using TMS-induced
heart-brain coupling", DOI `10.1038/s44220-024-00248-8`, building on the
neuro-cardiac-guided TMS line from Iseger et al.

The clinical claim is not what matters here. The **estimator design** is, and
it transfers to this protocol whether or not any TMS is ever involved.

## The design, and why it is the right shape

The 10 Hz "Dash" protocol is 5 s of stimulation followed by an 11 s intertrain
interval: a 16 s cycle, so **1/16 = 0.0625 Hz**. Stimulation slows the heart,
the interval lets it recover, and the interbeat-interval series therefore
carries a periodic component at exactly the cycle frequency. HBC is defined as
the oscillatory power of the interbeat interval at 0.0625 Hz, swept across
15 intensities in 2% MSO steps up to 120% RMT.

So rather than asking "did heart rate change after the perturbation", which is
a question about a slow transient competing with drift, they ask "how much
power is there at the one frequency we built into the input". That is a
single-frequency transfer-function measurement of the brain-to-body arm, and it
is the estimator this document argues the perturbation arms should use.

Three reasons it fits the problem in `docs/brain_body_system_models.md`:

1. **It targets a chosen frequency band.** A block or ramp
   perturbation concentrates its energy near DC, which is exactly where the
   slow envelope of interest sits, and where scanner drift and slow
   physiological wander sit too. 0.0625 Hz is above the slow-envelope band and
   well below the respiratory band at roughly 0.2-0.3 Hz, so it may reduce overlap with
   both; spectral leakage, drift and other physiological components remain.
2. **TR 2.0 s samples it properly.** Nyquist is 0.25 Hz and the cycle gives
   8 volumes per period, so the component is not aliased - unlike the breathing
   band, which at this TR is (see `docs/respiration_two_routes.md`).
3. **The input time course is known exactly,** because the experimenter wrote
   it. In the drug arm the input shape is an assumption with a cited source; in
   a perturbation arm it is a fact about timing. The magnitude and neural/peripheral mapping still need
   calibration; knowing the schedule does not by itself pin the coupling gain.

## What it is evidence for, and what it is not

The literature summary below is inherited from the prior draft and is not
an independent source audit in this numerical investigation.

**It is evidence that a periodic cortical perturbation produces a
phase-locked cardiac response that is anatomically specific.** Maximal HBC
power fell on a negative-sgACC frontal target in 12 of 14 participants, 86%
against a 40% chance level, `P < 0.001`; frontal HBC power was greater at
negative-sgACC than at neutral-sgACC sites, `t(13) = 2.327, P = 0.037,
d = 0.63`. Site selectivity is the part that matters for the present purpose,
because target dependence supports a brain-mediated component. It does not
by itself exclude sensory, pain or autonomic pathways. Normalized pain did not predict
HBC power (all `P >= 0.076`), nor did it identify negative-sgACC sites above
chance (42% frontal).

**It does not supply a prior on `g_bb`.** HBC power is reported min-max
normalized to [0, 1] across sites and intensities, so the paper carries no
absolute magnitude, and the between-target effect size `d = 0.63` is a
*site-discrimination* effect rather than the strength of the brain-to-body
coupling. Nothing here can be converted into a generating value, and nothing
should be.

**It does not import TMS into this protocol.** rTMS concurrent with HD-EEG-fMRI
is a different and much harder proposition than rTMS in a chair with a chest
strap, and the design does not need it: what is needed is a *brain-first
perturbation with an externally timed periodic schedule*, which a blocked task
or stimulus at a 16 s cycle supplies in-scanner. The estimator is the
transferable part; the actuator is a separate decision.

**Its clinical validity is contested and irrelevant here.** A University of
Iowa report found during-stimulation heart-rate deceleration did not predict
depressive symptom change. Whether HBC predicts treatment response is a
different claim from whether a periodic cortical perturbation moves the heart
measurably, and only the latter is load-bearing for this design.

**Respiration is not addressed in the paper.** 0.0625 Hz is outside the
respiratory band, so respiratory sinus arrhythmia does not sit on the estimate
directly, but slow breathing-rate drift contributes broadband interbeat-interval
power and paced or recorded respiration would tighten it. For this protocol
respiration is being recorded anyway, and
`docs/respiration_two_routes.md` already requires it on the imaging clock.

## Consequence for the two drug-free arms

The periodic design motivates testing both arms, but the simulation probes
support different conclusions for them. It does not justify duty-cycling both
arms at 16 s by default.

The historical probes are now saved in `scripts/brain_body_arm_probe.py`,
`brain_body_period_sweep.py`, and `brain_body_dcgain.py`. Their original aggregate
outputs are preserved in `docs/results/brain_body_*_legacy.txt`. They did not
record optimizer convergence or Hessian positive-definiteness, so these are
exploratory results, not validated precision estimates. The comparator called
"block" in the first probe was actually a Bateman-shaped transient.

- Brain-first (`g_bb`): at 480 s, the 16 s duty cycle barely changes median
  log SE (0.359 to 0.348). No useful advantage over the transient is established.
- Body-first (`g_bp`): at 480 s, median log SE falls from 0.409 to 0.200.
  This favors the duty cycle under these simulation settings.
- At 480 s, brain-first log SE has a sampled minimum at 32–64 s (0.242,
  0.239); body-first favors the shortest tested period, 8 s (0.184).

The receiving-state filter provides a heuristic, not a universal optimum:
`1/sqrt(1+(2*pi*tau/period)^2)` has its corner at `period=2*pi*tau`.
For `tau_a=10 s` this is about 63 s; for `tau_z=2 s`, about 13 s.
The best finite-duration design also depends on cycle count, noise, nuisance
parameters and sampling. Moreover, `simulate` rescales observation noise with
each design's own clean peak-to-peak excursion; these comparisons hold SNR,
not absolute instrument noise, fixed. Selecting distinct frequencies does not
alone demonstrate separation in finite records or account for harmonics.

### The SE discrepancy and the varied-start test

The claimed 479% curvature SE was a units error. `parameter_uncertainty`
computes the Hessian in log parameters and returns **log SE**, approximately
relative SE. At 64 s the historical value was 0.2394 (about 24%), not
`0.2394/0.05 = 4.79` (479%). This is comparable to the five-seed empirical
CV of 0.240. At 32 s, log SE 0.242 and empirical CV 0.203 are likewise of
similar size. Five seeds do not establish calibration.

The product `g_bb*tau_a` had smaller empirical CV (0.094–0.110), but this
does not establish that only the product is determined. The initialization
check varies `g_bb` by factors 0.25, 1 and 4 while varying `tau_a` inversely,
keeping their starting product fixed. Other parameters start at half truth.
All six fitted parameters remain free; no product constraint or prior is added.
Use `scripts/brain_body_initialization.py`, which checkpoints full per-fit
results, convergence, Hessian diagnostics and correlations in JSON. Its
product log SE uses the covariance, conditional on a positive-definite Hessian.

Step 2 sizing remains open until these diagnostics are assessed. The corners
above are candidates for a design sweep, not established minutes per arm.

### Varied-start result (2026-10-12)

Canonical results are in `docs/results/brain_body_initialization.json`: 30
float64 fits, five seeds, two periods and three starts per recording. A 16-fold
span in starting gain returns to effectively identical fitted values for each
recording: maximum within-seed relative ranges are 0.00775%/0.01044% for `g_bb`
and 0.00671%/0.01149% for `tau_a` at 32/64 s. Maximum likelihood ranges are
less than 1.7e-7. Thus the tested initializations do not explain the across-seed
spread; this is not a proof of global identifiability beyond these starts.

| Period | Gain empirical CV | Median gain log SE | Product empirical CV | Median product log SE, PD cells only |
|---|---:|---:|---:|---:|
| 32 s | 20.3% | 0.228 | 9.4% | 0.239 |
| 64 s | 24.0% | 0.239 | 11.0% | 0.204 |

The product has less empirical spread in this small sample, but its
covariance-based uncertainty does not establish uniformly better precision.
The reported gain log SE medians include diagnostic pseudo-inverse values;
they must not be read as a calibrated population precision claim.

28/30 fits report convergence. Both exceptions are seed 4 at 32 s; their
endpoints match the third, converged start. A 1000-step retry still reports
nonconvergence and is saved separately. At 64 s, seed 4 drives `q_z` toward
zero (~4.6e-16). One start gives a tiny negative information eigenvalue
(~-1.6e-14); the other starts are also at this effectively singular nuisance
boundary. Ordinary joint asymptotic SE interpretation is unreliable there.
The product SE is omitted for the non-PD fit rather than given a false finite
precision. These caveats remain visible in the JSON and handoff.

The initialization-dependence concern is substantially resolved for this
simulation. Duration sizing still requires a precision criterion, effect-size
sweep, fixed absolute measurement-noise sensitivity and boundary-aware
uncertainty checks. No minutes-per-arm recommendation follows from this run.
