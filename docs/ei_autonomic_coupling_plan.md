# Coupling an E/I layer to the autonomic layer: what exists, and one trap

This is the design note for connecting a neural excitation/inhibition layer to
the existing vascular and autonomic machinery. It records what the two
repositories already provide, one identifiability trap that would silently
void the E/I test, and the order the pieces should be built in.

Companion notes: `docs/respiration_two_routes.md` (why respiration needs a
neural route) and `docs/ei_spectral_observable.md` (why a two-band signature is
falsifiable where an amplitude change is not).

## The trap: a free signed loading destroys the E/I test

The recorded-data model gives every fast channel its own free gain on the shared
drive (`vpjax/validation/eeg_fmri_statespace.py:284`):

```python
y = gains[c] * x[Z] + (x[nidx[c]] if nuisance else 0.0)
```

with `gains[c] = signs[c] * get(f"gain_{c}")` at `:255`. That convention is
right for its purpose: on real data you do not know in advance whether a pupil
trace or an alpha envelope rises or falls with the drive, so you let the data
choose and you fit a positive magnitude.

It is exactly wrong for the E/I test. Suppose theta and gamma band power enter as
two channels, each with its own free signed gain. Then:

- the E/I hypothesis predicts theta down and gamma up from one parameter change;
- the global-excitability hypothesis predicts both bands moving together;
- but with two free signed gains, the global model **can fit the opposite-sign
  pattern exactly**.

The two hypotheses collapse to the same model, the route dLL is identically
zero, and the test reports nothing while appearing to run. The sign pattern is
the entire evidence, so it cannot also be a free parameter.

So the E/I observation model needs the two band loadings **tied to one scalar
with a fixed relative sign and magnitude**, and that loading vector has to be
*derived* rather than assumed — otherwise the test only confirms the assumption
that produced it. Deriving it from a neural mass simulation is the whole reason
to involve a circuit model at all, rather than writing a two-band regression.

## What already exists

**vpjax has the two injection points and the lag.** In
`vpjax/validation/statespace_recovery.py:146-155` the drift rows are

```python
-z / tau_z + beta * u,
z - kappa * s - gamma * (f - 1.0) + beta_vasc * u,
```

so `beta` is the neural route and `beta_vasc` the vascular one, deliberately
placed where the neural drive enters so both pass the identical Balloon
low-pass. On recorded data, a measured input enters through a first-order lag,
`dr/dt = (u - r)/tau_in` then `ds/dt += beta * r`
(`eeg_fmri_statespace.py:261-276`), with the sign fixed per input and a positive
magnitude fitted.

**But measured inputs currently have only the vascular route.** At
`eeg_fmri_statespace.py:266` the input is added to row 1 alone:

```python
dx = dx.at[1].add(vascular)
```

and the module docstring (`:33-41`) names breathing depth as *the* canonical
vascular input, acting "on the vasculature rather than reporting on the neural
drive". That is precisely the assumption the isoelectric knockout contradicts.
Giving a measured input a second gain on row 0 is a local change to this
function plus the parameter list, and it is the first thing to build.

**vpjax has no E/I layer.** Nothing in the package carries separate excitatory
and inhibitory states; the only "inhibition" is a scalar cortical-excitability
gain in `vpjax/cardiac/baroreceptor.py:108-142`. The `pharmacokinetics/` layer
was however built to feed one: `occupancy.py:159` `regional_drive` and `:184`
`competitive_drive` exist to hand a drive to a neural mass, and the header at
`occupancy.py:22` says so.

**vpjax has no respiration generative model** — only respiration as a *measured*
series (`validation/natview.py` `load_respiration`) and end-tidal CO2 as a
measured stimulus (`hemodynamics/cvr.py:134`). Given the conclusion that
respiration should be a measured two-route calibration input rather than a
simulated nuisance, that absence is the right state: nothing needs generating.

**vbjax has two models with an independent inhibitory knob**, both in
`vbjax/vbjax/neural_mass.py`:

- **Jansen-Rit** (`:5-31`): six states with `y1` the excitatory and `y2` the
  inhibitory post-synaptic potential. Inhibitory amplitude `B` and rate `b`
  enter only the `y5` row (`:30`); `a_3` scales E→I and `a_4` scales I→E. The
  smallest possible "E/I balance as one scalar" is a wrapper that scales `B`.
- **Dopa** (`:146-205`): one population but with genuinely separate excitatory
  and inhibitory synaptic states `Sa`/`Sg`, separate conductances `ga`/`gg`, and
  network coupling through separate connectomes `Ce`/`Ci` with weights
  `we`/`wi`.

Wilson-Cowan, Wong-Wang and the full Epileptor are **not** in vbjax; only a 2-D
Epileptor reduction. Jansen-Rit is the right choice here: it has explicit E and
I potentials, and it is the model whose spectral response to `B` is already
exercised by the PSD-fitting examples.

## Gotchas found in the survey, worth writing down

- `vbjax.monitor.make_fft` (`monitor.py:98`) and `compute_sarvas_gain`
  (`monitor.py:31`) are **stubs containing `pass`**. The working differentiable
  PSD prior art is example code: `vbjax/examples/jr-psd-fit-sbi.py:24-60`.
- `make_ode`'s step **discards `t`** (`vbjax/loops.py:221`), so a time-varying
  parameter cannot be driven through it. `make_sdde` passes an integer `t`
  (`:331`) and is the route — but its zero-delay fast path hard-codes `t=0`
  (`:299`), so `nh >= 1` is required.
- `vbjax/tests/test_uloop.py` is a design sketch with no tests and no exports.
- `BaroreflexParams/State/ODE` and `sympathetic_fraction` are **not** in
  `vpjax/cardiac/__init__.py`'s `__all__`; import from
  `vpjax.cardiac.baroreceptor` directly.
- In both `statespace_recovery` and `eeg_fmri_statespace` the drift and
  observation are rebuilt per `theta` inside a `build(theta)` closure, so any
  new parameter must be threaded through `_FIT_NAMES` / `param_names` as well.

## The architectural decision (taken 2026-10-13: option 4)

**Decision: the latent E/I block is the linearised graph Wilson-Cowan field of
gnf-jax (`m9h/gnf-jax`, MIT, a JAX port of Aqil's model), wrapped in
`vpjax/neural/ei_block.py` and installed through the `ei` extra. Option 3 is
kept only for deriving the two-band loading vector offline; options 1 and 2 are
dropped.** The reasoning: each Laplacian eigenmode of the linearised field is a
two-state Ornstein-Uhlenbeck block, so it drops into the local-linearisation
filter as an exact linear-Gaussian latent with gradients through every
parameter and no neural-mass ODE code in vpjax; the E and I spectra are
closed-form functions of the same fitted parameters, so the EEG band observable
is derived rather than a free sign (`band_power` in the adapter); and an
external time course enters at a named entry point (`p`, `q` or a coupling
gain), linearised about the steady state, which is the framing constraint
satisfied. Two limits, which are why option 3 survives: with a single region
(`eigs = [0.]`) only the constant mode remains, so the field's spatial structure
is absent until a graph is supplied; and a linearised two-state oscillator has
one resonance per mode and cannot by itself produce the opposite-sign
theta/gamma pattern of `ei_spectral_observable.md`. The two-band prediction
therefore still comes from a circuit run (Jansen-Rit or Neymotin-type), and the
block supplies the latent dynamics and a resonance/instability readout.

The options as they stood before the decision, kept for the record:


vpjax does not depend on vbjax: zero imports, and vbjax is absent from
`pyproject.toml`. The styles differ — vbjax uses `namedtuple` parameters with
`jax.lax.scan` Heun loops and caller-supplied noise; vpjax uses `equinox.Module`
with Diffrax vector fields `(t, y, args)`.

Three options, and this is a decision rather than a detail:

1. **Add vbjax as an optional dependency** and use Jansen-Rit directly. Least
   duplicated code, and the E/I model is one other people already cite. Costs a
   dependency and a style seam.
2. **Reimplement Jansen-Rit natively** in vpjax's Equinox/Diffrax style. Six
   states and four connectivity constants, so perhaps forty lines. No
   dependency, consistent style, but it is a second copy of a standard model
   that must be validated against the original.
3. **Use vbjax offline, only to derive the loading vector.** Run Jansen-Rit
   once, outside the fitting path, to get the two-band sign pattern and relative
   magnitude; commit that vector with its provenance; the state-space model then
   carries no neural-mass code at all. Smallest footprint, and it keeps the
   fitted model linear — but the loading becomes a fixed input rather than
   something the fit can refine.

Option 3 is the cheapest route to the result that matters, since the deliverable
is the loading vector and the sizing curve rather than a joint circuit fit.
Option 1 is the right end state if the E/I parameters are ever to be fitted
jointly with the haemodynamics.

4. **gnf-jax as the latent block** (chosen; see above). `vpjax.neural.linear_ei_block`
   linearises on any set of Laplacian eigenvalues, `statespace_spec` returns the
   filter's `f`/`Q`/`m0`/`P0`, `EIBlock.input_vector(entry)` is the linearised
   gain of an external time course at a named entry point, and
   `band_power` is the derived band observable. Tests in
   `vpjax/tests/test_ei_block.py` check the Lyapunov identity, that the spectrum
   integrates to the stationary variance, that a uniform input enters only the
   constant mode, that the band power is differentiable in the inhibitory gain,
   and that the filter's likelihood peaks at the generating time constant.
   Not yet done: composing the block with the Balloon in
   `eeg_fmri_statespace.py` (replace `z` by a loading over the E modes), and
   supplying a graph for the datasets on disk (see the connectome note in the
   handoff).

## Build order

1. **Give a measured input a neural route.** Add a second gain on row 0 in
   `eeg_fmri_statespace.py`, so measured respiration has both `beta` and
   `beta_neural`. Fit both against the nested single-route models. This is
   self-contained, needs no neural mass, and directly tests the
   `respiration_two_routes` claim on recorded data.
2. **Derive the two-band loading** from Jansen-Rit by sweeping the inhibitory
   amplitude and measuring theta and gamma band power. Output: a sign pattern
   and relative magnitude, with the sweep committed so the provenance is
   visible.
3. **Add a constrained two-band observation** to the simulation model, loading
   tied to one scalar by the derived vector.
4. **Three-way route comparison** — neural-E/I, neural-global, vascular — and an
   SNR sweep to find the recording requirement, in the shape of the existing
   sweep in `scripts/autonomic_confound_case.py`.

Step 1 is worth doing first and on its own: it is the step that could change the
interpretation of data already in hand.

No compound-specific parameters belong in any of this. As with
`drug_shaped_input`, an E/I change is a *shape* applied to a model parameter,
with magnitude and time course supplied externally and carrying a cited source.
