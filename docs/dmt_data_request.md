# Data request — Timmermann et al. (2023) DMT EEG-fMRI

Draft message to cbt14@ic.ac.uk, the contact named in the
`timmer500/DMT_Imaging` README. The ask is deliberately narrow: two
derived files that the release's own pipeline already produced, not the
raw recordings.

---

**Subject:** DMT_Imaging — request for unfiltered parcel series and pre-convolution IRASA regressors

Dear Dr Timmermann,

I am advising on the neuroimaging protocol for a placebo-controlled
simultaneous EEG-fMRI study of 5-MeO-DMT in Robin Carhart-Harris' lab at
UCSF, with recruitment planned for early 2027. We are using your 2023
EEG-fMRI release to test an analysis method before the protocol is
fixed, and I am grateful it is public — it is the only dataset we have
found that pairs a serotonergic psychedelic with simultaneous EEG.

The method fits a neurovascular state-space model (a drive process
feeding a Balloon model) jointly to the BOLD series and an EEG band
envelope, and then asks whether the drug's effect on the drive is
*identifiable* — whether the data constrain it at all — from the
curvature of the likelihood. The point of the exercise is to decide, in
advance, which acquisition choices make the drug effect estimable.

Two properties of the release stand in the way, and both are, I think,
one step back in your own pipeline rather than anything new:

1. **The parcellated BOLD series without the band-pass filter.** The
   released series carry no power above 0.08 Hz — measured on the global
   mean, 0.00% above 0.08 Hz and 1.3% below 0.01 Hz, against 2.5% and
   31% for an unfiltered run at the same TR. Our method predicts each
   volume from the previous one and treats the remainder as white noise,
   so a filtered series cannot be whitened under any parameters: fitting
   14 subject-conditions gives innovation autocorrelations of +0.50 to
   +0.61 (they should be near zero) while the time constants converge on
   the filter's corner rather than on the haemodynamics. Modelling the
   filter is not a way around it, since inverting an operator that has
   removed essentially all the high-frequency content is ill-posed.
   The parcellated series *before* the 0.01–0.08 Hz step — however
   minimally cleaned otherwise — would resolve this entirely.

2. **The IRASA band-power regressors before HRF convolution.** The
   released `RegressorsInterpscrubbedIRASA_*.mat` series appear to have
   been passed through `spm_Volterra` with an SPM basis set. Our model
   applies its own haemodynamic response through the Balloon, so a
   pre-convolved input is counted twice and biases exactly the transit
   time and signal decay we are trying to estimate. Usefully, the
   *sampling rate* is not a problem at all: in simulation an envelope
   sampled once per 2 s volume recovers the drug effect's amplitude to
   8% where fMRI alone leaves it at 44% and formally unidentifiable. So
   the interpolated, scrubbed IRASA power per region before that
   convolution step is all we would need.

3. **The mapping between the 14 EEG subjects and the 20 fMRI subjects.**
   The two folders have different subject counts and the README does not
   give a key, so we can currently only pair the modalities at the group
   level.

If any of these is awkward to extract, we would be glad of whatever is
easiest — even a single subject's unfiltered parcel series and
pre-convolution regressors would let us verify our loader against the
released versions, which is the thing we most need to be sure of.

We would of course cite the 2023 paper and the release, and we are happy
to share our code (it is open, in JAX) and anything we learn about the
method's behaviour on your data — including, candidly, if it turns out
the effect is not identifiable from these series, which is a result
worth knowing before anyone designs another study around it.

With thanks and best wishes,

Morgan Hough
morgan.hough@gmail.com

---

## Notes for us, not for the email

- Do **not** ask for raw EEG or unprocessed fMRI images in the first
  message. Each ask is one step back in their existing pipeline, which is
  far more likely to be granted than a re-release.
- The band-pass is the blocking item, not the convolution: without
  broadband series there is no usable arm at all, not even BOLD-only.
  Evidence is in `docs/results/` and `vpjax/validation/dmt_imaging.py`.
- Sampling rate is explicitly *not* part of the ask: TR-rate is enough
  (`scripts/simultaneity_case.py --eeg-dt 2.0`).
- The Zenodo `.bdf` release (zenodo.org/records/3992359) is the inhaled
  DMT EEG-only study, not this experiment: no fMRI, so it cannot
  exercise the joint model. It is still a reasonable artifact-free
  reference for benchmarking in-bore EEG cleaning.
- Motion parameters are also offered in the README; worth a follow-up
  once we know whether we need censoring, but not in the first message.
