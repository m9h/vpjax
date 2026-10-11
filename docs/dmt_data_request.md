# Data request — Timmermann et al. (2023) DMT EEG-fMRI

Draft message to cbt14@ic.ac.uk, the contact named in the
`timmer500/DMT_Imaging` README. The ask is deliberately narrow: two
derived files that the release's own pipeline already produced, not the
raw recordings.

---

**Subject:** DMT_Imaging — request for pre-convolution IRASA regressors and the EEG/fMRI subject key

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

Two small things from the release would let us do this properly:

1. **The IRASA band-power regressors before HRF convolution.** The
   released `RegressorsInterpscrubbedIRASA_*.mat` series appear to have
   been passed through `spm_Volterra` with an SPM basis set. Our model
   applies its own hemodynamic response through the Balloon, so a
   pre-convolved input is counted twice and biases exactly the transit
   time and signal decay we are trying to estimate. The interpolated,
   scrubbed IRASA power per region *before* that convolution step —
   whatever time base it is on — would be ideal.

2. **The mapping between the 14 EEG subjects and the 20 fMRI subjects.**
   The two folders have different subject counts and the README does not
   give a key, so we can currently only pair the modalities at the group
   level.

If either is awkward to extract, we would also be glad of whatever is
easiest — even one subject's pre-convolution regressors would let us
check that our loader reproduces the convolved series, which is the main
thing we need to be sure of.

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

- Do **not** ask for raw EEG or unprocessed fMRI in the first message.
  The narrow ask is far more likely to be granted, and the
  pre-convolution regressors are sufficient for the fusion arm.
- The Zenodo `.bdf` release (zenodo.org/records/3992359) is the inhaled
  DMT EEG-only study, not this experiment: no fMRI, so it cannot
  exercise the joint model. It is still a reasonable artifact-free
  reference for benchmarking in-bore EEG cleaning.
- Motion parameters are also offered in the README; worth a follow-up
  once we know whether we need censoring, but not in the first message.
