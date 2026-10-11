# A drug-free arousal arm, and why the protocol needs one

## The problem this is for

5-MeO-DMT produces a large autonomic response alongside whatever it does
to cortex: heart rate, blood pressure and breathing all move, and they
move on the same time scale as the subjective effect.  Every BOLD change
in the acute window is therefore confounded, and the objection is not
pedantic -- it is the standard reviewer's objection to psychedelic fMRI,
and it is correct as stated: a drug that only changed the subject's
circulation would produce acute BOLD changes in exactly the windows
where a drug that changed neural activity would.

`scripts/noise_confound_case.py` answered one version of this and the
answer was clean: a drug that only inflates *measurement* noise cannot
be mistaken for a neural effect, because noise is white while the drive
reaches the BOLD through the Balloon's low-pass, so the confound deposits
its power where no drive can reach.  That argument does not transfer to
a real autonomic response.  A heart-rate or CO2 change moves perfusion
*through the vasculature*, which is to say through the same low-pass the
neural drive uses, smoothly, on the drug's own time scale.  Nothing about
the filter separates the two.  `scripts/autonomic_confound_case.py`
measures what does.

## What the model says

The two routes are both in `augmented_drift`:

- `beta` -- the input drives the neural state `z`, and the vasculature
  follows.  Anything that observes `z` must see this.
- `beta_vasc` -- the input drives the flow-inducing signal `s` directly
  and leaves `z` alone.  The BOLD moves; the EEG does not.

From BOLD alone those are very nearly the same model, which is the
confound in one line.  A channel that observes the drive breaks the
degeneracy, because only one of the two routes predicts a change in it.
That makes simultaneous EEG not a precision luxury in a drug study but
the thing that makes "the drug changed neural activity" separable from
"the drug changed the subject's circulation" at all.

## The arm

Add a drug-free sustained-arousal run to the **placebo** session: same
subject, same scanner, same cap, same physiological recording, no drug.
It buys three things.

1. **A within-subject estimate of the vascular route.**  The arousal run
   has a known stimulus time course and a measurable autonomic response
   (HR, RVT, GSR, pupil), with no pharmacology.  Fit it and `beta_vasc`,
   `tau_in` and the arousal drive's time constant are identified by data
   in which they are the only thing happening.  Carry them into the drug
   run as subject-specific priors, where they are otherwise poorly
   identified because the drug moves everything at once.
2. **An empirical negative control.**  A large autonomic perturbation
   with no drug on board should return `beta` (the neural route's gain on
   a drug-shaped input) at zero.  That is the real version of the
   `pinned`/`modelled` null rows in the noise-confound table, with a
   structured physiological perturbation instead of injected noise.  If
   it does not come back at zero, the drug analysis is not trustworthy
   and we find that out in the placebo session rather than in the paper.
3. **A second drive that the data already demanded.**  On NATVIEW,
   pupil and the alpha envelope could not be served by one latent state:
   fitted jointly, the envelope keeps the drive and pupil's loading goes
   to zero.  A two-drive model (neural + arousal) is the structural
   answer, and an arousal arm is the condition that identifies the second
   drive instead of leaving it to be inferred from resting fluctuations.

### Why not breath-hold or a CO2 challenge

Those give a vascular reactivity scalar and they are worth having, but
they are short, attentionally demanding, and emotionally nothing like a
drug.  What makes an autonomic response confusable with a drug effect is
the *temporal envelope* -- rise, plateau, resolution over tens of
minutes.  A manipulation matched on that envelope is the one that tests
the discrimination; a 20-second breath-hold tests a different question.

## An off-the-shelf induction

Allen (2026) characterises a 13.5-minute first-person film of people
undertaking dangerous activities at height as a sustained anxiety
induction, and the validation is unusually complete for a stimulus one
might otherwise have to build:

- Group-mean continuous arousal and valence trajectories reproduce
  across independent samples at *r* = .99 and .94 (confirmatory
  *N* = 124, exploratory *n* = 21), so the time course is a known
  quantity rather than something each study re-estimates.
- Somatic anxiety rises substantially after viewing (STICSA somatic,
  *d*<sub>z</sub> = 1.03), with a racing heart and tension the commonest
  reported symptoms -- i.e. the response is autonomic, which is what the
  arm is for.
- The envelope has the right shape: arousal rises, holds through a
  sustained phase, then declines, with preregistered segment edges found
  by piecewise-linear segmentation with a participant bootstrap.
- The rating tool is open (`embodied-computation-group/online-affect-rating`,
  jsPsych + JATOS, 1-D or 2-D continuous affect), as are the 1 Hz
  per-participant rating series, the analysis code and the
  preregistration.
- The group reports having already scanned ~300 people watching this
  film with GSR and heart rate recorded; brain-body results are not out
  yet.

References: preprint <https://doi.org/10.31234/osf.io/szv9e_v1>; data and
code <https://doi.org/10.5281/zenodo.23151387>; preregistration
<https://osf.io/s2yvb>.

### Caveats before adopting it

- **The film is not an open stimulus.**  It is a copyrighted YouTube
  video, streamed to participants from the study server; the repository's
  own `docs/stimulus_source.md` says permission has not been obtained
  even for a still in the paper.  We would need permission or our own
  stimulus.  The ratings, timings, segmentation method and task are open
  and reusable either way, and the task works with any video.
- **The content depicts an unauthorised rooftop stunt that drew legal
  action.**  An ethics board may reasonably object to the stimulus even
  where it accepts the manipulation.
- **Not in the drug session.**  Inducing anxiety in a 5-MeO session is
  its own risk and is not what this is for.  The arm belongs in the
  placebo session or a separate baseline visit, which makes session
  order and its carry-over part of the design rather than a detail.
- **Fear of heights is a trait moderator** -- the validation study
  collected it on a 1-7 item and found larger somatic increases went with
  higher arousal and more unpleasant valence ratings.  Screen or covary.
- **STICSA is self-report.**  The physiological half of the claim rests
  on their unpublished scanner data, not on the preprint.

### What to ask for

The scanned cohort (film + BOLD + GSR + heart rate, ~300 subjects) is a
better-matched calibration set for the vascular route than anything
public: a large, temporally shaped autonomic perturbation with no drug,
in a group that preregisters and shares.  Worth asking for alongside the
DMT request in `docs/dmt_data_request.md`.  Two things would be enough:
the parcellated BOLD **before** any band-pass, and the concurrent
GSR/HR/respiration on the same clock.  The band-pass is the item that
killed the Timmermann release for this purpose, so it is worth naming
first and early, while their analysis is still in progress.
