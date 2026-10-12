# Respiration needs a neural route, not just a vascular one

The autonomic confound work in `scripts/autonomic_confound_case.py` rests on one
assumption: that the EEG is a clean arbiter between the two routes into the BOLD.
A drive that enters the neural state `z` is seen by the EEG; a drive that enters
the flow-inducing signal `s` is not. That asymmetry is what makes the route
comparison decisive once an EEG channel is present (route dLL +84.76 for neural
data against -995.56 for vascular, in `docs/results/autonomic_confound_grid.json`).

Respiration breaks the assumption, and the evidence that it does is a knockout
experiment rather than a correlational claim. This note records what the
literature establishes, what it means for the model, and why the conclusion is
less damaging than it first appears.

## The evidence

**Respiration drives neural activity, and the resulting BOLD network is
neurally mediated.** Tu and Zhang recorded electrophysiology, resting-state
fMRI and respiration simultaneously in rats
([10.7554/eLife.81555](https://doi.org/10.7554/eLife.81555)). Slow respiration
variations phase-lock to gamma-band power in anterior cingulate cortex, and
those same slow variations are linked to a characteristic resting-state fMRI
network that is mediated by the gamma-band activity. The decisive step is the
control: the respiration-related network **disappears when brain-wide neural
activity is silenced at an isoelectric state while respiration is maintained**.
A vascular or CO2-driven network would have survived that manipulation, because
the breathing continued. It did not survive, so the route is neural.

**The effect is whole-brain and spans every band, in humans.** Kluger and Gross
measured phase-amplitude coupling between respiration and resting MEG from 2 to
150 Hz ([10.1371/journal.pbio.3001457](https://doi.org/10.1371/journal.pbio.3001457)).
Respiration modulates oscillatory amplitude across all major frequency bands,
with distributed cortical and subcortical sources that map onto canonical
resting-state networks.

**It is visible on the scalp, but not uniformly across subjects.** Watanabe and
colleagues found significant coherence between high-density scalp EEG and
respiration in healthy subjects
([10.1016/j.neulet.2023.137079](https://doi.org/10.1016/j.neulet.2023.137079)),
but the number and location of electrodes showing coherence differed between
subjects, and they leave open whether the signal is volume-conducted from
olfactory regions or reflects local cortical activity. For a study whose EEG is
the arbiter, that between-subject variability is itself a design parameter: the
route gains have to be estimated per subject, not assumed.

## What it means for the model

Respiration is not assignable to `beta_vasc`. It has a genuine `beta` path as
well, so a respiration-locked EEG change is not by itself evidence of a drug
effect on the neural state. In the current two-route model respiration is a
*correlated two-route input*, which is a harder confound than either route
alone.

The saving grace is timescale, and it splits the problem cleanly in two.

**Respiration-phase effects are fast and rhythmic.** They sit at the breathing
frequency and its harmonics, around 0.2-0.3 Hz. A drug envelope is slow and
monotone - an onset, a rise and a decay over many minutes. These do not overlap
in frequency, so the fast component is separable by filtering and poses no
threat to an envelope estimate, provided the sampling is fast enough to resolve
it rather than alias it. At TR 2.0 s the BOLD Nyquist is 0.25 Hz, which is
inside the breathing band: this is an argument for a faster TR or for explicit
respiration regression, not something to leave to chance.

**Slow drift in breathing depth and rate is the real threat.** Respiratory
volume per time wanders over a session, and that drift is slow, monotone and
directly confusable with a drug envelope. It is also precisely the component Tu
and Zhang showed to have a neural route: their effect was on *slow* respiration
variations, not on respiratory phase. So the dangerous part of respiration is
the part with two routes.

## The design consequence, which is a gain rather than a loss

Because respiration can be *measured* - a belt for the envelope, capnography for
end-tidal CO2 - it does not have to be an unknown confound. It can enter the
model as a known input with both a `beta` and a `beta_vasc` path, and both gains
are then estimable from the data.

That turns respiration into a free, continuous, two-route calibration probe that
is present in every run, including placebo and including baseline. It measures
exactly the quantity the model needs and cannot otherwise obtain: how strongly a
known autonomic input reaches the BOLD through each route *in this subject, in
this session, on this scanner*. The drug envelope's own route gains are then
estimated against a within-session reference rather than against a prior.

This complements the sustained-arousal arm in `docs/anxiety_induction_arm.md`
rather than competing with it. The film gives one large, controlled, scheduled
arousal excursion. Respiration gives a continuous small-amplitude excursion in
every run. The film calibrates the large-signal regime; respiration calibrates
the operating point and does so repeatedly.

## What this does not establish

The rat knockout is strong evidence for a neural route in rats. The human
evidence (MEG, scalp EEG) establishes that respiration-coupled oscillatory
modulation exists and is whole-brain, but no human study has run the isoelectric
control, which is not possible. So the *existence* of a human neural route is an
inference from the oscillatory coupling, not a demonstration.

The relative size of the two routes is unknown and is not something to assume. It
is a quantity to estimate, which is the point of treating measured respiration
as an input with two free gains.

## Requirements this puts on the recording

1. A respiration belt and capnography on the imaging clock, at a sampling rate
   that resolves individual breaths, not a per-volume summary.
2. Enough temporal resolution in the BOLD to avoid aliasing the breathing band
   into the envelope band, or an explicit model of the aliasing.
3. The EEG retained before any respiration-related artefact removal, since a
   pipeline that regresses respiration out of the EEG destroys the neural route
   that has to be estimated. This is the same requirement as the "features
   before HRF convolution" ask in `docs/data_access_contacts.md`, applied to
   physiology.
