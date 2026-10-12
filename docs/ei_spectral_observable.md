# Why an E/I observable is harder to confound than an amplitude one

The current model has one neural state `z` and the EEG observes its amplitude.
That makes the EEG a *detector* of neural involvement but a weak one: it
establishes that something moved `z`, not what. This note argues that a spectral
observable derived from an excitation/inhibition formulation is a qualitatively
stronger constraint, and says why, using the mechanism from the ketamine
modelling literature.

## The mechanism worth copying

Neymotin and colleagues modelled hippocampal CA3 with pyramidal, basket and
oriens-lacunosum moleculare (OLM) cells and asked where an NMDA-receptor
reduction has to act to reproduce the experimentally observed effect
([10.1523/JNEUROSCI.0501-11.2011](https://doi.org/10.1523/JNEUROSCI.0501-11.2011)).
The answer was specific: the effect appeared only with an *isolated* reduction on
the OLM cells. The consequence chain is

- OLM activity falls, so theta power falls, because OLM cells carry theta;
- pyramidal cells are disinhibited, so basket cell activation rises;
- basket cells primarily generate gamma, so gamma power rises;
- and the theta modulation of gamma - the cross-frequency coupling - is
  disrupted rather than merely scaled.

So the signature of this particular E/I change is **theta down, gamma up, and
phase-amplitude coupling between them degraded**.

## Why the sign pattern is the valuable part

A confound that acts through a single scalar gain - a vascular drive, a change in
neurovascular coupling, a change in electrode impedance, a global arousal shift
that scales cortical excitability uniformly - moves band powers *together*. It
can raise everything or lower everything, and it can do so by different amounts
per band, but it has one sign.

An E/I change of the kind above moves two bands in *opposite* directions from a
single underlying parameter change. That is a one-parameter prediction about a
two-dimensional observable, so it is falsifiable in a way an amplitude change is
not. No single-route amplitude confound can produce it, because producing it
requires the inhibitory and excitatory populations to be affected differently,
which is the definition of an E/I change.

This is the same logic as the existing route comparison, one level up. There, the
EEG channel turned a non-identifiable amplitude question into a decisive one
because it broke a symmetry between two routes. Here, a two-band observable
breaks a symmetry between an E/I change and a global excitability change - and
that symmetry is *not* broken by adding the EEG amplitude channel alone, because
both hypotheses predict the same amplitude change.

## The honest difficulty

The neural-versus-vascular comparison was easy in the end because the two
hypotheses predict different things about *different channels*: one moves the
EEG, one does not. The E/I-versus-global comparison is harder because both
hypotheses live inside the EEG and both predict a spectral change. They differ
only in the pattern across bands, and that pattern has to be estimated from
noisy band-power time series whose own variance is large.

So the expected result of the planned simulation is not "decisive" but "decisive
under conditions", and the conditions are the deliverable: how much EEG recording
time, at what signal-to-noise, is needed before an opposite-sign two-band
pattern can be separated from a same-sign one. That is a sizing question of the
same shape as the drive-channel SNR sweep already in
`scripts/autonomic_confound_case.py`.

## The scalar summary, and its limits

The standard scalar proxy for E/I balance in EEG is the aperiodic exponent - the
slope of the power spectrum once periodic peaks are removed. It is attractive
because it is one number per epoch, so it drops into the existing state-space
machinery as a second observation channel beside the BOLD without any change to
the estimator.

It is also strictly weaker than the band pattern, for the reason above: a slope
is a single scalar, so it has one sign, and it therefore *cannot* express the
opposite-sign two-band signature that makes the E/I hypothesis falsifiable. A
slope change is consistent with both an E/I change and a global excitability
change. It is the right first observable to implement because it is cheap, and
the wrong one to rely on for the discrimination that matters.

The ordering that follows: implement the exponent channel first to establish the
plumbing and get a power curve, then the two-band channel to get the
discrimination.

## What this needs from the model layer

An E/I formulation needs at least two neural populations with separate states and
a parameter that scales inhibitory gain independently of excitatory gain - which
is what makes `vbjax`'s neural mass models the right place for this layer rather
than extending the single-state `z`. The coupling to the existing work is then:
the E/I layer supplies a spectral observable to the EEG channel, the Balloon
layer supplies the BOLD channel, and the autonomic layer supplies the vascular
route and the respiration input described in `docs/respiration_two_routes.md`.

No compound-specific parameters belong anywhere in this. The E/I change, like
the existing `drug_shaped_input`, is a *shape* applied to a model parameter, with
its magnitude and time course supplied externally and carrying a cited source.
