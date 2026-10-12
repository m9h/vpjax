# Which brain-body system model is it? Five topologies and the simulations that separate them

The route question so far has been *where in one subject's physiology does the
input act* — the neural drive or the flow-inducing signal. That is a question
about one arrow. The larger question is the **topology**: which way the arrows
between brain and body point at all. It matters because several topologies
predict the same thing about amplitude, and the usual readouts cannot tell them
apart, so a result can be reported as a brain effect when the data never
excluded the body being upstream.

This note sets out the candidate topologies as corners of a two-gain model,
says what distinguishes each pair, and defines the simulations. It is the
generalisation of `scripts/autonomic_confound_case.py` from two routes to five
systems.

## The minimal model

Three blocks, each of which already exists in the package:

- a **neural drive** `z` (OU, time constant `tau_z`) feeding the Balloon states
  `s, f, v, q` and so the BOLD — `vpjax/validation/statespace_recovery.py:146`;
- a **slow autonomic state** `a` standing for sympathetic/vagal tone and
  observed through heart rate — `vpjax/cardiac/vagal.py:88` already integrates
  `d_vagal = gain * u - vagal_tone / tau_vagal`, `d_hr = -coupling * vagal_tone
  - hr_deviation / tau_hr`;
- the **measurement route** into BOLD that bypasses `z` entirely, the existing
  `beta_vasc` on the `s` row.

Two coupling gains define the topology:

| gain | arrow | state in the package |
|---|---|---|
| `g_bb` | brain → body | `VagalODE`: neural stimulus `u` drives vagal tone |
| `g_bp` | body → brain | **gap** — see below |

and three entry points for an input (drug envelope, or a drug-free
manipulation): `beta` onto `z`, `beta_a` onto `a`, `beta_vasc` onto `s`.

**The body→brain path is the gap.** `vpjax/cardiac/baroreceptor.py:142`
`modulate_neural_drive` multiplies the drive by a cardiac-*phase* excitability,
which is a within-cycle gating effect, fast and phase-locked. It is not a slow
tonic path by which a sustained change in autonomic state raises or lowers
cortical excitability over minutes — and the slow path is the one that can be
confused with a drug effect. One scalar `g_bp` on `dz/dt` is what is missing,
and it is a small addition.

## The five topologies

Each is a corner of the `(g_bb, g_bp)` space together with a choice of where the
input enters.

1. **Efferent.** Input → `z`; `g_bb > 0`, `g_bp = 0`. The brain is perturbed and
   the body follows. This is what a drug study usually claims.
2. **Afferent.** Input → `a`; `g_bp > 0`, `g_bb = 0`. The periphery is perturbed
   and the brain follows. The "it is all just heart rate" alternative, and the
   one a BOLD-only analysis cannot exclude.
3. **Common drive.** Input → `z` *and* → `a`; both gains zero. Brain and body
   both move and correlate, but neither causes the other.
4. **Loop.** Both gains positive. The perturbation enters a closed
   brain-body loop, which is the physiologically likely answer and the hardest
   to establish.
5. **Measurement only.** Input → `a` and → `s`; no neural effect anywhere. The
   BOLD moves through the vascular route alone. This is the null the existing
   confound grid showed BOLD cannot reject on its own.

Topologies 1, 2 and 3 all predict that BOLD changes, that heart rate changes,
and that the two correlate. Amplitude cannot separate them. Four things can.

## What separates them

**(a) Lag asymmetry.** Topology 1 predicts `z` leads `a`; topology 2 the
reverse. This is the weakest discriminator and the one most damaged by the
Balloon, which low-passes `z` into BOLD and so costs most of the timing
information on that channel. It works on the fast channel, not on BOLD — which
is another reason the EEG is load-bearing.

**(b) Conditional independence given a measured input.** In topology 3,
`z` and `a` are conditionally independent given `u`. In 1, 2 and 4 they are
not. Where the input is measured — and for respiration it is — this is a sharp,
testable constraint rather than a timing argument.

**(c) A body-first perturbation with no drug.** Drive the periphery and nothing
else: paced breathing, a cold pressor, a Valsalva manoeuvre. Topologies 2 and 4
predict a brain response; 1, 3 and 5 predict none beyond the vascular route.
**This measures `g_bp` directly.**

**(d) A brain-first perturbation with no drug.** Drive the brain and let the
body follow: an affective film, a cognitive load block. Topologies 1 and 4
predict a body response; 2, 3 and 5 predict none. **This measures `g_bb`
directly.**

## The protocol consequence, and a gap in the current design

(c) and (d) identify the two gains *separately*, and both are needed to pin a
loop. The sustained-arousal arm in `docs/anxiety_induction_arm.md` is a
**brain-first** manipulation: a film drives central affect and the body follows.
It therefore measures `g_bb` and says nothing about `g_bp`.

There is at present **no body-first arm**, so `g_bp` — the gain that carries the
entire "the brain changes are downstream of the heart" alternative — is
unmeasured. Paced breathing is the obvious candidate: it is a graded, reversible,
drug-free peripheral manipulation, it needs no new hardware beyond the
respiration belt already required, and it can be blocked within a session.

Measured respiration gives a weaker continuous version of the same thing for
free, which is the argument in `docs/respiration_two_routes.md`. A paced block
is the large-signal version of it.

The payoff is a reduction in what the drug session has to establish. With
`g_bb` and `g_bp` already measured drug-free **in the same subject and session**,
the drug data only has to resolve the *entry point* — `z`, `a` or `s` — with the
coupling gains pinned. That is a far smaller question than estimating topology
and entry point jointly, and it is the difference between a study that can
attribute a drug effect to the brain and one that can only report a correlation.

## The simulations, in order

**Step 0 — structural identifiability, before any noise.** Ask whether the five
topologies are distinguishable *in principle* from a given recording set, using
the curvature machinery that already exists (`parameter_uncertainty`,
`profile_likelihood` in `vpjax/statespace/estimation.py`). If two topologies are
structurally indistinguishable from BOLD plus heart rate, no amount of data
fixes it and the recording set has to change. This is cheap and comes first, on
the principle that identifiability is a prerequisite rather than a result.

**Step 1 — the topology confusion matrix.** Simulate from each of the five,
fit all five to each, and report the matrix of log-likelihood differences, with
every gain not being fitted held at **zero** rather than at its generating
value, as in the existing grid. The diagonal should win; the informative part is
which off-diagonal pairs do not separate. Recording configurations to cross
against: BOLD only; BOLD + fast channel; BOLD + heart rate; all three; and all
three plus measured respiration.

**Step 2 — sizing the two drug-free arms.** For `g_bp` from a paced-breathing
block and `g_bb` from the film: how much of each is needed, at what effect size,
before the gain is estimable with a usable standard error. This is the same
shape as the SNR sweep in `scripts/autonomic_confound_case.py` and produces the
number the protocol actually needs — minutes of each arm.

**Expected hard cases, stated in advance so they are not presented as
surprises.** Topology 3 against 4 with weak gains is the genuinely difficult
discrimination, because a weak loop and a common drive make nearly the same
predictions. Topology 5 is the one that matters most to reject and the one BOLD
alone cannot reject. Topology 1 against 2 should be the easiest, given a fast
channel, and is the one most often assumed without test.

## Scope

No compound-specific parameters belong in any of this. As with
`drug_shaped_input`, the input is a *shape* applied at a named entry point, with
magnitude and time course supplied externally and carrying a cited source. The
deliverable is which system models a given recording set can tell apart, and how
much recording each takes — not a claim about any compound.
