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

## Step 0 result: physiology confers identifiability, the fast channel does not

Run with `scripts/brain_body_topologies.py`; full output in
`docs/results/brain_body_topologies.json`. A 900 s run, input peaking at 1 from
120 s, BOLD every 2 s, fast channel every 2 s, physiology every 1 s, every
channel scaled to the same SNR against its own excursion.

| topology | BOLD | +fast | +physiology | +both | flat direction under BOLD alone |
|---|---|---|---|---|---|
| efferent | 4/6 | 4/6 | **6/6** | **6/6** | `tau_a`, `g_bb` each unseen |
| afferent | 5/6 | 5/6 | **6/6** | **6/6** | `-0.71 beta_a + 0.71 g_bp` |
| common | 4/6 | 4/6 | **6/6** | **6/6** | `tau_a`, `beta_a` each unseen |
| loop | 6/7 | 6/7 | **7/7** | **7/7** | `-0.71 g_bb + 0.71 g_bp` |
| measurement | 3/5 | 3/5 | **5/5** | **5/5** | `tau_a`, `beta_a` each unseen |

**No topology is identifiable from BOLD alone. Every one becomes identifiable
the moment a peripheral channel is added. Adding the fast channel changes no
rank at all.**

That reads at first like a contradiction of the autonomic confound result, where
the EEG channel was decisive and BOLD alone was hopeless. It is not. The two
findings are about different parameters, and together they say something sharper
than either alone:

- the **fast channel** identifies *where the input enters the brain* — the
  `beta` against `beta_vasc` question — because it observes `z` directly and the
  vascular route bypasses `z`;
- the **peripheral channel** identifies *the body arm and the coupling* —
  `tau_a`, `beta_a`, `g_bb`, `g_bp` — because **no brain channel observes `a` at
  all**, so nothing about the autonomic state is recoverable without measuring
  it.

Neither substitutes for the other, and a study with only one of them has a named
blind spot rather than a general loss of precision. The fast channel is not
useless here: it improves conditioning substantially where a trade-off is
near-flat rather than exactly flat (`afferent` goes from 1.7e11 to 2.7e10, and
with physiology present `common` goes from 3.0e3 to 9.6e2). So the honest
summary is **physiology confers identifiability; the fast channel buys
precision**.

The flat directions are more informative than the ranks:

- **`efferent`, `common` and `measurement` lose whole parameters**, with a lone
  `1.00` loading — `tau_a` and the gain that drives `a`. When the body is purely
  downstream (or entirely decoupled), BOLD sees nothing of it whatsoever. This
  is not a trade-off that more data narrows; the parameters are simply absent
  from the likelihood.
- **`afferent` loses a product**: `beta_a` against `g_bp`, equal and opposite.
  BOLD cannot tell a strong input to the body with weak body-to-brain coupling
  from a weak input with strong coupling. Only their product is determined,
  which is precisely the quantity that is *not* the scientific claim — the claim
  is about `g_bp`.
- **`loop` loses exactly the split of the loop gain**: `g_bb` against `g_bp`,
  equal and opposite. BOLD determines `g_bb g_bp tau_z tau_a`, the closed-loop
  gain, but not how it divides between the directions. This is the same
  combination the stability analysis singles out, arrived at independently — the
  one quantity BOLD does fix is the one that governs whether the loop is stable,
  and the one it cannot fix is the one that says which way causation runs.

That last point is the protocol argument in its sharpest form. **A BOLD-only
study of a brain-body loop can establish that a loop exists and how strong it
is, while remaining unable in principle to say which direction dominates.** The
two drug-free arms are what break that degeneracy, and they break it because
each one drives a known end of the loop.

### The stability bound, which fell out on the way

The first nominal point put the loop at a closed-loop gain of exactly 1.0 and
the simulation diverged — BOLD to minus infinity rather than back to baseline.
Linearising the `(z, a)` subsystem explains it: the Jacobian determinant is
`(1 - g_bb g_bp tau_z tau_a) / (tau_z tau_a)`, so a loop gain of one parks an
eigenvalue at the origin. The integrated model reproduces the predicted
`1 / (1 - L)` amplification to within 2%, which is the check that the bound
describes the model and not just its linearisation.

This is a constraint on the topology space rather than a detail of the
parameterisation: a brain-body loop with positive gain in both directions is
only stable while that product stays below one, and the closer it runs to the
boundary the more it amplifies any input. Any claim about loop strength has to
state where it sits relative to that boundary.

### The unit loading on physiology is a convention, not an assumption

Step 0 above reads the physiological channel as `hr = a`. That looks like the
weak point of the whole result: a real channel reports its state up to an
unknown gain, and a free gain is one more parameter. Testing it
(`--loadings hr`) gives a sharp answer, and it is the opposite of a weakness.

| free loading | efferent | afferent | common | loop | measurement |
|---|---|---|---|---|---|
| none | 6/6 ✓ | 6/6 ✓ | 6/6 ✓ | 7/7 ✓ | 5/5 ✓ |
| `gain_fast` | 7/7 ✓ | 7/7 ✓ | 7/7 ✓ | 8/8 ✓ | 5/5 ✓ |
| `gain_hr` | 6/7 | 6/7 | 6/7 | 7/8 | 5/6 |

**A free loading on the fast channel costs nothing. A free loading on the
physiological channel costs exactly one flat direction, in every topology, with
every combination of channels.** The asymmetry has a one-line cause:

- the drive's scale is **anchored by the Balloon**, where `z` enters `ds/dt`
  with a coefficient of exactly one, so `gain_fast` is estimable alongside
  everything else;
- the autonomic state has **no anchor**. `a` appears only through `g_bp` (into
  the drive) and through its own loading, so sending `a -> c a` while sending
  `beta_a -> c beta_a`, `g_bb -> c g_bb`, `g_bp -> g_bp / c` and
  `gain_hr -> gain_hr / c` leaves every observation untouched.

That is an exact invariance, so the flat direction is structural and no run
length or extra channel removes it. The numerical null space confirms it term
for term: `+1` on each gain that *writes into* `a` and `-1` on each gain that
*reads out of* it, with equal magnitudes — `-0.71 g_bb + 0.71 gain_hr` for
`efferent`, `-0.58 beta_a + 0.58 g_bp + 0.58 gain_hr` for `afferent`, and so on,
each at `1/sqrt(k)` for its `k` loaded terms. The tests assert the agreement
rather than the numbers, so the invariance is checked and not merely recorded.

So `gain_hr = 1` is the **identifying convention**: it defines the autonomic
state in the units of the series actually recorded, which is exactly what makes
`beta_a` and `g_bp` interpretable — a body-to-brain gain in drive units per unit
of measured heart rate. Fixing it costs no generality, and it is the only way
the body arm is estimable at all. **Step 0's result stands unchanged**, and the
free drive loading is a bonus: the fast channel does not need its sign or scale
assumed, which is what the recorded-data model already does for it.

The practical rule for the protocol: the physiological series has to be recorded
in reportable units and used as the definition of the autonomic state, not
rescaled away by a preprocessing step that standardises it per run. Standardise
per run and the loading varies between runs, which reintroduces the very
parameter that cannot be estimated.

### What Step 0 does not say

It is a *local* test at one nominal point, so it gives a necessary condition and
not a sufficient one: passing means the parameters are resolvable to first order
there, not that they are globally identifiable or that a fit will find them.
Equal-SNR scaling is a deliberate fairness choice, not a claim that real
channels are equally clean. And identifiability of each topology *on its own*
is not the same as being able to tell two topologies *apart*, which is Step 1.

## Step 1 result: four of five separate cleanly, and the fifth pair is the one the study cares about

Step 0 asked whether each topology is identifiable on its own. Step 1 asks the
question a study actually faces: data came from one topology, all five are
fitted to it, and does the right one win? Settings are the ones Step 0 argued
for — physiology present, `gain_hr` fixed at 1, channels fast + BOLD + heart
rate at SNR 3 against each channel's own deterministic excursion, 900 s, TR
2.0 s, loop gain 0.25, three seeds. Omitted gains are held at **exactly zero**
in the fitted model, never at the value that generated the data. Full output in
`docs/results/brain_body_confusion.json`; rerun with
`scripts/brain_body_confusion.py`.

Median dAIC above the row's best, rows the generating topology and columns the
fitted one:

| data \ fit | efferent | afferent | common | loop | measurement |
|---|---|---|---|---|---|
| efferent | **0.00** | 45.99 | 3.03 | 1.63 | 46.26 |
| afferent | 50.67 | **0.00** | 40.06 | 31.26 | 77.05 |
| common | 3.08 | 25.46 | **0.00** | 2.41 | 39.35 |
| loop | 19.56 | 71.46 | 24.74 | **0.00** | 60.85 |
| measurement | 112.21 | 126.90 | 111.57 | 113.05 | **0.00** |

The diagonal wins every row, but that is the wrong thing to read first. A dAIC
under about 2 is not a preference, and two cells sit inside it.

**Measurement-only is rejected overwhelmingly, which is the result that most
matters.** A recording in which the input moves the vasculature and the body
directly and the brain contributes nothing loses by 111 to 127 dAIC against
every alternative, and every alternative loses to it by a comparable margin on
its own data. No seed comes close to confusing it. A study with these three
channels can say whether the brain is in the causal path at all, and that is the
first question anyone will ask of a physiological-confound story.

**The loop is detectable, and the nested pair is asymmetric in the right
direction.** `efferent` sits strictly inside `loop` (`g_bp` the extra
parameter), so that pair admits a likelihood ratio: on loop data, 2·dLL is 38.6,
38.5 and 21.6 across seeds against a 3.84 critical value, so the loop is
detected in every seed. On efferent data the same ratio is −0.13, 0.37 and
−0.37 — `g_bp` buys nothing, and AIC correctly charges 1.63 for carrying it.
The design can find a body-to-brain arm when there is one without inventing one
when there is not.

**Efferent versus common is not resolved**, and the median table conceals it.
Per seed on efferent data the margin over `common` is +0.26, **−0.81** and
+1.63: one seed in three names the wrong topology, and no seed produces a margin
worth reporting either way. On common data the margins over `efferent` are
+0.81, +2.24 and +3.08 — the same thin band. So "truth wins" is true of the
median and not of the evidence. The right summary is that a single
input-onset run cannot tell **the input drives the brain and the body follows**
from **the input drives brain and body independently**.

That is exactly the distinction this study wants, so it is worth being precise
about why one run cannot make it. Both topologies put the same input into z and
produce correlated brain and body excursions on the same clock; they differ only
in whether the body's excursion is *caused by* z or merely *concurrent with* it.
A single shared onset gives no leverage on that, which is what discriminators 3
and 4 — the drug-free perturbation arms — exist to supply: a body-first
perturbation moves `a` while leaving the input flat, and under `common` the
brain does not follow. Step 1's negative result is therefore the argument for
Step 2 rather than a problem with it.

### The predictions I recorded in advance, scored

Two of three right, and the one I got wrong matters. I predicted common
against loop as the genuinely hard pair; it is not — 2.41 on common data and
24.74 the other way, so the asymmetry resolves it. The hard pair is
**efferent against common**, mutually inside the noise, which I had not
flagged. I predicted measurement-only would be the one most important to
reject, and it is rejected by over 110 dAIC. I predicted efferent against
afferent would be easiest given a fast channel, and it is: 45.99 and 50.67.

Scope not yet covered: this matrix is the all-three-channel configuration
only. Step 0 already rules out BOLD alone, so the remaining cell worth running
is BOLD + heart rate *without* the fast channel — the configuration an
fMRI-only study with a pulse oximeter actually has. That is a one-flag rerun
(`--observers bold hr`) and belongs with Step 2.

### What the fit recovers, and what it does not

Topology selection working does not mean the parameters are measured. On the
diagonal, median estimate over truth:

| | coupling and body | hemodynamic nuisance |
|---|---|---|
| recovered | `g_bb` 0.90–0.92x, `g_bp` 0.91–1.00x, `beta_a` 0.87–0.91x, `tau_a` 0.98–1.07x, `beta_vasc` 0.94x | — |
| not recovered | — | `kappa` 0.31x–2.45x, `tau` 0.70x–2.07x, `tau_z` 0.95x–1.77x |

The brain-entry gain `beta` looks poor on its own (0.56x–1.05x), but it is
poor in a specific and harmless way: it trades against `tau_z`, and the
**product** `beta·tau_z` — the DC gain of z's response to the input, which is
the only combination the Balloon downstream actually sees — recovers to
0.963–1.013 in every seed and every topology carrying it. The loop gain
`g_bb·g_bp·tau_z·tau_a` recovers to 0.88–0.95 of its true 0.25 likewise. So the
reportable quantities are the coupling gains and the DC gains, not the rate
constants, and a results table should be written in those terms.

### Two caveats that are mine, not the design's

The per-cell `identifiable` flag is mostly `False`, which looks alarming beside
Step 0's clean verdicts and is a different test: Step 0 evaluates curvature at
the nominal point over the structural parameters, while this flag evaluates it
at the *fitted* optimum with the two process-noise variances `q_z` and `q_a`
also free. Those two add near-flat directions of their own, and the flat
combinations are in the nuisance block the table above already reports as
unrecovered. It does not contradict Step 0 and it does not touch topology
selection, but the flag should not be read as a structural verdict.

Second, three of the fifteen nested likelihood ratios come out *negative*
(−13.03, −1.11, −0.27). A superset model must fit at least as well, so a
negative ratio is an optimiser failure and not evidence — three restarts and 200
steps are not always enough. The magnitudes are small except on `afferent` data
where both models are misspecified anyway, so the conclusions above do not rest
on them, but a published version of this matrix needs more restarts.

## Scope

No compound-specific parameters belong in any of this. As with
`drug_shaped_input`, the input is a *shape* applied at a named entry point, with
magnitude and time course supplied externally and carrying a cited source. The
deliverable is which system models a given recording set can tell apart, and how
much recording each takes — not a claim about any compound.
