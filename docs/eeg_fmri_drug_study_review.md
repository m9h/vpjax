# Hypothesis-driven drug EEG-fMRI: what Timmermann/Pasquini obtained, who to learn from, and what vbjax + vpjax would extend

Written 2026-10-21 after a meeting with the Carhart-Harris lab. The lab wants to know what
can be done with simultaneous EEG-fMRI once the data exist, and is still at a
phenomenological stage of analysis. This note (1) summarises the kinds of result their own
group has already obtained on the DMT EEG-fMRI dataset, (2) lists the groups that ran
*hypothesis-testing* pharmacological EEG-fMRI studies and what their designs taught, and
(3) maps the published Virtual-Brain-family examples onto what the vbjax + vpjax stack adds.
Sources were retrieved through PubMed/PMC (DOIs given) and bioRxiv; nothing here is from
the lab's unpublished material.

## 1. What the Imperial DMT EEG-fMRI dataset has yielded so far

### 1.1 Timmermann et al. 2023, PNAS (DOI 10.1073/pnas.2218949120)

Design. 20 healthy volunteers, single-blind placebo-controlled counterbalanced crossover,
20 mg IV DMT fumarate over 30 s, 28-min eyes-closed resting EEG-fMRI (8 min pre, 20 min
post). 3 T Siemens Verio, 12-channel coil, TR 2 s, 3 mm isotropic, 35 slices. EEG: 31
channels BrainAmp MR + BrainCap MR, 5 kHz, 250 Hz hardware low-pass, SyncBox clock sync,
two ECG channels (used only for ballistocardiogram correction). Minute-wise verbal
intensity ratings came from a *separate* run on the same day and were used as a covariate.

Pipeline. AFNI/FSL/ANTs: despike, slice-time, motion, nonlinear MNI, scrubbing FD > 0.4
(volumes replaced by neighbours), 6 mm smoothing, band-pass 0.01-0.08 Hz, nine nuisance
regressors (6 motion, ventricles, draining veins, local white matter). No GSR in the main
analysis (GSR replicated in SI). EEG: AAS gradient + pulse-artefact templates, 250 Hz,
1-45 Hz, 2-s epochs, ICA; IRASA to separate oscillatory from 1/f power. Exclusions: 4/20
for motion in the 8-min post window, 3 more for the dynamic analyses, 2 for EEG artefact,
1 for a 7.5 Hz alpha peak; the joint EEG-fMRI analysis used n = 12. Head motion differed
between DMT and placebo (p = 0.003), handled by subsample replications.

Results, fMRI. Yeo-7 dual-regression integrity fell in 5/7 networks (not salience, not
limbic); between-network segregation fell for frontoparietal/salience/DMN; global
functional connectivity (GFC, mean Fisher-z to all other parcels, Schaefer-100 + 12
subcortical) rose in salience, frontoparietal and DMN, and whole-brain GFC rose
(t(15) = 3.11, p = 0.007). Sliding-window dynamics (44 s windows, 2 s step) regressed on
intensity ratings in linear mixed models: DMN/FP/SAL/LIM GFC tracks intensity; GFC rise is
strongest 1-6 min post injection; the same held against the published mean plasma curve.
The intensity-GFC beta map correlates with the Beliveau 5-HT2A PET map. The principal
connectivity gradient (BrainSpace diffusion embedding) is compressed: transmodal pole
down, unimodal up.

Results, EEG. Alpha down, delta and gamma up, Lempel-Ziv complexity up (cluster-corrected
permutation tests on 8-min post-injection blocks, baseline-subtracted). Time-resolved
separations from placebo: alpha 0-8 min, LZc 2-14 min, delta 0-4 min, beta 0-1 min.
Backward travelling-wave power down, forward up. LZc correlates with rated "richness"
(r = 0.70, p = 0.008), presented as a direct test of the entropic-brain hypothesis.

Results, EEG-fMRI. The band-power and LZc time courses were band-passed to 0.01-0.08 Hz,
HRF-convolved and Gaussian-smoothed to match the fMRI, then entered as predictors of
regional GFC in linear mixed models: frontal delta tracks widespread GFC; parietal alpha
is anticorrelated with transmodal GFC; occipital gamma and LZc track frontoparietal and
limbic GFC. The authors note these relationships only appear when within-subject
trajectories are exploited, not in between-subject averages.

What this is and is not. It is a thorough correlational, multivariate description, with
one a-priori hypothesis test (entropy vs richness) and one receptor-map comparison. The
EEG and fMRI are fused only at the level of smoothed, convolved summary time courses; no
generative model links them, no physiological regressor enters the fMRI model, and the
public release (`timmer500/DMT_Imaging`) is the band-passed, HRF-convolved end product
(see `data_access_contacts.md`).

### 1.2 Pasquini et al. 2025, Imaging Neuroscience (DOI 10.1162/imag_a_00541)

The EKG channel of the same recordings. 14 volunteers with the full multimodal
acquisition. Pipeline: whole-brain activity map per TR, Pearson similarity between TRs,
DMT-minus-placebo similarity matrix thresholded at r = 0.25 (data-driven) and partitioned
with community detection into "brain activity substates"; per-subject nodal beta maps per
substate; paired tests across conditions; partial least squares against the 25
post-scan visual-analogue items; heart rate from R-peaks at 0.5 Hz. Result: the first
post-injection substate (about the first 4 min) shows right superior temporal
hyperactivity (loads on hallucination items) and left hippocampal + medial parietal
deactivation (loads on time/space/self items). Heart rate rises under DMT in that
substate (t = 2.68, p = 0.02) and the *size* of the heart-rate rise correlates with the
hippocampal/medial-parietal deactivation across subjects (r = 0.56, p = 0.04): smaller
rise, deeper deactivation. Heart rate is used as a between-subject covariate, not as a
time series inside the BOLD model, so neural and vascular routes remain confounded; this
is the "confound in miniature" already flagged in `data_access_contacts.md`.

### 1.3 Lorenzo Pasquini's analytic lineage (what to expect from the UCSF side)

- Ayahuasca subacute, J Psychopharmacol 2020 (DOI 10.1177/0269881120909409): ICA/seed
  network connectivity pre/post, salience vs DMN, correlated with Hallucinogen Rating Scale.
- Anterior insula time-varying connectivity, NeuroImage 2019
  (DOI 10.1016/j.neuroimage.2019.116425): sliding-window seed FC + k-means states.
- Autonomic states, Psychophysiology 2022 (DOI 10.1111/psyp.14218): PCA + clustering of
  second-by-second multichannel physiology into brief ANS states (Sturm lab).
- DMT substates + EKG, 2025 (above).
- Psilocybin fronto-striato-thalamic modelling with Deco, Hum Brain Mapp 2026
  (DOI 10.1002/hbm.70596): whole-brain model, structure-function coupling, effective
  connectivity (bottom-up vs top-down), related to 5-HT2A and D2 receptor maps.
- Cross-species psychedelic gene expression preprint 2025 (DOI 10.21203/rs.3.rs-7625999/v1).

Pattern: data-driven dynamic-state discovery, PLS against questionnaires, physiology as a
covariate, and recently Deco-style whole-brain modelling with receptor maps. He has the
EKG handling and the dynamic-state machinery; what is missing on that side is a
generative forward model in which physiology is an input rather than a covariate.

## 2. Who tested hypotheses with pharmacological EEG-fMRI, and what each design teaches

| group | drug, design | hypothesis tested | what to take |
|---|---|---|---|
| Tracey lab, Oxford: Ní Mhuircheartaigh 2013 Sci Transl Med (DOI 10.1126/scitranslmed.3006007); Warnaby 2016 Anesthesiology (DOI 10.1097/ALN.0000000000001027); Warnaby 2017 (DOI 10.1097/ALN.0000000000001759) | propofol, ultra-slow TCI ramp, 16/15 volunteers, multisensory stimulation inside the scanner | an *individualised EEG endpoint* (slow-wave saturation) marks loss of perception; at that point the thalamocortical system becomes isolated from stimuli (evoked BOLD lost) while internal thalamocortical exchange persists; dorsal anterior insula is the cortical gate at loss of response | define the drug-state marker on the EEG per subject, then ask what the fMRI does *at that marker*; use evoked responses, not only rest; the dose axis is an effect-site concentration ramp, not a bolus |
| TUM (Jordan, Ilg, Ranft): Ranft 2016 Anesthesiology (DOI 10.1097/ALN.0000000000001322); Golkowski 2017 (DOI 10.1002/brb3.679) | sevoflurane at burst suppression, 3 and 2 vol%, 16 ventilated volunteers | reduced anterior-to-posterior information transfer (symbolic transfer entropy) and reduced permutation entropy go with reduced frontal and thalamocortical BOLD connectivity, sparing primary sensory networks; BOLD-EEG coherence under burst suppression | directed EEG measures as the hypothesis carrier, steady-state concentration plateaus as the design |
| Auckland (Muthukumaraswamy, Forsyth, McMillan): Forsyth 2018 Psychopharmacology (DOI 10.1007/s00213-018-5064-8); Forsyth 2019 HBM (DOI 10.1002/hbm.24889); McMillan 2019 J Psychopharmacol (DOI 10.1177/0269881118822263); McMillan 2019 PNBP (DOI 10.1016/j.pnpbp.2019.109838); Forsyth 2021 Brain Topogr (DOI 10.1007/s10548-021-00876-8) | ketamine vs midazolam vs placebo three-way crossover, 30 subjects, rest + working memory; ketamine vs remifentanil in MDD | can fMRI connectivity serve as a drug biomarker when checked against simultaneous EEG; does a single pharmacokinetic regressor capture the phMRI response | seven preprocessing pipelines gave different drug effects (ICA denoising strongest); no significant cross-modal relationship between EEG and BOLD connectivity *changes*; the subgenual ACC BOLD decrease was physiological noise (cardiac); EEG band time courses as phMRI regressors reveal distinct neural time courses. This is the cautionary tale: without a forward model, summary-statistic fusion does not replicate across pipelines |
| Winterer group: Zacharias 2019 HBM (DOI 10.1002/hbm.24791); Mobascher/Warbrick 2012 (DOI 10.1016/j.pscychresns.2012.09.005; 10.1162/jocn_a_00236) | S-ketamine placebo crossover, 24 subjects; nicotine oddball | the ketamine prefrontal connectivity drop is a *vigilance* effect visible in EEG; EEG-informed single-trial fMRI | vigilance must be measured, not assumed; the drug moves arousal and the arousal moves the fMRI |
| Wise lab, Cardiff: Diukova 2012 NeuroImage (DOI 10.1016/j.neuroimage.2012.04.041) | caffeine 250 mg crossover, 14 non-consumers, visual/motor/oddball tasks + ASL | separate neural from vascular drug effects: visual BOLD fell while the visual evoked potential was unchanged (vascular), frontal oddball BOLD rose with shorter P300 latency (neural); grey-matter CBF down 19% | the cleanest published template for the vascular-vs-neural question; needs an evoked EEG response the drug leaves intact as the neural anchor |
| Imperial/UCSF: Timmermann 2023, Pasquini 2025 (section 1) | DMT bolus | entropic-brain test; GFC vs 5-HT2A map; HR vs deactivation | the only psychedelic EEG-fMRI set; phenomenology-rich, model-free |

Common properties of the studies that produced an interpretable answer: (a) the hypothesis
named an EEG quantity and an fMRI quantity and the direction of their coupling in advance;
(b) the drug level was controlled or measured as a concentration axis (TCI ramp, vol%
plateaus, infusion) rather than a single bolus; (c) an evoked or task response gave a neural
anchor independent of the vascular pathway; (d) physiology was recorded and entered the
model. The psychedelic studies have none of (b)-(d) yet. Point (b) is the one the 5-MeO
protocol can satisfy by design because the compound is short-acting; the PK layer in vpjax
is what turns the dose event into that axis without hard-coding any compound parameters.

## 3. Virtual-Brain-family precedents, and what vbjax + vpjax would extend

### 3.1 Precedents

- Ritter et al. 2013 Brain Connect (DOI 10.1089/brain.2012.0120): the TVB proof of concept
  was itself an EEG-fMRI problem: alpha-rhythm neural masses on a connectome, BOLD from a
  Balloon monitor, matched to simultaneous EEG-fMRI.
- Schirner et al. 2018 eLife (DOI 10.7554/eLife.28927): the hybrid model. Subject-specific
  EEG source activity is injected as the drive to each connectome node and the model
  predicts the subject's 20-min fMRI time series, reproducing alpha-BOLD anticorrelation
  on short and long time scales, E/I balance and fMRI power-law scaling. This is the
  closest structural ancestor of the vpjax state-space fit: EEG as the observed drive,
  BOLD as the slow observation, the hemodynamic model in between.
- Deco et al. 2018 Curr Biol (DOI 10.1016/j.cub.2018.07.083) and Kringelbach et al. 2020
  PNAS (DOI 10.1073/pnas.1921475117): dynamic mean-field whole-brain model in which the
  neuronal gain is scaled by the regional 5-HT2A PET density, fitted to LSD and psilocybin
  fMRI; in 2020 the neurotransmitter system is a coupled dynamical variable with
  release/reuptake kinetics. The drug enters as one static global scaling parameter.
- Herzog et al. 2020/2023 Sci Rep (DOI 10.1038/s41598-020-74060-6; 10.1038/s41598-023-32649-7)
  and Jobst et al. 2021 NeuroImage (DOI 10.1016/j.neuroimage.2021.117809): the same
  family used to explain the entropy increase and the perturbational sensitivity under LSD.
- Piccinini, Tagliazucchi et al. 2025 Commun Biol (DOI 10.1038/s42003-025-07576-0): a
  time-dependent Hopf whole-brain model fitted to the Timmermann DMT fMRI in which the
  distance to the global bifurcation is parametrised by a pharmacokinetic equation;
  predicts a transient of heightened reactivity in fronto-parietal and visual cortex that
  follows the PK curve and correlates with 5-HT2A density. This is the closest published
  version of "PK drives a dynamical parameter", on BOLD only, without EEG.
- Luppi et al. 2022 Commun Biol (DOI 10.1038/s42003-022-03330-y): propofol as GABA-A
  density-weighted inhibition in a DMF model; Alnagger et al. 2025 Adv Sci
  (DOI 10.1002/advs.202511780): individualised Hopf models of DoC patients with simulated
  LSD/psilocybin.
- Angiolelli, Sorrentino, Jirsa et al. 2025 npj Syst Biol Appl (DOI 10.1038/s41540-025-00516-y),
  "The Virtual Parkinsonian patient": a neural-mass model with a local dopamine
  concentration state (Michaelis-Menten clearance, dopamine scales the excitatory
  conductance) is inverted against EEG + STN recordings to infer the dopaminergic tone ON
  vs OFF L-DOPA per patient. **This model is what vbjax ships as `dopa_dfun`** (adaptive
  QIF with states r, V, u, Sa, Sg, Dp; `vbjax/app/dopa/` has the sweep and BOLD tooling).
  It is the only TVB-family example where a *drug level* is a state variable inferred from
  electrophysiology.

What vbjax has today: Jansen-Rit, Montbrió-Pazó-Roxin, Epileptor (BVEP), bilinear DCM, the
dopamine-modulated QIF, a Balloon BOLD monitor, delays, and examples that fit a JR power
spectrum to EEG (`jr-psd-fit-*.py`) or an MPR network to HCP BOLD FC (`fc-opt.py`) but
never both observations together.

### 3.2 What the vpjax additions extend, concretely

1. **The drug as a time axis, not a scalar.** Deco/Kringelbach fit one gain scalar; Piccinini
   wrote the PK into the Hopf parameter by hand. vpjax's `pharmacokinetics.drive` turns an
   externally supplied, cited PK/PD assumption set into an effect-site occupancy series and
   `brain_body`/`statespace` accept it as a drive. Composed with vbjax, the 5-HT2A (or
   receptor-weighted) gain in a Deco-style or `dopa`-style node becomes occupancy(t) times
   a fitted beta, so the fitted quantity is "drive per unit peak occupancy" and the test is
   whether EEG and BOLD follow the effect-site curve rather than the plasma curve or an
   arousal curve. Nothing about the compound is in the code; the assumptions file carries
   the citation.
2. **Joint EEG + BOLD likelihood with identifiability.** Schirner injects EEG and scores the
   simulated BOLD by correlation; Angiolelli inverts against EEG only. vpjax's state-space
   layer fits both observations by innovation likelihood, with Fisher/curvature
   `parameter_uncertainty`, and the recorded result that BOLD-only is non-identifiable while
   a simultaneous EEG channel at TR rate identifies every parameter (3/3 seeds). That is
   the quantitative version of why the hybrid model needed the EEG at all.
3. **Physiology on the vascular input side.** None of the TVB examples carry respiration,
   heart rate or CSF/arousal channels. `brain_body.py` makes heart rate an observed state
   with five candidate brain-body topologies (efferent, afferent, common, loop,
   measurement) and the Step-1 confusion shows they are distinguishable with EEG + heart
   rate and not with BOLD alone. This is exactly the question Pasquini's heart-rate
   correlation leaves open, and the Diukova caffeine logic made mechanistic.
4. **The hemodynamic model is fitted, not fixed.** TVB's Balloon monitor is a fixed forward
   map. vpjax's hemodynamics (Balloon, Riera NVC, qBOLD, layered NVC) are differentiable
   and the specificity result (a drug that only triples BOLD noise yields zero fitted
   drive, LR 0) shows the Balloon low-pass is what makes a neural drive distinguishable
   from a vascular/noise change.
5. **Design before data.** The simultaneity-case and design-grid scripts (TR, duration,
   EEG rate, SNR) give the acquisition arguments (short TR for precision, in-bore EEG for
   identifiability, bottom slice at the fourth ventricle for the CSF channel) that the TVB
   literature takes as given.

### 3.3 Hypotheses the lab could commit to, in the form these tools can test

- H1 (receptor-gain time course): the EEG alpha/beta damping shift and the transmodal GFC
  rise follow effect-site occupancy with one shared lag; the alternative is an arousal
  process with its own time course (the Zacharias confound). Readouts: fitted `beta`, the
  drive time constant, and the likelihood ratio against the arousal-only model; requires
  simultaneous EEG.
- H2 (neural vs vascular): the medial-parietal/hippocampal BOLD deactivation is efferent
  (brain drives heart rate) rather than afferent or common-input. Readouts: the brain-body
  topology confusion on the recorded EKG + respiration + EEG; requires physiology on a
  common clock.
- H3 (preserved evoked response, Tracey/Diukova logic): the auditory steady-state and
  visual tagging responses through the peak are preserved in EEG while their BOLD
  counterparts change, or both change together. This separates vascular from neural
  reductions without a model, and anchors H1 and H2.
- H4 (cortical damping, Liley/neurojax): the resting spectrum under 5-MeO moves on the
  damping axis differently from propofol (faster alpha, sharper weaker beta, knee down) and
  from ketamine; readouts from `neurojax.bsd` `liley_mix2` vs `specparam2`, free-energy
  comparison per subject and block.

Each of these names an EEG quantity, an fMRI quantity, a direction, and a competing
explanation, which is what the anaesthesia groups had and the psychedelic studies so far
did not.


## 4. Confounds of the time-resolved complexity measure, and the interpretable alternative

The published time-resolved endpoint on the EEG side is Lempel-Ziv complexity (LZc),
computed per sliding window and correlated with ratings and with regional fMRI
connectivity. This section lists what LZc cannot separate, then what replaces it.

A note on scope: LZc is a genuinely good *detector*. Its best-established property is that
it orders wake, NREM, anaesthesia and disorders of consciousness consistently across labs
and montages, which gives any new state a cross-state anchor to a large prior literature.
Nothing below argues for dropping it. The argument is that it is the wrong *endpoint* for a
study whose stated question has moved from "did a graded state occur" to "what changed in
the tissue".

### 4.1 The measure is largely a restatement of the spectrum

LZc is computed on a binarised signal, normally the Hilbert amplitude envelope thresholded
at its median, sometimes the raw trace at its mean. The run-length statistics of that binary
string are governed by the threshold-crossing rate, which is set by the spectral centroid.
A flatter aperiodic slope, less alpha, or more high-frequency power each raise LZc more or
less mechanically.

Mediano, Rosas, Barrett & Bor formalised this and gave the test: decompose any
between-condition difference in a non-linear feature into a spectral part and a genuinely
phase-dependent part, by comparing the observed difference against phase-randomised
surrogates that preserve the amplitude spectrum (Phys Rev Lett 127, 124101, 2021,
doi:10.1103/PhysRevLett.127.124101; arXiv:2009.10015). Applied to LZc across several
datasets, a large share of the condition difference is spectral. This critique comes from
inside the same methodological community, which makes it usable in a proposal without
appearing adversarial.

Consequence for the published result: reporting "alpha down, delta and gamma up" and then
"LZc up" as separate findings is close to reporting one observation twice. The two are not
independent evidence for an entropy account.

Normalisation does not repair this. Dividing by a *shuffled* surrogate removes
amplitude-distribution effects but not spectral ones. Dividing by a *phase-randomised*
surrogate does remove the spectral part, but the resulting ratio has two biased estimators
in it and much larger variance, which matters at n = 12.

### 4.2 Broadband nuisance, and why the magnet makes it worse

Because LZc rewards novel substrings, it rewards added broadband variance of any origin.
The relevant sources in an EEG-fMRI run:

- Scalp EMG (temporalis, frontalis, neck) spreads broadband above roughly 20 Hz and is the
  classic contaminant of "gamma" and of every complexity measure built on it.
- Saccadic spike potentials and microsaccades add brief broadband transients, with rates
  that change with arousal and with eyes-closed restlessness.
- Residual gradient artefact surviving average-artefact subtraction, which grows whenever
  the head moves relative to its position at template formation.
- Residual ballistocardiogram surviving ICA, which scales with heart rate and pulse
  amplitude. The 2023 study carried only two ECG channels, for BCG correction alone.

The structural problem is not that these exist. It is that each one moves in the *same
direction* and on the *same time course* as the manipulation, because autonomic activation,
muscle tone and small head movements all track the state. Head motion already differed
between conditions in the published comparison (p = .003), and four of twenty subjects were
excluded for motion. A nuisance correlated with the design cannot be regressed out after the
fact without removing signal with it.

### 4.3 Estimator bias in sliding windows

LZ complexity converges to the entropy rate only asymptotically. In short windows its
expectation depends on window length, sample rate and the binarisation rule, with a positive
bias that decays slowly (order 1/log n). So a time-resolved LZc trace carries a bias
trajectory of its own whenever effective window occupancy changes: artefact rejection,
scrubbing, variable retained channel counts, drift in cap impedance over a long run. Any
monotone decline in data quality across a session appears as a monotone change in
"complexity". This is checkable and is rarely checked.

### 4.4 It is an arousal detector

The property that makes LZc valuable, consistent ordering across wake, sleep and
anaesthesia, is exactly the property that makes it confounded here: it is highly sensitive
to the arousal axis. Any manipulation that raises arousal raises LZc. Without an independent
arousal regressor in the model (pupillometry, EOG-derived drowsiness, heart rate,
respiratory rate), "complexity rose" and "subjects became more aroused" are not
distinguishable. Zacharias et al. (2019, doi:10.1002/hbm.24791) showed precisely this
failure mode for a prefrontal connectivity effect under S-ketamine.

### 4.5 The two inferential steps built on it are weak

**Complexity against ratings.** The time-resolved LZc trace and the rated intensity trace
are both smooth and both approximately unimodal across the run. Correlating them over time
inflates significance badly unless effective degrees of freedom are corrected for
autocorrelation. Across subjects, a single retrospective summary rating against a run-summary
LZc gives one point per subject; r = .70 at n = 12 has a 95% interval running from roughly
.21 to .91.

**Complexity against BOLD.** HRF-convolving the LZc trace and regressing it on regional
global connectivity shares the drive with the BOLD time course, so the two series are
correlated through the design regardless of any neural coupling. Separately, BOLD responds
to arterial CO2, cardiac output and blood pressure, all of which move with the state. A
positive LZc-BOLD coupling is therefore fully consistent with a systemic vascular path from
the manipulation to both signals. With no respiratory, CO2 or beat-level cardiac measure in
the BOLD model, that alternative cannot be tested. Pasquini's own finding that the
hippocampal and medial parietal deactivation tracks the heart-rate change across subjects
(r = .56) is direct evidence that this path is live in this dataset.

### 4.6 No forward model, no units, no direction

The decisive limitation. There is no mapping from a tissue-level parameter to a predicted
LZc value. You cannot state what change in synaptic time constant, excitatory gain,
inhibitory tone or conduction delay would produce a given ΔLZc, and you cannot invert one
into the other. From this follow four things:

- It cannot be compared on a common scale across montages, reference schemes, channel
  counts, species or recording modalities.
- It cannot constrain or be predicted by a biophysical model, so it cannot be falsified at
  the mechanism level.
- It is non-monotone in the underlying physiology: an E/I shift in *either* direction, added
  broadband noise, and merely reduced oscillatory burstiness all raise it.
- "Entropy increased" is therefore a redescription of the data, not an explanation of it.

### 4.7 Variant proliferation

LZ76 versus LZ78; LZc versus LZsum versus LZs; per-channel averaged versus
channel-concatenated; Hilbert-envelope versus raw binarisation; shuffled versus
phase-randomised normaliser. Different papers use the same word for different estimators,
and effect sizes do not transfer between them. The same holds for "entropy" covering sample
entropy, permutation entropy and context-tree weighting.

### 4.8 The alternative: parameters with units instead of scalars without

The move is to replace a summary statistic *of the data* with a parameter *of a generative
model*, and to demote complexity from endpoint to validation target.

1. **Fit a physiological spectral model.** A neural-mass or mean-field model fitted to the
   sensor or source cross-spectrum reports a change in a rate constant, a synaptic gain, a
   damping ratio or a conduction delay, each with a credible interval and a sign that means
   something. `neurojax.bsd` `liley_mix2` does this; DCM for cross-spectral densities is the
   established precedent. The decisive move is the last one: **simulate from the fitted model
   and compute LZc on the simulation**. If the fitted parameter change reproduces the
   observed complexity change, the complexity result is explained rather than contradicted.
   That subsumes the existing finding instead of attacking it.

2. **Separate aperiodic from periodic as the cheap floor.** A spectral-parametrisation
   comparator (exponent, knee, peaks) splits the two things LZc conflates: broadband tilt and
   narrowband power. Pre-register it as the descriptive baseline the generative model must
   beat on free energy or held-out likelihood.

3. **Add a driven response as the identifiability anchor.** Auditory steady-state, visual
   tagging, paired-pulse or oddball. The input is known and chosen, so gain and phase are
   measured at a frequency the experimenter owns; EMG and arousal do not sit at that
   frequency, so the signal-to-nuisance ratio is structurally better than for any resting
   broadband measure. A driven EEG response preserved while its BOLD counterpart changes is
   the clean neural-versus-vascular dissociation (Diukova et al. 2012,
   doi:10.1016/j.neuroimage.2012.04.041). Chuan Hou's dichoptic SSVEP work gives a ready
   E/I-interpretable readout (see `reference_driven_eeg_paradigm_contacts`).

4. **Model physiology as a pathway, not a covariate.** Beat-level ECG, respiration belt,
   pulse oximetry, and end-tidal CO2 where the setup permits. The vascular input then enters
   the BOLD model as its own pathway (the `brain_body` topologies), and the directed
   brain-heart test asks whether a cardiac change is efferent or afferent, rather than
   entering heart rate as a between-subject nuisance term after the fact.

5. **Use a joint EEG+BOLD state-space likelihood.** One latent state, two observation
   channels, one shared drive. Cross-modal coupling becomes a model parameter with an
   identifiability check attached, not a correlation between two post-hoc summaries. The
   vpjax result that the drive is non-identifiable from BOLD alone and identifiable once EEG
   is added is itself the argument for in-bore EEG, and it does not depend on any complexity
   measure being right.

6. **Check identifiability before power.** FIM and profile-likelihood curvature on the design
   grid, before any data exist.

### 4.9 How to put this to the lab

Complexity measures are excellent detectors and poor explanations. They were the correct
instrument for the question the field actually had, which was whether a graded state occurred
and whether it scaled with report. They are the wrong instrument for "what changed in the
tissue", which is the question the lab says it now wants to ask.

So the proposal is not to discard LZc. Keep it in three roles: as a manipulation check, as
the cross-state anchor to the anaesthesia and sleep literature, and as a validation target
the generative model is required to reproduce in simulation. Move the endpoint to a parameter
that has units, a sign, a credible interval and a forward model. The spectral-versus-phasic
decomposition is the honest first analysis to run on the existing data, because it tells you
how much of the published complexity effect is left once the spectrum is accounted for, and
that number sizes the whole argument.
## 5. What this note does not do

It does not request any data, and it does not propose dosing or pharmacokinetic values for
any compound; PK/PD enter only as externally supplied, cited assumption files consumed by
the software. The Imperial DMT dataset remains request-only (see `dmt_data_request.md`,
unsent). The PsiConnect analysis by Cooper et al. 2026 (bioRxiv 10.64898/2026.10.06.757069)
is the open-data fMRI-side target for the confound audit but has no simultaneous EEG.
