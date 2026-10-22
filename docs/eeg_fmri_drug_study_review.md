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

## 4. What this note does not do

It does not request any data, and it does not propose dosing or pharmacokinetic values for
any compound; PK/PD enter only as externally supplied, cited assumption files consumed by
the software. The Imperial DMT dataset remains request-only (see `dmt_data_request.md`,
unsent). The PsiConnect analysis by Cooper et al. 2026 (bioRxiv 10.64898/2026.10.06.757069)
is the open-data fMRI-side target for the confound audit but has no simultaneous EEG.
