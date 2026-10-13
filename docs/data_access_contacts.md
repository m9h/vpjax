# Who to ask for raw psychedelic neuroimaging data

Compiled 2026-10-11. Every email below is taken from a published author
affiliation or a repository contact field, never guessed; where no email
could be verified the row says so and names the institution instead.

## The thing that changes the plan

The definitive inventory of this literature is Girn et al. (2026),
*An international mega-analysis of psychedelic drug effects on brain
circuit function*, Nature Medicine, <https://doi.org/10.1038/s41591-026-04287-9>.
Its Table 1 is the field: **11 datasets, 273 unique participants**, and
its author list is the set of people who agreed to share each one.

That consortium is led from UCSF Neurology. Manesh Girn (first author,
manesh.girn@ucsf.edu) and Robin Carhart-Harris (senior author) are both
in the department this protocol is being written for, and **the pooled
dataset already exists** -- somebody has already negotiated access to
all eleven. The useful first move is therefore internal: ask whether we
can be added to the existing arrangement, and only then write to
individual sites for the parts the mega-analysis did not need (the
unfiltered series, the physiology, the EEG).

One site is informative by its absence: Copenhagen (PI Patrick Fisher,
Neurobiology Research Unit, Rigshospitalet) could not contribute "because
of data protection restrictions". That is a GDPR constraint on the
custodian, not reluctance, and it will not be solved by asking again more
nicely -- it needs a data transfer agreement.

## Already open: download, do not ask

| dataset | what is there | access |
|---|---|---|
| **Inhaled DMT EEG, 35 participants** (Pallavicini, Cavanna, Tagliazucchi, with Carhart-Harris & Timmermann) | **raw `.bdf`** EEG: DMT, eyes-open, eyes-closed, plus psychometrics | Zenodo <https://doi.org/10.5281/zenodo.3992359>, CC BY 4.0, open. No fMRI. |
| **Ayahuasca fMRI 2015** (Palhano-Fontes et al.) | resting + verbal-fluency + structural, before/after ayahuasca, as raw-image zips | Harvard Dataverse <https://doi.org/10.7910/DVN/28619>, **CC0**, open, v2, 6 files |
| **LSD 2016, Imperial** (Carhart-Harris et al.) | raw MRI, **15** of the mega-analysis's 19 participants | OpenNeuro **ds003059**, CC0, open |
| **ds006072 psilocybin / methylphenidate** (WashU precision mapping) | raw, minimally preprocessed and processed BIDS; the series are **unfiltered** (10-15% of power above 0.1 Hz), so usable for innovation-likelihood fits | OpenNeuro, open |

Against the mega-analysis's Table 1, **three of the eleven** are raw and
open: Imperial LSD 2016 (ds003059), WashU psilocybin (ds006072) and UFRN
ayahuasca 2015 (Dataverse). The Imperial DMT 2023 release is public but
derived and band-passed, so it does not count. PsiConnect (ds006110, 65
subjects, MRI + EEG, CC0) is raw and open too but was not in the
mega-analysis.

## Needs a request

| dataset | custodian to write to | verified contact | note |
|---|---|---|---|
| **DMT EEG-fMRI, n=20** (Timmermann et al. 2023 PNAS, <https://doi.org/10.1073/pnas.2218949120>) | Christopher Timmermann; Robin Carhart-Harris as senior author | `cbt14@ic.ac.uk` (from the release repo) — **likely stale**: 2025-26 papers list him at UCL Psychology and at Exeter Psychology, not Imperial | the public release is band-passed and HRF-convolved; see below |
| same dataset's **EKG** | Lorenzo Pasquini, UCSF Neurology/Neuroscape | no email verified; UCSF Neurology | he has already published on it — see the next section |
| **Psilocybin 2012 (n=15)**, and the 4 LSD 2016 participants not in ds003059, Imperial | Robin Carhart-Harris | `r.carhart-harris@imperial.ac.uk` is the 2016 address and is **stale**; he is at UCSF Neurology / Neuroscape | senior author on both; a search summary attributed an "available on request from Carhart-Harris" statement to a 2025 paper from the group, which I did not verify against the paper |
| **PsiConnect, n=62 psilocybin** (Monash), multi-echo fMRI + EEG + diffusion, resting plus guided meditation, music and movie, half the cohort meditation-trained, follow-up to one year. <https://doi.org/10.1038/s41597-026-07312-1> | Adeel Razi, Monash | `adeel.razi@monash.edu` | OpenNeuro **ds006110** / NEMAR **on006110**. The largest single-site psychedelic dataset, and the only big one with EEG. **Checked 2026-10-12: the EEG is not simultaneous.** The cap is an actiCAP slim 64 (not an MR cap) at 500 Hz, each session holds one continuous `task-series` EEG file covering movie/rest/meditation/music, and the Nature paper describes separate EEG blocks. No physiology in the func directories. Still the best open drug dataset for a different reason: TR 0.91 s, MB4, **four echoes**, so the breathing band is inside the sampled range and an S0/T2* split is available |
| **ds006072 descriptor / replication sessions** (<https://doi.org/10.1038/s41597-025-05189-0>) | Subha Subramanian (corresponding); Joshua Siegel; Nico Dosenbach | `ssubram5@bidmc.harvard.edu`; `Joshua.Siegel@nyulangone.org` (current; `jssiegel@wustl.edu` was the 2024 address) | Siegel and Dosenbach are now both at the NYU Langone Center for Psychedelic Medicine |
| **Psilocybin n=23 (2020)** and **LSD n=25 (2018)**, Zurich | Katrin Preller; Franz Vollenweider | none verified; Dept of Adult Psychiatry and Psychotherapy, University of Zurich | checked 2026-10-11: the Bitbucket deposit for Preller et al. 2018 eLife (`katrinpreller/lsd-effects-on-global-brain-connectivity`, under Downloads) holds only two CIFTI files of per-scan global-brain-connectivity maps (24 subjects × 6 scans, with and without GSR; one value per grayordinate, no time series). The licence is non-commercial academic use. It is derived data and cannot be fitted; the raw data must be requested |
| **Psilocybin n=50**, Maastricht | Natasha Mason; Pablo Mallaroni; Johannes Ramaekers; Kim Kuypers | none verified; Dept of Neuropsychology and Psychopharmacology, Maastricht | |
| **Psilocybin n=19**, Johns Hopkins | Frederick Barrett | `fbarrett@jhmi.edu` (a Frontiers paper names him as the access contact) | Center for Psychedelic and Consciousness Research |
| **Psilocybin / LSD / mescaline n=32 (2023)** and **LSD n=50 (2018)**, Basel | Felix Müller; Matthias Liechti | none verified; Clinical Pharmacology, University Hospital Basel | |
| **Ayahuasca n=50 (2019 trial)**, UFRN | Fernanda Palhano-Fontes; Draulio Araujo | `fernandapalhano@neuro.ufrn.br` (Dataverse contact field for the 2015 deposit) | the 2015 data are open (above); the larger placebo-controlled trial is "on reasonable request" |
| **Copenhagen psilocybin** (NeuroPharm) | Patrick Fisher; Drummond McCulloch | none verified; Neurobiology Research Unit, Rigshospitalet | excluded from the mega-analysis for data-protection reasons; needs a DTA, not an email |

## The DMT dataset: what to ask for, and why it is better than it looked

The public release (`timmer500/DMT_Imaging`) is band-passed to roughly
0.01-0.08 Hz -- 0.00% of the power sits above 0.08 Hz -- and the EEG
files are HRF-convolved IRASA regressors, so an innovation-based
state-space fit cannot whiten either. The OSF HADES projections of the
same data carry the same signature. That is why the release is unusable
for us and why the ask has to go one step back in their own pipeline.

What makes it worth the trouble is something the release does not
advertise: **the study recorded EKG simultaneously with the fMRI, on a
hardware-synchronised clock.** Pasquini et al. (2025), *Imaging
Neuroscience* <https://doi.org/10.1162/imag_a_00541>, describes it --
two electrodes, BrainAmp MR amplifier, 5 kHz sampling, 250 Hz low-pass,
Brain Products SyncBox for EKG-MRI clock sync, 28-minute recordings, in
the 14 volunteers with the full multimodal acquisition. They used it to
show that the post-injection hippocampal and medial-parietal
deactivations correlate with **increased heart rate**.

So this one dataset has all three things the vascular-route question
needs -- BOLD, simultaneous EEG, and simultaneous EKG on a common clock
-- and the people who have already handled its physiology (Pasquini,
Kettner, Carhart-Harris) are at UCSF Neuroscape, not Imperial. Note also
that their own published result is the confound in miniature: a BOLD
deactivation that tracks heart rate. Whether that is neural or vascular
is exactly what `scripts/autonomic_confound_case.py` says BOLD alone
cannot decide.

## The standard ask

The same three items, in the same order, work for every site. Ordering
matters: the first is the blocking one, and asking for raw images first
invites a long conversation about governance that none of this needs.

1. **The parcellated or ROI time series before any band-pass**, with the
   acquisition-rate sampling and whatever units the pipeline produced
   them in (not z-scored per parcel, if that is avoidable). The
   band-pass, not the parcellation and not the sampling rate, is the item
   that makes a dataset unusable for generative modelling.
2. **The physiological recordings on the imaging clock** -- cardiac,
   respiratory, and pupil where they exist -- with the synchronisation
   method, since that is what separates a vascular drug effect from a
   neural one.
3. **The EEG features before HRF convolution**, where EEG exists.

Worth saying in the request, because it costs them nothing and is the
general lesson of this survey: of the public psychedelic fMRI datasets we
could reach, two are band-pass filtered beyond use for generative
modelling, and one of those was filtered upstream of the people who
shared it. Publishing the parcellated series one step before the
band-pass, and the regressors before convolution, preserves every
downstream option at no cost to the original analysis.

## Method test beds: drug-free simultaneous EEG-fMRI with physiology

Added 2026-10-12. The brain-body simulations
(`docs/brain_body_system_models.md`) say the fast channel is what makes
the topology *selectable* and a peripheral channel is what makes the
body pathway *identifiable*, so the method has to be exercised on
recordings that carry both, on the imaging clock. None of the open drug
datasets do. These drug-free ones do, and two are already on disk.

| dataset | EEG | physiology | BOLD | n | status | use |
|---|---|---|---|---|---|---|
| **NIMH / Bolt et al.** (Nat Neurosci 2025, "Autonomic physiological coupling of the global fMRI signal"; preprint PMC11312447) | simultaneous, supplied **preprocessed** (.mat; AAS + BCG template + ICA in Analyzer, ref FCz) | **PPG + respiration**, trimmed to the scan | multi-echo, 3 echoes | rest 11 subj / 15 scans; **cued deep-breath task** 6 subj / 9 scans; cue-plus-button control | figshare `10.6084/m9.figshare.28225415.v3`, CC BY 4.0, 262 files, **14.0 GB**; code `github.com/tsb46/fmri_arousal` | the deep-breath task is a **body-first perturbation with experimenter-known timing** and the button task is its brain-first control: the closest existing thing to the two drug-free conditions in our design. TR not stated in the record |
| **NATVIEW** (Telesford et al., Sci Data 2023, `10.1038/s41597-023-02458-8`) | simultaneous, 64 ch, 5 kHz, SyncBox to the MR clock | **ECG = EEG ch 32; respiration belt** (BIOPAC MP150, 62.5 Hz, `StartTime 0` against the scan trigger); eye tracking 250 Hz | TR 2.1 s, 3.5 mm, no multiband (12-ch coil) | 22, two sessions | FCP-INDI S3 `s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/raw_data/<sub>/<ses>/` per file (no-sign-request), or `raw_data_gz/<sub>.tar.gz` at 3.5-9.6 GB each; CC BY 4.0. **`~/Workspace/data/natview`**: sub-01 ses-01 rest (849 MB) plus whatever the 2026-10-13 pull added; `scripts/validate_natview_statespace.py` fits BOLD / EEG / pupil / respiration arms; results in `~/Workspace/data/vpjax_results/natview/` | rest 600 s; film runs 300-600 s (`inscapes`, `tp` = *The Present*, `dme` = Despicable Me, `monkey`) are a brain-first perturbation with a known onset; per-subject respiration and ECG on one clock |
| **ds003768** (Gu, Han, ... Duyn, Liu, Penn State) | simultaneous, 32 ch, 5 kHz | **ECG = ch 32**, EOG ch 31; no respiration | TR 2.1 s | 33 | OpenNeuro, CC0. **`~/Workspace/data/openneuro/ds003768` holds sub-01..06 (3.0 GB)**; fits for six subjects x three ROIs in `docs/results/ds003768_rest_run1_sub01-06.csv` and `~/Workspace/data/vpjax_results/ds003768/` | two 10-min rest runs plus 15-min sleep runs with staged epochs in `sourcedata/`; the accompanying paper is about autonomic-neural sequences at arousals |
| **ds007216** (Kucyi, Drexel) | simultaneous, 31 ch + 4 CWL, 5 kHz | **ECG = ch 32**; no respiration | TR 2.0 s, MB3, 2 mm | 24, two sessions, ~80 min each | OpenNeuro, open; not downloaded | the only one at our TR with CWL; derivatives include Bergen + CW-regressed EEG |

What the two on-disk sets have already shown (README, "On a recording"):
BOLD alone reproduces the simulation's pathology on real data; the alpha
loading comes out negative in 16 of 18 fits; and in the well-determined
subjects the joint fit pulls `tau_z` to 0.5 s, the envelope's own
timescale, because at TR 2.1 s BOLD cannot object. That is the reason to
prefer a short-TR dataset for the shared-timescale question, and why
PsiConnect's 0.91 s would have mattered had its EEG been simultaneous.

**Searched and not found** (2026-10-12): no open simultaneous EEG-fMRI
dataset with a pharmacological condition -- ketamine (an Auckland thesis
and an ANZCTR registration, no deposit), propofol or sevoflurane (TUM and
Hospital del Mar registrations, no deposit), caffeine, nicotine,
zolpidem, methylphenidate, alcohol. The Imperial DMT study remains the
only drug recording with all three channels, and it needs the request
above.

Suggested order: NATVIEW beyond sub-01 (on disk, respiration + ECG + a
film block, script exists); then the NIMH deep-breath task as the
body-first condition; then the DMT request.

## Adjacent: the CWL datasets

Not psychedelic, but they settle the carbon-wire-loop question that feeds
the hardware decision, and both are open.

- **ATR, 39 adults, two scanners, CWL caps, oddball + N-back + rest,
  inside and outside the scanner** (traveling-subject subset).
  Tsutsumi et al. (2026) *Sci Data*
  <https://doi.org/10.1038/s41597-026-06734-1>; BIDS, raw, public;
  the DOI `10.34860/atr-EfP-2025` resolves to the ATR Brain Information
  Communication Research Lab Group. Contacts: `t.ogawa@atr.jp`,
  `t.kuroda@atr.jp`. Participants explicitly consented to inter-institutional
  sharing and public release.
- **Drexel, 24 adults, multi-session simultaneous EEG-fMRI with CWL
  sensors**, ~1.33 h per person over two days, plus experience sampling.
  Shareef-Trudeau et al. (2026) <https://doi.org/10.64898/2026.02.04.703882>;
  OpenNeuro **ds007216**, open. PI Aaron Kucyi, Drexel.

## Sources

Dataset inventory and site attributions from Girn et al. (2026), Nature
Medicine, Table 1 and author list, via PubMed (PMID 41942645). Author
affiliations and corresponding addresses from PubMed records for PMIDs
36940333, 39020167, 40473634, 40533500, 40800803, 41639123, 41676676,
42168232, 25693169, 27071089. Repository metadata from the Zenodo and
Harvard Dataverse APIs.
