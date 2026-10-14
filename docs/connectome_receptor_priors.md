# Connectome and receptor priors for the E/I block

Written 2026-10-14. The E/I latent block (`vpjax/neural/ei_block.py`) needs a
graph for its modes and a spatial pattern for where an external time course
enters. None of the datasets on disk ships diffusion imaging, so both come from
public group resources. This note records what was pulled, under what licence,
what was derived from it, and one caveat that decides which graph to use.

## Sources pulled (each in its own directory under `~/Workspace/data/`)

| resource | what | licence | local path |
|---|---|---|---|
| Xia et al. 2026, *Commun Biol* 9:1339 (10.1038/s42003-026-10558-5); code+data `jiexfmri/MSFD_GSP_analysis`, Zenodo 10.5281/zenodo.20558532 | group multiscale structural connectome (MSC) on HCP-MMP1.0, 360×360, from 100 HCP subjects: tractography strength (TS), inverse geodesic distance (GD), microstructural profile covariance (MPC), rank-normalised and fused by normalised-angle similarity; all seven combinations in `HCP_MMP_Fusion_all.mat`; hierarchy table (first FC gradient, BigBrain gradient, cognitive PC) | CC BY 4.0 (Zenodo record) | `msfd_gsp/` |
| Hansen et al. 2022, *Nat Neurosci* 25:1569 (`netneurolab/hansen_receptors`) | PET tracer maps in MNI152: 5-HT1A (WAY, CUMI), 5-HT1B, 5-HT2A (altanserin, Cimbi-36), 5-HT4, 5-HT6, 5-HTT (DASB) | CC BY-NC-SA 4.0; cite the source studies (their Table S3) | `hansen_receptors/` |
| Mills 2016, figshare 10.6084/m9.figshare.3498446 | HCP-MMP1.0 on fsaverage, `lh/rh.HCP-MMP1.annot` (181 labels incl. `???`) | CC BY 4.0 | `hcp_mmp_fsaverage/` |
| neuromaps 0.0.7 (Markello 2022) | MNI152→fsaverage registration fusion; installed in `.venv` | BSD-3 | `~/neuromaps-data/` |
| the paper PDF | | publisher OA | `papers/s42003-026-10558-5.pdf` |

Not pulled: raw HCP (registration at db.humanconnectome.org is the user's to do;
the MSC above already carries what we need from it), MICA-MICs (OSF `j532r`,
public; 50 subjects with DWI and qT1 MPC, useful later for a second group graph
or per-subject variability, not for our subjects), BigBrain, AHBA.

## What was built

- `vpjax/neural/connectome.py`: `load_msc` (any of the seven matrices,
  symmetrised, zero diagonal), `laplacian_modes` (normalised or combinatorial,
  eigenvalues non-positive with the constant mode first, as gnf-jax expects),
  `hcp_mmp_names`, `parcellate_fsaverage`, `mode_weights`,
  `spectral_concentration`. Region order is Glasser's, left 1..180 then right;
  checked against the hierarchy table (regions 0 and 180 sit at the sensory end
  of the first gradient).
- `scripts/receptor_modes.py`: MNI152 → fsaverage 164k → parcel means → z-score
  → modal coefficients on the chosen graph. Outputs
  `docs/results/receptor_hcp_mmp.csv` (360 × 8 regional densities) and
  `docs/results/receptor_hcp_mmp.json` (eigenvalues and per-map mode weights,
  ready for `EIBlock.input_vector(entry, mode_weights=...)`).
- Tests: `vpjax/tests/test_connectome.py` (ring-graph spectra, round trip of
  the modal projection, MSC load and orthonormality, annotation names and
  parcellation identity; data-dependent tests skip when the files are absent).

## The caveat that decides the graph

The fused MSC is 99.7% dense with affinities in [0, 0.94], i.e. close to a
complete graph. Its normalised-Laplacian spectrum is 0, −0.94, −0.96, −0.99,
… −1.00: every non-constant mode is nearly degenerate, and its first mode does
not even separate the hemispheres (|r| = 0.01 with hemisphere). That is fine for
Xia et al., who use the eigenvectors only as an ordered basis for reconstruction.
It is not fine for gnf-jax, whose kernels are functions of the eigenvalues
(`exp(t λ)` and relatives): on a near-degenerate spectrum every mode gets the
same kernel weight and the field has no spatial structure. The tractography
matrix alone (`SC`, 12% density) has a proper spectrum (0, −0.17, −0.22, −0.26,
…, −1.22) and its first mode is the hemispheric split (|r| = 0.82).

So: **use `SC` (or a distance-thresholded variant) for the E/I block's modes and
kernels; use the fused `SC_MPC_GD` where a basis for projecting BOLD is all that
is wanted.** Both are one `key=` away in `load_msc`.

## Where the serotonin maps sit on the modes

Fraction of each z-scored map's energy in the first ten modes:

| map | fused MSC | SC |
|---|---|---|
| 5-HT1A (WAY, n=36) | 0.46 | 0.21 |
| 5-HT1A (CUMI, n=8) | 0.42 | 0.22 |
| 5-HT2A (altanserin, n=19) | 0.43 | 0.36 |
| 5-HT2A (Cimbi-36, n=29) | 0.20 | 0.15 |
| 5-HTT (DASB, n=100) | 0.52 | 0.37 |
| 5-HT1B / 5-HT4 / 5-HT6 | 0.22 / 0.29 / 0.34 | 0.10 / 0.21 / 0.25 |

The 1A and transporter maps are smooth (low-mode) patterns; the Cimbi-36 2A map
is the patchiest of the set. On the SC graph a spatially structured drug entry
would therefore need tens of modes, not the first few. That is a design input
for how many modes the block carries, which is a cost question (two states per
mode in the filter).

## From receptor pharmacology to entry points

This is the bridge the user asked for between "what does 5-MeO-DMT target" and
the model, kept at the level of *which row and which spatial pattern* an
external time course perturbs. It contains no pharmacokinetic quantity; the
block's drivers remain externally supplied with a cited `source`.

Binding data from ChEMBL for mebufotenin (CHEMBL7257; 124 activity records,
rodent and human, 1982–2003 *J Med Chem* / *Bioorg Med Chem Lett*), Ki unless
noted:

| target | affinity | note |
|---|---|---|
| 5-HT1A | 7.8 nM; 20 nM ([3H]8-OH-DPAT) | highest-affinity site; agonist |
| 5-HT6 (human) | 16 nM (two assays) | |
| 5-HT7 | 30 nM | |
| 5-HT1 (cortex, [3H]5-HT) | 50 nM | |
| 5-HT1B | 90 nM; 350 nM | |
| 5-HT2A | 15–21 nM with an agonist radioligand (DOI); 390–1600 nM with antagonist radioligands (ketanserin, MDL 100907) | the agonist/antagonist-label gap is the high-affinity G-protein-coupled state; functionally a lower-potency partial agonist than at 1A |

Reading for the block, with the usual caveats that affinity is not occupancy
and rodent is not human:

- **1A is the primary target** by one to two orders of magnitude over 2A on
  antagonist-labelled affinity. 5-HT1A is Gi/o-coupled: somatodendritic
  autoreceptors on raphe neurons (reduced serotonergic tone globally) and
  postsynaptic receptors that hyperpolarise pyramidal cells and some
  interneurons, densest in limbic and temporal cortex. In the block that is a
  *negative* perturbation of the excitatory drive row (`entry="p"`, or a
  reduction of `alpha_ee` if the effect is on recurrent gain), with the 1A map's
  modal coefficients as `mode_weights`.
- **2A is the secondary target**, Gq-coupled, depolarising layer-5 pyramidal
  cells; densest in association cortex. In the block, a *positive* perturbation
  of `p` or `alpha_ee` with the 2A map's coefficients. This is the psilocybin /
  LSD route, and the reason the two compound classes are expected to differ in
  spatial pattern as much as in sign.
- 5-HT6 and 5-HT7 affinities are in the same range as 1A and are usually
  ignored in the psychedelic literature; their maps (5-HT6 is in the set) make
  that an assumption we can test rather than take.
- Two time courses, one per receptor pattern, at the same named entry point
  with opposite signs, is the smallest honest representation. Which of the two
  dominates the BOLD and EEG response is then a fitted quantity, and the two
  maps' different spectral content (above) is what would let the data tell them
  apart.

What this note does not do: assign any magnitude, time course or occupancy to
either pattern. Those come from the externally supplied driver with its own
source, exactly as before.
