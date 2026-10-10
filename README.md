# vpjax — Virtual Physiology in JAX

**The vascular/metabolic complement to [vbjax](https://github.com/ins-amu/vbjax) (Virtual Brain in JAX).**

vpjax provides differentiable models of cerebrovascular physiology:
neural activity → metabolic demand → blood flow → BOLD/ASL/qBOLD signals.

```
vbjax:  neural mass model → synaptic activity → coupling
  ↓
vpjax:  activity → CMRO₂ (metabolic demand)
        → vasodilation (NO, adenosine, neural signaling)
        → CBF (vessel compliance, autoregulation)
        → blood volume, oxygenation (Balloon model)
        → BOLD signal (T2* forward model)
        → ASL signal (kinetic model)
        → qBOLD signal (multi-echo GRE → regional OEF)
```

## Why vpjax?

vbjax's `make_bold()` provides a simplified Balloon-Windkessel hemodynamic model — adequate for BOLD fitting but missing:

- **Metabolic intermediaries** (oxygen consumption, lactate, ATP)
- **Vascular geometry** (vessel density, compliance vary across cortex)
- **ASL forward model** (predict pCASL signal, not just BOLD)
- **Multi-modal constraints** (MRS, qMRI, angiography inform the physiology)
- **Regional OEF** from quantitative BOLD (qBOLD) biophysical model
- **Local Linearization** integration for stiff neurovascular ODEs

vpjax fills these gaps with models from Riera (neurovascular coupling), Bulte (quantitative BOLD/qBOLD), Germuska (calibrated fMRI), and Lu (TRUST), fully differentiable in JAX.

## Architecture

```
vpjax/
├── pharmacokinetics/   # Dose → plasma → effect site → receptor occupancy
│   ├── dosing.py       # Dose events; static times, differentiable amounts
│   ├── compartments.py # 1-3 compartment models, closed form (matrix exp)
│   ├── occupancy.py    # Effect-site lag, Hill/Emax, regional weighting
│   ├── error.py        # Combined additive + proportional residual error
│   ├── inversion.py    # Per-subject fit (Optimistix Levenberg-Marquardt)
│   └── nlme.py         # Population model (NumPyro, optional)
│
├── statespace/         # Latent state from multirate, asynchronous sensors
│   ├── propagators.py  # LL mean step; Van Loan process noise
│   ├── filters.py      # LL filter + RTS smoother, masked observations
│   ├── multirate.py    # Asynchronous streams → union grid + masks
│   └── estimation.py   # Innovation likelihood, identifiability, diagnostics
│
├── hemodynamics/       # Balloon-Windkessel → extended Riera model
│   ├── balloon.py      # Standard B-W (from vbjax, reference)
│   ├── riera.py        # Full neurovascular coupling (Riera 2006/2007)
│   ├── bold.py         # T2* forward model (multi-echo aware)
│   └── optics.py       # Optical properties for fNIRS/DOT (dot-jax interface)
│
├── metabolism/          # Neural activity → metabolic demand
│   ├── cmro2.py        # CMRO₂ from neural activity
│   ├── oef.py          # Oxygen extraction fraction dynamics
│   └── fick.py         # Fick's principle: CMRO₂ = CBF × OEF × CaO₂
│
├── vascular/           # Blood vessel models
│   ├── compliance.py   # Vessel compliance (pressure-volume, Grubb)
│   ├── autoregulation.py # Cerebral autoregulation (Lassen curve)
│   └── geometry.py     # Vascular tree morphometry
│
├── perfusion/          # ASL forward/inverse models
│   ├── asl.py          # Simple ASL observation (CBF change)
│   ├── vaso.py         # Simple VASO observation (CBV change)
│   ├── kinetic.py      # Buxton general kinetic model (pCASL signal)
│   ├── trust.py        # TRUST: venous T2 → SvO₂ (Lu 2008)
│   └── calibration.py  # M0, blood T1 calibration
│
├── qbold/              # Quantitative BOLD (Bulte, He & Yablonskiy)
│   ├── signal_model.py # Multi-echo GRE signal: S(TE) = f(OEF, DBV, R2, S0)
│   ├── oef_mapping.py  # Per-voxel OEF from multi-echo data
│   ├── dbv.py          # Deoxygenated blood volume estimation
│   └── calibrated.py   # Gas-free calibrated BOLD (Bulte/Davis)
│
├── qsm/                # Quantitative Susceptibility Mapping
│   ├── susceptibility.py # χ from iron, myelin, blood oxygenation
│   ├── r2star_fitting.py # R2* from multi-echo GRE magnitude
│   └── phase.py        # Multi-echo phase combination & frequency mapping
│
├── vaso/               # VAscular Space Occupancy (CBV measurement)
│   ├── signal_model.py # SS-SI-VASO signal: S ∝ (1 - CBV)
│   ├── boco.py         # BOLD contamination correction
│   ├── devein.py       # Ascending vein removal (layer-specific)
│   └── cbv_mapping.py  # ΔCBV/CBV₀ estimation
│
├── layers/             # Cortical depth-resolved physiology
│   ├── layering.py     # Equivolume layer definition (LAYNII/Nighres)
│   ├── profiles.py     # Depth-dependent sampling of volumetric maps
│   ├── iron_myelin.py  # Iron/myelin separation (R2* + QSM + BPF)
│   └── layer_nvc.py    # Layer-specific neurovascular coupling
│
├── integrators/        # ODE integration methods
│   └── local_linearization.py  # LL filter (Riera/Ozaki)
│
├── presets.py          # 3T/7T parameter bundles & pipeline helpers
└── tests/
```

### qBOLD: Regional OEF from Multi-Echo GRE

The key upgrade from global OEF (TRUST) to regional OEF. The qBOLD signal model:

```
S(TE) = S₀ · F(OEF, DBV, TE) · exp(-R₂·TE)

where F(OEF, DBV, TE) models the intravoxel frequency distribution
from randomly oriented deoxygenated blood vessels (He & Yablonskiy 2007).

At short TE: signal dominated by R₂ (tissue T2)
At long TE:  signal dominated by R₂' (reversible, from deoxy-Hb)
The difference: R₂' = R₂* - R₂ ∝ OEF × DBV × B₀
```

WAND provides 7-echo GRE (TE = 5-35ms) — ideal for fitting this model.
Combined with pCASL CBF: CMRO₂(regional) = CBF(regional) × OEF(regional) × CaO₂

This replaces the limitation of TRUST (global OEF only) with per-voxel OEF.

### Cortical Layer-Resolved Physiology (LAYNII)

WAND ses-06 structural data is **sub-millimeter** (0.67-0.7mm isotropic) — sufficient for cortical depth analysis (~2-3 layers across the ~2.5mm cortical ribbon). Combined with LAYNII or Nighres:

```
vpjax/
├── layers/                 # Cortical depth-resolved physiology
│   ├── layering.py         # Equivolume layer definition (LAYNII/Nighres interface)
│   ├── profiles.py         # Depth-dependent sampling of any volumetric map
│   ├── iron_myelin.py      # Separate iron (paramagnetic) from myelin (diamagnetic)
│   └── layer_nvc.py        # Layer-specific neurovascular coupling
│                             (feedforward=deep layers, feedback=superficial)
```

**What WAND provides per cortical layer:**

| Map | From | Measures at each depth |
|---|---|---|
| Quantitative T1 | MP2RAGE (0.7mm) | Myelination gradient |
| R2* | MEGRE magnitude (0.67mm) | Combined iron + myelin |
| QSM | MEGRE **phase** (0.67mm, currently unused!) | Iron (+) vs myelin (−) separated |
| BPF | QMT (ses-02, lower res) | Direct myelin content |
| T1w/T2w ratio | ses-03 | Myelin proxy — validate per layer vs QMT |

**Iron-myelin separation per layer** from R2* + QSM + BPF is the killer application — no single contrast can do this alone.

**Connection to vpjax:** Layer-specific neurovascular coupling parameters differ between superficial cortex (feedback connections, layer I-III) and deep cortex (feedforward, layer V-VI). This maps directly onto Valdes-Sosa's ξ-αNET feedforward/feedback hierarchy and the geodesic cortical flow directions from Liu et al. 2026.

### VASO (planned, for CBV measurement)

```
vpjax/
├── vaso/                   # VAscular Space Occupancy
│   ├── signal_model.py     # SS-SI-VASO signal: S ∝ (1 - CBV)
│   ├── boco.py             # BOLD contamination correction
│   ├── devein.py           # Ascending vein removal (layer-specific)
│   └── cbv_mapping.py      # ΔCBV/CBV₀ estimation
```

VASO measures **cerebral blood volume (CBV)** directly — the missing Balloon-Windkessel state variable. WAND does not currently have VASO, but if added:

```
Complete Balloon-Windkessel observation:
  s (vasodilatory signal) → inferred from model
  f (CBF)                 → pCASL ✓
  v (CBV)                 → VASO (planned)
  q (deoxy-Hb content)    → qBOLD ✓

With VASO: zero free hemodynamic parameters — all observables measured.
```

**LAYNII** (Huber et al. 2021) provides VASO-specific processing: `LN_BOCO` (BOLD correction), `LN2_DEVEIN` (ascending vein removal), `LN_LEAKY_LAYERS` (inter-layer leakage model). Processing tools in LAYNII or via Neurodesk container.

## Key Innovation

**Every variable in the model has an independent measurement** (using the WAND dataset):

| Model variable | vpjax module | WAND measurement |
|---|---|---|
| Neural activity | (from vbjax) | MEG |
| Metabolic demand | `metabolism/cmro2.py` | pCASL × TRUST |
| Blood flow | `hemodynamics/riera.py` | pCASL CBF |
| Oxygen extraction | `metabolism/oef.py` | TRUST SvO₂ |
| BOLD signal | `hemodynamics/bold.py` | fMRI |
| Vascular geometry | `vascular/geometry.py` | Angiography |
| Neurotransmitters | (constraint) | MRS GABA/Glu |
| Myelination | (constraint) | QMT BPF |
| Conduction velocity | (constraint, from sbi4dwi) | AxCaliber |

## Drug Studies: the PK/PD Layer

For a pharmacological experiment the measured signal change is not the
quantity of interest — it is the convolution of a concentration time
course with a neural effect and a vascular effect, and those two are not
separable from BOLD alone. The `pharmacokinetics/` subpackage supplies
the top of that chain so the rest can be identified:

| Layer | Module | What it contributes |
|---|---|---|
| Dose → plasma | `pharmacokinetics/compartments.py` | Exact concentration time course, closed form |
| Plasma → effect site | `pharmacokinetics/occupancy.py` | The lag that resolves concentration–effect hysteresis |
| Occupancy → regional drive | `pharmacokinetics/occupancy.py` | Receptor density turns a scalar into a spatial pattern |
| Drive → neural activity | (vbjax neural mass) | Modulated excitability |
| Drive → vascular tone | `hemodynamics/riera.py` | Direct vasoactive effect, independent of neural |
| States → signals | `hemodynamics/bold.py`, `perfusion/` | BOLD, ASL, VASO observation models |

One drive entering both the neural and the vascular model is the point:
fitting EEG and BOLD jointly against a shared concentration curve is
what distinguishes a neural effect from a vasoactive one. Fitting them
separately cannot.

Conventions follow NONMEM and Pumas — clearance-and-volume
parameterisation, log-normal between-subject variability, combined
residual error, closed-form linear disposition — so that any fit can be
cross-checked against an established estimator. That check is worth
doing: a PK model with a mis-scaled volume still produces a plausible
curve, and agreement with an independent implementation is the cheapest
way to find out whether the model is the one intended.

```python
import jax
jax.config.update("jax_enable_x64", True)   # concentrations span orders of magnitude

from vpjax.pharmacokinetics import (
    TWO_COMPARTMENT, PKParams, infusion, concentration,
    effect_site_concentration, occupancy, EffectSiteParams,
)

params = PKParams(CL=..., V1=..., Q2=..., V2=...)   # from published PK
regimen = infusion(amount=..., duration=...)
cp = concentration(params, regimen, t, TWO_COMPARTMENT)
ce = effect_site_concentration(cp, dt=dt, params=EffectSiteParams(ke0=...))
occ = occupancy(ce, ec50=..., gamma=...)
```

`examples/pkpd_drive.py` runs the whole chain from a JSON of explicit
assumptions and records them in the output, so a downstream fit cannot
be mistaken for an independent measurement of the PK.

## Fusing EEG and fMRI: the State-Space Layer

EEG and fMRI observe the same latent state at rates three orders of
magnitude apart. Fitting them separately and comparing the results
discards the constraint that makes a neurovascular model identifiable;
upsampling the slow modality to the fast grid invents data. The
`statespace/` subpackage estimates the shared latent state directly, from
whichever channels are present at each instant.

```python
from vpjax.statespace import build_observation_grid, fit_statespace, parameter_uncertainty

grid = build_observation_grid(
    {
        "eeg_delta": {"time_s": t_eeg,  "values": delta_power, "noise_sd": eeg_sd},
        "bold":      {"time_s": t_bold, "values": bold_rois,   "noise_sd": bold_sd},
    },
    max_dt=0.25,          # keep the linearization fresh between volumes
)

fit = fit_statespace(build, grid, init=theta0, restarts=12)
unc = parameter_uncertainty(build, grid, fit["log_theta"])
```

Three things this module insists on, each because the obvious
alternative fails quietly:

- **Nothing is interpolated.** Each channel carries a presence mask and
  an absent channel contributes nothing. Observations are assimilated as
  sequential scalar updates, which for diagonal `R` is identical to the
  joint update but makes masking and the per-channel likelihood
  decompose exactly.
- **Multi-start is not optional.** In the Balloon model `kappa` and
  `tau` both set the width of the impulse response, so the likelihood
  has a ridge. A single gradient-based fit started off the ridge slides
  along it and reports convergence several hundred log-likelihood units
  below the truth.
- **Whiteness is necessary, not sufficient.** `residual_diagnostics`
  tells you whether the model described the data. It does not tell you
  whether the parameters that did so are the right ones — a flat
  direction lets badly wrong parameters whiten the residuals perfectly.
  `parameter_uncertainty` is the check that catches it, from the
  curvature of the likelihood, and it needs no ground truth.

### What simultaneous EEG actually buys

`scripts/validate_statespace_recovery.py` fits the same simulated run
twice. At resting-state amplitude over 300 s (TR 2 s, 10 Hz EEG envelope
at SNR 3):

| | κ | τ | τ_z | q_z | verdict |
|---|---|---|---|---|---|
| BOLD only, relative SE | 0.27 | 0.22 | 17.3 | 34.6 | not identifiable |
| BOLD + EEG, relative SE | 0.03 | 0.03 | 0.13 | 0.06 | identifiable |

The BOLD-only arm's innovations are white (variance 0.97) and its
parameters are wrong by three orders of magnitude on `q_z`: from 151
volumes, a fast large-amplitude drive through a slow balloon is
indistinguishable from a slow small drive through a fast one. So the
fast channel is not buying precision here, it is buying identifiability.

That figure describes the estimator under this model, not any recording.
A model misspecified in a way EEG cannot see would show the same benefit
here and none in practice, so treat it as an upper bound.

### Simultaneous, or just EEG?

The obvious objection is that EEG recorded in a separate session, outside
the bore, would supply the same constraint without the gradient and
pulse artifacts. `scripts/simultaneity_case.py` tests that over several
seeds (three here; 360 s, TR 2 s, 10 Hz envelope, 4 restarts):

| arm | identifiable | κ | τ | τ_z | q_z |
|---|---|---|---|---|---|
| BOLD only | 0/3 | 0.21 | 0.18 | 3.5 | 7.1 |
| BOLD + EEG, envelope SNR 3 | 3/3 | 0.024 | 0.027 | 0.12 | 0.05 |
| BOLD + EEG, envelope SNR 1 | 3/3 | 0.040 | 0.042 | 0.14 | 0.11 |
| BOLD + EEG, envelope SNR 0.25 | 3/3 | 0.11 | 0.12 | 0.24 | 0.34 |
| BOLD + EEG, envelope SNR 0.1 | 0/3 | 0.18 | 0.23 | 0.39 | 0.67 |
| BOLD + EEG, envelope SNR 0.05 | 0/3 | 0.20 | 0.19 | 0.73 | 1.4 |
| BOLD only, τ_z known | 3/3 | 0.21 | 0.27 | — | 0.24 |
| BOLD only, τ_z and q_z known | 3/3 | 0.18 | 0.28 | — | — |

(median relative standard errors across seeds.)

Two things follow. The in-scanner envelope does not have to be clean: at
SNR 0.25, mostly noise after artifact removal, it still identifies every
parameter, with precision degrading smoothly by about 4× across the
sweep; the breakeven lies between 0.25 and 0.1, where the likelihood is
still curved in every direction but the drive diffusion's standard
error passes 50%. And a separate session, idealised as perfect knowledge of the
drive's statistics, restores formal identifiability but leaves the
hemodynamic parameters at 20–28% — worse than simultaneous EEG at SNR
0.25, and 5–7× worse than at SNR 1. What a separate session cannot
supply is the drive's *realisation*, and that is what resolves the
κ–τ ridge. For a drug whose effect is a single non-stationary pass
through one run there is no trial structure to average over, so the
realisation is the only thing there is.

The same script scores the drug contrast directly: a shape-only input
(unit-peak two-exponential, every time scale from the command line)
enters the drive with a gain `beta`, and `beta` is estimated alongside
the rest. With BOLD alone `beta` was identifiable in one seed of three,
at a relative SE of 0.33; with the envelope it was identifiable in all
three at 0.12. The fast channel does not just sharpen the baseline
parameters, it is what makes the drug effect an estimable quantity.

### On a recording

`scripts/validate_ds003768_statespace.py` runs the same two arms on a
10-minute resting run from OpenNeuro ds003768 (32-channel Brain Products
EEG at 5 kHz inside a 3T Prisma, TR 2.1 s), using the global BOLD mean
and the occipital alpha envelope at 10 Hz after gradient and pulse
artifact subtraction (`validation/eeg_artifacts.py`, a deliberately
plain AAS baseline). Nothing is known here, so the noise variances and
the EEG loading are estimated too. On sub-01, run 1:

- **BOLD only** reproduces the simulation's pathology on real data: the
  innovations are white (variance 0.99, lag-1 0.00) and the fit is not
  identifiable (curvature ratio −2e-5), with the drive collapsed to white
  noise (τ_z = 0.02 s, q_z = 24, relative SEs 3.4 and 6.8).
- **BOLD + EEG with one shared state** is rejected by whiteness: the fit
  pins the state to the envelope, drives the EEG noise variance to zero,
  and leaves the BOLD innovations autocorrelated at 0.89. The alpha
  envelope has structure at 0.1–0.3 s that no drive passing through the
  Balloon can share with a 2 s BOLD series.
- **BOLD + EEG with an EEG-specific nuisance state** (a fast OU process
  the BOLD never sees; the default in `eeg_fmri_statespace`) is
  calibrated on both channels (BOLD 0.99 / 0.01, EEG 1.00 / 0.11) and
  248 log-likelihood units better. The shared drive comes out at
  τ_z = 2.2 s with a negative alpha loading, as the alpha–BOLD literature
  would predict, but its parameters are weakly determined (relative SEs
  0.3–0.9) and the arm is not identifiable as a whole: the EEG noise
  variance is redundant with a 0.18 s nuisance process sampled at 0.1 s.

The useful number is the one the last arm implies. The shared drive
accounts for about 10% of the envelope's SD — roughly 1% of its
variance — so the *effective* SNR of a global alpha envelope with respect
to the drive a global BOLD series sees is around 0.1 — right at the
sweep's breakeven. That is the gap between the simulation and a recording,
and it is a property of the global average at rest, not of the method:
regional pairings (occipital alpha against visual cortex) and a run with
a drive worth the name — a task, or a drug — raise the shared fraction,
and the identifiability machinery says by how much before anyone is
scanned. Whether a protocol clears the bar is now a computation with
inputs the lab controls.

## Dependencies

- JAX
- vbjax (neural mass models)
- jaxtyping, equinox (optional, for typed modules)

### QSM Pipeline (from WAND phase data)

The 7-echo GRE **phase** data (`*_part-phase_MEGRE.nii.gz`) in ses-06 is currently unused — it enables **Quantitative Susceptibility Mapping (QSM)**:

```
Multi-echo GRE phase data
  → Phase unwrapping (ROMEO / Laplacian)
  → Background field removal (V-SHARP / PDF / LBV)
  → Dipole inversion (TKD / MEDI / TGV-QSM / STAR-QSM)
  → Susceptibility map (χ in ppm)
```

**QSM tools (via Neurodesk):**

| Tool | Language | Method | Install |
|---|---|---|---|
| **QSMxT** | Python/Nextflow | Automated BIDS pipeline, multiple algorithms | Neurodesk `qsmxt` |
| **SEPIA** | MATLAB | Comprehensive, GUI + scripting | Neurodesk |
| **ROMEO** | Julia | Fast phase unwrapping (Dymerska et al.) | Neurodesk |
| **TGV-QSM** | Python | Total generalized variation regularization | Neurodesk |
| **MEDI** | MATLAB | Morphology-enabled dipole inversion (Cornell) | Neurodesk |
| **STI Suite** | MATLAB | Susceptibility tensor imaging (Stanford) | Manual |

**QSMxT** is recommended — it's BIDS-aware, runs the full pipeline automatically, and is in Neurodesk. Produces susceptibility maps in the same space as the magnitude T2* maps.

**Why QSM matters for vpjax:**
- Susceptibility χ = χ_iron (paramagnetic, +) + χ_myelin (diamagnetic, −)
- Combined with R2* and QMT BPF → full iron/myelin decomposition per voxel
- Layer-resolved QSM (via LAYNII) → iron/myelin per cortical depth
- Iron content in basal ganglia changes with age → connects to the Valdes-Sosa lifespan model

## The CMRO₂ Hierarchy

vpjax provides three levels of oxygen metabolism estimation, each more spatially specific:

| Level | Method | OEF | CBF | CMRO₂ | Data needed |
|---|---|---|---|---|---|
| **1. Global** | TRUST + pCASL | Global (sagittal sinus) | Regional | CBF×OEF_global | pCASL + TRUST |
| **2. Regional** | qBOLD + pCASL | **Per-voxel** (multi-echo) | Regional | CBF×OEF_regional | pCASL + 7-echo GRE |
| **3. Dynamic** | Riera model | Time-varying | Time-varying | Full dynamics | MEG + fMRI + ASL |

WAND provides data for all three levels in the same subjects.

## References

- Riera JJ et al. (2006/2007). Nonlinear local electrovascular coupling. HBM.
- Buxton RB et al. (1998). Dynamics of blood flow and oxygenation changes during brain activation. MRM.
- Friston KJ et al. (2000). Nonlinear responses in fMRI: the Balloon model. NeuroImage.
- Germuska M et al. Dual-calibrated fMRI (CUBRIC).
- Lu H, Ge Y (2008). TRUST MRI: quantitative assessment of venous oxygenation. MRM.
- Bulte DP et al. Quantitative BOLD and gas-free calibrated fMRI (Oxford WIN).
- He X, Yablonskiy DA (2007). Quantitative BOLD: mapping of human cerebral deoxygenated blood volume and oxygen extraction fraction. MRM.
- Ozaki T (1992). Local linearization approach. Statistica Sinica.
- Valdes-Sosa PA et al. (2000). Variable resolution electromagnetic tomography (VARETA).

## License

Apache 2.0
