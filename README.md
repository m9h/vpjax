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
