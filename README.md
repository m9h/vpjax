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
│   └── bold.py         # T2* forward model (multi-echo aware)
│
├── metabolism/          # Neural activity → metabolic demand
│   ├── cmro2.py        # CMRO₂ from neural activity
│   ├── oef.py          # Oxygen extraction fraction dynamics
│   └── fick.py         # Fick's principle: CMRO₂ = CBF × OEF × CaO₂
│
├── vascular/           # Blood vessel models
│   ├── compliance.py   # Vessel compliance (pressure-volume)
│   ├── autoregulation.py # Cerebral autoregulation
│   └── geometry.py     # Vascular tree from angiography
│
├── perfusion/          # ASL forward/inverse models
│   ├── kinetic.py      # Buxton general kinetic model (ASL signal)
│   ├── trust.py        # TRUST: venous T2 → SvO₂ (Lu 2008)
│   └── calibration.py  # M0, blood T1 calibration
│
├── qbold/              # Quantitative BOLD (Bulte, He & Yablonskiy)
│   ├── signal_model.py # Multi-echo GRE signal: S(TE) = f(OEF, DBV, R2, S0)
│   ├── oef_mapping.py  # Per-voxel OEF from multi-echo data
│   ├── dbv.py          # Deoxygenated blood volume estimation
│   └── calibrated.py   # Gas-free calibrated BOLD (Bulte)
│
├── integrators/        # SDE integration methods
│   └── local_linearization.py  # LL filter (Riera/Ozaki)
│
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
