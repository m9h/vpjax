# vpjax -- Virtual Physiology in JAX

**Differentiable models of cerebrovascular physiology.**

vpjax maps neural activity through metabolic demand and blood flow to
multi-modal brain signals (BOLD, ASL, qBOLD, VASO), fully differentiable in JAX.
It is the vascular/metabolic complement to
[vbjax](https://github.com/ins-amu/vbjax) (Virtual Brain in JAX).

```text
vbjax:  neural mass model -> synaptic activity -> coupling
  |
vpjax:  activity -> CMRO2 (metabolic demand)
        -> vasodilation (NO, adenosine, neural signaling)
        -> CBF (vessel compliance, autoregulation)
        -> blood volume, oxygenation (Balloon model)
        -> BOLD signal (T2* forward model)
        -> ASL signal (kinetic model)
        -> qBOLD signal (multi-echo GRE -> regional OEF)
```

## Installation

```bash
uv pip install vpjax
# With all optional dependencies:
uv pip install "vpjax[full,validation]"
```

## Tutorials

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/balloon_windkessel
tutorials/sleep_physiology
tutorials/riera_nvc
tutorials/cardiac_coupling
```

## API Reference

```{toctree}
:maxdepth: 2
:caption: API Reference

reference/vpjax
```

## Project

```{toctree}
:maxdepth: 1
:caption: Project

changelog
contributing
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
