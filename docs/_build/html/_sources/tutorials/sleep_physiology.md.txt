# Sleep Physiology: State-Dependent Neurovascular Coupling

This tutorial explores vpjax's sleep package, which models how
neurovascular coupling changes across the five sleep stages and how
these changes drive the slow vascular oscillations that power
glymphatic clearance.

## Background

Sleep is not merely "the brain turned off." Each sleep stage produces
distinct hemodynamic signatures:

- **Wake**: Strong, fast NVC; high norepinephrine (NE) tone
- **N1/N2**: Progressively weaker NVC gain; spindle-related hemodynamic transients
- **N3 (deep NREM)**: Weakest NVC; large slow vasomotion at ~0.02 Hz driven by NE oscillations
- **REM**: NVC returns near wake levels; high cholinergic tone

The slow vasomotion during N3 drives CSF pulsations -- the glymphatic
system that clears metabolic waste from the brain.

## Sleep Stages and NVC Modulation

vpjax defines the five standard sleep stages and provides
stage-dependent Balloon-Windkessel parameters:

```python
from vpjax.sleep.nvc_state import (
    WAKE, N1, N2, N3, REM,
    balloon_params_for_stage,
    nvc_gain_for_stage,
)
from vpjax._types import BalloonParams

# All five stages are integer-coded
assert WAKE == 0
assert N1 == 1
assert N2 == 2
assert N3 == 3
assert REM == 4

# Each stage returns valid BalloonParams
for stage in [WAKE, N1, N2, N3, REM]:
    p = balloon_params_for_stage(stage)
    assert isinstance(p, BalloonParams)
    assert float(p.tau) > 0
```

### NVC Gain Decreases Through NREM

The key physiological prediction: neurovascular coupling weakens
as sleep deepens through NREM stages:

```python
g_wake = nvc_gain_for_stage(WAKE)
g_n1 = nvc_gain_for_stage(N1)
g_n3 = nvc_gain_for_stage(N3)
assert float(g_wake) > float(g_n1) > float(g_n3)
```

### REM Resembles Wake

REM sleep reactivates cortical circuits and NVC returns
closer to waking levels than to deep NREM:

```python
g_rem = nvc_gain_for_stage(REM)
assert abs(float(g_rem - g_wake)) < abs(float(g_rem - g_n3))
```

### Continuous Sleep Depth

For modeling smooth transitions between stages, vpjax provides
a continuous interpolation parameterized by sleep depth
(0 = wake, 1 = deep NREM):

```python
import jax
import jax.numpy as jnp
from vpjax.sleep.nvc_state import nvc_gain_continuous

g0 = nvc_gain_continuous(jnp.array(0.0))    # wake
g05 = nvc_gain_continuous(jnp.array(0.5))   # light sleep
g1 = nvc_gain_continuous(jnp.array(1.0))    # deep NREM
assert float(g0) > float(g05) > float(g1)

# Fully differentiable -- can optimize sleep depth
g = jax.grad(nvc_gain_continuous)(jnp.array(0.5))
assert jnp.isfinite(g)
assert float(g) < 0  # gain decreases with depth
```

### Riera Model Parameters Per Stage

The multi-compartment Riera model also adapts to sleep stage,
with reduced NO coupling gain during deep sleep:

```python
from vpjax.sleep.nvc_state import riera_params_for_stage
from vpjax.hemodynamics.riera import RieraParams

p_wake = riera_params_for_stage(WAKE)
p_n3 = riera_params_for_stage(N3)
assert isinstance(p_wake, RieraParams)
assert float(p_n3.c_no) < float(p_wake.c_no)
```

## Norepinephrine-Driven Vasomotion

During NREM sleep, locus coeruleus (LC) norepinephrine exhibits
slow oscillations (~0.02 Hz, period ~50 s) that drive large-amplitude
vasomotion -- rhythmic contraction and dilation of cerebral arterioles.

### NE Oscillation

```python
from vpjax.sleep.vasomotion import norepinephrine_oscillation

t = jnp.linspace(0, 100, 1000)  # 100 seconds
ne = norepinephrine_oscillation(t)

# NE oscillates (not flat)
assert float(jnp.std(ne)) > 0.01

# ~2 full cycles in 100s (0.02 Hz)
crossings = jnp.sum(jnp.abs(jnp.diff(jnp.sign(ne - jnp.mean(ne)))) > 0)
assert 2 <= int(crossings) <= 8
```

### CBV Vasomotion

The NE oscillation drives CBV changes around the baseline:

```python
from vpjax.sleep.vasomotion import VasomotionParams, cbv_vasomotion

t = jnp.linspace(0, 100, 1000)
cbv = cbv_vasomotion(t)

# Oscillates around CBV = 1.0
assert float(jnp.mean(cbv)) == pytest.approx(1.0, abs=0.05)
assert float(jnp.std(cbv)) > 0.001
```

### Amplitude Scaling

Larger NE amplitude produces larger CBV oscillations:

```python
t = jnp.array([25.0])
small = VasomotionParams(cbv_amplitude=jnp.array(0.01))
large = VasomotionParams(cbv_amplitude=jnp.array(0.05))

cbv_s = cbv_vasomotion(t, small)
cbv_l = cbv_vasomotion(t, large)
assert abs(float(cbv_l[0]) - 1.0) > abs(float(cbv_s[0]) - 1.0)
```

### Wake vs. NREM Vasomotion

Vasomotion is minimal during wake and strongest during deep NREM:

```python
t = jnp.linspace(0, 100, 1000)
wake_p = VasomotionParams(cbv_amplitude=jnp.array(0.005))
nrem_p = VasomotionParams(cbv_amplitude=jnp.array(0.03))

std_wake = float(jnp.std(cbv_vasomotion(t, wake_p)))
std_nrem = float(jnp.std(cbv_vasomotion(t, nrem_p)))
assert std_nrem > std_wake
```

### BOLD Vasomotion Signal

The vasomotion produces a low-frequency BOLD fluctuation
(~0.02 Hz) that is observable in sleeping-state fMRI:

```python
from vpjax.sleep.vasomotion import bold_vasomotion

t = jnp.linspace(0, 200, 2000)
bold = bold_vasomotion(t)
assert float(jnp.std(bold)) > 0.0001
```

## CSF Coupling and Glymphatic Clearance

The CBV oscillations from vasomotion drive cerebrospinal fluid (CSF)
pulsations through the Virchow-Robin perivascular spaces. This is the
forward model for the glymphatic system.

### CSF Flow from CBV

```python
from vpjax.sleep.csf_coupling import csf_flow_from_cbv

cbv = jnp.array([1.0, 1.02, 1.04, 1.02, 1.0])
csf = csf_flow_from_cbv(cbv)
assert csf.shape == cbv.shape
assert float(jnp.std(csf)) > 0
```

### Delayed CSF Response

CSF flow lags behind CBV changes due to fluid inertia and
compliance of the perivascular spaces:

```python
from vpjax.sleep.csf_coupling import csf_flow_from_cbv_delayed

t = jnp.linspace(0, 100, 1000)
cbv = 1.0 + 0.03 * jnp.sin(2 * jnp.pi * 0.02 * t)
csf = csf_flow_from_cbv_delayed(cbv, t)
assert float(jnp.std(csf)) > 0
```

### Glymphatic Clearance

Cumulative CSF flow estimates total glymphatic waste clearance:

```python
from vpjax.sleep.csf_coupling import glymphatic_clearance

t = jnp.linspace(0, 300, 3000)  # 5 minutes

# NREM-like oscillation
cbv_nrem = 1.0 + 0.03 * jnp.sin(2 * jnp.pi * 0.02 * t)
clearance_nrem = glymphatic_clearance(cbv_nrem, t)
assert float(clearance_nrem) > 0

# No oscillation (wake-like) -> minimal clearance
cbv_flat = jnp.ones_like(t)
clearance_flat = glymphatic_clearance(cbv_flat, t)
assert float(clearance_flat) < 0.01

# Stronger oscillation -> more clearance
cbv_strong = 1.0 + 0.05 * jnp.sin(2 * jnp.pi * 0.02 * t)
clearance_strong = glymphatic_clearance(cbv_strong, t)
assert float(clearance_strong) > float(clearance_nrem)
```

### Differentiability

The entire sleep-glymphatic pipeline is differentiable, enabling
optimization of vasomotion amplitude for maximal clearance:

```python
def loss(amp):
    t = jnp.linspace(0, 100, 1000)
    cbv = 1.0 + amp * jnp.sin(2 * jnp.pi * 0.02 * t)
    return glymphatic_clearance(cbv, t)

g = jax.grad(loss)(jnp.array(0.03))
assert jnp.isfinite(g)
```

## The Full Sleep Pipeline

Putting it all together, the sleep forward model is:

$$
\text{Sleep stage} \xrightarrow{\text{NVC modulation}} \text{NE oscillation}
\xrightarrow{\text{vasomotion}} \text{CBV}
\xrightarrow{\text{CSF coupling}} \text{Glymphatic clearance}
$$

Each arrow is a differentiable function, so the entire pipeline can be
inverted: given measured BOLD or ASL time-series during sleep, infer
the underlying sleep physiology.

## References

- Fultz NE et al. (2019). Coupled electrophysiological, hemodynamic, and cerebrospinal fluid oscillations in human sleep. *Science* 366:628-631.
- Hauglund NL et al. (2020). Meningeal lymphangiogenesis and enhanced glymphatic activity in mice lacking locus coeruleus norepinephrine. *J Exp Med* 217:e20201986.
- Ozbay PS et al. (2018). Contribution of systemic vascular effects to fMRI activity in white matter. *NeuroImage* 176:541-549.
