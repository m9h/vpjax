# Multi-Compartment Neurovascular Coupling (Riera Model)

This tutorial introduces the Riera neurovascular coupling model, which
extends the single-compartment Balloon-Windkessel with a multi-compartment
vascular tree and dual vasodilatory pathways.

## Background

The standard Balloon-Windkessel model lumps all vasculature into a single
venous compartment. The Riera model {cite:p}`riera2006nonlinear,riera2007nonlinear`
decomposes this into three compartments reflecting vascular anatomy:

- **Arteriolar** ($v_a$): compliant, actively regulated by NO and adenosine
- **Capillary** ($v_c$): stiff, site of oxygen exchange
- **Venous** ($v_v$): compliant, passive outflow -- where the BOLD signal originates

### Dual Vasodilatory Pathways

Two distinct signaling pathways drive vasodilation:

1. **Nitric Oxide (NO)**: Fast, neurally driven. Interneurons release NO
   directly onto arteriolar smooth muscle. Time constant $\sim 1/\kappa_\text{NO} \approx 1.5$ s.

2. **Adenosine**: Slower, metabolically driven. When CMRO$_2$ rises, ATP
   consumption produces adenosine, which diffuses to arterioles.
   Time constant $\sim 1/\kappa_\text{ade} \approx 2.5$ s.

### The ODE System

The 8-dimensional state vector is:

$$
\mathbf{y} = (s_\text{NO},\; s_\text{ade},\; f_a,\; v_a,\; v_c,\; v_v,\; q_v,\; \text{CMRO}_2)
$$

The dynamics:

$$
\begin{aligned}
\dot{s}_\text{NO} &= c_\text{NO}\,u - \kappa_\text{NO}\,s_\text{NO} - \gamma_\text{NO}\,(f_a - 1) \\
\dot{s}_\text{ade} &= c_\text{ade}\,(\text{CMRO}_2 - 1) - \kappa_\text{ade}\,s_\text{ade} - \gamma_\text{ade}\,(f_a - 1) \\
\dot{f}_a &= s_\text{NO} + s_\text{ade} \\
\tau_a\,\dot{v}_a &= f_a - v_a^{1/\alpha_a} \\
\tau_c\,\dot{v}_c &= v_a^{1/\alpha_a} - v_c^{1/\alpha_c} \\
\tau_v\,\dot{v}_v &= v_c^{1/\alpha_c} - v_v^{1/\alpha_v} \\
\tau_v\,\dot{q}_v &= \frac{f_a\,E(f_a)}{E_0} - \frac{v_v^{1/\alpha_v}\,q_v}{v_v} \\
\tau_m\,\dot{\text{CMRO}}_2 &= (1 + \phi\,u) - \text{CMRO}_2
\end{aligned}
$$

Blood cascades from arterioles through capillaries to veins, each with
its own compliance exponent $\alpha$ and transit time $\tau$.

## Setting Up

```python
import jax
import jax.numpy as jnp

from vpjax.hemodynamics.riera import (
    RieraNVC, RieraParams, RieraState,
    riera_total_cbv, riera_to_balloon,
)
from vpjax._types import BalloonState
from vpjax.hemodynamics.bold import observe_bold
```

## Steady State

At rest, all fractional volumes are 1.0 and vasodilatory signals are zero:

```python
y0 = RieraState.steady_state()

model = RieraNVC(params=RieraParams())
dy = model(jnp.array(0.0), y0, jnp.array(0.0))

# All derivatives should be zero
assert jnp.allclose(dy.s_no, 0.0, atol=1e-6)
assert jnp.allclose(dy.s_ade, 0.0, atol=1e-6)
assert jnp.allclose(dy.f_a, 0.0, atol=1e-6)
assert jnp.allclose(dy.v_a, 0.0, atol=1e-6)
assert jnp.allclose(dy.v_v, 0.0, atol=1e-6)
assert jnp.allclose(dy.q_v, 0.0, atol=1e-6)
```

Total CBV (weighted sum of compartments) is unity at baseline:

```python
cbv = riera_total_cbv(y0)
assert jnp.allclose(cbv, 1.0, atol=1e-6)
```

The Balloon-equivalent mapping also gives (1, 1):

```python
v, q = riera_to_balloon(y0)
assert jnp.allclose(v, 1.0, atol=1e-6)
assert jnp.allclose(q, 1.0, atol=1e-6)
```

## Stimulus Response

### NO Pathway Responds First

Apply a unit stimulus and check that the NO vasodilatory signal
responds immediately:

```python
dy = model(jnp.array(0.0), y0, jnp.array(1.0))
assert float(dy.s_no) > 0.0  # NO signal rises with stimulus
```

### CMRO$_2$ Increases

Metabolic demand (CMRO$_2$) rises toward a new steady state:

```python
assert float(dy.cmro2) > 0.0
```

### Producing BOLD

Integrate the Riera ODE with Euler steps, then convert to
Balloon-equivalent variables for BOLD observation:

```python
model = RieraNVC(params=RieraParams())
y = RieraState.steady_state()

dt = 0.01
for _ in range(500):  # 5 seconds of stimulation
    dy = model(jnp.array(0.0), y, jnp.array(1.0))
    y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)

# Map multi-compartment state to single-compartment BOLD
v, q = riera_to_balloon(y)
pseudo_state = BalloonState(s=jnp.array(0.0), f=y.f_a, v=v, q=q)
bold = observe_bold(pseudo_state)

# Should have deviated from baseline
assert jnp.abs(bold) > 0.001
```

### CMRO$_2$ Dynamics

After sustained stimulation, CMRO$_2$ rises above baseline:

```python
y = RieraState.steady_state()
dt = 0.01
for _ in range(300):
    dy = model(jnp.array(0.0), y, jnp.array(1.0))
    y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)

assert float(y.cmro2) > 1.0  # above resting level
```

## Multi-Compartment CBV

The `riera_total_cbv` function computes a weighted sum of compartmental
volumes using typical resting fractions:

$$
\text{CBV}_\text{total} = 0.20\,v_a + 0.05\,v_c + 0.75\,v_v
$$

This reflects the anatomical reality that venous vasculature contains
~75% of cerebral blood volume.

## The CMRO$_2$ Hierarchy

The Riera model provides the **Level 3 (dynamic)** estimate in vpjax's
three-level CMRO$_2$ hierarchy:

| Level | Method | Spatial resolution | Temporal resolution |
|-------|--------|--------------------|---------------------|
| 1     | TRUST + pCASL | Global | Static |
| 2     | qBOLD + pCASL | Regional (per-voxel) | Static |
| 3     | Riera model | Regional | **Dynamic** |

```python
# Level 3: Track CMRO2 over time
cmro2_trace = []
y = RieraState.steady_state()
for _ in range(200):
    dy = model(jnp.array(0.0), y, jnp.array(1.0))
    y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)
    cmro2_trace.append(float(y.cmro2))

# CMRO2 should rise progressively
assert cmro2_trace[-1] > cmro2_trace[0]
```

## Differentiability

Gradients flow through the Riera model, enabling parameter estimation.
For example, compute $\partial s_\text{NO} / \partial c_\text{NO}$:

```python
y0 = RieraState.steady_state()

def loss(c_no):
    p = RieraParams(c_no=c_no)
    m = RieraNVC(params=p)
    dy = m(jnp.array(0.0), y0, jnp.array(1.0))
    return dy.s_no

g = jax.grad(loss)(jnp.array(1.0))
assert jnp.isfinite(g)
assert float(g) != 0.0
```

## Parameters

The default parameters come from Riera et al. (2006, 2007) and
Sotero & Trujillo-Barreto (2007):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `kappa_no` | $\kappa_\text{NO}$ | 0.65 | NO signal decay rate (s$^{-1}$) |
| `kappa_ade` | $\kappa_\text{ade}$ | 0.40 | Adenosine signal decay rate (s$^{-1}$) |
| `c_no` | $c_\text{NO}$ | 1.0 | NO coupling gain |
| `c_ade` | $c_\text{ade}$ | 0.5 | Adenosine coupling gain |
| `tau_a` | $\tau_a$ | 0.5 | Arteriolar transit time (s) |
| `tau_c` | $\tau_c$ | 1.0 | Capillary transit time (s) |
| `tau_v` | $\tau_v$ | 2.0 | Venous transit time (s) |
| `alpha_a` | $\alpha_a$ | 0.20 | Arteriolar Grubb exponent |
| `alpha_c` | $\alpha_c$ | 0.10 | Capillary Grubb exponent |
| `alpha_v` | $\alpha_v$ | 0.32 | Venous Grubb exponent |
| `E0` | $E_0$ | 0.34 | Resting oxygen extraction fraction |
| `phi` | $\phi$ | 0.5 | Metabolic-neural coupling strength |
| `tau_m` | $\tau_m$ | 3.0 | Metabolic response time constant (s) |

## References

- Riera JJ et al. (2006). Nonlinear local electrovascular coupling. I: A theoretical model. *HBM* 27:896-914.
- Riera JJ et al. (2007). Nonlinear local electrovascular coupling. II: From data to neuronal masses. *NeuroImage* 36:1179-1196.
- Sotero RC, Trujillo-Barreto NJ (2007). Biophysical model for integrating neuronal activity, EEG, fMRI and metabolism. *NeuroImage* 36:671-687.
