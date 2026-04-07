# From Neural Activity to Multi-Modal Brain Signals

This tutorial walks through the core vpjax pipeline: stimulating the
Balloon-Windkessel hemodynamic model and observing the response through
three complementary MRI modalities -- BOLD, ASL, and VASO.

## Background

The Balloon-Windkessel model {cite:p}`buxton1998dynamics,friston2000nonlinear`
describes how neural activity drives changes in cerebral blood flow and volume,
which in turn determine the MRI signals we measure. The state vector is:

$$
\mathbf{y} = \begin{pmatrix} s \\ f \\ v \\ q \end{pmatrix}
$$

where $s$ is the vasodilatory signal, $f$ is blood inflow (CBF), $v$ is blood
volume (CBV), and $q$ is deoxyhemoglobin content.

### The ODE

The dynamics are governed by:

$$
\begin{aligned}
\dot{s} &= u - \kappa\, s - \gamma\,(f - 1) \\
\dot{f} &= s \\
\tau\,\dot{v} &= f - v^{1/\alpha} \\
\tau\,\dot{q} &= \frac{f\,E(f)}{E_0} - \frac{v^{1/\alpha}\,q}{v}
\end{aligned}
$$

where $u(t)$ is the neural stimulus, $\kappa$ is the signal decay rate,
$\gamma$ is the flow-dependent elimination, $\tau$ is the venous transit time,
$\alpha$ is the Grubb exponent, and $E(f) = 1 - (1-E_0)^{1/f}$ is the
flow-dependent oxygen extraction.

## Setting Up

```python
import jax
import jax.numpy as jnp

from vpjax._types import BalloonParams, BalloonState
from vpjax.hemodynamics.balloon import BalloonWindkessel, solve_balloon
from vpjax.hemodynamics.bold import BOLDParams, observe_bold
from vpjax.perfusion.asl import observe_asl
from vpjax.perfusion.vaso import observe_vaso
```

## Steady State

At rest (zero stimulus), all state variables sit at their baseline values:

```python
y0 = BalloonState.steady_state()
assert float(y0.s) == 0.0   # no vasodilatory drive
assert float(y0.f) == 1.0   # baseline flow
assert float(y0.v) == 1.0   # baseline volume
assert float(y0.q) == 1.0   # baseline deoxyhemoglobin

# The RHS should be zero at baseline
model = BalloonWindkessel(params=BalloonParams())
dy = model(jnp.array(0.0), y0, jnp.array(0.0))
assert jnp.allclose(dy.s, 0.0, atol=1e-6)
```

All observation functions also return zero at rest -- they report
*changes* from baseline:

```python
assert jnp.allclose(observe_bold(y0), 0.0, atol=1e-6)
assert jnp.allclose(observe_asl(y0), 0.0, atol=1e-6)
assert jnp.allclose(observe_vaso(y0), 0.0, atol=1e-6)
```

## Simulating a Block Stimulus

Apply a 3-second block stimulus and integrate for 30 seconds:

```python
dt = 0.01
T = 30.0
n = int(T / dt)

# 3s block starting at t=1s
stim = jnp.zeros(n).at[int(1.0 / dt):int(4.0 / dt)].set(1.0)
ts, traj = solve_balloon(BalloonParams(), stim, dt=dt)
```

The `solve_balloon` helper uses Diffrax's Tsit5 solver internally,
interpolating the stimulus as a piecewise-constant control signal.

## Multi-Modal Observations

Now observe the hemodynamic trajectory through three MRI modalities:

```python
bold = observe_bold(traj)   # T2*-weighted: sensitive to dHb
asl = observe_asl(traj)     # Perfusion-weighted: sensitive to CBF
vaso = observe_vaso(traj)   # Volume-weighted: sensitive to CBV
```

### Key Physiological Signatures

**BOLD** shows a positive response (reduced deoxyhemoglobin concentration
as increased flow washes out dHb):

```python
assert float(jnp.max(bold)) > 0.001
```

**ASL** shows a positive response (increased blood flow):

```python
assert float(jnp.max(asl)) > 0.01
```

**VASO** shows a *negative* response (increased blood volume reduces the
available water signal):

```python
assert float(jnp.min(vaso)) < -0.001
```

### Hemodynamic Lag

A critical prediction of the Balloon model: BOLD peaks *after* ASL because
BOLD depends on the slow venous volume and dHb dynamics, while ASL
directly reflects blood flow:

```python
t_bold_peak = ts[jnp.argmax(bold)]
t_asl_peak = ts[jnp.argmax(asl)]
# ASL peaks before (or at the same time as) BOLD
assert float(t_asl_peak) <= float(t_bold_peak) + 1.0
```

### Return to Baseline

After the stimulus ends, all signals return to rest:

```python
assert jnp.abs(bold[-1]) < 0.01
assert jnp.abs(asl[-1]) < 0.01
assert jnp.abs(vaso[-1]) < 0.01
```

## Differentiability

Because vpjax uses JAX and Equinox throughout, gradients flow through
the entire pipeline. For example, compute the gradient of peak BOLD
with respect to the signal decay rate $\kappa$:

```python
def peak_bold(kappa):
    params = BalloonParams(kappa=kappa)
    dt = 0.05
    n = int(20.0 / dt)
    stim = jnp.zeros(n).at[int(1.0 / dt):int(4.0 / dt)].set(1.0)
    _, traj = solve_balloon(params, stim, dt=dt)
    bold = observe_bold(traj)
    return jnp.max(bold)

grad_fn = jax.grad(peak_bold)
g = grad_fn(jnp.array(0.65))
assert jnp.isfinite(g)
assert float(g) != 0.0
```

This enables parameter estimation via gradient descent -- fitting the
hemodynamic model to measured BOLD/ASL/VASO data.

## Summary

| Modality | Observation function | Sensitive to | Sign during activation |
|----------|---------------------|-------------|----------------------|
| BOLD     | `observe_bold()`    | dHb (q, v)  | Positive             |
| ASL      | `observe_asl()`     | CBF (f)     | Positive             |
| VASO     | `observe_vaso()`    | CBV (v)     | Negative             |

The Balloon-Windkessel model provides the foundation for all of vpjax.
The next tutorial extends this to the multi-compartment Riera model with
explicit metabolic coupling.

## References

- Buxton RB et al. (1998). Dynamics of blood flow and oxygenation changes during brain activation: the Balloon model. *MRM* 39:855-864.
- Friston KJ et al. (2000). Nonlinear responses in fMRI: the Balloon model, Volterra kernels, and other hemodynamics. *NeuroImage* 12:466-477.
- Stephan KJ et al. (2007). Comparing hemodynamic models with DCM. *NeuroImage* 38:387-401.
