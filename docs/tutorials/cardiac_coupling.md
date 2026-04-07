# Heart-Brain Coupling: Vagal and Baroreceptor Models

This tutorial covers vpjax's cardiac package, which models bidirectional
heart-brain interactions:

- **Brain to Heart**: the frontal-vagal pathway (DLPFC stimulation causes
  heart rate deceleration via the vagus nerve)
- **Heart to Brain**: baroreceptor-mediated modulation of cortical excitability
  (systolic blood pressure pulses inhibit cortical processing)
- **Heart to BOLD**: cardiac pulsatility as a confound in fMRI signals

## The Vagal Pathway

### Background

Stimulation of the dorsolateral prefrontal cortex (DLPFC) -- for example
via transcranial magnetic stimulation (TMS) -- activates a descending
pathway through the medial prefrontal cortex to the vagus nerve. Increased
vagal tone slows the heart.

### Basic Vagal Response

```python
import jax
import jax.numpy as jnp

from vpjax.cardiac.vagal import (
    VagalParams, VagalState, VagalODE,
    vagal_hr_response, hr_to_rr_interval,
)
```

The simplest interface: neural input to heart rate change:

```python
# No input -> no HR change
hr_change = vagal_hr_response(jnp.array(0.0))
assert jnp.allclose(hr_change, 0.0, atol=1e-6)

# DLPFC stimulation -> HR deceleration (negative)
hr_change = vagal_hr_response(jnp.array(1.0))
assert float(hr_change) < 0.0
```

### The Vagal ODE

For dynamic modeling (e.g., simulating TMS pulse trains), use the
full ODE system:

```python
model = VagalODE(params=VagalParams())
y0 = VagalState.steady_state()

# At rest, derivatives are zero
dy = model(jnp.array(0.0), y0, jnp.array(0.0))
assert jnp.allclose(dy.vagal_tone, 0.0, atol=1e-6)
assert jnp.allclose(dy.hr_deviation, 0.0, atol=1e-6)
```

### TMS Entrainment

Simulate an intermittent theta burst (iTBS)-like protocol: 2 seconds
on, 8 seconds off (~0.1 Hz):

```python
model = VagalODE(params=VagalParams())
y = VagalState.steady_state()
dt = 0.01

hr_trace = []
for i in range(3000):  # 30 seconds
    t = i * dt
    # iTBS-like: 2s on, 8s off
    stim = jnp.where((t % 10.0) < 2.0, 1.0, 0.0)
    dy = model(jnp.array(t), y, stim)
    y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)
    hr_trace.append(float(y.hr_deviation))

hr = jnp.array(hr_trace)

# HR should oscillate with the stimulation pattern
assert float(jnp.std(hr[1000:])) > 0.01
# Should show deceleration
assert float(jnp.min(hr)) < 0.0
```

### RR Interval Conversion

Convert heart rate deviation to RR interval (useful for
comparison with ECG-derived metrics):

```python
rr_base = hr_to_rr_interval(jnp.array(0.0))     # baseline ~857 ms
rr_decel = hr_to_rr_interval(jnp.array(-5.0))    # 5 bpm slower

# Slower HR -> longer RR interval
assert float(rr_decel) > float(rr_base)
```

### Differentiability

The vagal model supports gradient-based parameter estimation:

```python
g = jax.grad(lambda gain: vagal_hr_response(
    jnp.array(1.0), VagalParams(gain=gain)
))(jnp.array(0.5))
assert jnp.isfinite(g)
assert float(g) != 0.0
```

## The Baroreceptor Pathway

### Background

Arterial baroreceptors in the carotid sinus and aortic arch fire during
systole, sending inhibitory signals to the cortex via the nucleus tractus
solitarius (NTS). This produces a cardiac-phase-dependent modulation of
perception and neural excitability.

### Cortical Excitability

```python
from vpjax.cardiac.baroreceptor import (
    BaroreceptorParams, cortical_excitability,
    modulate_neural_drive, arterial_pressure,
)
```

Excitability is lowest during peak systole (~0.3$\pi$ after the R-peak)
and highest during diastole (~$\pi$):

```python
exc_systole = cortical_excitability(jnp.array(0.3 * jnp.pi))
exc_diastole = cortical_excitability(jnp.array(jnp.pi))
assert float(exc_systole) < float(exc_diastole)
```

The modulation is periodic with the cardiac cycle:

```python
exc_0 = cortical_excitability(jnp.array(0.0))
exc_2pi = cortical_excitability(jnp.array(2 * jnp.pi))
assert jnp.allclose(exc_0, exc_2pi, atol=1e-4)
```

### Modulating Neural Drive

The baroreceptor effect can modulate the effective neural input
to the hemodynamic model:

```python
drive = jnp.array(1.0)  # constant neural input

mod_systole = modulate_neural_drive(
    drive, cardiac_phase=jnp.array(0.3 * jnp.pi)
)
mod_diastole = modulate_neural_drive(
    drive, cardiac_phase=jnp.array(jnp.pi)
)

# Systolic inhibition reduces effective drive
assert float(mod_systole) < float(mod_diastole)
```

### Arterial Pressure Waveform

Generate a realistic arterial blood pressure waveform from cardiac phase:

```python
phases = jnp.linspace(0, 2 * jnp.pi, 100)
bp = jax.vmap(arterial_pressure)(phases)

# Systolic peak > diastolic trough
assert float(jnp.max(bp)) > float(jnp.min(bp))
# Physiological range (60-140 mmHg)
assert float(jnp.min(bp)) > 40
assert float(jnp.max(bp)) < 180
```

## Cardiac Pulsatility Confound

### Background

The beating heart transmits pressure waves through the cerebral
vasculature. These produce ~1 Hz fluctuations in CBV and therefore
in BOLD and ASL signals -- a confound that must be modeled or removed.

### CBV Pulsation

```python
from vpjax.cardiac.pulsatility import (
    PulsatilityParams, cbv_pulsation,
    bold_cardiac_confound, asl_cardiac_confound,
)

phases = jnp.linspace(0, 2 * jnp.pi, 100)
cbv = jax.vmap(cbv_pulsation)(phases)

# CBV oscillates around baseline
assert float(jnp.max(cbv)) > 1.0
assert float(jnp.min(cbv)) < 1.0
```

### BOLD Cardiac Confound

```python
t = jnp.linspace(0, 10, 10000)  # 10 seconds at 1 ms
bold_confound = bold_cardiac_confound(t)

# Nonzero variance
assert float(jnp.std(bold_confound)) > 0.0001
# Zero-mean (it's a confound, not a signal)
assert abs(float(jnp.mean(bold_confound))) < 0.01
```

### ASL Cardiac Confound

ASL is also affected by cardiac pulsatility, which modulates
the labeling efficiency and transit times:

```python
asl_confound = asl_cardiac_confound(t)
assert float(jnp.std(asl_confound)) > 0.0001
```

### Compliance Scaling

Higher vascular compliance amplifies the pulsatile volume changes:

```python
low_comp = PulsatilityParams(compliance=jnp.array(0.01))
high_comp = PulsatilityParams(compliance=jnp.array(0.05))

amp_low = float(cbv_pulsation(jnp.array(0.0), low_comp) - 1.0)
amp_high = float(cbv_pulsation(jnp.array(0.0), high_comp) - 1.0)
assert abs(amp_high) > abs(amp_low)
```

## Connecting to the Hemodynamic Model

The cardiac models integrate with the Balloon-Windkessel pipeline:

1. **Baroreceptor modulation**: Scale the neural input by cardiac-phase-dependent
   excitability before feeding it to the Balloon ODE
2. **Pulsatility confound**: Add the cardiac BOLD confound to the
   hemodynamic BOLD prediction for realistic simulated data
3. **Vagal feedback**: Model the complete loop where cortical activity
   (from vbjax) drives vagal tone, which modulates heart rate, which
   changes the baroreceptor rhythm, which feeds back to cortical excitability

```text
Cortex (vbjax)
  |
  +---> Balloon ODE (vpjax.hemodynamics) ---> BOLD
  |       ^
  |       | baroreceptor modulation
  |       |
  +---> Vagal ODE (vpjax.cardiac) ---> HR change
          |                                |
          +--- RR interval --------> cardiac phase
                                          |
                                    baroreceptor firing
                                          |
                                    cortical excitability
```

## References

- Thayer JF et al. (2012). A meta-analysis of heart rate variability and neuroimaging studies. *Neurosci Biobehav Rev* 36:747-756.
- Garfinkel SN et al. (2014). Knowing your own heart: Distinguishing interoceptive accuracy from interoceptive awareness. *Biol Psychol* 104:65-74.
- Azzalini D et al. (2019). Visceral signals shape brain dynamics and cognition. *Trends Cogn Sci* 23:488-509.
- Chang C et al. (2009). Influence of heart rate on the BOLD signal. *NeuroImage* 44:857-869.
