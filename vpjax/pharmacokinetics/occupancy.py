"""Pharmacodynamics: plasma concentration to receptor occupancy to drive.

Three layers sit between the PK model and the neural/vascular models
elsewhere in vpjax:

1. An **effect-site compartment**.  Plasma concentration is not the
   quantity the brain responds to; the effect site lags it, and the lag
   is identifiable from the hysteresis between concentration and effect.
   A single first-order transfer with rate ``ke0`` is the standard
   device (Sheiner 1979; Hull 1978), and it is what makes concentration
   and effect collapse onto one curve.

2. A **Hill (Emax) occupancy curve** mapping effect-site concentration
   to fractional receptor occupancy.

3. A **regional weighting** by receptor density, so that a scalar
   occupancy becomes a spatial drive pattern.  Density maps come from
   PET atlases; the module takes them as an argument and makes no
   assumption about their provenance or normalisation beyond what is
   documented on each function.

The output of layer 3 is the quantity to hand to a vbjax neural mass as
a modulation of excitability, and to the vpjax vascular models as a
modulation of tone — the same drive entering both, which is the whole
point of fitting EEG and BOLD jointly rather than separately.

References
----------
Sheiner LB et al. (1979) Clin Pharmacol Ther 25:358-371
    "Simultaneous modeling of pharmacokinetics and pharmacodynamics"
Hull CJ et al. (1978) Br J Anaesth 50:1113-1123
    "A pharmacodynamic model for pancuronium"
Holford NHG, Sheiner LB (1981) Clin Pharmacokinet 6:429-453
    "Understanding the dose-effect relationship"
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class EffectSiteParams(eqx.Module):
    """First-order plasma-to-effect-site transfer.

    Attributes
    ----------
    ke0 : effect-site equilibration rate constant (1/time).  The
          effect-site half-time is ``ln 2 / ke0``.
    """

    ke0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.5))


def effect_site_concentration(
    conc: Float[Array, "N"],
    dt: float | Float[Array, "N-1"],
    params: EffectSiteParams | None = None,
    ce0: Float[Array, ""] | float = 0.0,
) -> Float[Array, "N"]:
    """Filter a plasma concentration series into an effect-site series.

    Solves ``dCe/dt = ke0 (Cp - Ce)`` exactly under a zero-order hold on
    ``Cp`` within each sample interval, which is the right assumption
    when *conc* comes from a sampled schedule rather than a closed form:

        Ce[k+1] = Ce[k] e^{-ke0 dt} + Cp[k] (1 - e^{-ke0 dt})

    Parameters
    ----------
    conc   : plasma concentration, shape (N,)
    dt     : scalar sample interval, or per-interval lengths, shape (N-1,)
    params : EffectSiteParams
    ce0    : initial effect-site concentration

    Returns
    -------
    Effect-site concentration, shape (N,)
    """
    if params is None:
        params = EffectSiteParams()

    n = conc.shape[0]
    dts = jnp.broadcast_to(jnp.asarray(dt, dtype=conc.dtype), (n - 1,))

    def step(ce, inputs):
        cp, h = inputs
        decay = jnp.exp(-params.ke0 * h)
        ce_next = ce * decay + cp * (1.0 - decay)
        return ce_next, ce_next

    ce_init = jnp.asarray(ce0, dtype=conc.dtype)
    _, ces = jax.lax.scan(step, ce_init, (conc[:-1], dts))
    return jnp.concatenate([ce_init[None], ces])


class HillParams(eqx.Module):
    """Sigmoid Emax occupancy curve.

    Attributes
    ----------
    ec50 : concentration producing half the maximal effect
    gamma: Hill coefficient — steepness of the curve
    emax : maximal effect.  Leave at 1.0 for fractional occupancy; set
           it to a signal amplitude when the curve is being used as a
           direct effect model rather than an occupancy model.
    e0   : baseline effect at zero concentration
    """

    ec50: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    gamma: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    emax: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    e0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.0))


def hill_effect(
    conc: Float[Array, "..."],
    params: HillParams | None = None,
) -> Float[Array, "..."]:
    """Evaluate ``e0 + emax C^gamma / (ec50^gamma + C^gamma)``.

    Computed in log space so that a large Hill coefficient does not
    overflow, and guarded at ``C = 0`` where the gradient with respect
    to *gamma* is otherwise undefined.
    """
    if params is None:
        params = HillParams()

    safe = jnp.maximum(conc, 1e-30)
    ratio = jnp.exp(params.gamma * (jnp.log(safe) - jnp.log(params.ec50)))
    frac = jnp.where(conc > 0.0, ratio / (1.0 + ratio), 0.0)
    return params.e0 + params.emax * frac


def occupancy(
    conc: Float[Array, "..."],
    ec50: Float[Array, "..."],
    gamma: Float[Array, "..."] | float = 1.0,
) -> Float[Array, "..."]:
    """Fractional receptor occupancy in [0, 1].

    A thin wrapper on :func:`hill_effect` with ``emax = 1``, ``e0 = 0``,
    kept separate because occupancy has a fixed range and should not be
    given a free amplitude during fitting.
    """
    return hill_effect(
        conc,
        HillParams(
            ec50=jnp.asarray(ec50),
            gamma=jnp.asarray(gamma, dtype=float),
            emax=jnp.array(1.0),
            e0=jnp.array(0.0),
        ),
    )


def regional_drive(
    occ: Float[Array, "N"],
    density: Float[Array, "R"],
    normalise: bool = True,
) -> Float[Array, "R N"]:
    """Spread a scalar occupancy time course over regions by receptor density.

    Parameters
    ----------
    occ       : occupancy time course, shape (N,)
    density   : regional receptor density, shape (R,).  Any monotone
                measure will do — binding potential, SUVR, or
                normalised atlas values.
    normalise : divide *density* by its mean, so that the regional
                average drive equals *occ* and the amplitude of the
                drive is not confounded with the units of the atlas.

    Returns
    -------
    Regional drive, shape (R, N)
    """
    w = density / jnp.mean(density) if normalise else density
    return w[:, None] * occ[None, :]


def competitive_drive(
    occ_a: Float[Array, "N"],
    occ_b: Float[Array, "N"],
    density_a: Float[Array, "R"],
    density_b: Float[Array, "R"],
    weight_a: Float[Array, ""] | float = 1.0,
    weight_b: Float[Array, ""] | float = -1.0,
) -> Float[Array, "R N"]:
    """Combine two receptor systems with opposite-signed regional drives.

    Exists because a drug acting at more than one receptor cannot be
    represented by a single occupancy map: the two systems have
    different regional densities and may act in opposite directions, and
    the regional pattern of the *net* drive is then not proportional to
    either map.  Separating them is what makes the relative weights
    estimable from imaging data at all.

    The weights are signed effect sizes per unit occupancy; their scale
    is only identifiable up to the amplitude of whatever downstream
    model consumes the drive, so fix one of them or the amplitude.
    """
    return (
        weight_a * regional_drive(occ_a, density_a)
        + weight_b * regional_drive(occ_b, density_b)
    )
