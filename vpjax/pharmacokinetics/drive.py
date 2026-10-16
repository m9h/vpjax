"""From an explicit assumptions file to the drive series a state-space model consumes.

This is the bridge between :mod:`vpjax.pharmacokinetics` and the brain-body /
EEG-fMRI state-space fits in :mod:`vpjax.validation`, which until now took a
*shape* (:func:`vpjax.validation.statespace_recovery.drug_shaped_input`) at a
named entry.  Here the shape is replaced by the effect-site occupancy computed
from a dosing regimen and PK/PD parameters that the caller supplies in a JSON
file with a cited ``source``.  Nothing is defaulted: a plausible-looking
curve built from guessed parameters would enter the fit as if it were data.

The assumptions file has the same layout as ``examples/pkpd_drive.py``::

    {"units": {"time": "min", "volume": "L", "amount": "mg"},
     "source": "where these numbers come from",
     "structure": {"n_peripheral": 1, "absorption": false},
     "pk": {"CL": ..., "V1": ..., "Q2": ..., "V2": ...},
     "regimen": {"kind": "infusion", "amount": ..., "duration": ...},
     "pd": {"ke0": ..., "ec50": ..., "gamma": ...},
     "duration": ..., "dt": ...}

Times inside the file are in ``units.time``; the drive returned here is on a
seconds axis because every model in :mod:`vpjax.validation` runs in seconds.
"""

from __future__ import annotations

import json

import numpy as np

from vpjax.pharmacokinetics.compartments import PKParams, PKStructure, concentration, secondary_parameters
from vpjax.pharmacokinetics.dosing import bolus, infusion
from vpjax.pharmacokinetics.occupancy import EffectSiteParams, effect_site_concentration, occupancy, regional_drive

REQUIRED = ("units", "source", "structure", "pk", "regimen", "pd", "duration", "dt")
SECONDS_PER = {"s": 1.0, "sec": 1.0, "second": 1.0, "min": 60.0, "minute": 60.0, "h": 3600.0, "hr": 3600.0, "hour": 3600.0}


def load_assumptions(path_or_spec) -> dict:
    """Read and validate an assumptions file (or pass a dict through the same checks)."""
    spec = path_or_spec if isinstance(path_or_spec, dict) else json.loads(open(path_or_spec).read())
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise ValueError(f"assumptions are missing: {', '.join(missing)}")
    if not str(spec["source"]).strip():
        raise ValueError("'source' must say where the PK/PD parameters come from")
    unit = str(spec["units"].get("time", "")).lower()
    if unit not in SECONDS_PER:
        raise ValueError(f"units.time must be one of {sorted(SECONDS_PER)}, got {unit!r}")
    return spec


def build_regimen(spec: dict):
    kind = spec["kind"]
    if kind == "bolus":
        return bolus(spec["amount"], time=spec.get("time", 0.0), compartment=spec.get("compartment", 0))
    if kind == "infusion":
        return infusion(spec["amount"], spec["duration"], time=spec.get("time", 0.0),
                        compartment=spec.get("compartment", 0))
    raise ValueError(f"unknown regimen kind {kind!r}; expected 'bolus' or 'infusion'")


def effect_site_drive(spec: dict, density: np.ndarray | None = None) -> dict:
    """Dose -> plasma -> effect site -> occupancy (-> regional drive) on a seconds axis.

    Returns a dict with ``time_s``, ``plasma``, ``effect_site``, ``occupancy``,
    ``drive`` (shape ``(regions, N)``; one region without *density*), the peak
    occupancy and the metadata that records every assumption used.
    """
    spec = load_assumptions(spec)
    scale = SECONDS_PER[str(spec["units"]["time"]).lower()]
    structure = PKStructure(n_peripheral=int(spec["structure"]["n_peripheral"]),
                            absorption=bool(spec["structure"]["absorption"]))
    pk = PKParams(**{k: np.asarray(v, dtype=float) for k, v in spec["pk"].items()})
    regimen = build_regimen(spec["regimen"])
    dt = float(spec["dt"])
    t = np.arange(0.0, float(spec["duration"]) + dt * 1e-6, dt)
    cp = np.asarray(concentration(pk, regimen, t, structure))
    ce = np.asarray(effect_site_concentration(cp, dt=dt, params=EffectSiteParams(ke0=np.asarray(spec["pd"]["ke0"], float))))
    occ = np.asarray(occupancy(ce, ec50=np.asarray(spec["pd"]["ec50"], float), gamma=float(spec["pd"].get("gamma", 1.0))))
    if density is not None:
        density = np.asarray(density, dtype=float)
        if density.ndim != 1:
            raise ValueError("density must be one value per region")
        drive = np.asarray(regional_drive(occ, density))
    else:
        drive = occ[None, :]
    sec = secondary_parameters(pk, structure)
    meta = {
        "model": "linear compartment PK -> first-order effect site -> Hill occupancy",
        "assumptions": spec, "source": spec["source"], "units": spec["units"],
        "seconds_per_time_unit": scale,
        "secondary": {"t_half_terminal_s": float(sec["t_half_terminal"]) * scale,
                      "half_lives_s": [float(x) * scale for x in np.asarray(sec["half_lives"])],
                      "Vss": float(sec["Vss"])},
        "peak_occupancy": float(np.max(occ)),
        "time_to_peak_s": float(t[int(np.argmax(occ))]) * scale,
        "interpretation": "simulated drive conditional on the stated PK/PD assumptions; not an estimated or validated drug effect",
    }
    return {"time_s": t * scale, "plasma": cp, "effect_site": ce, "occupancy": occ, "drive": drive,
            "peak_occupancy": float(np.max(occ)), "metadata": meta}


def drive_series_from_pk(t_s: np.ndarray, pk_drive: dict, onset_s: float = 0.0,
                         region: int = 0, normalise: bool = True) -> np.ndarray:
    """Evaluate a PK drive on a model time axis (seconds), shifted to start at *onset_s*.

    With *normalise* the series peaks at one, so the model's ``beta`` keeps its
    meaning as the peak drive and the absolute occupancy scale lives in the
    returned metadata; without it the series is the occupancy itself.  Before
    the onset the drive is zero; after the PK window ends it holds its last value.
    """
    t_s = np.asarray(t_s, dtype=float)
    u = np.asarray(pk_drive["drive"][region], dtype=float)
    src_t = np.asarray(pk_drive["time_s"], dtype=float)
    if normalise:
        peak = float(np.max(np.abs(u)))
        if peak <= 0:
            raise ValueError("PK drive is identically zero; nothing to normalise")
        u = u / peak
    out = np.interp(t_s - onset_s, src_t, u, left=0.0)
    return np.where(t_s >= onset_s, out, 0.0)
