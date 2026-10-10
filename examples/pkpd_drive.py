"""Simulate the dose-to-drive chain and write it for a neural/vascular model.

Takes an explicit JSON file of PK and PD assumptions and produces the
regional drive time course that a vbjax neural mass or a vpjax vascular
model consumes.  Every number comes from the file; nothing is inferred
or defaulted, because a plausible-looking concentration curve built from
guessed parameters is worse than no curve at all — it propagates into
the neural fit as if it were data.

The assumptions file must contain::

    {
      "units":      {"time": "h", "volume": "L", "amount": "mg"},
      "source":     "citation for where these parameters come from",
      "structure":  {"n_peripheral": 1, "absorption": false},
      "pk":         {"CL": 40.0, "V1": 25.0, "Q2": 30.0, "V2": 60.0},
      "regimen":    {"kind": "infusion", "amount": 12.0, "duration": 0.08},
      "pd":         {"ke0": 6.0, "ec50": 0.05, "gamma": 1.4},
      "duration":   2.0,
      "dt":         0.002
    }

``density`` is optional: a path to a .npy file of regional receptor
density (one value per region, in the same order as the connectome the
neural model uses).  Without it a single global drive is written.

Nothing here is a validated drug effect.  The output records its own
assumptions so that a downstream fit cannot be mistaken for an
independent measurement of them.
"""

import argparse
import json

import jax
import numpy as np

from vpjax.pharmacokinetics import (
    EffectSiteParams,
    PKParams,
    PKStructure,
    bolus,
    concentration,
    effect_site_concentration,
    infusion,
    occupancy,
    regional_drive,
    secondary_parameters,
)

_REQUIRED = ("units", "source", "structure", "pk", "regimen", "pd", "duration", "dt")


def build_regimen(spec):
    kind = spec["kind"]
    if kind == "bolus":
        return bolus(spec["amount"], time=spec.get("time", 0.0),
                     compartment=spec.get("compartment", 0))
    if kind == "infusion":
        return infusion(spec["amount"], spec["duration"],
                        time=spec.get("time", 0.0),
                        compartment=spec.get("compartment", 0))
    raise ValueError(f"unknown regimen kind {kind!r}; expected 'bolus' or 'infusion'")


def main():
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assumptions", help="JSON: units, source, structure, pk, regimen, pd")
    parser.add_argument("output", help="drive .npz")
    parser.add_argument("--density", help=".npy of regional receptor density")
    args = parser.parse_args()

    spec = json.loads(open(args.assumptions).read())
    missing = [k for k in _REQUIRED if k not in spec]
    if missing:
        raise ValueError(f"assumptions file is missing: {', '.join(missing)}")
    if not spec["source"].strip():
        raise ValueError("'source' must cite where the PK parameters come from")

    structure = PKStructure(
        n_peripheral=int(spec["structure"]["n_peripheral"]),
        absorption=bool(spec["structure"]["absorption"]),
    )
    pk = PKParams(**{k: np.asarray(v, dtype=float) for k, v in spec["pk"].items()})
    regimen = build_regimen(spec["regimen"])

    dt = float(spec["dt"])
    t = np.arange(0.0, float(spec["duration"]) + dt * 1e-6, dt)

    cp = concentration(pk, regimen, t, structure)
    ce = effect_site_concentration(
        cp, dt=dt, params=EffectSiteParams(ke0=np.asarray(spec["pd"]["ke0"], float))
    )
    occ = occupancy(ce, ec50=np.asarray(spec["pd"]["ec50"], float),
                    gamma=float(spec["pd"].get("gamma", 1.0)))

    if args.density:
        density = np.load(args.density)
        if density.ndim != 1:
            raise ValueError("density must be one value per region")
        drive = np.asarray(regional_drive(occ, density))
    else:
        density = None
        drive = np.asarray(occ)[None, :]

    sec = secondary_parameters(pk, structure)
    metadata = {
        "model": "linear compartment PK -> first-order effect site -> Hill occupancy",
        "assumptions": spec,
        "units": spec["units"],
        "source": spec["source"],
        "density": args.density,
        "secondary": {
            "t_half_terminal": float(sec["t_half_terminal"]),
            "half_lives": [float(x) for x in np.asarray(sec["half_lives"])],
            "Vss": float(sec["Vss"]),
        },
        "interpretation": (
            "simulated drive conditional on the stated PK/PD parameters; "
            "not an estimated or validated drug effect"
        ),
    }

    with open(args.output, "wb") as file:
        np.savez_compressed(
            file,
            time=t,
            plasma=np.asarray(cp),
            effect_site=np.asarray(ce),
            occupancy=np.asarray(occ),
            drive=drive,
            metadata=json.dumps(metadata),
        )

    print(f"wrote {args.output}: drive {drive.shape}, "
          f"peak occupancy {float(np.max(occ)):.3f}, "
          f"terminal t1/2 {float(sec['t_half_terminal']):.3g} {spec['units']['time']}")


if __name__ == "__main__":
    main()
