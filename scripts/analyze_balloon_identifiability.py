#!/usr/bin/env python3
"""Local identifiability analysis of the Balloon-Windkessel model.

Computes the Walter & Pronzato Jacobian-rank test
(:func:`vpjax.identifiability.balloon_identifiability`) under several
experimental scenarios and prints a comparison table.  The scenarios
cover:

  - BOLD-only with the full 5-parameter set vs the (kappa, tau) and
    (kappa,) subsets used by the existing vpjax fitters
  - Multi-modal observation (BOLD + ASL, BOLD + ASL + VASO)
  - Two stimulus shapes: a synthetic block and the data-driven CO2
    proxy actually extracted from a DLBS subject's hypercapnia run

The point of this script is *not* to validate identifiability for
publication — that needs a wider sweep and the full nonlinear Hessian.
It documents (and records as JSON) the first-order conditioning of the
Balloon system at the DCM-2007 nominal point on a real protocol, so we
have a numerical reference for which fit_names combinations are worth
optimising on which acquisition.

Usage::

    python analyze_balloon_identifiability.py             # synthetic block + DLBS
    python analyze_balloon_identifiability.py --no-dlbs   # synthetic block only

Output JSON lands at
``/data/datasets/smri-fm-cmp/vpjax/ds004856/sub-1003/ses-wave1/balloon_identifiability.json``
when the DLBS path is available.
"""
from __future__ import annotations

import argparse
import json
import os

# Force CPU before importing JAX — small Jacobians, GPU launch overhead
# dominates and CPU is ~25× faster end-to-end on this hardware.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from vpjax.identifiability import balloon_identifiability


SCENARIOS = [
    ("all 5",       ("kappa", "gamma", "tau", "alpha", "E0")),
    ("kappa,tau",   ("kappa", "tau")),
    ("kappa only",  ("kappa",)),
]
OBSERVER_SETS = [
    ("BOLD",            ("bold",)),
    ("BOLD+ASL",        ("bold", "asl")),
    ("BOLD+ASL+VASO",   ("bold", "asl", "vaso")),
]


def block_stimulus(tr: float = 2.0, dt: float = 0.1, n_t: int = 60) -> jnp.ndarray:
    """20-s on / 40-s off block at peak amplitude 0.3 (Balloon-stable)."""
    t_total = n_t * int(round(tr / dt))
    return 0.3 * jnp.ones(t_total).at[:int(20.0 / dt)].set(0.0)


def dlbs_data_driven_stimulus(
    bold_path: str, mask_path: str, tr: float, dt: float,
) -> jnp.ndarray:
    """Whole-brain mean → low-pass → ±0.3 amplitude (matches run_cvr)."""
    import nibabel as nib
    from vpjax.hemodynamics.cvr import extract_global_stimulus
    bold = nib.load(bold_path).get_fdata(dtype=np.float32)
    mask = nib.load(mask_path).get_fdata(dtype=np.float32)
    stim, _ = extract_global_stimulus(bold, mask, tr=tr, dt=dt)
    return stim


def report(label: str, scenarios: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'fit_names':<14}{'observers':<22}{'rank':<7}{'cond #':<12}"
          f"{'σ_min':<12}{'σ_max':<12}")
    for s in scenarios:
        sigs = s["singular_values"]
        print(f"{s['fit_names_str']:<14}{s['observers_str']:<22}"
              f"{s['rank']}/{s['n_params']:<5}"
              f"{s['condition_number']:.2e}   "
              f"{min(sigs):.2e}   {max(sigs):.2e}")
        for col in s["collinear_sets"]:
            terms = " ".join(
                f"{c:+.3f}·{n}" for c, n in zip(col['coefficients'], col['params'])
            )
            print(f"   null σ={col['singular_value']:.2e}: {terms}")


def run_one(
    fit_names: tuple[str, ...],
    observers: tuple[str, ...],
    stimulus: jnp.ndarray,
    tr: float,
    dt: float,
) -> dict:
    out = balloon_identifiability(
        fit_names, stimulus, dt=dt, tr=tr, observers=observers,
    )
    return {
        "fit_names": list(fit_names),
        "fit_names_str": ",".join(fit_names),
        "observers": list(observers),
        "observers_str": "+".join(observers),
        "rank": int(out["rank"]),
        "n_params": int(out["n_params"]),
        "is_identifiable": bool(out["is_identifiable"]),
        "condition_number": float(out["condition_number"]),
        "singular_values": [float(v) for v in np.asarray(out["singular_values"])],
        "collinear_sets": out["collinear_sets"],
    }


def sweep(stimulus: jnp.ndarray, tr: float, dt: float) -> list[dict]:
    rows: list[dict] = []
    for fit_label, fit_names in SCENARIOS:
        for obs_label, observers in OBSERVER_SETS:
            rows.append(run_one(fit_names, observers, stimulus, tr, dt))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-dlbs", action="store_true",
                        help="skip the DLBS data-driven scenario")
    args = parser.parse_args()

    tr, dt = 2.0, 0.1

    # Synthetic block.
    block = block_stimulus(tr=tr, dt=dt, n_t=60)
    block_rows = sweep(block, tr, dt)
    report("Synthetic block stimulus (peak ±0.3)", block_rows)

    payload: dict = {
        "tr": tr, "dt": dt,
        "block_stimulus": block_rows,
    }

    # DLBS data-driven stimulus from sub-1003 wave-1.
    dlbs_bold = (
        "/data/datasets/smri-fm-cmp/fsl/ds004856/cvr/"
        "sub-1003_ses-wave1_run-1/mc.nii.gz"
    )
    dlbs_mask = (
        "/data/datasets/smri-fm-cmp/fsl/ds004856/cvr/"
        "sub-1003_ses-wave1_run-1/brain_mask.nii.gz"
    )
    if not args.no_dlbs and Path(dlbs_bold).exists():
        stim = dlbs_data_driven_stimulus(dlbs_bold, dlbs_mask, tr, dt)
        rows = sweep(stim, tr, dt)
        report("DLBS sub-1003 ses-wave1 data-driven CO2 stimulus", rows)
        payload["dlbs_data_driven"] = {
            "subject": "sub-1003",
            "session": "ses-wave1",
            "stimulus_peak_abs": float(jnp.max(jnp.abs(stim))),
            "rows": rows,
        }

        out_path = Path(
            "/data/datasets/smri-fm-cmp/vpjax/ds004856/sub-1003/"
            "ses-wave1/balloon_identifiability.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nsaved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
