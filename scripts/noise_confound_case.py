"""Can a drug that only adds measurement noise fake a drug effect on the drive?

A real worry for a drug study, and not hypothetical: on ds006072 the
acute psilocybin runs carry three to five times as much high-frequency
BOLD power as the baseline runs (10-15% of the band above 0.1 Hz at
baseline, up to 54% acutely), which is what motion and physiological
contamination look like when a drug raises heart rate, breathing and
movement.  If that extra noise can be absorbed by the drive parameters,
every drug study using a generative model inherits a false-positive
mode, and the inference is worthless.

The test is a specificity test.  Simulate a run with **no drug effect on
the drive at all** (``beta = 0``) but the BOLD observation noise
multiplied by *noise_factor* during the acute window, then fit with a
free drug gain and ask what comes back.  Three conditions:

``pinned``   the fit uses the baseline noise everywhere, as an analyst
             would who estimated it from the pre-infusion period.  The
             extra noise has nowhere to go but the state parameters, so
             this is the worst case and bounds the confound.
``modelled`` the fit is told the noise is elevated in that window, as an
             analyst would who measured it per phase.  Shows whether
             knowing about it is enough.
``positive`` a genuine drive effect with constant noise, so the same
             machinery is seen to recover a real one (otherwise a null
             result only proves the test is insensitive).

What makes separation possible in principle is that the drive reaches
the BOLD through the Balloon, which low-passes it, while measurement
noise is white: a noise increase puts power where no drive can.  Whether
that holds in practice at a given TR and run length is what this
measures.

    python scripts/noise_confound_case.py --seeds 0 1 2 --noise-factor 3 \\
        --onset 480 --rise 60 --decay 600 --duration 1680 --output confound.json
"""

import argparse
import json

import numpy as np

from vpjax.statespace import build_observation_grid, fit_statespace, parameter_uncertainty
from vpjax.validation.statespace_recovery import (
    _build_model,
    drug_shaped_input,
    simulate,
)

FIT_NAMES = ("kappa", "tau", "tau_z", "q_z", "beta")


def inflate_acute_noise(sim: dict, window: tuple[float, float], factor: float, seed: int):
    """Add noise to the BOLD samples inside *window*, returning the per-sample SD."""
    rng = np.random.default_rng(seed + 9999)
    t, base = sim["t_bold"], float(sim["bold_sd"])
    inside = (t >= window[0]) & (t < window[1])
    extra = np.sqrt(max(factor**2 - 1.0, 0.0)) * base
    sim = dict(sim)
    sim["bold"] = sim["bold"] + np.where(inside, rng.normal(scale=extra, size=t.size), 0.0)
    return sim, np.where(inside, factor * base, base)


def fit(sim: dict, bold_sd, free_beta: bool, restarts: int, seed: int, gain: float) -> dict:
    """Fit with a per-sample BOLD noise SD, with the drug gain free or at zero.

    The gain is estimated on the log scale, where zero is an infinitely
    distant boundary: a fit that wants no drug effect runs it to 1e-9 and
    its standard error degenerates, so a Wald ratio there is meaningless.
    The statistic that does not degenerate is the likelihood ratio
    against the nested model with the gain fixed at zero, which is what
    :func:`main` reports.
    """
    streams = {
        "bold": {"time_s": sim["t_bold"], "values": sim["bold"], "noise_sd": np.max(bold_sd)},
        "eeg": {"time_s": sim["t_eeg"], "values": sim["eeg"], "noise_sd": sim["eeg_sd"]},
    }
    grid = build_observation_grid(streams, max_dt=0.25)
    sim = dict(sim, truth=dict(sim["truth"], beta=gain if free_beta else 0.0))
    names = FIT_NAMES if free_beta else FIT_NAMES[:-1]
    build = _build_model(grid, sim, names)
    truth = sim["truth"]
    init = np.array([1.5 * truth["kappa"], 0.6 * truth["tau"], 1.7 * truth["tau_z"],
                     0.5 * truth["q_z"], 0.3][: len(names)])
    out = fit_statespace(build, grid, init=init, restarts=restarts, seed=seed, max_steps=300)
    unc = parameter_uncertainty(build, grid, out["log_theta"])
    est = {n: float(v) for n, v in zip(names, out["theta"])}
    se = {n: float(v) for n, v in zip(names, unc["standard_error"])}
    return {
        "estimate": est, "standard_error": se,
        "identifiable": bool(unc["identifiable"]),
        "curvature_ratio": float(unc["curvature_ratio"]),
        "log_likelihood": float(out["log_likelihood"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--duration", type=float, default=1680.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--eeg-dt", type=float, default=2.0)
    p.add_argument("--onset", type=float, default=480.0)
    p.add_argument("--rise", type=float, default=60.0)
    p.add_argument("--decay", type=float, default=600.0)
    p.add_argument("--beta", type=float, default=0.5, help="gain for the positive control")
    p.add_argument("--noise-factor", type=float, default=3.0)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--output")
    a = p.parse_args()

    window = (a.onset, min(a.onset + 3 * a.rise + a.decay, a.duration))
    u = drug_shaped_input(a.onset, a.rise, a.decay)
    base = dict(duration=a.duration, tr=a.tr, eeg_dt=a.eeg_dt)
    print(f"acute window {window[0]:.0f}-{window[1]:.0f} s, noise x{a.noise_factor} inside it; "
          f"the drug input peaks at 1 and the positive control uses beta={a.beta}")
    out = {"settings": vars(a), "window": window, "conditions": {}}

    print(f"{'condition':9s} {'truth':>6s} {'beta med':>9s} {'as % of':>8s} "
          f"{'2 dLL vs beta=0':>16s} {'ident':>6s}")
    for label in ("pinned", "modelled", "positive"):
        rows, nulls = [], []
        for seed in a.seeds:
            if label == "positive":
                sim = simulate(seed=seed, drive_input=u, beta=a.beta, **base)
                sd = float(sim["bold_sd"])
            else:
                sim = simulate(seed=seed, drive_input=u, beta=0.0, **base)
                sim, sd = inflate_acute_noise(sim, window, a.noise_factor, seed)
                if label == "pinned":
                    sd = float(sim["bold_sd"])
            rows.append(fit(sim, sd, True, a.restarts, seed + 1, a.beta))
            nulls.append(fit(sim, sd, False, a.restarts, seed + 1, a.beta))
        for r, n in zip(rows, nulls):
            r["log_likelihood_null"] = n["log_likelihood"]
            r["lr"] = 2.0 * (r["log_likelihood"] - n["log_likelihood"])
        out["conditions"][label] = rows
        beta = np.array([r["estimate"]["beta"] for r in rows])
        lr = np.array([r["lr"] for r in rows])
        truth = a.beta if label == "positive" else 0.0
        print(f"{label:9s} {truth:>6.2f} {np.median(beta):>+9.5f} "
              f"{100 * np.median(beta) / a.beta:>7.2f}% {np.median(lr):>16.2f} "
              f"{sum(r['identifiable'] for r in rows):>4d}/{len(rows)}")
    print()
    print("A likelihood ratio near zero means the data give no reason to prefer a drug")
    print("effect on the drive; 3.84 is the nominal 5% point of chi2(1), and the boundary")
    print("makes that conservative.  Beta is reported as a fraction of the gain the")
    print("positive control recovers, so it is a false-positive rate in effect size.")

    if a.output:
        json.dump(out, open(a.output, "w"), indent=1, default=float)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
