"""Can an autonomic response with no neural effect fake a neural drug effect?

This is the harder version of ``scripts/noise_confound_case.py``.  That
script showed that a drug which only inflates *measurement* noise cannot
be mistaken for a drive effect, because noise is white and the drive
reaches the BOLD through the Balloon's low-pass: the confound puts power
where no drive can.  A real drug does not do only that.  It raises heart
rate, blood pressure and breathing, and those move perfusion through the
vasculature itself -- smoothly, on the same time scale as the drug, and
through the very filter the neural drive uses.  The low-pass argument
gives no protection at all here, which is why this test is separate.

The two routes are both in :func:`vpjax.validation.statespace_recovery.augmented_drift`:

``beta``      the input drives ``z``, the neural state, and the
              vasculature follows.  An EEG channel observes ``z``, so a
              change on this route *must* show up in the EEG.
``beta_vasc`` the input drives the flow-inducing signal ``s`` directly,
              leaving ``z`` untouched.  The BOLD moves; the EEG does not.

So the prediction is specific and falsifiable: from BOLD alone the two
routes should be very nearly the same model, and the autonomic input
should be reported as a neural effect.  Adding a channel that observes
the drive should break the degeneracy, because only one of the two routes
predicts a change in it.  If that is right, simultaneous EEG is not a
precision luxury in a drug study -- it is what makes "the drug changed
neural activity" separable from "the drug changed the subject's
circulation" at all.

Note what the headline amplitude is: the established gain of 0.5 drives
the BOLD by several percent peak to peak, well above any real drug
effect.  That is deliberate -- it is the configuration the earlier
specificity and simultaneity runs used, so the numbers are comparable --
but it means the 2x2 shows whether the routes are *separable in
principle*, not how small an effect is separable.  The drive-channel SNR
sweep is where the requirement on the recording lives.

The vascular gain is calibrated so the noise-free BOLD excursion matches
the neural arm's.  An unmatched confound is not a confound: a smaller
response is distinguishable by size alone, and reporting that would
overstate the result.

Each dataset is fitted four ways: the neural-route model and the
vascular-route model, each with and without the EEG channel; and each
against its own nested model with the gain removed, for a likelihood
ratio.  ``route dLL`` is the neural-route log-likelihood minus the
vascular-route one, so it is positive when the data prefer the route
that actually generated them and near zero when the two are
indistinguishable.

A drug-free anxiety induction is the empirical version of this test, and
one now exists off the shelf: Allen (2026, PsyArXiv 10.31234/osf.io/szv9e)
characterises a 13.5-minute first-person film of people at height as a
sustained anxiety induction, with group-mean arousal and valence
trajectories reproducible at r = .99 and .94 across samples and a large
rise in somatic anxiety (STICSA, d_z = 1.03), racing heart and tension
being the commonest symptoms.  That is a large, reproducible, temporally
*shaped* autonomic perturbation with no drug on board -- the real
negative control for this simulation, and a within-subject calibration
of the vascular route in the placebo session.

    python scripts/autonomic_confound_case.py --seeds 0 1 2 --output autonomic.json
"""

import argparse
import json

import numpy as np

from vpjax.statespace import (
    build_observation_grid,
    fit_statespace,
    parameter_uncertainty,
)
from vpjax.validation.statespace_recovery import (
    _build_model,
    drug_shaped_input,
    simulate,
)

STATE_NAMES = ("kappa", "tau", "tau_z", "q_z")


def peak_bold(**kwargs) -> float:
    """Peak-to-peak BOLD excursion of a noise-free, drive-free run."""
    sim = simulate(bold_sd=0.0, q_z=1e-12, eeg_snr=1.0, **kwargs)
    return float(np.ptp(sim["bold"]))


def match_vascular_gain(beta: float, tol: float = 1e-3, max_iter: int = 6,
                        **kwargs) -> float:
    """The vascular gain whose BOLD excursion matches a neural gain's.

    The excursion grows with the gain, but sublinearly: the Balloon's
    oxygen extraction saturates, so a proportional correction
    under-shoots and iterating it converges slowly.  Treating the
    relation as a local power law and taking a secant step in log-log
    space lands inside the tolerance in two or three simulations.
    """
    target = peak_bold(beta=beta, **kwargs)
    log_target = np.log(target)
    gains = [beta, beta * target / peak_bold(beta_vasc=beta, **kwargs)]
    excursions = [np.log(peak_bold(beta_vasc=gains[0], **kwargs)), None]

    for _ in range(max_iter):
        excursions[1] = np.log(peak_bold(beta_vasc=gains[1], **kwargs))
        if abs(np.expm1(excursions[1] - log_target)) <= tol:
            return float(gains[1])
        slope = (excursions[1] - excursions[0]) / (
            np.log(gains[1]) - np.log(gains[0])
        )
        step = (log_target - excursions[1]) / slope
        gains = [gains[1], float(np.exp(np.log(gains[1]) + step))]
        excursions = [excursions[1], None]
    raise RuntimeError(
        f"vascular gain did not match the neural excursion to {tol:.0%}"
    )


def fit(
    sim: dict,
    gain: str | None,
    use_eeg: bool,
    reference: dict[str, float],
    restarts: int,
    seed: int,
) -> dict:
    """Fit one route to one dataset.

    *gain* names the route whose gain is free (``beta``, ``beta_vasc``,
    or ``None`` for the nested model with no drug effect at all).  Every
    gain not being fitted is held at **zero**, not at the value that
    generated the data: the whole point is that the fitted model does not
    know which route the input took.
    """
    streams = {
        "bold": {
            "time_s": sim["t_bold"],
            "values": sim["bold"],
            "noise_sd": sim["bold_sd"],
        }
    }
    if use_eeg:
        streams["eeg"] = {
            "time_s": sim["t_eeg"],
            "values": sim["eeg"],
            "noise_sd": sim["eeg_sd"],
        }
    grid = build_observation_grid(streams, max_dt=0.25)

    truth = dict(sim["truth"], beta=0.0, beta_vasc=0.0)
    names = STATE_NAMES + ((gain,) if gain else ())
    if gain:
        truth[gain] = reference[gain]
    build = _build_model(grid, dict(sim, truth=truth), names)

    init = [1.5 * truth["kappa"], 0.6 * truth["tau"],
            1.7 * truth["tau_z"], 0.5 * truth["q_z"]]
    if gain:
        init.append(0.6 * reference[gain])
    out = fit_statespace(build, grid, init=np.array(init), restarts=restarts,
                         seed=seed, max_steps=300)
    unc = parameter_uncertainty(build, grid, out["log_theta"])
    return {
        "gain": gain,
        "use_eeg": use_eeg,
        "estimate": {n: float(v) for n, v in zip(names, out["theta"])},
        "standard_error": {
            n: float(v) for n, v in zip(names, unc["standard_error"])
        },
        "identifiable": bool(unc["identifiable"]),
        "curvature_ratio": float(unc["curvature_ratio"]),
        "log_likelihood": float(out["log_likelihood"]),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--duration", type=float, default=1680.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--eeg-dt", type=float, default=2.0,
                   help="a TR-rate drive channel is enough for identifiability")
    p.add_argument("--onset", type=float, default=480.0)
    p.add_argument("--rise", type=float, default=60.0)
    p.add_argument("--decay", type=float, default=600.0)
    p.add_argument("--beta", type=float, default=0.5,
                   help="neural gain of the real effect, and the amplitude the "
                        "autonomic confound is matched to")
    p.add_argument("--eeg-snr", type=float, default=3.0,
                   help="drive SD over drive-channel noise SD, for the 2x2")
    p.add_argument("--snr-sweep", type=float, nargs="*",
                   default=[3.0, 1.0, 0.5, 0.25, 0.1],
                   help="drive-channel SNRs at which to re-ask whether the "
                        "route is still resolvable; empty to skip")
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--only", choices=("all", "grid", "sweep"), default="all",
                   help="run the 2x2, the SNR sweep, or both (default)")
    p.add_argument("--output")
    a = p.parse_args()

    u = drug_shaped_input(a.onset, a.rise, a.decay)
    base = dict(duration=a.duration, tr=a.tr, eeg_dt=a.eeg_dt,
                drive_input=u, eeg_snr=a.eeg_snr)
    shape = dict(duration=a.duration, tr=a.tr, eeg_dt=a.eeg_dt, drive_input=u)

    beta_vasc = match_vascular_gain(a.beta, **shape)
    amplitude = peak_bold(beta=a.beta, **shape)
    reference = {"beta": a.beta, "beta_vasc": beta_vasc}
    print(f"input peaks at 1 between {a.onset:.0f} s and the end of a "
          f"{a.duration:.0f} s run; TR {a.tr} s, drive channel every {a.eeg_dt} s")
    print(f"neural beta={a.beta:.3f} and vascular beta_vasc={beta_vasc:.3f} both give "
          f"a {100 * amplitude:.2f}% peak-to-peak BOLD excursion")
    print()

    out = {
        "settings": vars(a),
        "reference": reference,
        "bold_excursion": amplitude,
        "arms": {},
    }
    header = (f"{'data':10s} {'channels':9s} {'fitted gain':>11s} {'estimate':>9s} "
              f"{'% of real':>9s} {'2 dLL vs 0':>11s} {'route dLL':>10s} {'ident':>6s}")
    if a.only != "sweep":
        print(header)

    for data_route in ("neural", "vascular") if a.only != "sweep" else ():
        for use_eeg in (True, False):
            rows = {g: [] for g in ("beta", "beta_vasc", None)}
            for seed in a.seeds:
                gains = ({"beta": a.beta} if data_route == "neural"
                         else {"beta_vasc": beta_vasc})
                sim = simulate(seed=seed, **base, **gains)
                for g in rows:
                    rows[g].append(
                        fit(sim, g, use_eeg, reference, a.restarts, seed + 1)
                    )
            label = "BOLD+EEG" if use_eeg else "BOLD only"
            key = f"{data_route}/{'joint' if use_eeg else 'bold'}"
            out["arms"][key] = rows

            route_dll = np.median([
                n["log_likelihood"] - v["log_likelihood"]
                for n, v in zip(rows["beta"], rows["beta_vasc"])
            ])
            for g in ("beta", "beta_vasc"):
                est = np.array([r["estimate"][g] for r in rows[g]])
                lr = np.array([
                    2.0 * (r["log_likelihood"] - z["log_likelihood"])
                    for r, z in zip(rows[g], rows[None])
                ])
                ident = f"{sum(r['identifiable'] for r in rows[g])}/{len(a.seeds)}"
                # A nested fit that diverges has no likelihood; the ratio for
                # that seed is undefined, so the median is over the rest and
                # the count of seeds it rests on is shown beside it.
                n_lr = int(np.sum(np.isfinite(lr)))
                first = g == "beta"
                print(f"{data_route if first else '':10s} {label if first else '':9s} "
                      f"{g:>11s} {np.median(est):>+9.4f} "
                      f"{100 * np.median(est) / reference[g]:>8.1f}% "
                      f"{np.nanmedian(lr) if n_lr else float('nan'):>8.2f} "
                      f"({n_lr}/{len(lr)}) "
                      f"{route_dll if first else float('nan'):>10.2f} {ident:>6s}")

    if a.snr_sweep and a.only != "grid":
        print()
        print("The discrimination is only as good as the drive channel.  Re-asking the")
        print("same question as the drive channel's noise rises: 'leak' is the neural")
        print("gain reported on autonomic-only data, 'kept' is the gain on real neural")
        print("data, both as a percentage of the real effect, with the route dLL beside")
        print("each.  The sweep is what sets the cap, cleaning and CWL requirement.")
        print()
        print("It varies the drive channel's quality at a fixed effect size, so it")
        print("answers how good the cap must be, not how small an effect is separable.")
        print("Shrinking the effect instead would degrade the BOLD channel too, which")
        print("this sweep leaves untouched; that is a separate run, not this one.")
        print()
        print(f"{'drive SNR':>9s} {'leak':>7s} {'leak LR':>9s} {'leak dLL':>9s} "
              f"{'kept':>7s} {'kept LR':>9s} {'kept dLL':>9s}")
        sweep = {}
        for snr in a.snr_sweep:
            row = {}
            for data_route, gains in (("vascular", {"beta_vasc": beta_vasc}),
                                      ("neural", {"beta": a.beta})):
                fits, nulls, other = [], [], []
                for seed in a.seeds:
                    sim = simulate(seed=seed, **dict(base, eeg_snr=snr), **gains)
                    fits.append(fit(sim, "beta", True, reference, a.restarts, seed + 1))
                    nulls.append(fit(sim, None, True, reference, a.restarts, seed + 1))
                    other.append(
                        fit(sim, "beta_vasc", True, reference, a.restarts, seed + 1)
                    )
                row[data_route] = {
                    "beta": float(np.median([f["estimate"]["beta"] for f in fits])),
                    "lr": float(np.median([
                        2.0 * (f["log_likelihood"] - z["log_likelihood"])
                        for f, z in zip(fits, nulls)
                    ])),
                    "route_dll": float(np.median([
                        f["log_likelihood"] - o["log_likelihood"]
                        for f, o in zip(fits, other)
                    ])),
                    "fits": fits, "nulls": nulls, "vascular_route": other,
                }
            sweep[snr] = row
            leak, kept = row["vascular"], row["neural"]
            print(f"{snr:>9.2f} {100 * leak['beta'] / a.beta:>6.1f}% "
                  f"{leak['lr']:>9.2f} {leak['route_dll']:>9.2f} "
                  f"{100 * kept['beta'] / a.beta:>6.1f}% "
                  f"{kept['lr']:>9.2f} {kept['route_dll']:>9.2f}")
        out["snr_sweep"] = sweep

    print()
    if a.only == "sweep":
        print("Read 'leak dLL' first: it is the statistic that decides the route, and a")
        print("large negative value is the autonomic confound being rejected.  The leak")
        print("percentage beside it is what a single-route analysis would have")
        print("concluded instead, which is why route comparison is not optional.")
    else:
        print("Read the diagonal: 'neural' data fitted with beta, 'vascular' data fitted")
        print("with beta_vasc.  The off-diagonal rows are the confound -- what the wrong")
        print("route reports.  'route dLL' is the neural-route log-likelihood minus the")
        print("vascular-route one for the same data, so a value near zero means the data")
        print("cannot tell which of the subject's systems the drug acted on.  Read it")
        print("before the gain estimates: a gain inside a route the data reject by")
        print("hundreds of log-likelihood units is not a measurement of anything.")

    if a.output:
        json.dump(out, open(a.output, "w"), indent=1, default=float)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
