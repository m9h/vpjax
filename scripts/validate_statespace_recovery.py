"""Validate the LL filter by recovering known parameters from simulation.

Runs the gate that has to pass before the filter is pointed at a real
recording: simulate an EEG-fMRI run from the vpjax augmented Balloon
model, fit it back, and check both the parameter recovery and the
whiteness of the innovations.  Optionally fits the same data from BOLD
alone, which is the measurement of what the EEG channel contributes.

Nothing here touches real data; the output describes the estimator, not
any subject.

    python scripts/validate_statespace_recovery.py --duration 300 --compare
"""

import argparse
import json

import jax
import numpy as np

from vpjax.validation.statespace_recovery import (
    compare_modalities,
    fit_simulation,
    format_comparison,
    is_calibrated,
    simulate,
)


def main():
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=300.0, help="run length (s)")
    parser.add_argument("--tr", type=float, default=2.0, help="fMRI TR (s)")
    parser.add_argument("--eeg-dt", type=float, default=0.1,
                        help="EEG envelope sample interval (s)")
    parser.add_argument("--tau-z", type=float, default=2.0,
                        help="true neural drive time constant (s)")
    parser.add_argument("--q-z", type=float, default=0.01,
                        help="true neural drive diffusion (per s); the default "
                             "gives resting-state-scale BOLD (~1.5%% SD), "
                             "where the LL filter is calibrated")
    parser.add_argument("--bold-sd", type=float, default=0.002,
                        help="BOLD observation noise SD (fractional)")
    parser.add_argument("--eeg-snr", type=float, default=3.0,
                        help="drive SD over EEG channel noise SD")
    parser.add_argument("--restarts", type=int, default=12,
                        help="extra optimizer starts; the kappa-tau ridge "
                             "makes a single start unreliable")
    parser.add_argument("--max-dt", type=float, default=0.25,
                        help="longest propagation step (s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true",
                        help="also fit from BOLD alone and contrast")
    parser.add_argument("--output", help="write results as JSON")
    args = parser.parse_args()

    sim = simulate(
        duration=args.duration, tr=args.tr, eeg_dt=args.eeg_dt,
        tau_z=args.tau_z, q_z=args.q_z,
        bold_sd=args.bold_sd, eeg_snr=args.eeg_snr, seed=args.seed,
    )
    print(
        f"simulated {args.duration:g} s: {sim['eeg'].size} EEG envelope "
        f"samples, {sim['bold'].size} volumes; "
        f"drive SD {np.std(sim['x'][:, 0]):.3f} (EEG SNR "
        f"{sim['eeg_snr']:.1f}), BOLD SD {np.std(sim['bold']):.4f} "
        f"(BOLD SNR {np.std(sim['bold']) / sim['bold_sd']:.1f})"
    )
    print()

    if args.compare:
        result = compare_modalities(
            sim, restarts=args.restarts, max_dt=args.max_dt
        )
        print(format_comparison(result))
        payload = result
    else:
        fit = fit_simulation(
            sim, use_eeg=True, restarts=args.restarts, max_dt=args.max_dt
        )
        print(f"{'parameter':<10} {'truth':>10} {'initial':>10} {'estimate':>10}"
              f" {'rel err':>9}")
        for n in fit["estimate"]:
            print(f"{n:<10} {fit['truth'][n]:>10.4g} {fit['initial'][n]:>10.4g}"
                  f" {fit['estimate'][n]:>10.4g}"
                  f" {fit['relative_error'][n]:>9.3f}")
        print()
        print(f"log-likelihood {fit['log_likelihood']:.3f} over "
              f"{fit['n_obs']:.0f} observations; converged {fit['success']}")
        print("innovation whiteness:")
        for name, d in fit["diagnostics"]["per_channel"].items():
            print(f"  {name:<6} n={d['n']:>6.0f}  variance {d['variance']:>8.3f}"
                  f"  lag-1 {d['lag1']:>+7.3f}")
        print()
        print("calibrated" if is_calibrated(fit) else
              "NOT calibrated — the estimates do not describe the data")
        payload = fit

    if args.output:
        def plain(o):
            if isinstance(o, dict):
                return {k: plain(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [plain(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o

        drop = {"filter"}
        with open(args.output, "w") as f:
            json.dump(
                plain({k: v for k, v in payload.items() if k not in drop}),
                f, indent=2,
            )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
