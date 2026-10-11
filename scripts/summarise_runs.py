"""Tabulate recorded-run fits from validate_ds003768_statespace / validate_natview_statespace.

One row per run × ROI × arm, with the quantities that decide the
protocol question: the loading of each fast channel on the shared drive
(estimate, relative SE, the log-likelihood margin of its sign), the
drive time constant, identifiability, and innovation whiteness per
channel.  Reads any mix of the two drivers' JSON outputs.

    python scripts/summarise_runs.py results/*.json [--csv out.csv]
"""

import argparse
import csv
import glob
import json
import sys


def arms_of(doc: dict):
    """Yield (arm_label, fit_dict) for either driver's output."""
    if "arms" in doc:                          # natview driver
        yield from doc["arms"].items()
    else:                                      # ds003768 driver
        yield "bold", doc["bold_only"]
        yield "bold+eeg", doc["both"]


def sign_margin(fit: dict) -> float | None:
    """Best minus second-best log-likelihood over the sign combinations tried."""
    vals = sorted((v for v in fit.get("log_likelihood_by_sign", {}).values()
                   if v == v), reverse=True)
    return vals[0] - vals[1] if len(vals) > 1 else None


def rows_from(path: str):
    doc = json.load(open(path))
    run = doc.get("run", path)
    roi = doc.get("roi", "global")
    for arm, fit in arms_of(doc):
        est, se = fit["estimate"], fit["standard_error"]
        row = {
            "run": run, "roi": roi, "arm": arm,
            "n_obs": int(fit["n_obs"]), "loglik": round(fit["log_likelihood"], 1),
            "converged": fit["success"], "identifiable": fit["identifiable"],
            "curvature": f"{fit['curvature_ratio']:.2e}",
            "tau_z": f"{est.get('tau_z', float('nan')):.2f}",
            "tau_z_se": f"{se.get('tau_z', float('nan')):.2f}",
            "sign_margin": None if sign_margin(fit) is None else round(sign_margin(fit), 1),
            "floored": ",".join(fit.get("noise_floored", [])),
            "dropped": ",".join(fit.get("inputs_dropped", {})),
        }
        signs = fit.get("sign", {})
        if not isinstance(signs, dict):
            signs = {"eeg": signs}
        for name in list(est):
            if name.startswith(("gain_", "beta_")):
                c = name.split("_", 1)[1]
                row[f"{name}"] = f"{signs.get(c, 1.0) * est[name]:+.3g}"
                row[f"{name}_se"] = f"{se[name]:.2f}"
        for ch, d in fit["diagnostics"]["per_channel"].items():
            row[f"{ch}_var"] = f"{d['variance']:.2f}"
            row[f"{ch}_lag1"] = f"{d['lag1']:+.2f}"
        yield row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--csv")
    a = p.parse_args()
    rows = [r for pat in a.paths for path in sorted(glob.glob(pat)) for r in rows_from(path)]
    if not rows:
        sys.exit("no fits found")
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    out = csv.DictWriter(open(a.csv, "w") if a.csv else sys.stdout, fieldnames=cols)
    out.writeheader()
    out.writerows(rows)
    if a.csv:
        print(f"wrote {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
