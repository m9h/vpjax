"""Fit a reduced shared drive to a neurojax observation export.

Requires explicit loading/noise assumptions in a JSON file. They are model
choices, not estimated autonomic branch activity or validated drug effects.
"""

import argparse
import json

import numpy as np

from vpjax.autonomic import infer_shared_drive, load_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations")
    parser.add_argument(
        "assumptions", help="JSON: loadings, noise_sd, process_sd, prior_sd"
    )
    parser.add_argument("output", help="posterior .npz")
    parser.add_argument("--step-s", type=float, default=30.0)
    args = parser.parse_args()
    assumptions = json.loads(open(args.assumptions).read())
    all_streams = load_observations(args.observations)
    streams = {name: all_streams[name] for name in assumptions["loadings"]}
    for name, stream in streams.items():
        if stream["unit"] != "1":
            raise ValueError(f"{name} must be baseline-standardized")
        if not stream["valid"].any():
            raise ValueError(f"{name} has no valid observations")
    if not np.isfinite(args.step_s) or args.step_s <= 0:
        raise ValueError("step-s must be positive and finite")
    start = min(s["time_s"][0] for s in streams.values())
    stop = max(s["time_s"][-1] for s in streams.values())
    grid = np.arange(start, stop + args.step_s * 1e-6, args.step_s)
    result = infer_shared_drive(streams, grid, **assumptions)
    metadata = {
        "model": "linear-Gaussian shared drive; random walk; fixed loadings",
        "assumptions": assumptions,
        "valid_counts": result["valid_counts"],
        "source": args.observations,
        "observation_metadata": {
            name: stream["metadata"] for name, stream in streams.items()
        },
        "uncertainty": (
            "conditional on fixed loadings and noise; not physiological identification"
        ),
    }
    with open(args.output, "wb") as file:
        np.savez_compressed(
            file,
            time_s=np.asarray(result["time_s"]),
            mean=np.asarray(result["mean"]),
            sd=np.asarray(result["sd"]),
            metadata=np.array(json.dumps(metadata)),
        )
    print(f"Wrote {len(grid)} posterior samples to {args.output}")


if __name__ == "__main__":
    main()
