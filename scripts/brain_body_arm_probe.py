"""Block vs duty-cycled perturbation: log SE is approximately relative SE.

Run: JAX_PLATFORMS=cpu .venv/bin/python -m scripts.brain_body_arm_probe
The Bateman comparator is a transient, not a rectangular block.
"""
import numpy as np
from vpjax.validation.brain_body import simulate, fit_topology

CASES = [
    ("brain-first  g_bb", "efferent", ("fast", "hr"), "g_bb",
     ("tau_z", "q_z", "tau_a", "q_a", "beta", "g_bb")),
    ("body-first   g_bp", "afferent", ("fast", "hr"), "g_bp",
     ("tau_z", "q_z", "tau_a", "q_a", "beta_a", "g_bp")),
]

print("%-20s %-7s %5s  %9s %9s %9s %7s" % (
    "arm", "drive", "dur", "truth", "est", "log_se", "rel_pct"))
for label, topo, obs, target, names in CASES:
    for drive, kw in [("bateman", dict(drive="bateman", onset=60.0, rise=30.0, decay=150.0)),
                      ("duty", dict(drive="duty", period=16.0, onset=60.0))]:
        for dur in (240.0, 480.0):
            ests, ses = [], []
            for seed in (0, 1, 2):
                sim = simulate(topo, duration=dur, tr=2.0, fast_dt=2.0, hr_dt=1.0,
                               snr=3.0, seed=seed, **kw)
                r = fit_topology(sim, topo, obs, fit_names=names,
                                 restarts=3, max_steps=200)
                ests.append(r["estimate"][target])
                ses.append(r["standard_error"][target])
            e, s = float(np.median(ests)), float(np.median(ses))
            print("%-20s %-7s %5.0f  %9.4f %9.4f %9.4f %7.2f" % (
                label, drive, dur, sim["truth"][target], e, s, 100 * s))
