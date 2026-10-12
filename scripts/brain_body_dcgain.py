"""Is the DC gain into the body the identified combination, as beta*tau_z was?

The original se/truth ~ 4.8 was a units error: log SE was already relative. If
g_bb trades against tau_a the way beta trades against tau_z, then g_bb*tau_a
-- the steady-state gain from z into a -- should recover far tighter.  Spread
across seeds is used as the empirical check, since the per-parameter SE does
not carry the covariance.
"""
import numpy as np
from vpjax.validation.brain_body import simulate, fit_topology, nominal

nom = nominal()
NAMES = ("tau_z", "q_z", "tau_a", "q_a", "beta", "g_bb")
SEEDS = (0, 1, 2, 3, 4)
for period in (32.0, 64.0):
    rows = []
    for seed in SEEDS:
        sim = simulate("efferent", duration=480.0, tr=2.0, fast_dt=2.0, hr_dt=1.0,
                       snr=3.0, seed=seed, drive="duty", period=period, onset=0.0)
        r = fit_topology(sim, "efferent", ("fast", "hr"), fit_names=NAMES,
                         restarts=3, max_steps=200)
        rows.append(r["estimate"])
    def rep(f, truth, label):
        v = np.array([f(e) for e in rows]) / truth
        print("   %-14s ratios %s  median %.3f  CV %.3f" % (
            label, " ".join("%.3f" % x for x in v), np.median(v),
            np.std(v) / abs(np.mean(v))))
    print("period %.0f s" % period)
    rep(lambda e: e["g_bb"], nom["g_bb"], "g_bb")
    rep(lambda e: e["tau_a"], nom["tau_a"], "tau_a")
    rep(lambda e: e["g_bb"] * e["tau_a"], nom["g_bb"] * nom["tau_a"], "g_bb*tau_a")
    rep(lambda e: e["beta"] * e["tau_z"], nom["beta"] * nom["tau_z"], "beta*tau_z")
