"""Does each arm's best cycle period follow the RECEIVING state's time constant?

Heuristic from the first-order attenuation 1/sqrt(1+(w*tau)^2):
  brain-first (g_bb) is read out of `a`, tau_a=10s -> candidate slower cycle,
      period ~ 2*pi*tau_a ~ 63 s; the 16 s HBC cycle is attenuated ~4x.
  body-first  (g_bp) is read out of `z`, tau_z=2s  -> candidate faster cycle.
"""
import numpy as np
from vpjax.validation.brain_body import simulate, fit_topology, nominal

nom = nominal()
CASES = [
    ("brain-first g_bb", "efferent", "g_bb", "tau_a",
     ("tau_z", "q_z", "tau_a", "q_a", "beta", "g_bb")),
    ("body-first  g_bp", "afferent", "g_bp", "tau_z",
     ("tau_z", "q_z", "tau_a", "q_a", "beta_a", "g_bp")),
]
PERIODS = (8.0, 16.0, 32.0, 64.0, 128.0)
print("%-18s %7s %7s %8s  %9s %9s %7s" % (
    "arm", "period", "f(Hz)", "atten", "est", "log_se", "rel_pct"))
for label, topo, target, tau_name, names in CASES:
    tau = nom[tau_name]
    for period in PERIODS:
        w = 2 * np.pi / period
        atten = 1.0 / np.sqrt(1.0 + (w * tau) ** 2)
        ests, ses = [], []
        for seed in (0, 1, 2):
            sim = simulate(topo, duration=480.0, tr=2.0, fast_dt=2.0, hr_dt=1.0,
                           snr=3.0, seed=seed, drive="duty", period=period,
                           onset=0.0)
            r = fit_topology(sim, topo, ("fast", "hr"), fit_names=names,
                             restarts=3, max_steps=200)
            ests.append(r["estimate"][target])
            ses.append(r["standard_error"][target])
        e, s = float(np.median(ests)), float(np.median(ses))
        print("%-18s %7.0f %7.4f %8.3f  %9.4f %9.4f %7.2f" % (
            label, period, 1.0 / period, atten, e, s, 100 * s))
