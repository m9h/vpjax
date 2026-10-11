"""Load the public Timmermann et al. (2023) DMT EEG-fMRI release.

`github.com/timmer500/DMT_Imaging`, the data accompanying "Human brain
effects of DMT assessed via EEG-fMRI" (PNAS 120:e2218949120).  This is a
derived release, not raw recordings, and the shapes of what it does
contain decide what can be asked of it:

**The released BOLD series are band-pass filtered, and that makes them
unusable for an innovation-based fit.**  Measured on the release: 0.00%
of the global series' power lies above 0.08 Hz and 1.3% below 0.01 Hz,
against 2.5% and 31% for an unfiltered run of the same TR (ds003768).
An innovation likelihood asks the model to predict each volume from the
last and treats the remainder as white; a series with no power above
0.08 Hz cannot produce white innovations under any parameters, and in
fact the fits come back with innovation lag-1 autocorrelation near +0.55
across subjects while the variance is correctly 1.0.  The time constants
are then fitted to the filter rather than to the physiology.
:func:`check_passband` measures this and :func:`load_parcels` refuses by
default.  Modelling the filter instead is not a way out: it removes
essentially all the power above 0.08 Hz, so inverting it is ill-posed,
its exact specification is not documented, and the high-frequency
content is precisely what pins the Balloon's time constants.  What this
analysis needs is the parcellated series *without* the band-pass step.

**fMRI** ``fMRI/Long<subject><condition>.mat`` holds ``BOLD_AAL``, 112
AAL parcels x 840 volumes, one file per subject per condition.  The
series are **z-scored per parcel**: every parcel has mean 0 and SD 1, so
the fractional amplitude is gone.  A Balloon fit needs an amplitude, so
:func:`bold_from_zscored` multiplies by an assumed fractional SD that
the caller supplies and the result records.  Everything that scales with
amplitude -- the drive diffusion ``q_z``, the observation noise -- is
then in units of that assumption, and only *ratios* between conditions
or subjects mean anything.  The identifiability verdict and the time
constants are unaffected, which is what this dataset is for.

**EEG** ``EEG/RegressorsInterpscrubbedIRASA_<region>.mat`` holds
``regDMT``, ``regPCB`` and ``regdiff``, each 14 subjects x 840 volumes x
5 bands, for five scalp regions.  The bands are delta, theta, alpha,
beta and gamma, IRASA-separated so they are oscillatory power with the
aperiodic component removed, interpolated over scrubbed volumes and
already on the fMRI time base.

**These regressors are already HRF-convolved** -- the release's own
scripts build them with ``spm_Volterra`` on an SPM basis set and carry a
``convolved`` switch.  They therefore must not be used as observations
of the neural drive: the state-space model applies its own hemodynamics
through the Balloon, so a convolved input would count the HRF twice and
bias exactly the transit time and decay the identifiability question is
about.  :func:`load_eeg_bands` refuses to return them unless
``convolved_ok=True`` is passed, and records the warning in its output.
What they can legitimately do is enter as an *input* whose own lag is
estimated, which absorbs the double convolution into a nuisance latency,
or serve as a condition label.

**Timing**, from the release's scripts: 28 min at TR 2 s (840 volumes),
infusion at 8 min with a 30 s injection confound, the acute window
taken as 8:30-13:30 and the late window 16:30-28:00.

Subject coverage differs between the two modalities (20 fMRI, 14 EEG)
and the release gives no mapping between them, so EEG-fMRI pairing here
is per-condition at the group level, not per subject.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TR = 2.0
N_VOLUMES = 840
INFUSION_S = 480.0
INJECTION_CONFOUND_S = 30.0
ACUTE_S = (510.0, 810.0)
LATE_S = (990.0, 1680.0)

BANDS = ("delta", "theta", "alpha", "beta", "gamma")
REGIONS = ("Occipital", "Parietal", "Temporal", "Central", "Frontal")

# Yeo-7 network membership of the 112 AAL parcels, from the release's
# own dRSFCpairwise.m (MATLAB 1-based indices, converted on use).
_NETWORKS_MATLAB = {
    "visual": list(range(1, 10)) + list(range(51, 59)),
    "somatomotor": list(range(10, 16)) + list(range(59, 67)),
    "dorsal_attention": list(range(16, 24)) + list(range(67, 74)),
    "ventral_attention": list(range(24, 31)) + list(range(74, 79)),
    "limbic": list(range(31, 34)) + list(range(79, 81)),
    "frontoparietal": list(range(34, 38)) + list(range(81, 90)),
    "default": list(range(38, 51)) + list(range(90, 101)),
    "subcortical": list(range(101, 113)),
}
NETWORKS = {k: np.asarray(v) - 1 for k, v in _NETWORKS_MATLAB.items()}


def subjects(root: str | Path) -> list[str]:
    """Subject labels with both conditions present, from the fMRI folder."""
    fmri = Path(root) / "fMRI"
    found: dict[str, set[str]] = {}
    for path in sorted(fmri.glob("Long*.mat")):
        m = re.fullmatch(r"Long(S\d+)(DMT|PCB)", path.stem)
        if m:
            found.setdefault(m.group(1), set()).add(m.group(2))
    return [s for s, c in sorted(found.items()) if c == {"DMT", "PCB"}]


def check_passband(
    ts: np.ndarray, tr: float = TR, cutoff_hz: float = 0.1, min_fraction: float = 0.005
) -> dict:
    """Fraction of a series' power above *cutoff_hz*, and whether it is enough.

    A band-passed series has essentially none, and an innovation-based
    fit cannot whiten it (see the module docstring).  The threshold is
    deliberately low: an unfiltered resting run of this TR carries a few
    per cent above 0.1 Hz, a filtered one carries parts in ten thousand,
    so anything in between is already suspicious.
    """
    y = np.asarray(ts, dtype=float)
    k = np.arange(y.size)
    y = y - np.polyval(np.polyfit(k, y, 1), k)
    freq = np.fft.rfftfreq(y.size, tr)
    power = np.abs(np.fft.rfft(y)) ** 2
    total = power[1:].sum()
    fraction = float(power[freq > cutoff_hz].sum() / total) if total > 0 else 0.0
    return {
        "fraction_above_cutoff": fraction,
        "cutoff_hz": cutoff_hz,
        "broadband": fraction >= min_fraction,
        "nyquist_hz": 0.5 / tr,
    }


def load_parcels(
    root: str | Path, subject: str, condition: str, filtered_ok: bool = False
) -> np.ndarray:
    """``BOLD_AAL`` for one subject and condition, shape (112, 840), z-scored.

    Raises unless *filtered_ok*, because the released series are
    band-pass filtered and cannot support an innovation-based fit; see
    the module docstring.  Pass ``filtered_ok=True`` to load them for
    anything that does not depend on the high frequencies, such as
    reproducing the release's own connectivity analyses.
    """
    from scipy.io import loadmat

    path = Path(root) / "fMRI" / f"Long{subject}{condition}.mat"
    data = loadmat(str(path), squeeze_me=True)["BOLD_AAL"]
    data = np.asarray(data, dtype=float)
    if data.shape != (112, N_VOLUMES):
        raise ValueError(f"{path.name}: expected (112, {N_VOLUMES}), got {data.shape}")
    if not filtered_ok:
        band = check_passband(data.mean(axis=0))
        if not band["broadband"]:
            raise ValueError(
                f"{path.name}: only {100 * band['fraction_above_cutoff']:.3f}% of the "
                f"power is above {band['cutoff_hz']} Hz, so this series is band-pass "
                "filtered and an innovation-based state-space fit cannot whiten it; "
                "pass filtered_ok=True to load it anyway, but do not read the fitted "
                "time constants as physiology"
            )
    return data


def network_series(parcels: np.ndarray, network: str) -> np.ndarray:
    """Mean over the parcels of one Yeo-7 network, or ``"global"``."""
    if network == "global":
        return parcels.mean(axis=0)
    if network not in NETWORKS:
        raise KeyError(f"unknown network {network!r}; choose from {sorted(NETWORKS)} or 'global'")
    return parcels[NETWORKS[network]].mean(axis=0)


def bold_from_zscored(ts: np.ndarray, assumed_fractional_sd: float = 0.01) -> np.ndarray:
    """Give a z-scored series an amplitude, after removing a linear trend.

    The release's series are z-scored per parcel, so the Balloon's
    fractional-change observation has no scale to match.  Multiplying by
    *assumed_fractional_sd* supplies one; 1% is the usual resting
    grey-matter figure at 3T.  Record the value with any result, since
    ``q_z`` and the noise variance scale with its square.
    """
    y = np.asarray(ts, dtype=float)
    k = np.arange(y.size)
    y = y - np.polyval(np.polyfit(k, y, 1), k)
    sd = y.std()
    if not sd > 0:
        raise ValueError("series is constant after detrending")
    return y / sd * float(assumed_fractional_sd)


def volume_times(n: int = N_VOLUMES, tr: float = TR) -> np.ndarray:
    """Volume centres in seconds on the run clock."""
    return (np.arange(n) + 0.5) * tr


def phase_mask(times: np.ndarray, phase: str) -> np.ndarray:
    """Boolean mask for ``"pre"``, ``"acute"`` or ``"late"``."""
    if phase == "pre":
        return times < INFUSION_S
    if phase == "acute":
        return (times >= ACUTE_S[0]) & (times < ACUTE_S[1])
    if phase == "late":
        return (times >= LATE_S[0]) & (times < LATE_S[1])
    raise KeyError(f"unknown phase {phase!r}")


def load_eeg_bands(
    root: str | Path, region: str, condition: str, convolved_ok: bool = False
) -> dict:
    """IRASA band power for one region and condition, shape (14, 840, 5).

    Raises unless *convolved_ok*, because these series are already
    HRF-convolved; see the module docstring.  The returned dict records
    that in ``caveat`` so it travels with any result.
    """
    from scipy.io import loadmat

    if region not in REGIONS:
        raise KeyError(f"unknown region {region!r}; choose from {REGIONS}")
    if condition not in ("DMT", "PCB", "diff"):
        raise KeyError("condition must be DMT, PCB or diff")
    if not convolved_ok:
        raise ValueError(
            "these regressors are already HRF-convolved (spm_Volterra in the "
            "release's scripts), so they must not be used as observations of "
            "the neural drive; pass convolved_ok=True to load them anyway, and "
            "use them as an input with an estimated lag, or as a label"
        )
    path = Path(root) / "EEG" / f"RegressorsInterpscrubbedIRASA_{region}.mat"
    key = {"DMT": "regDMT", "PCB": "regPCB", "diff": "regdiff"}[condition]
    values = np.asarray(loadmat(str(path), squeeze_me=True)[key], dtype=float)
    if values.ndim != 3 or values.shape[1:] != (N_VOLUMES, len(BANDS)):
        raise ValueError(f"{path.name}: expected (n, {N_VOLUMES}, 5), got {values.shape}")
    return {
        "values": values,
        "region": region,
        "condition": condition,
        "bands": BANDS,
        "time_s": volume_times(values.shape[1]),
        "n_subjects": int(values.shape[0]),
        "caveat": "already HRF-convolved in the release; not a pre-hemodynamic envelope",
    }
