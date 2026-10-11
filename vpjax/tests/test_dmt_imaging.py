"""Tests for the DMT release loader, on synthetic .mat files."""

import numpy as np
import pytest

from vpjax.validation.dmt_imaging import (
    ACUTE_S,
    BANDS,
    INFUSION_S,
    N_VOLUMES,
    NETWORKS,
    TR,
    bold_from_zscored,
    load_eeg_bands,
    load_parcels,
    network_series,
    phase_mask,
    subjects,
    volume_times,
)


@pytest.fixture
def release(tmp_path):
    """A miniature stand-in for the release's folder layout."""
    from scipy.io import savemat

    (tmp_path / "fMRI").mkdir()
    (tmp_path / "EEG").mkdir()
    rng = np.random.default_rng(0)
    for sub in ("S01", "S02", "S07"):
        for cond in ("DMT", "PCB"):
            x = rng.normal(size=(112, N_VOLUMES))
            x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
            savemat(str(tmp_path / "fMRI" / f"Long{sub}{cond}.mat"), {"BOLD_AAL": x})
    # A subject with only one condition must be skipped.
    savemat(str(tmp_path / "fMRI" / "LongS99DMT.mat"),
            {"BOLD_AAL": rng.normal(size=(112, N_VOLUMES))})
    bands = {k: rng.normal(size=(14, N_VOLUMES, 5)) for k in ("regDMT", "regPCB", "regdiff")}
    savemat(str(tmp_path / "EEG" / "RegressorsInterpscrubbedIRASA_Occipital.mat"), bands)
    return tmp_path


class TestLayout:
    def test_only_complete_subjects(self, release):
        assert subjects(release) == ["S01", "S02", "S07"]

    def test_parcels_shape_checked(self, release):
        from scipy.io import savemat
        assert load_parcels(release, "S01", "DMT").shape == (112, N_VOLUMES)
        savemat(str(release / "fMRI" / "LongS98PCB.mat"), {"BOLD_AAL": np.zeros((50, 10))})
        savemat(str(release / "fMRI" / "LongS98DMT.mat"), {"BOLD_AAL": np.zeros((50, 10))})
        with pytest.raises(ValueError, match="expected"):
            load_parcels(release, "S98", "DMT")


class TestNetworks:
    def test_membership_partitions_the_parcels(self):
        allp = np.concatenate([v for v in NETWORKS.values()])
        assert allp.min() == 0 and allp.max() == 111
        assert np.unique(allp).size == 112        # a partition, no overlaps or gaps

    def test_network_mean_matches_manual(self, release):
        p = load_parcels(release, "S01", "DMT")
        assert np.allclose(network_series(p, "visual"), p[NETWORKS["visual"]].mean(axis=0))
        assert np.allclose(network_series(p, "global"), p.mean(axis=0))
        with pytest.raises(KeyError):
            network_series(p, "nope")


class TestAmplitude:
    def test_assumed_sd_is_applied_after_detrending(self):
        t = np.arange(N_VOLUMES, dtype=float)
        x = (t - t.mean()) / t.std() + 0.3 * np.sin(2 * np.pi * t / 50)
        y = bold_from_zscored(x, 0.02)
        assert np.isclose(y.std(), 0.02, rtol=1e-6)
        assert abs(np.polyfit(t, y, 1)[0]) < 1e-12      # trend removed

    def test_q_z_units_scale_with_the_square(self):
        # Doubling the assumed amplitude doubles the series, so any
        # variance-like quantity fitted to it scales fourfold; the test
        # pins the linear part, which is what the docstring promises.
        rng = np.random.default_rng(1)
        x = rng.normal(size=N_VOLUMES)
        assert np.allclose(bold_from_zscored(x, 0.02), 2 * bold_from_zscored(x, 0.01))

    def test_constant_series_rejected(self):
        with pytest.raises(ValueError, match="constant"):
            bold_from_zscored(np.ones(10))


class TestTiming:
    def test_volume_times_and_phases(self):
        t = volume_times()
        assert t.size == N_VOLUMES and np.isclose(t[-1], N_VOLUMES * TR - TR / 2)
        assert phase_mask(t, "pre").sum() == 240          # 480 s / 2 s
        acute = phase_mask(t, "acute")
        assert t[acute].min() >= ACUTE_S[0] and t[acute].max() < ACUTE_S[1]
        assert not phase_mask(t, "pre")[t >= INFUSION_S].any()
        with pytest.raises(KeyError):
            phase_mask(t, "during")


class TestEEGGuard:
    def test_convolved_regressors_refused_by_default(self, release):
        with pytest.raises(ValueError, match="already HRF-convolved"):
            load_eeg_bands(release, "Occipital", "DMT")

    def test_opt_in_returns_bands_with_the_caveat(self, release):
        out = load_eeg_bands(release, "Occipital", "DMT", convolved_ok=True)
        assert out["values"].shape == (14, N_VOLUMES, 5)
        assert out["bands"] == BANDS and out["n_subjects"] == 14
        assert "HRF-convolved" in out["caveat"]
        assert out["time_s"].size == N_VOLUMES

    def test_unknown_region_or_condition(self, release):
        with pytest.raises(KeyError):
            load_eeg_bands(release, "Insula", "DMT", convolved_ok=True)
        with pytest.raises(KeyError):
            load_eeg_bands(release, "Occipital", "drug", convolved_ok=True)
