"""Group connectome loader and Laplacian modes for the E/I block."""

import os

import numpy as np
import pytest

from vpjax.neural.connectome import (
    hcp_mmp_names, laplacian_modes, load_msc, mode_weights, parcellate_fsaverage,
    spectral_concentration,
)

MSC = os.path.expanduser("~/Workspace/data/msfd_gsp/HCP_MMP_Fusion_all.mat")
ANNOT = os.path.expanduser("~/Workspace/data/hcp_mmp_fsaverage")


def _ring(n=12):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = W[(i + 1) % n, i] = 1.0
    return W


@pytest.mark.parametrize("normalised", [True, False])
def test_laplacian_modes_on_a_ring(normalised):
    eigs, U = laplacian_modes(_ring(), normalised=normalised)
    assert eigs[0] == 0.0 and np.all(eigs <= 0) and np.all(np.diff(eigs) <= 1e-12)
    np.testing.assert_allclose(U.T @ U, np.eye(12), atol=1e-10)
    np.testing.assert_allclose(np.abs(U[:, 0]), 1 / np.sqrt(12), atol=1e-10)
    if normalised:
        assert eigs.min() >= -2.0 - 1e-12


def test_mode_weights_round_trip():
    eigs, U = laplacian_modes(_ring())
    p = np.sin(np.arange(12) * 2 * np.pi / 12)
    w = mode_weights(U, p)
    np.testing.assert_allclose(U @ w, p, atol=1e-10)
    assert abs(spectral_concentration(U, p, 12) - 1.0) < 1e-10
    with pytest.raises(ValueError):
        mode_weights(U, np.full(12, np.nan))


@pytest.mark.skipif(not os.path.exists(MSC), reason="MSC file not on this machine")
def test_msc_loads_and_decomposes():
    W = load_msc(MSC)
    assert W.shape == (360, 360) and np.allclose(W, W.T) and W.diagonal().max() == 0.0
    eigs, U = laplacian_modes(W)
    assert eigs[0] == 0.0 and eigs.min() > -2.0
    np.testing.assert_allclose(U.T @ U, np.eye(360), atol=1e-8)
    with pytest.raises(ValueError):
        load_msc(MSC, key="nope")


@pytest.mark.skipif(not os.path.exists(ANNOT), reason="annotations not on this machine")
def test_hcp_mmp_names_and_parcellation():
    names = hcp_mmp_names(ANNOT)
    assert len(names) == 360 and names[0] == "L_V1_ROI" and names[180] == "R_V1_ROI"
    import nibabel as nib
    lab, _, _ = nib.freesurfer.read_annot(os.path.join(ANNOT, "lh.HCP-MMP1.annot"))
    v = lab.astype(float)  # value == label index, so each parcel mean equals its index
    out = parcellate_fsaverage(v, v, ANNOT)
    np.testing.assert_allclose(out[:180], np.arange(1, 181))
