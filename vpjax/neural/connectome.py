"""Graphs for the E/I block: the multiscale structural connectome on HCP-MMP1.0.

The datasets on disk (NATVIEW, ds003768, NIMH/Bolt) ship a T1w and no
diffusion imaging, so a subject-specific tractography graph is not
available.  This module loads the *group* multiscale structural
connectome (MSC) of Xia et al. 2026 (Commun Biol 9:1339,
10.1038/s42003-026-10558-5; code and data github.com/jiexfmri/
MSFD_GSP_analysis, Zenodo 10.5281/zenodo.20558532): a 360-region
HCP-MMP1.0 affinity matrix fusing tractography strength (TS), inverse
geodesic distance (GD) and microstructural profile covariance (MPC)
from 100 unrelated HCP subjects, built by rank-normalising each
modality, concatenating row-wise and taking the normalised-angle cosine
similarity.  The paper's Methods reproduce resting and task fMRI with
the first 200 eigenmodes of this graph better than with tractography
alone, which is the property wanted of a basis for
:func:`vpjax.neural.linear_ei_block`.

Region order is Glasser's: left hemisphere parcels 1..180 then right
181..360, which is the order of the fsaverage annotations
(figshare 10.6084/m9.figshare.3498446, CC BY 4.0) that
:func:`hcp_mmp_names` reads.  The MSC file carries no names; the
assumption is checked against the hierarchy table shipped with it
(region 0 and region 180 sit at the sensory end of the first functional
gradient, i.e. V1 in each hemisphere).

Two Laplacians are offered because the two sources differ: gnf-jax's
kernels are written for the combinatorial Laplacian ``-(D - W)``, while
Xia et al. decompose the symmetric normalised one
``I - D^{-1/2} W D^{-1/2}``.  Both return eigenvalues non-positive and
sorted with the constant mode first, which is what
:class:`vpjax.neural.EIBlock` expects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MSC_KEYS = ("SC_MPC_GD", "SC_MPC", "SC_GD", "MPC_GD", "SC", "GD", "MPC")


def load_msc(path: str | Path, key: str = "SC_MPC_GD") -> np.ndarray:
    """Load one adjacency from ``HCP_MMP_Fusion_all.mat`` as a (360, 360) float array.

    Fused matrices are dense, symmetric, in [0, 1) with zero diagonal.
    ``SC`` alone is the raw streamline count (uint16, asymmetric from
    the source's row-wise normalisation) and is symmetrised here.
    """
    from scipy.io import loadmat

    if key not in MSC_KEYS:
        raise ValueError(f"key must be one of {MSC_KEYS}, got {key!r}")
    W = np.asarray(loadmat(str(path))[key], dtype=float)
    if W.shape != (360, 360):
        raise ValueError(f"expected a 360x360 matrix, got {W.shape}")
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    if (W < 0).any():
        raise ValueError("negative weights in the adjacency")
    return W


def laplacian_modes(W: np.ndarray, normalised: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (non-positive, descending from 0) and orthonormal modes of a graph.

    *normalised* selects ``-(I - D^{-1/2} W D^{-1/2})`` (eigenvalues in
    [-2, 0]); otherwise ``-(D - W)``.  Returns ``(eigs, U)`` with modes
    as columns, the constant (or degree-weighted constant) mode first.
    """
    W = np.asarray(W, dtype=float)
    d = W.sum(axis=1)
    if (d <= 0).any():
        raise ValueError("isolated node: zero degree")
    if normalised:
        s = 1.0 / np.sqrt(d)
        L = np.eye(W.shape[0]) - (s[:, None] * W * s[None, :])
    else:
        L = np.diag(d) - W
    L = 0.5 * (L + L.T)
    lam, U = np.linalg.eigh(L)
    order = np.argsort(lam)
    lam, U = lam[order], U[:, order]
    lam[0] = 0.0
    return -lam, U


def hcp_mmp_names(annot_dir: str | Path) -> list[str]:
    """The 360 parcel names, left 1..180 then right, from the fsaverage annotations."""
    import nibabel as nib

    names = []
    for h in ("lh", "rh"):
        _, _, n = nib.freesurfer.read_annot(str(Path(annot_dir) / f"{h}.HCP-MMP1.annot"))
        n = [x.decode() if isinstance(x, bytes) else str(x) for x in n]
        if len(n) != 181 or n[0] != "???":
            raise ValueError(f"unexpected annotation layout for {h}: {len(n)} labels")
        names += n[1:]
    return names


def parcellate_fsaverage(lh_values: np.ndarray, rh_values: np.ndarray, annot_dir: str | Path) -> np.ndarray:
    """Mean of per-vertex fsaverage (164k) values within each HCP-MMP parcel, (360,)."""
    import nibabel as nib

    out = []
    for h, v in (("lh", lh_values), ("rh", rh_values)):
        lab, _, _ = nib.freesurfer.read_annot(str(Path(annot_dir) / f"{h}.HCP-MMP1.annot"))
        v = np.asarray(v, dtype=float)
        if v.shape[0] != lab.shape[0]:
            raise ValueError(f"{h}: {v.shape[0]} values for {lab.shape[0]} vertices")
        for k in range(1, 181):
            m = (lab == k) & np.isfinite(v)
            out.append(v[m].mean() if m.any() else np.nan)
    return np.asarray(out)


def mode_weights(U: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Modal coefficients ``U^T pattern`` of a regional pattern, for ``EIBlock.input_vector``."""
    pattern = np.asarray(pattern, dtype=float)
    if not np.isfinite(pattern).all():
        raise ValueError("pattern has non-finite entries")
    return U.T @ pattern


def spectral_concentration(U: np.ndarray, pattern: np.ndarray, k: int) -> float:
    """Fraction of a pattern's energy carried by the first *k* modes."""
    w = mode_weights(U, pattern)
    return float((w[:k] ** 2).sum() / (w**2).sum())
