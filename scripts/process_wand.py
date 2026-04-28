#!/usr/bin/env python3
"""Process WAND sub-08033 ses-06 with vpjax.

Pipeline:
  1. Load 7-echo MEGRE magnitude data + brain mask
  2. Fit R2*/S0 per voxel (vpjax) — validate against existing qmri derivative
  3. qBOLD OEF mapping per voxel from multi-echo data
  4. Load QSM chi map + BPF → iron-myelin decomposition
  5. Combine CBF + OEF → regional CMRO2 via Fick principle
  6. Save all outputs to derivatives/vpjax/sub-08033/

Usage:
  python scripts/process_wand.py                    # run all steps
  python scripts/process_wand.py --step r2star      # just R2* fitting
  python scripts/process_wand.py --step oef         # just qBOLD OEF
  python scripts/process_wand.py --step ironmyelin  # iron-myelin decomp
  python scripts/process_wand.py --step cmro2       # regional CMRO2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WAND = Path("/data/raw/wand")
SUB = "sub-08033"
SES = "ses-06"

ANAT_DIR = WAND / SUB / SES / "anat"
DERIV_QMRI = WAND / "derivatives" / "qmri" / SUB / SES
DERIV_PERF = WAND / "derivatives" / "perfusion" / SUB
DERIV_QSM = WAND / "derivatives" / "qsmxt-2026-03-29-183031" / SUB / SES / "anat"
DERIV_BPF = WAND / "derivatives" / "qmri" / SUB / "ses-02"

OUT_DIR = WAND / "derivatives" / "vpjax" / SUB / SES
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Echo times (seconds) from BIDS sidecars
ECHO_TIMES = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
N_ECHOES = len(ECHO_TIMES)
B0 = 6.98  # Tesla (7T scanner)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nii(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI, return (data, img) for header/affine reuse."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32), img


def save_nii(data: np.ndarray, ref_img: nib.Nifti1Image, path: Path, desc: str):
    """Save array as NIfTI using ref header. Print summary stats."""
    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(out, str(path))
    masked = data[data != 0]
    if masked.size > 0:
        print(f"  {desc}: saved {path.name}  "
              f"shape={data.shape}  "
              f"range=[{masked.min():.4f}, {masked.max():.4f}]  "
              f"median={np.median(masked):.4f}")
    else:
        print(f"  {desc}: saved {path.name}  shape={data.shape}  (all zeros)")


def load_megre_magnitude() -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load 7-echo magnitude data, return (X, Y, Z, 7) array + ref image.

    Crops to minimum slab size across echoes (echo 7 may have fewer slices
    due to shorter readout at TE=35ms).
    """
    imgs = []
    for i in range(1, N_ECHOES + 1):
        p = ANAT_DIR / f"{SUB}_{SES}_echo-0{i}_part-mag_MEGRE.nii.gz"
        data, img = load_nii(p)
        imgs.append(data)
    # Crop to minimum dimensions across echoes
    min_shape = tuple(min(d.shape[ax] for d in imgs) for ax in range(3))
    imgs_cropped = [d[:min_shape[0], :min_shape[1], :min_shape[2]] for d in imgs]
    ref_img = nib.load(str(ANAT_DIR / f"{SUB}_{SES}_echo-01_part-mag_MEGRE.nii.gz"))
    print(f"  Echo shapes: {[d.shape for d in imgs]} → cropped to {min_shape}")
    return np.stack(imgs_cropped, axis=-1), ref_img


# ---------------------------------------------------------------------------
# Step 1: R2* fitting
# ---------------------------------------------------------------------------

def step_r2star():
    """Fit R2* and S0 per voxel from 7-echo MEGRE magnitude."""
    from vpjax.qsm.r2star_fitting import fit_r2star_loglinear

    print("\n=== Step 1: R2* Fitting ===")
    t0 = time.time()

    megre, ref_img = load_megre_magnitude()
    mask_data, _ = load_nii(DERIV_QMRI / "megre_brain_mask.nii.gz")
    mask = mask_data > 0.5

    print(f"  MEGRE shape: {megre.shape}, mask voxels: {mask.sum():,}")

    te = jnp.array(ECHO_TIMES)

    # Flatten masked voxels for batch processing
    flat_data = megre[mask]  # (N_voxels, 7)
    print(f"  Fitting {flat_data.shape[0]:,} voxels...")

    flat_jax = jnp.array(flat_data)

    # fit_r2star_loglinear handles batched input natively (N, T) → (N,), (N,)
    # Process in chunks to manage GPU memory (0.67mm iso = ~10M voxels)
    chunk_size = 500_000
    n_voxels = flat_jax.shape[0]
    r2star_flat = np.zeros(n_voxels, dtype=np.float32)
    s0_flat = np.zeros(n_voxels, dtype=np.float32)

    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        chunk = flat_jax[start:end]
        r2, s0 = fit_r2star_loglinear(chunk, te)
        r2star_flat[start:end] = np.asarray(r2)
        s0_flat[start:end] = np.asarray(s0)
        print(f"    chunk {start:,}-{end:,} done")

    # Reconstruct volumes
    r2star_vol = np.zeros(mask.shape, dtype=np.float32)
    s0_vol = np.zeros(mask.shape, dtype=np.float32)
    r2star_vol[mask] = r2star_flat
    s0_vol[mask] = s0_flat

    save_nii(r2star_vol, ref_img, OUT_DIR / "R2star_map.nii.gz", "R2*")
    save_nii(s0_vol, ref_img, OUT_DIR / "S0_map.nii.gz", "S0")

    # Validate against existing R2* map
    ref_r2, _ = load_nii(DERIV_QMRI / "R2star_map.nii.gz")
    both_mask = mask & (ref_r2 > 0) & (r2star_vol > 0)
    if both_mask.sum() > 100:
        corr = np.corrcoef(r2star_vol[both_mask], ref_r2[both_mask])[0, 1]
        mae = np.mean(np.abs(r2star_vol[both_mask] - ref_r2[both_mask]))
        print(f"  Validation vs qmri R2*: r={corr:.4f}, MAE={mae:.2f} s^-1")

    print(f"  R2* fitting done in {time.time() - t0:.1f}s")
    return r2star_vol, s0_vol, ref_img, mask


# ---------------------------------------------------------------------------
# Step 2: qBOLD OEF mapping
# ---------------------------------------------------------------------------

def step_oef(r2star_vol=None, s0_vol=None, ref_img=None, mask=None):
    """Per-voxel OEF from qBOLD model applied to 7-echo MEGRE."""
    from vpjax.qbold.signal_model import QBOLDParams, qbold_signal, compute_r2prime
    from vpjax.qbold.oef_mapping import fit_oef_voxel
    from vpjax.qbold.dbv import DBVParams

    print("\n=== Step 2: qBOLD OEF Mapping ===")
    t0 = time.time()

    # Load data if not passed from step 1
    if r2star_vol is None:
        r2star_path = OUT_DIR / "R2star_map.nii.gz"
        if not r2star_path.exists():
            print("  R2* map not found — run --step r2star first")
            return
        r2star_vol, ref_img = load_nii(r2star_path)
        mask_data, _ = load_nii(DERIV_QMRI / "megre_brain_mask.nii.gz")
        mask = mask_data > 0.5

    megre, _ = load_megre_magnitude()
    te = jnp.array(ECHO_TIMES)
    qbold_params = QBOLDParams(B0=jnp.array(B0))

    flat_data = jnp.array(megre[mask])  # (N, 7)
    n_voxels = flat_data.shape[0]
    print(f"  Fitting qBOLD for {n_voxels:,} voxels...")

    # Use the volume fitting function (already vmapped internally)
    from vpjax.qbold.oef_mapping import fit_oef_volume

    chunk_size = 100_000  # smaller chunks — qBOLD fitting is heavier (200 grad steps each)
    oef_flat = np.zeros(n_voxels, dtype=np.float32)
    dbv_flat = np.zeros(n_voxels, dtype=np.float32)
    r2_tissue_flat = np.zeros(n_voxels, dtype=np.float32)

    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        chunk = flat_data[start:end]
        result = fit_oef_volume(chunk, te, qbold_params)
        oef_flat[start:end] = np.asarray(result["oef"])
        dbv_flat[start:end] = np.asarray(result["dbv"])
        r2_tissue_flat[start:end] = np.asarray(result["r2"])
        print(f"    chunk {start:,}-{end:,} done")

    # Reconstruct volumes
    oef_vol = np.zeros(mask.shape, dtype=np.float32)
    dbv_vol = np.zeros(mask.shape, dtype=np.float32)
    oef_vol[mask] = oef_flat
    dbv_vol[mask] = dbv_flat

    # Compute R2' from fitted OEF + DBV
    r2prime_vol = np.zeros(mask.shape, dtype=np.float32)
    r2p = compute_r2prime(
        jnp.array(oef_flat),
        jnp.array(dbv_flat),
        qbold_params,
    )
    r2prime_vol[mask] = np.asarray(r2p)

    save_nii(oef_vol, ref_img, OUT_DIR / "OEF_map.nii.gz", "OEF")
    save_nii(dbv_vol, ref_img, OUT_DIR / "DBV_map.nii.gz", "DBV")
    save_nii(r2prime_vol, ref_img, OUT_DIR / "R2prime_map.nii.gz", "R2'")

    print(f"  qBOLD OEF fitting done in {time.time() - t0:.1f}s")
    return oef_vol, dbv_vol


# ---------------------------------------------------------------------------
# Step 3: Iron-myelin decomposition
# ---------------------------------------------------------------------------

def step_ironmyelin(ref_img=None, mask=None):
    """Iron-myelin separation from R2* + QSM chi map."""
    from vpjax.layers.iron_myelin import IronMyelinParams, decompose_r2star_qsm

    print("\n=== Step 3: Iron-Myelin Decomposition ===")
    t0 = time.time()

    # Load R2* (vpjax or existing)
    r2star_path = OUT_DIR / "R2star_map.nii.gz"
    if r2star_path.exists():
        r2star_vol, ref_img = load_nii(r2star_path)
    else:
        r2star_vol, ref_img = load_nii(DERIV_QMRI / "R2star_map.nii.gz")

    if mask is None:
        mask_data, _ = load_nii(DERIV_QMRI / "megre_brain_mask.nii.gz")
        mask = mask_data > 0.5

    # Load QSM chi map
    chi_path = DERIV_QSM / f"{SUB}_{SES}_MEGRE_Chimap.nii"
    if not chi_path.exists():
        print(f"  QSM chi map not found at {chi_path}")
        return
    chi_vol, chi_img = load_nii(chi_path)
    print(f"  R2* shape: {r2star_vol.shape}, chi shape: {chi_vol.shape}")

    # Resample chi to R2* space if shapes differ
    if chi_vol.shape != r2star_vol.shape:
        print(f"  WARNING: shape mismatch R2*={r2star_vol.shape} vs chi={chi_vol.shape}")
        print("  Using R2* shape — will need registration for production use")
        # For now, crop/pad to match
        min_shape = tuple(min(a, b) for a, b in zip(r2star_vol.shape, chi_vol.shape))
        r2_crop = r2star_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
        chi_crop = chi_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
        mask_crop = mask[:min_shape[0], :min_shape[1], :min_shape[2]]
    else:
        r2_crop, chi_crop, mask_crop = r2star_vol, chi_vol, mask

    # Flatten and decompose
    params = IronMyelinParams()
    r2_flat = jnp.array(r2_crop[mask_crop])
    chi_flat = jnp.array(chi_crop[mask_crop])

    decompose_fn = jax.vmap(lambda r, c: decompose_r2star_qsm(r, c, params))
    iron_flat, myelin_flat = decompose_fn(r2_flat, chi_flat)

    # Reconstruct
    iron_vol = np.zeros(r2_crop.shape, dtype=np.float32)
    myelin_vol = np.zeros(r2_crop.shape, dtype=np.float32)
    iron_vol[mask_crop] = np.asarray(iron_flat)
    myelin_vol[mask_crop] = np.asarray(myelin_flat)

    save_nii(iron_vol, ref_img, OUT_DIR / "iron_map.nii.gz", "Iron")
    save_nii(myelin_vol, ref_img, OUT_DIR / "myelin_map.nii.gz", "Myelin")

    print(f"  Iron-myelin decomposition done in {time.time() - t0:.1f}s")
    return iron_vol, myelin_vol


# ---------------------------------------------------------------------------
# Step 4: Regional CMRO2 via Fick
# ---------------------------------------------------------------------------

def step_cmro2():
    """Combine CBF (oxford_asl) + OEF (qBOLD) → regional CMRO2."""
    from vpjax.metabolism.fick import FickParams, fick_cmro2, compute_cao2

    print("\n=== Step 4: Regional CMRO2 (Fick) ===")
    t0 = time.time()

    oef_path = OUT_DIR / "OEF_map.nii.gz"
    if not oef_path.exists():
        print("  OEF map not found — run --step oef first")
        return

    oef_vol, ref_img = load_nii(oef_path)
    mask_data, _ = load_nii(DERIV_QMRI / "megre_brain_mask.nii.gz")
    mask = mask_data > 0.5

    # Load CBF from oxford_asl (in native perfusion space)
    cbf_path = DERIV_PERF / "oxford_asl" / "native_space" / "perfusion_calib.nii.gz"
    if not cbf_path.exists():
        print(f"  CBF map not found at {cbf_path}")
        return
    cbf_vol, cbf_img = load_nii(cbf_path)
    print(f"  OEF shape: {oef_vol.shape}, CBF shape: {cbf_vol.shape}")

    if cbf_vol.shape != oef_vol.shape:
        print("  WARNING: CBF and OEF are in different spaces — need registration")
        print("  Skipping CMRO2 computation (requires spatial alignment)")
        print("  To fix: register CBF to ses-06 MEGRE space with ANTs or FSL FLIRT")
        return

    # Compute arterial O2 content
    fick_params = FickParams()
    cao2 = compute_cao2(fick_params)

    # CMRO2 = CBF × OEF × CaO2
    oef_flat = jnp.array(oef_vol[mask])
    cbf_flat = jnp.array(cbf_vol[mask])
    cmro2_flat = fick_cmro2(cbf_flat, oef_flat, cao2)

    cmro2_vol = np.zeros(mask.shape, dtype=np.float32)
    cmro2_vol[mask] = np.asarray(cmro2_flat)

    save_nii(cmro2_vol, ref_img, OUT_DIR / "CMRO2_regional.nii.gz", "CMRO2")

    # Compare to existing CMRO2
    existing_cmro2_path = DERIV_PERF / "CMRO2_map.nii.gz"
    if existing_cmro2_path.exists():
        ref_cmro2, _ = load_nii(existing_cmro2_path)
        if ref_cmro2.shape == cmro2_vol.shape:
            both = mask & (ref_cmro2 > 0) & (cmro2_vol > 0)
            if both.sum() > 100:
                corr = np.corrcoef(cmro2_vol[both], ref_cmro2[both])[0, 1]
                print(f"  Validation vs existing CMRO2: r={corr:.4f}")

    print(f"  Regional CMRO2 done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 5: Generate figures for WAND paper
# ---------------------------------------------------------------------------

def step_figures():
    """Generate publication figures from vpjax outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    print("\n=== Step 5: Generating Figures ===")

    mask_data, _ = load_nii(DERIV_QMRI / "megre_brain_mask.nii.gz")
    mask = mask_data > 0.5

    # Pick an axial slice near centre of brain
    z_mid = mask.shape[2] // 2

    # --- Figure 1: R2* validation (vpjax vs existing qmri) ---
    vpjax_r2_path = OUT_DIR / "R2star_map.nii.gz"
    ref_r2_path = DERIV_QMRI / "R2star_map.nii.gz"
    if vpjax_r2_path.exists() and ref_r2_path.exists():
        vpjax_r2, ref_img = load_nii(vpjax_r2_path)
        ref_r2, _ = load_nii(ref_r2_path)
        both = mask & (ref_r2 > 0) & (vpjax_r2 > 0)
        corr = np.corrcoef(vpjax_r2[both], ref_r2[both])[0, 1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(vpjax_r2[:, :, z_mid].T, cmap="hot", vmin=0, vmax=80, origin="lower")
        axes[0].set_title("vpjax R2* (s$^{-1}$)")
        plt.colorbar(im0, ax=axes[0], shrink=0.7)

        im1 = axes[1].imshow(ref_r2[:, :, z_mid].T, cmap="hot", vmin=0, vmax=80, origin="lower")
        axes[1].set_title("QUIT R2* (s$^{-1}$)")
        plt.colorbar(im1, ax=axes[1], shrink=0.7)

        axes[2].hexbin(ref_r2[both], vpjax_r2[both], gridsize=80, cmap="inferno",
                        mincnt=1, extent=[0, 80, 0, 80])
        axes[2].plot([0, 80], [0, 80], "w--", lw=1)
        axes[2].set_xlabel("QUIT R2* (s$^{-1}$)")
        axes[2].set_ylabel("vpjax R2* (s$^{-1}$)")
        axes[2].set_title(f"Correlation r = {corr:.4f}")
        axes[2].set_aspect("equal")

        for ax in axes[:2]:
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig1_r2star_validation.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Fig 1: R2* validation saved (r={corr:.4f})")

    # --- Figure 2: qBOLD OEF + DBV + R2' maps ---
    oef_path = OUT_DIR / "OEF_map.nii.gz"
    dbv_path = OUT_DIR / "DBV_map.nii.gz"
    r2p_path = OUT_DIR / "R2prime_map.nii.gz"
    if oef_path.exists() and dbv_path.exists():
        oef_vol, _ = load_nii(oef_path)
        dbv_vol, _ = load_nii(dbv_path)

        maps = [("OEF", oef_vol, "RdYlBu_r", 0.0, 0.7),
                ("DBV", dbv_vol, "YlOrRd", 0.0, 0.10)]
        if r2p_path.exists():
            r2p_vol, _ = load_nii(r2p_path)
            maps.append(("R2' (s$^{-1}$)", r2p_vol, "magma", 0, 20))

        fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 5))
        if len(maps) == 1:
            axes = [axes]
        for ax, (title, vol, cmap, vmin, vmax) in zip(axes, maps):
            im = ax.imshow(vol[:, :, z_mid].T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig2_qbold_oef_dbv.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("  Fig 2: qBOLD OEF/DBV maps saved")

    # --- Figure 3: Iron-myelin decomposition ---
    iron_path = OUT_DIR / "iron_map.nii.gz"
    myelin_path = OUT_DIR / "myelin_map.nii.gz"
    chi_path = DERIV_QSM / f"{SUB}_{SES}_MEGRE_Chimap.nii"
    if iron_path.exists() and myelin_path.exists():
        iron_vol, _ = load_nii(iron_path)
        myelin_vol, _ = load_nii(myelin_path)

        # Load R2* and chi for the source maps panel
        r2_vol = vpjax_r2 if vpjax_r2_path.exists() else None
        chi_vol = None
        if chi_path.exists():
            chi_vol, _ = load_nii(chi_path)

        n_panels = 2 + (1 if r2_vol is not None else 0) + (1 if chi_vol is not None else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        idx = 0

        z_iron = iron_vol.shape[2] // 2  # may differ from z_mid if cropped

        if r2_vol is not None:
            im = axes[idx].imshow(r2_vol[:, :, z_mid].T, cmap="hot", vmin=0, vmax=80, origin="lower")
            axes[idx].set_title("R2* (s$^{-1}$)")
            plt.colorbar(im, ax=axes[idx], shrink=0.7)
            axes[idx].axis("off")
            idx += 1

        if chi_vol is not None:
            z_chi = chi_vol.shape[2] // 2
            im = axes[idx].imshow(chi_vol[:, :, z_chi].T, cmap="RdBu_r", vmin=-0.1, vmax=0.1, origin="lower")
            axes[idx].set_title("QSM $\\chi$ (ppm)")
            plt.colorbar(im, ax=axes[idx], shrink=0.7)
            axes[idx].axis("off")
            idx += 1

        im = axes[idx].imshow(iron_vol[:, :, z_iron].T, cmap="Reds", vmin=0, vmax=np.percentile(iron_vol[iron_vol > 0], 98) if np.any(iron_vol > 0) else 1, origin="lower")
        axes[idx].set_title("Iron (a.u.)")
        plt.colorbar(im, ax=axes[idx], shrink=0.7)
        axes[idx].axis("off")
        idx += 1

        im = axes[idx].imshow(myelin_vol[:, :, z_iron].T, cmap="Blues", vmin=0, vmax=np.percentile(myelin_vol[myelin_vol > 0], 98) if np.any(myelin_vol > 0) else 1, origin="lower")
        axes[idx].set_title("Myelin (a.u.)")
        plt.colorbar(im, ax=axes[idx], shrink=0.7)
        axes[idx].axis("off")

        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig3_iron_myelin_decomposition.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("  Fig 3: Iron-myelin decomposition saved")

    # --- Figure 4: Multi-echo signal decay example (single voxel) ---
    megre_path = ANAT_DIR / f"{SUB}_{SES}_echo-01_part-mag_MEGRE.nii.gz"
    if megre_path.exists() and oef_path.exists():
        from vpjax.qbold.signal_model import QBOLDParams, qbold_signal
        megre, _ = load_megre_magnitude()
        oef_vol, _ = load_nii(oef_path)
        dbv_vol, _ = load_nii(dbv_path)

        # Pick a high-OEF voxel within the mask
        oef_masked = oef_vol.copy()
        oef_masked[~mask] = 0
        voxel = np.unravel_index(np.argmax(oef_masked[:, :, z_mid]), oef_masked[:, :, z_mid].shape)
        ix, iy, iz = voxel[0], voxel[1], z_mid

        signal_obs = megre[ix, iy, iz, :]
        oef_val = oef_vol[ix, iy, iz]
        dbv_val = dbv_vol[ix, iy, iz]

        te_fine = jnp.linspace(0.003, 0.038, 100)
        qp = QBOLDParams(B0=jnp.array(B0), S0=jnp.array(float(signal_obs[0])))
        model_fit = qbold_signal(te_fine, jnp.array(oef_val), jnp.array(dbv_val), qp)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ECHO_TIMES * 1000, signal_obs, "ko", ms=8, label="Measured")
        ax.plot(np.asarray(te_fine) * 1000, np.asarray(model_fit), "r-", lw=2,
                label=f"qBOLD fit (OEF={oef_val:.2f}, DBV={dbv_val:.3f})")
        ax.set_xlabel("Echo Time (ms)")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title(f"Multi-echo decay — voxel ({ix},{iy},{iz})")
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig4_echo_decay_fit.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("  Fig 4: Echo decay fit example saved")

    print(f"  All figures saved to {FIG_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="vpjax WAND processing pipeline")
    parser.add_argument("--step", choices=["r2star", "oef", "ironmyelin", "cmro2", "figures", "all"],
                        default="all")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args()

    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    print(f"vpjax WAND pipeline — {SUB} {SES}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  Output: {OUT_DIR}")

    if args.step in ("r2star", "all"):
        r2, s0, ref, mask = step_r2star()
    else:
        r2, s0, ref, mask = None, None, None, None

    if args.step in ("oef", "all"):
        step_oef(r2, s0, ref, mask)

    if args.step in ("ironmyelin", "all"):
        step_ironmyelin(ref, mask)

    if args.step in ("cmro2", "all"):
        step_cmro2()

    if args.step in ("figures", "all"):
        step_figures()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
