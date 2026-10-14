"""Serotonin receptor PET maps on HCP-MMP1.0 and on the group connectome's modes.

Spatial priors for where an external time course enters the E/I block:
each receptor map is taken from MNI152 to fsaverage (neuromaps
registration fusion), averaged within the 360 HCP-MMP parcels, z-scored,
and projected on the eigenmodes of the multiscale structural connectome.
The outputs are regional densities (CSV) and modal coefficients (JSON)
for ``EIBlock.input_vector(entry, mode_weights=...)``.  Nothing here is
a property of any compound: the maps are tracer binding in healthy
volunteers (Hansen et al. 2022, 10.1038/s41593-022-01186-3; data
CC BY-NC-SA 4.0, cite the source studies in their Table S3).

    python scripts/receptor_modes.py --maps ~/Workspace/data/hansen_receptors \\
        --msc ~/Workspace/data/msfd_gsp/HCP_MMP_Fusion_all.mat \\
        --annot ~/Workspace/data/hcp_mmp_fsaverage --out docs/results/receptor_hcp_mmp
"""

import argparse
import csv
import glob
import json
import os

import nibabel as nib
import numpy as np

from vpjax.neural.connectome import (
    hcp_mmp_names, laplacian_modes, load_msc, mode_weights, parcellate_fsaverage,
    spectral_concentration,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maps", required=True, help="directory of MNI152 NIfTI receptor maps")
    p.add_argument("--msc", required=True)
    p.add_argument("--annot", required=True)
    p.add_argument("--key", default="SC_MPC_GD")
    p.add_argument("--out", required=True, help="output stem (.csv and .json are written)")
    a = p.parse_args()

    from neuromaps import transforms

    names = hcp_mmp_names(a.annot)
    W = load_msc(a.msc, a.key)
    eigs, U = laplacian_modes(W)
    regional, modal, summary = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(a.maps, "*.nii*"))):
        label = os.path.basename(f).split(".nii")[0]
        img = nib.load(f)
        if img.ndim == 4:
            img = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine, img.header)
        lh, rh = transforms.mni152_to_fsaverage(img, fsavg_density="164k", method="linear")
        r = parcellate_fsaverage(lh.agg_data(), rh.agg_data(), a.annot)
        if not np.isfinite(r).all():
            raise RuntimeError(f"{label}: {(~np.isfinite(r)).sum()} parcels without vertices")
        z = (r - r.mean()) / r.std()
        regional[label] = z
        w = mode_weights(U, z)
        modal[label] = w.tolist()
        summary[label] = {f"energy_first_{k}": round(spectral_concentration(U, z, k), 4) for k in (10, 50, 100, 200)}
        summary[label]["dominant_modes"] = [int(i) for i in np.argsort(-(w**2))[:5]]
        print(label, summary[label], flush=True)

    with open(a.out + ".csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["region"] + list(regional))
        for i, n in enumerate(names):
            wr.writerow([n] + [f"{regional[k][i]:.4f}" for k in regional])
    json.dump({"msc_key": a.key, "laplacian": "normalised", "eigenvalues": eigs.tolist(),
               "mode_weights": modal, "summary": summary}, open(a.out + ".json", "w"))
    print(f"wrote {a.out}.csv and {a.out}.json")


if __name__ == "__main__":
    main()
