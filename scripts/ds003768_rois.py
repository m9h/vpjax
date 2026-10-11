"""Region-of-interest BOLD series for a ds003768 run, via FSL and Harvard-Oxford.

The global BOLD mean shares about 1% of its variance with the occipital
alpha envelope (see README, "On a recording"). The obvious next pairing
is regional: visual cortex against occipital alpha, with a motor-strip
control region that the alpha envelope should *not* explain. This script
builds those masks in native EPI space and writes the mean series of
each, plus the global mean, to an npz the state-space driver can read.

Registration is the plain FSL chain: mean EPI -> T1w (6 dof) -> MNI152
2 mm (12 dof), inverted and applied to the Harvard-Oxford cortical
max-probability atlas with nearest-neighbour resampling. Adequate for
lobe-scale ROIs on 3 mm data; not for anything finer.

    python scripts/ds003768_rois.py --root /data/ds003768 --subject 01 --run 1 \\
        --out sub-01_rest1_rois.npz
"""

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np

ROIS = {
    "visual": [
        "Occipital Pole", "Lateral Occipital Cortex, superior division",
        "Lateral Occipital Cortex, inferior division", "Intracalcarine Cortex",
        "Cuneal Cortex", "Lingual Gyrus", "Supracalcarine Cortex",
    ],
    "motor": ["Precentral Gyrus", "Postcentral Gyrus"],
}


def fsl(*args):
    subprocess.run([str(a) for a in args], check=True, capture_output=True)


def atlas_values(fsldir: Path, names: list[str]) -> list[int]:
    """Label values in the max-prob image are XML index + 1."""
    xml = ET.parse(fsldir / "data/atlases/HarvardOxford-Cortical.xml")
    by_name = {lab.text.strip(): int(lab.get("index")) + 1 for lab in xml.iter("label")}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"not in Harvard-Oxford cortical atlas: {missing}")
    return [by_name[n] for n in names]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--subject", default="01")
    p.add_argument("--task", default="rest")
    p.add_argument("--run", default="1")
    p.add_argument("--work", help="directory for registration intermediates")
    p.add_argument("--out", required=True)
    p.add_argument("--bold", help="explicit BOLD path (overrides --root layout)")
    p.add_argument("--t1", help="explicit T1w path (overrides --root layout)")
    a = p.parse_args()

    fsldir = Path(os.environ["FSLDIR"])
    root = Path(a.root)
    stem = f"sub-{a.subject}_task-{a.task}_run-{a.run}"
    bold_path = Path(a.bold) if a.bold else root / f"sub-{a.subject}/func/{stem}_bold.nii.gz"
    t1_path = Path(a.t1) if a.t1 else root / f"sub-{a.subject}/anat/sub-{a.subject}_T1w.nii.gz"
    if a.bold:
        stem = Path(a.bold).name.replace("_bold.nii.gz", "")
    work = Path(a.work or Path(a.out).with_suffix("")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    mni = fsldir / "data/standard/MNI152_T1_2mm_brain.nii.gz"
    atlas = fsldir / "data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"

    # --- registration -------------------------------------------------------
    epi_mean = work / "epi_mean.nii.gz"
    t1_brain = work / "t1_brain.nii.gz"
    fsl("fslmaths", bold_path, "-Tmean", epi_mean)
    fsl("bet", t1_path, t1_brain, "-R", "-f", "0.4")
    fsl("flirt", "-in", epi_mean, "-ref", t1_brain, "-dof", "6",
        "-omat", work / "epi2t1.mat", "-out", work / "epi2t1.nii.gz")
    fsl("flirt", "-in", t1_brain, "-ref", mni, "-dof", "12",
        "-omat", work / "t12mni.mat", "-out", work / "t12mni.nii.gz")
    fsl("convert_xfm", "-omat", work / "epi2mni.mat", "-concat", work / "t12mni.mat", work / "epi2t1.mat")
    fsl("convert_xfm", "-omat", work / "mni2epi.mat", "-inverse", work / "epi2mni.mat")
    atlas_epi = work / "atlas_epi.nii.gz"
    fsl("flirt", "-in", atlas, "-ref", epi_mean, "-applyxfm", "-init", work / "mni2epi.mat",
        "-interp", "nearestneighbour", "-out", atlas_epi)

    # --- extraction ---------------------------------------------------------
    img = nib.load(str(bold_path))
    data = img.get_fdata(dtype=np.float32)
    labels = np.asarray(nib.load(str(atlas_epi)).dataobj).astype(int)
    mean_img = data.mean(axis=-1)
    brain = mean_img > np.percentile(mean_img, 10)

    series = {"global": data[brain].mean(axis=0)}
    counts = {"global": int(brain.sum())}
    for roi, names in ROIS.items():
        mask = np.isin(labels, atlas_values(fsldir, names)) & brain
        if mask.sum() < 20:
            raise RuntimeError(f"{roi} mask has only {int(mask.sum())} voxels; registration failed?")
        series[roi] = data[mask].mean(axis=0)
        counts[roi] = int(mask.sum())
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine), str(work / f"mask_{roi}.nii.gz"))

    tr = float(json.load(open(str(bold_path).replace(".nii.gz", ".json")))["RepetitionTime"])
    np.savez(a.out, tr=tr, voxel_counts=json.dumps(counts),
             **{f"bold_{k}": v.astype(np.float64) for k, v in series.items()})
    print(f"{stem}: TR {tr} s, voxels {counts}; wrote {a.out}")


if __name__ == "__main__":
    main()
