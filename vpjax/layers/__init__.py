"""Cortical depth-resolved physiology."""

from vpjax.layers.layering import assign_layers, equivolume_depths, layer_mask
from vpjax.layers.profiles import normalize_profile, sample_profile, sample_profile_weighted
from vpjax.layers.iron_myelin import (
    IronMyelinParams,
    chi_from_iron_myelin,
    decompose_r2star_qsm,
    decompose_with_bpf,
    r2star_from_iron_myelin,
)
from vpjax.layers.layer_nvc import (
    LayerNVCParams,
    ascending_vein_contamination,
    devein_bold,
    layer_stimulus,
)

__all__ = [
    "assign_layers",
    "equivolume_depths",
    "layer_mask",
    "sample_profile",
    "sample_profile_weighted",
    "normalize_profile",
    "IronMyelinParams",
    "decompose_r2star_qsm",
    "decompose_with_bpf",
    "r2star_from_iron_myelin",
    "chi_from_iron_myelin",
    "LayerNVCParams",
    "layer_stimulus",
    "ascending_vein_contamination",
    "devein_bold",
]
