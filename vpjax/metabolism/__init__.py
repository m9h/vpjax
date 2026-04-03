"""Metabolism: neural activity → oxygen consumption."""

from vpjax.metabolism.cmro2 import (
    CMRO2Params,
    compute_cmro2,
    compute_cmro2_absolute,
    compute_cmro2_from_cbf_oef,
)
from vpjax.metabolism.oef import (
    OEFParams,
    compute_oef,
    extraction_fraction,
    oef_from_coupled_ratio,
)
from vpjax.metabolism.fick import (
    FickParams,
    compute_cao2,
    fick_cbf,
    fick_cmro2,
    fick_oef,
)

__all__ = [
    "CMRO2Params",
    "compute_cmro2",
    "compute_cmro2_absolute",
    "compute_cmro2_from_cbf_oef",
    "OEFParams",
    "compute_oef",
    "extraction_fraction",
    "oef_from_coupled_ratio",
    "FickParams",
    "compute_cao2",
    "fick_cmro2",
    "fick_oef",
    "fick_cbf",
]
