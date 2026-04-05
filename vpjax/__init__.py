"""vpjax — Virtual Physiology in JAX.

Differentiable models of cerebrovascular physiology:
neural activity → metabolic demand → blood flow → BOLD/ASL/qBOLD signals.

Complement to vbjax (Virtual Brain in JAX).
"""

__version__ = "0.1.0"

# Core types
from vpjax._types import BalloonParams, BalloonState, HemodynamicState

# Hemodynamic models
from vpjax.hemodynamics.balloon import BalloonWindkessel, solve_balloon
from vpjax.hemodynamics.bold import BOLDParams, observe_bold
from vpjax.hemodynamics.optics import to_optical_properties
from vpjax.hemodynamics.riera import RieraNVC, RieraParams, RieraState

# Perfusion observers
from vpjax.perfusion.asl import observe_asl
from vpjax.perfusion.vaso import observe_vaso

# Subpackages (available as vpjax.metabolism, vpjax.qbold, etc.)
from vpjax import hemodynamics
from vpjax import metabolism
from vpjax import perfusion
from vpjax import qbold
from vpjax import vascular
from vpjax import layers
from vpjax import integrators
from vpjax import vaso
from vpjax import qsm
from vpjax import cardiac
from vpjax import sleep
from vpjax import brainstem

__all__ = [
    # Types
    "BalloonParams",
    "BalloonState",
    "HemodynamicState",
    # Hemodynamic models
    "BalloonWindkessel",
    "solve_balloon",
    "RieraNVC",
    "RieraParams",
    "RieraState",
    # Observers
    "BOLDParams",
    "observe_bold",
    "observe_asl",
    "observe_vaso",
    "to_optical_properties",
    # Subpackages
    "hemodynamics",
    "metabolism",
    "perfusion",
    "qbold",
    "vascular",
    "layers",
    "integrators",
    "vaso",
    "qsm",
    "cardiac",
    "sleep",
    "brainstem",
]
