"""vpjax — Virtual Physiology in JAX.

Differentiable models of cerebrovascular physiology:
neural activity → metabolic demand → blood flow → BOLD/ASL signals.

Complement to vbjax (Virtual Brain in JAX).
"""

__version__ = "0.1.0"

# Core types
from vpjax._types import BalloonParams, BalloonState, HemodynamicState

# Hemodynamic models
from vpjax.hemodynamics.balloon import BalloonWindkessel, solve_balloon
from vpjax.hemodynamics.bold import BOLDParams, observe_bold
from vpjax.hemodynamics.optics import to_optical_properties

# Perfusion observers
from vpjax.perfusion.asl import observe_asl
from vpjax.perfusion.vaso import observe_vaso

__all__ = [
    # Types
    "BalloonParams",
    "BalloonState",
    "HemodynamicState",
    # Models
    "BalloonWindkessel",
    "solve_balloon",
    # Observers
    "BOLDParams",
    "observe_bold",
    "observe_asl",
    "observe_vaso",
    "to_optical_properties",
]
