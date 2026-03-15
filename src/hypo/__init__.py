"""
refracpo: Physical optics analysis toolkit for refractive optical systems.

Modules
-------
coordinate
    Coordinate system utilities for the full model, including coordinate-frame definitions
    and transforms. Use:
        from refracpo.coordinate import coord_sys, global_coord
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _version
from typing import Any
import numpy as np

# Physical Constants
c = 299792458 # speed of light in m/s
mu = 4 * np.pi * 10**(-7)  # permeability in H/m
epsilon = 8.854187817 * 10**(-12)  # permittivity in F/m
Z0 = np.sqrt(mu / epsilon, dtype=np.float64)  # impedance of free space in Ohm
#Z0 = 1

#__all__ = ["__version__", "coordinate", "c", "mu", "epsilon", "Z0"]
__all__ = ["__version__", "coordinate", 'rim','surface','lenspy']


try:
    __version__ = _version("hypo")
except PackageNotFoundError:
    __version__ = "0.0.0"


def __getattr__(name: str) -> Any:
    """Lazy import selected submodules to keep top-level import fast."""
    if name == "coordinate":
        obj = import_module(".coordinate", __name__)
        globals()[name] = obj  # cache module object
        return obj
    raise AttributeError(f"module 'hypo' has no attribute {name!r}")