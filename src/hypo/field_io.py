"""
Field/surface HDF5 I/O utilities.

This module is a modern replacement for `RWcur.py` ("write/read current"),
with clearer function names and basic validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

from .vecops import Vector


PathLike = Union[str, Path]


def _as_1d(a):
    """Return a flattened 1D NumPy array."""
    return np.asarray(a).ravel()


def _require_attrs(obj, attrs, obj_name: str):
    for name in attrs:
        if not hasattr(obj, name):
            raise AttributeError(f"{obj_name} must provide attribute '{name}'")


def write_surface_field_h5(
    file_h5: h5py.File,
    face,
    face_n,
    E: Vector,
    H: Vector,
    t_p,
    t_s,
    group_name: str = "face",
) -> None:
    """
    Save one sampled surface + fields into an HDF5 group.

    Parameters
    ----------
    file_h5 : h5py.File
        Open HDF5 file handle in write/append mode.
    face : object
        Must provide x, y, z, w.
    face_n : object
        Must provide nx/ny/nz semantics via x, y, z. Optional N.
    E, H : Vector
        Electric and magnetic fields.
    T, R : array-like or scalar
        Transmission/reflection coefficients or summaries.
    group_name : str
        HDF5 group name (for example "f1" or "f2").
    """
    if group_name in file_h5:
        del file_h5[group_name]
    group = file_h5.create_group(group_name)

    _require_attrs(face, ("x", "y", "z", "w"), "face")
    _require_attrs(face_n, ("x", "y", "z"), "face_n")

    group.create_dataset("x", data=_as_1d(face.x))
    group.create_dataset("y", data=_as_1d(face.y))
    group.create_dataset("z", data=_as_1d(face.z))
    group.create_dataset("w", data=_as_1d(face.w))

    group.create_dataset("nx", data=_as_1d(face_n.x))
    group.create_dataset("ny", data=_as_1d(face_n.y))
    group.create_dataset("nz", data=_as_1d(face_n.z))
    if hasattr(face_n, "N") and getattr(face_n, "N") is not None:
        group.create_dataset("N", data=_as_1d(face_n.N))

    group.create_dataset("Ex", data=_as_1d(E.x))
    group.create_dataset("Ey", data=_as_1d(E.y))
    group.create_dataset("Ez", data=_as_1d(E.z))
    group.create_dataset("Hx", data=_as_1d(H.x))
    group.create_dataset("Hy", data=_as_1d(H.y))
    group.create_dataset("Hz", data=_as_1d(H.z))

    group.create_dataset("tp", data=np.asarray(t_p))
    group.create_dataset("ts", data=np.asarray(t_s))


def read_surface_field_h5(filename: PathLike, group_name: str = "f2"):
    """
    Read one surface-field group from HDF5.

    Returns
    -------
    face : Vector
    face_n : Vector
    H : Vector
    E : Vector
    """
    with h5py.File(str(filename), "r") as f:
        if group_name not in f:
            raise KeyError(f"Group '{group_name}' not found in {filename}")
        g = f[group_name]

        face = Vector(
            g["x"][:].ravel(),
            g["y"][:].ravel(),
            g["z"][:].ravel(),
        )
        face.w = g["w"][:].ravel()

        face_n = Vector(
            g["nx"][:].ravel(),
            g["ny"][:].ravel(),
            g["nz"][:].ravel(),
        )
        face_n.N = g["N"][:].ravel() if "N" in g else None
        H = Vector(g["Hx"][:].ravel(), g["Hy"][:].ravel(), g["Hz"][:].ravel())
        E = Vector(g["Ex"][:].ravel(), g["Ey"][:].ravel(), g["Ez"][:].ravel())
    return face, face_n, H, E


# ---------------------------------------------------------------------------
# Backward-compatible aliases (matching legacy RWcur naming)
# ---------------------------------------------------------------------------
def saveh5_surf(file_h5, face, face_n, E, H, t_p, t_s, name="face"):
    """Backward-compatible alias of write_surface_field_h5."""
    return write_surface_field_h5(file_h5, face, face_n, E, H,  t_p, t_s, group_name=name)


def read_cur(filename, group_name: str = "f2"):
    """Backward-compatible alias of read_surface_field_h5."""
    return read_surface_field_h5(filename, group_name=group_name)
