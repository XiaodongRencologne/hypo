"""
Legacy compatibility layer for historical RWcur naming.

Use `refracpo.field_io` for new code.
"""

from .field_io import read_cur, read_surface_field_h5, saveh5_surf, write_surface_field_h5

__all__ = [
    "saveh5_surf",
    "read_cur",
    "write_surface_field_h5",
    "read_surface_field_h5",
]

