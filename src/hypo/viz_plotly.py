"""
Reusable Plotly 3D visualization helpers for hypo optical elements.

This module is intentionally decoupled from any single element class (such as
``simple_Lens`` in :mod:`hypo.lenspy`). It works purely through duck-typing on
the small set of interfaces already shared across the package's geometry
objects, so it can be reused for other elements (reflectors, feeds, ...) later
without editing this file's existing functions or the element classes
themselves:

- ``surf``: an object with ``surf.sag(x, y) -> z`` in its own local frame
  (e.g. anything derived from :class:`hypo.surface.Surface`).
- ``rim``: an object with ``rim.sampling(Nx, Ny, quadrature=...) -> (x, y, w)``
  (e.g. :class:`hypo.rim.Elliptical_rim`, :class:`hypo.rim.Rect_rim`).
- ``face_coord_sys``: a :class:`hypo.coordinate.coord_sys` whose
  ``Local_to_Global(x, y, z)`` maps that face's local samples directly into
  the global/world frame. Because ``coord_sys`` composes its transform with
  its own ``ref_coord`` at construction time, this works regardless of how
  deeply the face frame is nested (face -> body -> mount -> ... -> global).

Low-level building blocks
--------------------------
- ``sample_face_mesh``: sample + triangulate one face, return global mesh.
- ``rim_boundary_xy``: local (x, y) points tracing an aperture rim outline.
- ``new_figure``: create a ``go.Figure`` with sane 3-D layout defaults.
- ``add_face_trace``: add one triangulated face as a ``Mesh3d`` trace.
- ``add_side_wall_trace``: connect two faces around a shared rim boundary.

High-level convenience wrappers
--------------------------------
- ``plot_lens``: display a :class:`hypo.lenspy.simple_Lens` (both faces + the
  side wall around the aperture).

Add further ``plot_<element>`` wrappers here as new element types need a
Plotly view; they should be built from the same low-level helpers above.
"""

from typing import Any, Optional, Tuple

import numpy as np


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for hypo.viz_plotly (pip install plotly).") from exc
    return go


def _require_delaunay():
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise ImportError("scipy is required for hypo.viz_plotly (pip install scipy).") from exc
    return Delaunay


def rim_boundary_xy(rim: Any, n: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` points tracing an aperture rim's outline.

    Supports the rim shapes already defined in :mod:`hypo.rim`:

    - Elliptical rims (objects exposing ``cx``, ``cy``, ``a``, ``b``), such as
      :class:`hypo.rim.Elliptical_rim`.
    - Rectangular rims (objects exposing ``cx``, ``cy``, ``sizex``,
      ``sizey``), such as :class:`hypo.rim.Rect_rim`.

    Parameters
    ----------
    rim:
        Aperture rim object.
    n:
        Number of boundary points to generate (spread evenly around/along the
        perimeter).
    """
    if n < 3:
        raise ValueError("n must be >= 3.")

    if hasattr(rim, "a") and hasattr(rim, "b") and hasattr(rim, "cx"):
        phi = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
        x = rim.cx + rim.a * np.cos(phi)
        y = rim.cy + rim.b * np.sin(phi)
        return x, y

    if hasattr(rim, "sizex") and hasattr(rim, "sizey") and hasattr(rim, "cx"):
        # Distribute points proportionally along the four edges of the rectangle.
        perimeter = 2.0 * (rim.sizex + rim.sizey)
        n_x = max(1, round(n * rim.sizex / perimeter))
        n_y = max(1, (n - 2 * n_x) // 2)
        x0, x1 = rim.cx - rim.sizex / 2, rim.cx + rim.sizex / 2
        y0, y1 = rim.cy - rim.sizey / 2, rim.cy + rim.sizey / 2
        top_x = np.linspace(x0, x1, n_x, endpoint=False)
        right_y = np.linspace(y1, y0, n_y, endpoint=False)
        bottom_x = np.linspace(x1, x0, n_x, endpoint=False)
        left_y = np.linspace(y0, y1, max(1, n - 2 * n_x - n_y), endpoint=False)
        x = np.concatenate([top_x, np.full_like(right_y, x1), bottom_x, np.full_like(left_y, x0)])
        y = np.concatenate([np.full_like(top_x, y1), right_y, np.full_like(bottom_x, y0), left_y])
        return x, y

    raise TypeError(
        "Don't know how to trace the boundary of this rim type; "
        "pass explicit boundary points instead of using rim_boundary_xy()."
    )


def sample_face_mesh(
    surf: Any,
    rim: Any,
    face_coord_sys: Any,
    Nx: int = 60,
    Ny: int = 60,
    quadrature: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one refracting/reflecting face and triangulate it for plotting.

    The face is sampled on its aperture rim in the face-local ``(x, y)``
    plane, triangulated there (a Delaunay triangulation is well-behaved on a
    roughly planar/convex-ish point set such as a clipped aperture grid), and
    then carried into the global frame through ``face_coord_sys``.

    Returns
    -------
    x, y, z:
        Global mesh vertex coordinates.
    triangles:
        ``(M, 3)`` integer array of vertex indices, one row per triangle.
    """
    Delaunay = _require_delaunay()
    x, y, _w = rim.sampling(Nx, Ny, quadrature=quadrature)
    z = surf.sag(x, y)
    triangles = Delaunay(np.column_stack((x, y))).simplices
    xg, yg, zg = face_coord_sys.Local_to_Global(x, y, z)
    return np.asarray(xg), np.asarray(yg), np.asarray(zg), triangles


def new_figure(title: Optional[str] = None):
    """Create a ``go.Figure`` with axis-equal 3-D layout defaults (mm)."""
    go = _require_plotly()
    fig = go.Figure()
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)",
        ),
    )
    return fig


def add_face_trace(
    fig: Any,
    surf: Any,
    rim: Any,
    face_coord_sys: Any,
    Nx: int = 60,
    Ny: int = 60,
    quadrature: str = "uniform",
    color: str = "lightblue",
    opacity: float = 0.9,
    name: str = "face",
):
    """Sample one face and add it to ``fig`` as a ``Mesh3d`` trace."""
    go = _require_plotly()
    x, y, z, triangles = sample_face_mesh(surf, rim, face_coord_sys, Nx, Ny, quadrature)
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color=color, opacity=opacity, name=name,
        showlegend=True, flatshading=False,
    ))
    return fig


def add_side_wall_trace(
    fig: Any,
    surf_a: Any,
    coord_a: Any,
    surf_b: Any,
    coord_b: Any,
    rim: Any,
    n: int = 128,
    color: str = "lightgray",
    opacity: float = 0.9,
    name: str = "side wall",
):
    """Add a wall mesh connecting two faces around a shared aperture rim.

    ``surf_a``/``coord_a`` and ``surf_b``/``coord_b`` describe the two faces
    (e.g. the front and back of a lens); both are evaluated on the same
    ``rim`` boundary outline so the wall closes cleanly between them.
    """
    go = _require_plotly()
    xb, yb = rim_boundary_xy(rim, n)
    n_pts = len(xb)

    za = surf_a.sag(xb, yb)
    zb = surf_b.sag(xb, yb)
    xa_g, ya_g, za_g = coord_a.Local_to_Global(xb, yb, za)
    xb_g, yb_g, zb_g = coord_b.Local_to_Global(xb, yb, zb)

    side_x = np.concatenate([xa_g, xb_g])
    side_y = np.concatenate([ya_g, yb_g])
    side_z = np.concatenate([za_g, zb_g])

    tri_i, tri_j, tri_k = [], [], []
    for idx in range(n_pts):
        idx_next = (idx + 1) % n_pts
        top_i, top_j = idx, idx_next
        bot_i, bot_j = n_pts + idx, n_pts + idx_next
        # Split the (top_i, top_j, bot_j, bot_i) quad into two triangles.
        tri_i.extend([top_i, top_i])
        tri_j.extend([top_j, bot_j])
        tri_k.extend([bot_j, bot_i])

    fig.add_trace(go.Mesh3d(
        x=side_x, y=side_y, z=side_z,
        i=tri_i, j=tri_j, k=tri_k,
        color=color, opacity=opacity, name=name,
        showlegend=True, flatshading=True,
    ))
    return fig


def plot_lens(
    lens: Any,
    Nx: int = 60,
    Ny: int = 60,
    N_boundary: int = 128,
    face1_color: str = "lightblue",
    face2_color: str = "lightsalmon",
    side_color: str = "lightgray",
    opacity: float = 0.9,
    show: bool = True,
):
    """Render a two-surface lens (such as ``hypo.lenspy.simple_Lens``) in Plotly.

    Only public attributes are used (``rim``, ``surface1``/``surface2``,
    ``coord_sys_f1``/``coord_sys_f2``, ``name``), so this works with any
    object that exposes that same shape without requiring changes to the
    element class itself.

    Parameters
    ----------
    lens:
        Object exposing ``rim``, ``surface1``, ``surface2``,
        ``coord_sys_f1``, ``coord_sys_f2`` (see :class:`hypo.lenspy.simple_Lens`).
    Nx, Ny:
        Sampling counts for each face mesh.
    N_boundary:
        Number of points used to build the side wall around the aperture rim.
    face1_color, face2_color, side_color:
        Plotly color names/hex for each mesh.
    opacity:
        Opacity applied to all three meshes.
    show:
        If True, immediately display the figure (``fig.show()``).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = new_figure(title=getattr(lens, "name", None))
    add_face_trace(
        fig, lens.surface1, lens.rim, lens.coord_sys_f1,
        Nx=Nx, Ny=Ny, color=face1_color, opacity=opacity, name="face 1",
    )
    add_face_trace(
        fig, lens.surface2, lens.rim, lens.coord_sys_f2,
        Nx=Nx, Ny=Ny, color=face2_color, opacity=opacity, name="face 2",
    )
    add_side_wall_trace(
        fig,
        lens.surface1, lens.coord_sys_f1,
        lens.surface2, lens.coord_sys_f2,
        lens.rim, n=N_boundary, color=side_color, opacity=opacity,
    )

    if show:
        fig.show()
    return fig
