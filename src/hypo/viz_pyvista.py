"""
Reusable PyVista 3D visualization helpers for hypo optical elements.

This module mirrors :mod:`hypo.viz_plotly`'s design: it is intentionally
decoupled from any single element class (such as ``simple_Lens`` in
:mod:`hypo.lenspy`) and works purely through duck-typing on the same small
set of interfaces already shared across the package's geometry objects, so
it can be reused for other elements (reflectors, feeds, ...) later without
editing this file's existing functions or the element classes themselves:

- ``surf``: an object with ``surf.sag(x, y) -> z`` in its own local frame
  (e.g. anything derived from :class:`hypo.surface.Surface`).
- ``rim``: an object with ``rim.sampling(Nx, Ny, quadrature=...) -> (x, y, w)``
  (e.g. :class:`hypo.rim.Elliptical_rim`, :class:`hypo.rim.Rect_rim`).
- ``face_coord_sys``: a :class:`hypo.coordinate.coord_sys` whose
  ``Local_to_Global(x, y, z)`` maps that face's local samples directly into
  the global/world frame.

Two pieces are borrowed from :mod:`hypo.viz_plotly`: ``rim_boundary_xy`` (pure
rim geometry) and ``sample_face_mesh`` (rim sampling + scipy-``Delaunay``
triangulation -- see below for why PyVista's own triangulator isn't used
instead). Neither import pulls in plotly itself (it's only imported lazily,
inside the functions that actually need it), so the two modules stay
independent renderers -- this one adds no plotly dependency.

Why the rim outline is sampled explicitly
------------------------------------------
``rim.sampling`` alone only returns *interior* points (e.g. a Cartesian grid
clipped inside the aperture, offset roughly half a cell in from the true
edge). Triangulating that alone leaves the mesh's outer boundary sitting
strictly inside the aperture rim rather than on it, so a side wall built from
the exact rim curve would not meet it -- a visible gap around the edge. To
avoid that, :func:`sample_face_mesh` appends the exact rim outline from
:func:`hypo.viz_plotly.rim_boundary_xy` to the sample set before
triangulating; those points become the last ``n_boundary`` rows of its
output, in the same order for every face sampled with the same
``rim``/``n_boundary`` -- so two faces' boundary rings line up point-for-point
and a wall (or a fully merged solid) can stitch them with no seam.

Low-level building blocks
--------------------------
- ``sample_face_mesh``: sample + triangulate one face, return global
  ``(points, triangles)``.
- ``face_polydata``: wrap ``sample_face_mesh`` as a ``pyvista.PolyData``.
- ``ring_wall_mesh``: build a quad-strip wall connecting two matching point
  rings, return ``(points, triangles)``.
- ``new_plotter``: create a ``pyvista.Plotter`` with sane defaults (axes,
  isometric view).
- ``set_view``: point a plotter's camera at a standard 3D / XY / YZ / ZX view.

High-level convenience wrappers
--------------------------------
- ``plot_lens``: display a :class:`hypo.lenspy.simple_Lens` as three separate
  meshes (face 1, face 2, side wall), each independently colored.
- ``lens_solid_mesh`` / ``lens_solid_polydata`` / ``plot_lens_solid``: merge
  both faces and the side wall into a single watertight point/triangle set
  (shared vertices at the seam -- not just coincident, duplicate ones), for
  rendering the lens as one solid body or for further mesh processing (e.g.
  exporting to STL via ``pyvista.PolyData.save``).

Add further ``plot_<element>`` wrappers here as new element types need a
PyVista view; they should be built from the same low-level helpers above.
"""

from typing import Any, Optional, Tuple

import numpy as np

from .viz_plotly import rim_boundary_xy, sample_face_mesh as _plotly_sample_face_mesh


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for hypo.viz_pyvista (pip install pyvista).") from exc
    return pv


def _pv_faces(triangles: np.ndarray) -> np.ndarray:
    """Convert an ``(M, 3)`` triangle index array to PyVista's flat ``faces`` format."""
    triangles = np.asarray(triangles, dtype=np.int64)
    return np.column_stack([np.full(len(triangles), 3, dtype=np.int64), triangles]).ravel()


def sample_face_mesh(
    surf: Any,
    rim: Any,
    face_coord_sys: Any,
    Nx: int = 60,
    Ny: int = 60,
    quadrature: str = "uniform",
    n_boundary: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample one refracting/reflecting face and triangulate it for plotting.

    The face is sampled on its aperture rim in the face-local ``(x, y)``
    plane (interior grid plus the exact rim outline -- see the module
    docstring) and carried into the global frame through ``face_coord_sys``.

    Triangulation is delegated to :func:`hypo.viz_plotly.sample_face_mesh`
    (a thin, scipy-``Delaunay``-based routine with no plotly dependency at
    import time) rather than PyVista's own ``delaunay_2d``: on this point
    set -- a dense interior grid plus a separate exact rim-outline ring --
    ``delaunay_2d`` was found to spuriously include several interior points
    on the triangulation's outer boundary (an artifact of VTK's
    incremental algorithm), leaving small gaps around the rim instead of a
    clean boundary matching ``n_boundary`` exactly. ``scipy.spatial.Delaunay``
    does not have that problem and is already a hard dependency of this
    package.

    Returns
    -------
    points:
        ``(N, 3)`` global mesh vertex coordinates (interior samples followed
        by the ``n_boundary`` rim-outline vertices).
    triangles:
        ``(M, 3)`` integer array of vertex indices, one row per triangle.
    """
    xg, yg, zg, triangles = _plotly_sample_face_mesh(surf, rim, face_coord_sys, Nx, Ny, quadrature, n_boundary)
    points = np.column_stack([np.asarray(xg), np.asarray(yg), np.asarray(zg)])
    return points, triangles


def face_polydata(
    surf: Any,
    rim: Any,
    face_coord_sys: Any,
    Nx: int = 60,
    Ny: int = 60,
    quadrature: str = "uniform",
    n_boundary: int = 128,
):
    """Sample one face and return it as a ``pyvista.PolyData``."""
    pv = _require_pyvista()
    points, triangles = sample_face_mesh(surf, rim, face_coord_sys, Nx, Ny, quadrature, n_boundary)
    return pv.PolyData(points, _pv_faces(triangles))


def ring_wall_mesh(ring_a: np.ndarray, ring_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build a quad-strip wall mesh connecting two matching, equal-length point rings.

    ``ring_a`` and ``ring_b`` are ``(n, 3)`` arrays tracing the same closed
    loop (e.g. a lens's front- and back-face rim outlines, in matching
    point order), such as the trailing rows :func:`sample_face_mesh` returns.

    Returns
    -------
    points:
        ``(2n, 3)`` vertex coordinates (``ring_a`` followed by ``ring_b``).
    triangles:
        ``(2n, 3)`` integer array of vertex indices, one row per triangle.
    """
    ring_a = np.asarray(ring_a)
    ring_b = np.asarray(ring_b)
    n = len(ring_a)
    if len(ring_b) != n:
        raise ValueError("ring_a and ring_b must have the same number of points.")

    points = np.vstack([ring_a, ring_b])
    tri = []
    for idx in range(n):
        idx_next = (idx + 1) % n
        top_i, top_j = idx, idx_next
        bot_i, bot_j = n + idx, n + idx_next
        # Split the (top_i, top_j, bot_j, bot_i) quad into two triangles.
        tri.append([top_i, top_j, bot_j])
        tri.append([top_i, bot_j, bot_i])
    return points, np.asarray(tri, dtype=np.int64)


def new_plotter(title: Optional[str] = None, parallel_projection: bool = True, **kwargs):
    """Create a ``pyvista.Plotter`` with sane defaults (axes, isometric view).

    ``parallel_projection`` defaults to True (orthographic camera, no
    perspective foreshortening) so that rotating the solid doesn't distort
    its apparent proportions -- pass ``False`` to keep PyVista's default
    perspective camera instead.
    """
    pv = _require_pyvista()
    plotter = pv.Plotter(**kwargs)
    plotter.show_axes()
    if title:
        plotter.add_title(title, font_size=10)
    plotter.view_isometric()
    if parallel_projection:
        plotter.enable_parallel_projection()
    return plotter


# Camera-view methods keyed by view name, matching pyvista.Plotter's own
# view_* methods (each an axis-aligned, orthographic-style camera position).
_VIEW_METHODS = {
    "3d": "view_isometric",
    "xy": "view_xy",
    "yz": "view_yz",
    "zx": "view_zx",
    "xz": "view_zx",  # alias: same front-on plane as "zx"
}


def set_view(plotter: Any, view: str = "3d"):
    """Point a plotter's camera at a standard view.

    ``view`` is one of ``"xy"`` (top), ``"yz"`` (side), ``"zx"``/``"xz"``
    (front), or ``"3d"`` (isometric, the default).
    """
    key = view.lower()
    if key not in _VIEW_METHODS:
        raise ValueError(f"Unknown view {view!r}; choose from 'xy', 'yz', 'zx' (alias 'xz'), or '3d'.")
    getattr(plotter, _VIEW_METHODS[key])()
    return plotter


def _lens_face_meshes(lens: Any, Nx: int, Ny: int, N_boundary: int, quadrature: str):
    p1, tri1 = sample_face_mesh(lens.surface1, lens.rim, lens.coord_sys_f1, Nx, Ny, quadrature, N_boundary)
    p2, tri2 = sample_face_mesh(lens.surface2, lens.rim, lens.coord_sys_f2, Nx, Ny, quadrature, N_boundary)
    return p1, tri1, p2, tri2


def lens_solid_mesh(
    lens: Any,
    Nx: int = 60,
    Ny: int = 60,
    N_boundary: int = 128,
    quadrature: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge a lens's two faces and side wall into one watertight point/triangle set.

    Both faces are sampled with :func:`sample_face_mesh`, which appends the
    exact rim outline as the trailing ``N_boundary`` vertices of each face's
    mesh. Since both faces share the same ``rim`` and ``N_boundary``, those
    trailing vertices already line up ring-for-ring between the two faces --
    so the side wall is built by directly stitching triangles between them
    (reusing those same vertex indices), with no new/duplicate vertices and
    no seam gap.

    Returns
    -------
    points:
        ``(N, 3)`` vertex coordinates (face 1 vertices followed by face 2
        vertices; each face's own trailing rim-outline vertices double as
        its wall ring).
    triangles:
        ``(M, 3)`` integer array of vertex indices, one row per triangle.
    """
    p1, tri1, p2, tri2 = _lens_face_meshes(lens, Nx, Ny, N_boundary, quadrature)

    n1, n2 = len(p1), len(p2)
    off1, off2 = 0, n1
    ring1, ring2 = off1 + (n1 - N_boundary), off2 + (n2 - N_boundary)

    points = np.vstack([p1, p2])

    wall = []
    for idx in range(N_boundary):
        idx_next = (idx + 1) % N_boundary
        top_i, top_j = ring1 + idx, ring1 + idx_next
        bot_i, bot_j = ring2 + idx, ring2 + idx_next
        wall.append([top_i, top_j, bot_j])
        wall.append([top_i, bot_j, bot_i])
    wall_tri = np.asarray(wall, dtype=np.int64)

    triangles = np.vstack([tri1 + off1, tri2 + off2, wall_tri])
    return points, triangles


def lens_solid_polydata(
    lens: Any,
    Nx: int = 60,
    Ny: int = 60,
    N_boundary: int = 128,
    quadrature: str = "uniform",
):
    """Build the merged watertight lens solid as a ``pyvista.PolyData``."""
    pv = _require_pyvista()
    points, triangles = lens_solid_mesh(lens, Nx, Ny, N_boundary, quadrature)
    return pv.PolyData(points, _pv_faces(triangles))


def plot_lens(
    lens: Any,
    Nx: int = 60,
    Ny: int = 60,
    N_boundary: int = 128,
    face1_color: str = "lightblue",
    face2_color: str = "lightsalmon",
    side_color: str = "lightgray",
    opacity: float = 0.9,
    plotter: Any = None,
    view: str = "3d",
    orientation_widget: bool = True,
    show: bool = True,
):
    """Render a two-surface lens (such as ``hypo.lenspy.simple_Lens``) in PyVista.

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
        PyVista color names/hex for each mesh.
    opacity:
        Opacity applied to all three meshes.
    plotter:
        Existing ``pyvista.Plotter`` to draw into. If ``None``, a new one is
        created via :func:`new_plotter`.
    view:
        Initial camera view; see :func:`set_view`.
    orientation_widget:
        If True, add PyVista's clickable camera-orientation widget (a small
        navigation cube in the corner) so the view can be snapped to any
        face interactively.
    show:
        If True and ``plotter`` was created here, immediately display it
        (``plotter.show()``).

    Returns
    -------
    plotter : pyvista.Plotter
    """
    pv = _require_pyvista()
    own_plotter = plotter is None
    if plotter is None:
        plotter = new_plotter(title=getattr(lens, "name", None))

    p1, tri1, p2, tri2 = _lens_face_meshes(lens, Nx, Ny, N_boundary, "uniform")
    face1 = pv.PolyData(p1, _pv_faces(tri1))
    face2 = pv.PolyData(p2, _pv_faces(tri2))
    wall_points, wall_tri = ring_wall_mesh(p1[-N_boundary:], p2[-N_boundary:])
    wall = pv.PolyData(wall_points, _pv_faces(wall_tri))

    plotter.add_mesh(face1, color=face1_color, opacity=opacity, name="face1")
    plotter.add_mesh(face2, color=face2_color, opacity=opacity, name="face2")
    plotter.add_mesh(wall, color=side_color, opacity=opacity, name="side_wall")

    set_view(plotter, view)
    if orientation_widget:
        try:
            plotter.add_camera_orientation_widget()
        except Exception:
            pass  # Not every render backend supports the interactive widget.

    if own_plotter and show:
        plotter.show(jupyter_backend="trame")
    return plotter


def plot_lens_solid(
    lens: Any,
    Nx: int = 60,
    Ny: int = 60,
    N_boundary: int = 128,
    color: str = "lightblue",
    opacity: float = 1.0,
    plotter: Any = None,
    view: str = "3d",
    orientation_widget: bool = True,
    show: bool = True,
):
    """Render a two-surface lens as a single solid body (one merged mesh).

    Unlike :func:`plot_lens` (three independently colored meshes), this
    produces one watertight ``pyvista.PolyData`` covering both faces and the
    side wall (see :func:`lens_solid_mesh`), so the lens reads as a single
    solid object with one color.

    Parameters mirror :func:`plot_lens`; ``color``/``opacity`` apply to the
    single merged mesh.

    Returns
    -------
    plotter : pyvista.Plotter
    """
    own_plotter = plotter is None
    if plotter is None:
        plotter = new_plotter(title=getattr(lens, "name", None))

    solid = lens_solid_polydata(lens, Nx, Ny, N_boundary)
    plotter.add_mesh(solid, color=color, opacity=opacity, name=getattr(lens, "name", "lens"))

    set_view(plotter, view)
    if orientation_widget:
        try:
            plotter.add_camera_orientation_widget()
        except Exception:
            pass  # Not every render backend supports the interactive widget.

    if own_plotter and show:
        plotter.show(jupyter_backend="trame")
    return plotter
