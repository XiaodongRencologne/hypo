
#%%
"""
Lens physical-optics workflow for a two-surface dielectric element.

This module provides a compact high-level wrapper around the lower-level
surface, coordinate, and PO propagation utilities in ``Kirchhoffpy``.

The central class ``simple_Lens`` models a lens with:

- an aperture rim that defines where each face is sampled,
- two explicit surface objects that provide ``sag(x, y)`` and ``normal(x, y)``,
- a lens-local coordinate system plus one local frame for each face,
- an optional anti-reflection coating description,
- and a workflow that stores equivalent surface fields to disk for reuse.

Typical workflow
----------------
1. Build two surface objects, for example from ``surfacev2``.
2. Construct ``simple_Lens`` with those surfaces and the lens coordinate frame.
3. Call ``PO_analysis(...)`` with an illuminating source.
4. Call ``source(...)`` to radiate the stored lens output currents to a target
   plane or spherical grid.

Coordinate convention
---------------------
- ``self.coord_sys`` is the lens body coordinate system.
- ``coord_sys_f1`` is the local frame of the first lens face.
- ``coord_sys_f2`` is the local frame of the second lens face.
- Face-2 is translated by the center thickness and rotated by ``pi`` around
  ``x`` so that its local outward normal follows the package convention.

Units
-----
- Frequency input is expected in GHz.
- The wave number ``k`` is computed in mm^-1.
- Surface sampling coordinates are therefore expected to be consistent with mm.

"""
import os
import copy
import h5py
import numpy as np
import torch as T
from typing import Any, Optional, Sequence, Tuple, Union

from .rim import Elliptical_rim
from .surface import Surface
from .coordinate import coord_sys
from .vecops import Vector

from .field_storage import Spherical_grd, plane_grd

from .Lenspo import lensPO
from .POpyGPU import PO_far_GPU2 as PO_far_GPU, PO_GPU_2 as PO_GPU

from .RWcur import saveh5_surf,read_cur
from . import c

#%%
class simple_Lens:
    """Two-surface dielectric lens model driven by physical optics.

    The class stores the geometric description of the lens, samples both
    refracting surfaces, computes equivalent currents on them, and can then use
    the stored currents as a secondary source for near-field or far-field
    propagation.

    Notes
    -----
    This class is intentionally orchestration-focused. The heavy numerical work
    is delegated to:

    - ``lensPO`` for the two-interface transmission workflow,
    - ``PO_GPU`` for near-field propagation,
    - ``PO_far_GPU`` for far-field propagation,
    - and the surface objects for geometry evaluation.
    """
    def __init__(self,
                 n: float,
                 thickness: float,
                 D: float,
                 surface1: Surface,
                 surface2: Surface,
                 coord_system: coord_sys,
                 name: str='simplelens',
                 AR_file: Optional[str]=None,
                 Device: T.device=T.device('cuda'),
                 outputfolder: str='output/') -> None:
        """Create a lens object from two surface descriptions.

        Parameters
        ----------
        n:
            Refractive index of the lens dielectric.
        thickness:
            Center thickness of the lens, measured along the lens local z-axis.
        D:
            Clear aperture diameter.
        surface1, surface2:
            Surface objects implementing ``sag(x, y)`` and ``normal(x, y)``.
            They are interpreted in their own face-local coordinates.
        coord_system:
            Lens body coordinate system.
        name:
            Base name used when writing current files.
        AR_file:
            Optional anti-reflection data file passed through to ``lensPO``.
        Device:
            Reserved device parameter kept for API compatibility.
        outputfolder:
            Directory where computed surface-current files are written.
        """
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        # Lens clear aperture and the two rotationally symmetric faces.
        self.rim = Elliptical_rim([0, 0], D / 2, D / 2)
        self.surface1 = surface1
        self.surface2 = surface2
        self.t = thickness  # Center thickness of the lens.
        self.diameter = D  # Clear aperture diameter.

        self.coord_sys = coord_system
        # Face 1 is defined in the lens local frame; face 2 is shifted by the
        # center thickness and flipped so its local normal points outward.
        self.coord_sys_f1 = coord_sys([0, 0, 0], angle=[0, 0, 0], ref_coord=self.coord_sys)
        self.coord_sys_f2 = coord_sys([0, 0, self.t], angle=[np.pi, 0, 0], ref_coord=self.coord_sys)

        # Optional anti-reflection coating description used by the PO solver.
        self.AR_file = AR_file
        # Refractive index of the dielectric lens material.
        self.n = n 

        # Bookkeeping and output locations.
        self.name = name # lens name
        self.outfolder = outputfolder
        
        # Runtime analysis state filled after a PO run.
        self.method = None
        self.target_face = None
        self.surf_cur_file = None

    @staticmethod
    def _validate_sampling_shape(name: str, samples: Sequence[int]) -> None:
        """Require 2-D sampling counts in the form ``(radial, azimuthal)``.

        The rim sampler expects exactly two counts. This helper keeps the error
        close to the public API entry points.
        """
        if len(samples) != 2:
            raise ValueError(f"{name} must contain exactly two sampling counts.")

    def PO_analysis(self,
                    source: Any,
                    N1: Sequence[int],
                    N2: Sequence[int],
                    freq: float,
                    device: str ='cuda',
                    po_name: str='_cur.h5',
                    order: str='f1_f2') -> None:
        """Run a two-surface physical-optics analysis through the lens.

        The input source illuminates the first chosen face. Equivalent fields
        and transmission coefficients are computed on both faces and stored in
        an HDF5 current file for later reuse.

        Parameters
        ----------
        source:
            Source object that provides

            - ``source.coord_sys``: the source local coordinate system
            - ``source.source(target_points, k)``: field evaluation on points

        N1, N2:
            Surface sampling counts for face 1 and face 2. Each must be a
            length-2 sequence accepted by ``Elliptical_rim.sampling``.
        freq:
            Frequency in GHz.
        device:
            Torch device or backend tag forwarded to the PO solvers.
        po_name:
            Output suffix used for the stored current HDF5 file.
        order:
            Propagation direction across the lens:

            - ``'f1_f2'``: source enters face 1 and exits face 2
            - ``'f2_f1'``: source enters face 2 and exits face 1

        Side effects
        ------------
        This method updates in-memory field attributes such as ``self.f2_E_t``
        and writes the final surface-current file to ``self.surf_cur_file``.

        Workflow summary
        ----------------
        1. Sample both faces on the aperture rim.
        2. Convert sampled points/normals into the lens frame.
        3. Convert face-1 samples into the source frame and query the source.
        4. Rotate the incident fields back into the lens frame.
        5. Run ``lensPO`` for the two-surface transmission workflow.
        6. Save both faces and their transmitted fields to HDF5.
        """
        self._validate_sampling_shape("N1", N1)
        self._validate_sampling_shape("N2", N2)
        if freq <= 0:
            raise ValueError("freq must be positive.")
        if not hasattr(source, "coord_sys") or not hasattr(source, "source"):
            raise TypeError("source must provide 'coord_sys' and a callable 'source(...)' method.")

        freq_factor = 10 ** 9
        frequency = freq * freq_factor
        wavelength = c * 1000 / frequency
        # Wave number in mm^-1 because c is converted consistently above.
        k = 2 * np.pi / wavelength

        method = lensPO
        # Choose the geometric order in which the wave intersects the two
        # surfaces. This only swaps which surface/frame pair is treated as the
        # entry face and which is treated as the exit face.
        if order == 'f1_f2':
            surf1 = self.surface1
            surf2 = self.surface2
            coord_sys1 = self.coord_sys_f1
            coord_sys2 = self.coord_sys_f2
        elif order == 'f2_f1':
            surf1 = self.surface2
            surf2 = self.surface1
            coord_sys1 = self.coord_sys_f2
            coord_sys2 = self.coord_sys_f1
        else:
            raise ValueError(" plase check the wave propagation direction!!!! ")
        # Sample both surfaces on the clear aperture.
        #
        # The PO lens solver assumes the source-face normal points toward the
        # incident side for its interface convention, so the first normal is
        # explicitly negated after sampling.
        f1, f1_n = self.sampling(N1, surf1)
        f1_n = -1 * f1_n
        f2, f2_n = self.sampling(N2, surf2)
        
        # Move sampled points and normals from each face-local system into the
        # common lens coordinate system.
        f1.x, f1.y, f1.z = coord_sys1.Local_to_Ref(f1.x, f1.y, f1.z)
        f1_n.x, f1_n.y, f1_n.z = coord_sys1.Local_to_Ref(f1_n.x, f1_n.y, f1_n.z, Vector=True)
        f2.x, f2.y, f2.z = coord_sys2.Local_to_Ref(f2.x, f2.y, f2.z)
        f2_n.x, f2_n.y, f2_n.z = coord_sys2.Local_to_Ref(f2_n.x, f2_n.y, f2_n.z, Vector=True)

        # Evaluate the incident source in its own coordinate system.
        # The source object is treated as the authority for field definition, so
        # the sampled face-1 geometry is converted from the lens frame into the
        # source frame before the source fields are requested.
        f1_p = copy.deepcopy(f1)
        f1_p_n = copy.deepcopy(f1_n)
        f1_p.x, f1_p.y, f1_p.z = self.coord_sys.To_coord_sys(source.coord_sys, f1_p.x, f1_p.y, f1_p.z)
        
        f1_p_n.x, f1_p_n.y, f1_p_n.z = self.coord_sys.To_coord_sys(
            source.coord_sys,
            f1_p_n.x,
            f1_p_n.y,
            f1_p_n.z,
            Vector=True,
        )
        
        # Incident electric and magnetic fields on face 1.
        E_in, H_in = source.source(f1_p, k)

        # ``lensPO`` expects the input fields to be expressed in the lens-local
        # frame, so rotate the source-returned fields back from source coords to
        # lens coords.
        matrix_transfer = np.matmul(self.coord_sys.mat_g_l, source.coord_sys.mat_l_g)
        E_in.tocoordsys(matrix=matrix_transfer)
        H_in.tocoordsys(matrix=matrix_transfer)
        
        # Solve the coupled PO transmission problem across the two surfaces.
        self.f2_E_t, self.f2_H_t, self.f1_E_t, self.f1_H_t, tp2, ts2, tp1, ts1 = method(
            f1,
            f1_n,
            f1.w,
            f2,
            f2_n,
            E_in,
            H_in,
            k,
            self.n,
            AR_filename=self.AR_file,
            frequency=str(freq) + 'GHz',
            device=device,
        )
        
        # Persist sampled surfaces, equivalent fields, and Fresnel-like
        # transmission factors for later field propagation.
        self.surf_cur_file = os.path.join(self.outfolder, self.name + po_name)
        with h5py.File(self.surf_cur_file, 'w') as file:
            saveh5_surf(file, f1, f1_n, self.f1_E_t, self.f1_H_t, tp1, ts1, name='f1')
            saveh5_surf(file, f2, f2_n, self.f2_E_t, self.f2_H_t, tp2, ts2, name='f2')

    def source(self,
               target: Any,
               freq: float,
               far_near: str='near',
               device: str='cuda',
               cur_file: Optional[str]=None) -> Optional[Tuple[Vector, Vector]]:
        """Propagate stored lens currents to a requested target grid or points.

        Parameters
        ----------
        target:
            Either a predefined grid object such as ``Spherical_grd`` or
            ``plane_grd``, or a raw target point container accepted by
            ``PO_GPU``.
        k:
            Wave number in mm^-1.
        far_near:
            Select far-field or near-field propagation for supported grid
            targets. Raw point-set targets always use the near-field kernel.
        device:
            Torch device forwarded to the PO propagators.
        cur_file:
            Optional explicit current file. If omitted, the most recent file
            produced by ``PO_analysis`` is used.

        Returns
        -------
        ``None`` for grid targets, because the resulting fields are written into
        ``target.E`` and ``target.H`` in place.

        ``(E, H)`` for raw target point sets, because there is no grid object to
        mutate.
        """
        Lambda = c*1000/freq/10**9
        k = 2*np.pi/Lambda
        # Read the equivalent currents computed on the output face.
        if cur_file is None and self.surf_cur_file is None:
            raise ValueError("No stored current file is available. Run PO_analysis first or pass cur_file.")
        if far_near.lower() not in {"near", "far"}:
            raise ValueError("far_near must be either 'near' or 'far'.")

        if cur_file is None:
            face2, face2_n, H2, E2 = read_cur(self.surf_cur_file)
        else:
            face2, face2_n, H2, E2 = read_cur(cur_file)
        if isinstance(target, (Spherical_grd, plane_grd)):
            if not hasattr(target, "coord_sys") or not hasattr(target, "grid"):
                raise TypeError("target grid must provide 'coord_sys' and 'grid'.")
            # Map the stored output-face geometry from the lens frame into the
            # requested target frame before evaluating the radiation integral.
            face2.x, face2.y, face2.z = self.coord_sys.Local_to_Global(face2.x, face2.y, face2.z)
            face2.x, face2.y, face2.z = target.coord_sys.Global_to_Local(face2.x, face2.y, face2.z)

            # The same rotation is applied to normals and field vectors so that
            # geometry and fields remain expressed in one consistent frame.
            target_transform = np.matmul(target.coord_sys.mat_g_l, self.coord_sys.mat_l_g)
            data = np.matmul(target_transform, np.array([face2_n.x, face2_n.y, face2_n.z]))
            face2_n.x = data[0, :]
            face2_n.y = data[1, :]
            face2_n.z = data[2, :]
            H2.tocoordsys(matrix=target_transform)
            E2.tocoordsys(matrix=target_transform)

            if far_near.lower() == 'far':
                # Far-field radiation integral onto a spherical observation grid.
                target.E, target.H = PO_far_GPU(
                    face2,
                    face2_n,
                    face2.w,
                    target.grid,
                    E2,
                    H2,
                    k,
                    device=device,
                )
            else:
                # Near-field radiation integral onto a planar or spherical grid.
                target.E, target.H = PO_GPU(
                    face2,
                    face2_n,
                    face2.w,
                    target.grid,
                    E2,
                    H2,
                    k,
                    1,  # n refractive index
                    device=device,
                )
        else:
            # Fallback path for a raw target point set instead of a grid object.
            E, H = PO_GPU(
                face2,
                face2_n,
                face2.w,
                target,
                E2,
                H2,
                k,
                1,  # n refractive index
                device=device,
            )
            return E, H
    # Surface sampling helper used by the PO solver.

    def sampling(self, 
                 f_N: Sequence[int], 
                 surf: Surface,
                 quadrature='gaussian') -> Tuple[Vector, Vector]:
        """Sample a lens face on the rim aperture and evaluate its normal.

        Parameters
        ----------
        f_N:
            Two sampling counts passed to the aperture-rim Gaussian sampler.
        surf:
            Surface object providing ``sag`` and ``normal`` in face-local
            coordinates.

        Returns
        -------
        f:
            Sampled surface points with coordinates ``(x, y, z)`` and quadrature
            weights ``w``.
        f_n:
            Unit surface normal at each sampled point.
        """
        self._validate_sampling_shape("f_N", f_N)
        f = Vector()
        f.x, f.y, f.w = self.rim.sampling(
            f_N[0],
            f_N[1],
            quadrature=quadrature,
            Nr_part=1,
            phi0=0,
            phi1=2 * np.pi,
            Phi_type='less',
        )
        f.z = surf.sag(f.x, f.y)
        f_n = surf.normal(f.x, f.y)
        return f, f_n
    
    def to_pyvista_solid(
        self,
        Nx: int = 121,
        Ny: int = 121,
        N_boundary: int = 256,
        plotter=None,
        add_mesh_kwargs: Optional[dict] = None,
    ):
        """
        Build a solid lens model in pyvista (face1 + face2 + side wall).

        Parameters
        ----------
        Nx, Ny : int
            Sampling counts for face meshes.
        N_boundary : int
            Number of boundary points used to construct side wall.
        include_error : bool
            If True, include configured surface errors in geometry.
        include_thickness : bool
            If True, face2 z includes `self.thickness`.
        plotter : pyvista.Plotter or None
            If provided, add the solid mesh to this plotter.
        add_mesh_kwargs : dict or None
            Optional kwargs passed to `plotter.add_mesh`.

        Returns
        -------
        solid : pyvista.PolyData
            Triangulated/cleaned solid lens mesh.
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError("pyvista is required for Lens.to_pyvista_solid().") from exc

        if N_boundary < 3:
            raise ValueError("N_boundary must be >= 3.")

        # Sample faces.
        face1, _ = self.sampling(
            [Nx, Ny], quadrature="uniform")
        face2, _ = self.sampling(
            [Nx, Ny], quadrature="uniform")
        # Triangulate each face from sampled points.
        face1 = pv.PolyData(np.column_stack((face1.x, face1.y, face1.z))).delaunay_2d()
        face2 = pv.PolyData(np.column_stack((face2.x, face2.y, face2.z))).delaunay_2d()

        # Build side wall from rim boundary loops.
        phi = np.linspace(0.0, 2.0 * np.pi, int(N_boundary), endpoint=False)
        xb = self.rim.cx + self.rim.a * np.cos(phi)
        yb = self.rim.cy + self.rim.b * np.sin(phi)

        z1b= self.surface1.sag(xb, yb)
        z2b= self.eval_face2(xb, yb)
        if include_thickness:
            z2b = z2b + self.thickness

        side_points = np.column_stack((
            np.concatenate([xb, xb]),
            np.concatenate([yb, yb]),
            np.concatenate([z1b, z2b]),
        ))
        n = int(N_boundary)
        faces = []
        for i in range(n):
            i_next = (i + 1) % n
            top_i = i
            top_j = i_next
            bot_i = n + i
            bot_j = n + i_next
            # Quad: top_i -> top_j -> bot_j -> bot_i
            faces.extend([4, top_i, top_j, bot_j, bot_i])
        side = pv.PolyData(side_points, np.asarray(faces, dtype=np.int64))
        side = side.triangulate()

        # Merge to one solid shell.
        solid = face1.merge(face2).merge(side).clean().triangulate()

        # Convert lens-local mesh points to global coordinate system.
        pts = solid.points
        xg, yg, zg = self.coord_sys.Local_to_Global(pts[:, 0], pts[:, 1], pts[:, 2], Vector=False)
        solid.points = np.column_stack((xg, yg, zg))

        if plotter is not None:
            kwargs = {"color": "lightblue", "opacity": 1.0}
            if add_mesh_kwargs is not None:
                kwargs.update(add_mesh_kwargs)
            plotter.add_mesh(solid, name="lens_solid", **kwargs)

        return solid

