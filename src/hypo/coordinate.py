"""
Coordinate system utilities for the RefracPO physical-optics model.

Conventions
-----------
Angles
    All Euler angles are in radians.

Euler axes
    The `axes` argument specifies the Euler rotation axis sequence **with respect to the
    reference coordinate system axes**. In other words, the rotations are applied in the
    order given by `axes` about the reference-frame axes.

Vector vs point transforms
    The `Vector` flag distinguishes direction vectors from point coordinates:

    - Vector=False (default): treat (x, y, z) as **points**. Transforms include both rotation
      and translation (origin shift).
    - Vector=True: treat (x, y, z) as **vectors/directions** (e.g., field vectors, surface
      normals). Only the rotational part is applied; **no origin correction/translation** is
      performed.
"""
import numpy as np
from .transform import euler2mat, cartesian2spherical, cartesian2cylinder

__all__ = ["coord_sys", "global_coord"]

class _global_coord_sys():
    """
    Internal helper: global/root coordinate system used as the default reference for `coord_sys`.

    IMPORTANT
    ---------
    This class is an implementation detail. Users should NOT import, instantiate, or rely on
    `_global_coord_sys` directly. Use the module-level singleton `global_coord` instead.

    Purpose
    -------
    - Provide a default coordinaty frame or Mechanical coordinate system, so that `coord_sys` always has a valid `ref_coord`.
    - Expose the minimal set of attributes/methods expected by `coord_sys`.

    Notes
    -----
    - The global frame is defined such that local == reference == global.
    - The `Vector` flag follows the same convention as `coord_sys`:
      Vector=False -> point (translation applied); Vector=True -> direction (no translation).
    """

    def __init__(self):
        # Local origin expressed in the local frame (here identical to global).
        self.origin=np.zeros((3,1))

        # Local origin expressed in the global frame.
        self.origin_g=np.zeros((3,1))

        # Rotation matrix from local -> global. For the global frame, this is identity.
        self.mat_l_g=np.eye(3)

        # Rotation matrix from global -> local. For the global frame, this is identity.
        self.mat_g_l=np.eye(3)

        # Rotation matrix from reference -> local (identity here).
        self.mat_r_l=np.eye(3)

        # Rotation matrix from local -> reference (identity here).
        self.mat_l_r=np.eye(3)

    def Local_to_Ref(self,x,y,z,Vector=False):
        """Identity mapping for the global frame (local == reference)."""
        return x,y,z

    def Ref_to_Local(self,x,y,z,Vector=False):
        """Identity mapping for the global frame (reference == local)."""
        return x,y,z

    def Local_to_Global(self,x,y,z,Vector=False):
        """Identity mapping for the global frame (local == global)."""
        return x,y,z

    def Global_to_Local(self,x,y,z,Vector=False):
        """Identity mapping for the global frame (global == local)."""
        return x,y,z

    def ToSpherical(self,x,y,z):
        """Convert Cartesian (x,y,z) to spherical coordinates (r, theta, phi)."""
        return cartesian2spherical(x,y,z)

    def ToCylinder(self,x,y,z):
        """Convert Cartesian (x,y,z) to cylindrical coordinates (rho, phi, z)."""
        return cartesian2cylinder(x,y,z)


# Global reference coordinate system instance (used as default ref_coord).
global_coord=_global_coord_sys()


# %%
class coord_sys():
    '''
    Coordinate frame with respect to a reference frame (default: global_coord).

    This class defines a local coordinate system by specifying:
    - `origin`: local origin expressed in the reference coordinate system;
    - `angle` + `axes`: Euler-angle (rad) rotation defining the local orientation w.r.t. the axis order in 'axes'
    - `rotation_matrix`: explicitly provided 3x3 rotation matrix (local -> reference);
    - `ref_coord`: the reference coordinate frame (can be another coord_sys instance).

    Key matrices (naming is "to/from", as implied by usage)
    ------------------------------------------------------
    - mat_l_r: local  -> reference rotation (3x3)
    - mat_r_l: reference -> local rotation (transpose of mat_l_r)
    - mat_l_g: local  -> global rotation (composition via ref_coord)
    - mat_g_l: global -> local rotation (transpose of mat_l_g)

    Points vs vectors
    -----------------
    All transform methods accept `Vector`:
    - Vector=False: treat (x,y,z) as *point coordinates*; translation (origin) is applied.
    - Vector=True: treat (x,y,z) as *direction vectors*; translation is NOT applied.

    Input/output shape convention
    -----------------------------
    x, y, z are assumed to be array-like of shape (N,) or (1,N) such that
    `np.append([x,y],[z],axis=0)` yields a (3, N) array.
    The outputs are returned as (x_out, y_out, z_out), each with shape (N,).
    '''

    def __init__(self,
                 origin =[0,0,0],
                 angle = [0,0,0],
                 axes='xyz',
                 rotation_matrix = None,
                 ref_coord=global_coord):
        '''
        Parameters
        ----------
        origin:
            Local origin expressed in the reference coordinate system; will be reshaped to (3,1).
        angle:
            Euler rotation angles (angle[0], angle[1], angle[2]) used by `euler2mat`.
            Must be provided if rotation_matrix is None.
        axes:
            Euler axis sequence passed to `euler2mat` (e.g., 'xyz', 'zyx').
        rotation_matrix:
            If provided, use this 3x3 matrix as mat_l_r (local -> reference).
            Orientation must be specified either by rotation_matrix (preferred if provided) or by Euler angles angle.
            If neither is provided, the default angle=[0,0,0] yields an identity rotation.
        ref_coord:
            Reference coordinate system. If ref_coord is itself a coord_sys, transforms compose.

        Internal notes
        --------------
        - mat_r_l is always the transpose of mat_l_r (assumes pure rotation, orthonormal matrix).
        - origin_g is computed by composing the reference frame's global transform with the local origin.
        '''
        self.origin=np.array(origin).reshape(3,1)

        # Define local->reference rotation. If not provided explicitly, construct from Euler angles.
        if rotation_matrix is None:
            self.mat_l_r = euler2mat(angle[0],angle[1],angle[2],axes=axes)
        else:
            self.mat_l_r = rotation_matrix

        # Reference->local rotation (inverse for orthonormal rotation matrices).
        self.mat_r_l=np.transpose(self.mat_l_r)

        # Compose local->global rotation via the reference frame.
        # If ref_coord is global_coord, ref_coord.mat_l_g is identity.
        self.mat_l_g=np.matmul(ref_coord.mat_l_g,self.mat_l_r)
        self.mat_g_l=np.transpose(self.mat_l_g)

        # Local origin expressed in the global coordinate system:
        # origin_g = ref_origin_g + (ref_rot_l_g @ local_origin_in_ref)
        self.origin_g=ref_coord.origin_g+np.matmul(ref_coord.mat_l_g,self.origin)

    def Local_to_Ref(self,x,y,z,Vector=False):
        '''
        Convert coordinates from this local frame to the reference frame.

        Args:
            x, y, z: Cartesian coordinates in the local frame (array-like).
            Vector: If True, treat inputs as direction vectors (no translation).
                    If False, treat inputs as points (apply translation by self.origin).

        Returns:
            (x_ref, y_ref, z_ref) in the reference frame.
        '''
        xyz=np.append([x,y],[z],axis=0)       # (3, N): assemble local coordinates
        xyz=np.matmul(self.mat_l_r,xyz)       # rotate: local -> reference
        if not Vector:
            xyz+=self.origin                  # translate points by local origin in reference frame
        return xyz[0,:], xyz[1,:], xyz[2,:]

    def Ref_to_Local(self,x,y,z,Vector=False):
        '''
        Convert coordinates from the reference frame to this local frame.

        Notes:
            - This is the inverse of Local_to_Ref for pure rotation + translation.
            - The comment "commonly this function is useless" suggests this is less used
              in the current workflow, but it is mathematically well-defined.

        Args:
            x, y, z: Cartesian coordinates in the reference frame (array-like).
            Vector: If True, treat inputs as direction vectors (no translation subtraction).
                    If False, treat inputs as points (subtract translation by self.origin).

        Returns:
            (x_loc, y_loc, z_loc) in the local frame.
        '''
        xyz=np.append([x,y],[z],axis=0)       # (3, N): assemble reference coordinates
        if not Vector:
            xyz=xyz-self.origin              # remove local origin offset (in reference frame)
        xyz=np.matmul(self.mat_r_l,xyz)       # rotate: reference -> local
        return xyz[0,:], xyz[1,:], xyz[2,:]

    def Local_to_Global(self,x,y,z,Vector=False):
        '''
        Convert coordinates from this local frame to the global frame.

        Args:
            x, y, z: Cartesian coordinates in the local frame (array-like).
            Vector: If True, treat inputs as direction vectors (no translation).
                    If False, treat inputs as points (apply translation by self.origin_g).

        Returns:
            (x_g, y_g, z_g) in the global frame.
        '''
        xyz=np.append([x,y],[z],axis=0)       # (3, N)
        xyz=np.matmul(self.mat_l_g,xyz)       # rotate: local -> global (via reference composition)
        if not Vector:
            xyz = xyz + self.origin_g         # translate points by local origin in global frame
        return xyz[0,:], xyz[1,:], xyz[2,:]

    def Global_to_Local(self,x,y,z,Vector=False):
        '''
        Convert coordinates from the global frame to this local frame.

        Args:
            x, y, z: Cartesian coordinates in the global frame (array-like).
            Vector: If True, treat inputs as direction vectors (no translation subtraction).
                    If False, treat inputs as points (subtract translation by self.origin_g).

        Returns:
            (x_loc, y_loc, z_loc) in the local frame.
        '''
        xyz=np.append([x,y],[z],axis=0)       # (3, N)
        if not Vector:
            xyz=xyz-self.origin_g             # remove local origin offset (in global frame)
        xyz=np.matmul(self.mat_g_l,xyz)       # rotate: global -> local
        return xyz[0,:], xyz[1,:], xyz[2,:]

    def ToSpherical(self,x,y,z):
        '''
        Convert Cartesian coordinates to spherical coordinates.

        Returns:
            r, theta, phi (as defined by cartesian2spherical in .transform).
        '''
        r, theta, phi = cartesian2spherical(x,y,z)
        return r, theta, phi

    def ToCylinder(self,x,y,z):
        '''
        Convert Cartesian coordinates to cylindrical coordinates.

        Returns:
            rho, phi, z (as defined by cartesian2cylinder in .transform).
        '''
        pho, phi, z = cartesian2cylinder(x,y,z)
        return pho, phi, z

    def To_coord_sys(self,target_coord,x,y,z,Vector=False):
        '''
        Transform coordinates from this local frame into another coordinate system.

        Implementation detail:
            This is done via the global frame as an intermediate:
                local (self) -> global -> local (target)

        Args:
            target_coord: target coordinate system (another coord_sys instance).
            x, y, z: coordinates in *this* local frame.
            Vector: point vs vector flag; forwarded to both transforms.

        Returns:
            (x_t, y_t, z_t) expressed in the target coord_sys local frame.
        '''
        xp,yp,zp = self.Local_to_Global(x,y,z,Vector=Vector)
        x_t,y_t,z_t = target_coord.Global_to_Local(xp,yp,zp,Vector=Vector)
        return x_t,y_t,z_t