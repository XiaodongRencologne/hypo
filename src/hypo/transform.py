import numpy as np

"""
Low-level geometric transforms used across the RefracPO model.

Conventions
-----------
Angles
    All angles are in radians.

Euler axes (reference-frame axis sequence)
    The `axes` argument (e.g., 'xyz', 'zyx') specifies the order of elementary rotations
    about the *reference coordinate system axes*.

Vector/point convention (used by higher-level coordinate module)
    This module provides rotation matrices and coordinate conversions. Translation handling
    (origin shift) is performed explicitly in Transform_* functions or by `coord_sys`.
"""

#### coordinates points transform.
def euler2mat(ai,al,ak,axes='xyz'):
    '''
    Calculate a 3x3 coordinate transform matrix based on Euler angles.

    Parameters
    ----------
    ai, al, ak:
        Euler rotation angles (radians). They are consumed in the order specified by `axes`.
        For example, axes='xyz' uses:
            angle[0] about x, then angle[1] about y, then angle[2] about z
        (with respect to the *reference-frame axes*).

    axes:
        A 3-character string specifying the rotation axis sequence (default: 'xyz').

    Returns
    -------
    M : (3, 3) ndarray
        Composite rotation matrix.

    Interpretation (IMPORTANT)
    --------------------------
    As described in the original comment:

        "The resulting matrix multiplied by the coordinates of a point in the new coordinate
        system after rotations gives the coordinates of the point in the original (reference) coordinate
        system."

    In other words, this function returns a matrix that maps coordinates expressed in the
    *rotated/new frame* into the *original/reference frame*.

    Practical note
    --------------
    - Because pure rotation matrices are orthonormal, the inverse rotation is the transpose.
      You will see `mat = np.transpose(mat)` used elsewhere when the opposite mapping is needed.
    - Sign conventions of the elementary rotations are encoded below; consistency matters more
      than any specific convention, so downstream code should follow this function's definition.
    '''
    angle=[ai,al,ak]

    # Elementary right-handed rotations about x/y/z axes.
    # Each lambda returns a 3x3 rotation matrix for the given angle phi (radians).
    axis={'x': lambda phi: np.array([[1.0,0.0,0.0],
                                    [0.0,np.cos(phi),np.sin(phi)],
                                    [0.0,-np.sin(phi),np.cos(phi)]]).ravel().reshape(3,3),

          'y': lambda phi: np.array([[np.cos(phi),0.0,-np.sin(phi)],
                                    [0.0,1.0,0.0],
                                    [np.sin(phi),0.0,np.cos(phi)]]).ravel().reshape(3,3),

          'z': lambda phi: np.array([[np.cos(phi),np.sin(phi),0.0],
                                    [-np.sin(phi),np.cos(phi),0.0],
                                    [0.0,0.0,1.0]]).ravel().reshape(3,3)
      }

    # Start with identity and left-multiply in the order specified by `axes`.
    # Implementation detail:
    #   M <- R_axis(angle[i]) @ M
    # so after looping, M is the composed rotation according to the chosen sequence.
    M=np.eye(3)
    i=0
    for n in axes:
        M=np.matmul(axis[n](angle[i]),M)
        i+=1
    return M


def cartesian2spherical(x,y,z):
    '''
    Convert Cartesian coordinates into spherical coordinates.

    Parameters
    ----------
    x, y, z:
        Cartesian coordinates (array-like). They may be scalars or NumPy arrays with a common shape.

    Returns
    -------
    r:
        Radial distance: sqrt(x^2 + y^2 + z^2)
    theta:
        Polar angle (radians), defined here as:
            theta = arccos(z / r)
        Therefore theta is measured from the +z axis (0 at +z, pi at -z).
    phi:
        Azimuthal angle (radians), defined here as:
            phi = arctan2(y, x)
        Typical range is (-pi, pi].

    Notes / pitfalls
    ---------------
    - When r == 0, z/r is undefined. This function does not explicitly guard against r=0,
      so callers should avoid passing the origin or handle it upstream if needed.
    - The (theta, phi) convention matches many physics texts: theta = polar angle from +z,
      phi = azimuth in x-y plane.
    '''
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arccos(z/r)
    phi=np.arctan2(y,x)
    return r, theta, phi


def cartesian2cylinder(x,y,z):
    '''
    Convert Cartesian coordinates to cylindrical coordinates.

    Parameters
    ----------
    x, y, z:
        Cartesian coordinates (array-like).

    Returns
    -------
    pho:
        Cylindrical radius rho = sqrt(x^2 + y^2).
        (Variable name is `pho` in code; conceptually this is rho.)
    phi:
        Azimuthal angle (radians): arctan2(y, x), range (-pi, pi].
    z:
        The original z coordinate (passed through unchanged).

    Notes
    -----
    - This function treats cylindrical coordinates as (rho, phi, z).
    - `z=z` is a no-op kept for clarity.
    '''
    pho=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    z=z

    return pho, phi, z


def Transform_local2global (angle,origin,local,axes='xyz'):
    '''
    Convert coordinates from a local coordinate system to the global coordinate system.

    Parameters
    ----------
    angle:
        Euler angles [a0, a1, a2] in radians.
    origin:
        Translation of the local origin expressed in the global frame.
    local:
        An object with attributes `x`, `y`, `z` representing local coordinates.
        Each attribute is expected to be array-like with a compatible shape, such that:
            L = np.append([local.x, local.y], [local.z], axis=0)
        yields a (3, N) array.
    axes:
        Euler axis sequence passed to euler2mat.

    Returns
    -------
    x_g, y_g, z_g:
        Coordinates expressed in the global frame.

    Implementation details
    ----------------------
    - `euler2mat` returns a matrix mapping (new/rotated frame) -> (original/reference frame).
      Here, the code transposes it:
          mat = np.transpose(mat)
      which corresponds to using the inverse rotation (because rotation matrices are orthonormal).
    - The transform applied is:
          G = (mat @ L) + origin
      i.e., rotate then translate.

    Relation to coord_sys
    ---------------------
    This is a standalone helper. In the main codebase, `coord_sys.Local_to_Global(...)` provides
    the same conceptual operation with a consistent API and `Vector` semantics.
    '''
    origin=np.array(origin)
    L=np.append([local.x,local.y],[local.z],axis=0)
    mat=euler2mat(angle[0],angle[1],angle[2],axes=axes)
    mat=np.transpose(mat)
    G=np.matmul(mat,L)
    G=G+origin.reshape(-1,1)
    return G[0,:], G[1,:], G[2,:]


def Transform_global2local (angle,origin,G,axes='xyz'):
    '''
    Convert coordinates from the global coordinate system to a local coordinate system.

    Parameters
    ----------
    angle:
        Euler angles [a0, a1, a2] in radians.
    origin:
        Translation of the local origin expressed in the global frame.
    G:
        An object with attributes `x`, `y`, `z` representing global coordinates.
        Each attribute is expected to be array-like such that assembling
            g = np.append([G.x, G.y], [G.z], axis=0)
        yields a (3, N) array.
    axes:
        Euler axis sequence passed to euler2mat.

    Returns
    -------
    x_l, y_l, z_l:
        Coordinates expressed in the local frame.

    Implementation details
    ----------------------
    - The transform applied is:
          g = G - origin
          local = euler2mat(...) @ g
    - Unlike Transform_local2global, there is no transpose here; the code uses `euler2mat`
      directly. This is consistent with the interpretation that `euler2mat` maps from the
      rotated/new frame to the original/reference frame.
    - The docstring in the original code said "convert coordinates from local to global";
      this function name and implementation indicate it is global -> local.
    '''
    origin=np.array(origin)
    g=np.append([G.x,G.y],[G.z],axis=0)
    g=g-origin.reshape(-1,1)
    mat=euler2mat(angle[0],angle[1],angle[2],axes=axes)
    local=np.matmul(mat,g)

    return local[0,...], local[1,...], local[2,...]