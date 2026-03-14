"""Surface builders and lightweight surface classes for refractive PO models.

This module provides a small set of reusable surface descriptions used by the
refractive physical-optics workflow:

- rotationally symmetric profiles read from RSF files,
- conic surfaces,
- even-asphere surfaces,
- biconic surfaces,
- and generic 2-D polynomial surfaces.

Design philosophy
-----------------
Each analytic or file-backed surface is exposed in two layers:

1. A ``build_*_sag_normal(...)`` function that returns two callables
   ``(sag_func, normal_func)``.
2. A small wrapper class derived from ``Surface`` (or ``RotationalSurface``)
   that stores and exposes those callables with a uniform interface.

This split keeps the mathematical surface description reusable in functional
code while still supporting an object-oriented interface in higher-level lens
models.

Common convention
-----------------
All surfaces are represented as graph surfaces

    z = z(x, y)

in their own local coordinate system. The returned surface normal follows the
package convention

    n ~ (dz/dx, dz/dy, -1)

and is normalized to unit length before being returned as a ``Vector``.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from .vecops import Vector


def build_rsf_sag_normal(profile_file, units="cm"):
    """Build sag and normal callables from an RSF radial sag file.

    The RSF file is expected to contain two header lines followed by columns
    ``(r, z)``. Input values are converted to mm before interpolation.

    Parameters
    ----------
    profile_file:
        Path to the radial sag file.
    units:
        Unit tag of the file contents. Supported values are ``'mm'``, ``'cm'``,
        and ``'m'``.

    Notes
    -----
    The file is interpreted as a rotationally symmetric profile:

        r = sqrt(x^2 + y^2)

    A cubic spline is fit to the sag ``z(r)``, and the normal is recovered
    from the spline derivative ``dz/dr``.
    """
    factor = {"mm": 1.0, "cm": 10.0, "m": 1000.0}[units]
    data = np.genfromtxt(profile_file, skip_header=2)
    radius = data[:, 0] * factor
    sag = data[:, 1] * factor

    spline = CubicSpline(radius, sag)
    slope = spline.derivative()

    def sag_func(x, y):
        r = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
        return spline(r)

    def normal_func(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.sqrt(x**2 + y**2)
        r_safe = np.where(r == 0.0, 1e-12, r)
        dz_dr = slope(r_safe)

        n = Vector()
        n.x = dz_dr * x / r_safe
        n.y = dz_dr * y / r_safe
        n.x = np.where(r == 0.0, 0.0, n.x)
        n.y = np.where(r == 0.0, 0.0, n.y)
        n.z = -np.ones_like(r, dtype=np.float64)
        n.N = np.sqrt(n.x**2 + n.y**2 + 1.0)
        n.x = n.x / n.N
        n.y = n.y / n.N
        n.z = n.z / n.N
        return n

    return sag_func, normal_func


def build_poly_sag_normal(coefficients, normalization_radius=1.0):
    """Build sag and normal callables for a 2-D polynomial surface.

    The surface is defined as

        z(x, y) = sum(c_ij (x / R)^i (y / R)^j)

    where ``R = normalization_radius`` and ``c_ij`` are given by the input
    coefficient matrix. ``coefficients`` may be either

    - an array-like coefficient matrix, or
    - a ``.surfc`` file containing the matrix.

    The normal is computed from the graph surface ``z = z(x, y)``:

        n ~ (dz/dx, dz/dy, -1)

    before normalization.

    Notes
    -----
    This builder is useful for freeform surfaces represented by a rectangular
    coefficient matrix ``c_ij``. The polynomial basis follows NumPy's
    ``polyval2d`` convention, where the first axis corresponds to powers of
    ``x`` and the second axis corresponds to powers of ``y``.
    """
    if normalization_radius == 0:
        raise ValueError("normalization_radius must be non-zero.")

    if isinstance(coefficients, str):
        if coefficients.split(".")[-1].lower() != "surfc":
            raise ValueError("Coefficient file must use the .surfc extension.")
        coeff_matrix = np.genfromtxt(coefficients, delimiter=",")
    else:
        coeff_matrix = np.asarray(coefficients, dtype=np.float64)

    if coeff_matrix.ndim != 2:
        raise ValueError("coefficients must be a 2-D array or a .surfc file.")

    radius = float(normalization_radius)

    # Differentiate the 2-D polynomial coefficient matrix analytically.
    x_orders = np.arange(coeff_matrix.shape[0], dtype=np.float64).reshape(-1, 1)
    dx_coeffs = coeff_matrix * x_orders
    y_orders = np.arange(coeff_matrix.shape[1], dtype=np.float64)
    dy_coeffs = coeff_matrix * y_orders

    def sag_func(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.polynomial.polynomial.polyval2d(x / radius, y / radius, coeff_matrix)

    def normal_func(x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        n = Vector()
        n.x = np.polynomial.polynomial.polyval2d(x / radius, y / radius, dx_coeffs[1:, :]) / radius
        n.y = np.polynomial.polynomial.polyval2d(x / radius, y / radius, dy_coeffs[:, 1:]) / radius
        n.z = -np.ones_like(n.x, dtype=np.float64)
        n.N = np.sqrt(n.x**2 + n.y**2 + 1.0)
        n.x = n.x / n.N
        n.y = n.y / n.N
        n.z = n.z / n.N
        return n

    return sag_func, normal_func


def build_conic_sag_normal(radius, conic_const=0.0):
    """Build sag and normal callables for a rotationally symmetric conic.

    The surface is defined in Cartesian coordinates by a radial sag

        z(r) = c r^2 / (1 + sqrt(1 - (1 + k) c^2 r^2))

    where

    - ``r^2 = x^2 + y^2``
    - ``c = 1 / radius`` is the curvature
    - ``k = conic_const`` is the conic constant

    This is the standard explicit sag form for conic sections used in optics.
    Typical values are:

    - ``k = 0``: sphere
    - ``-1 < k < 0``: ellipsoid
    - ``k = -1``: paraboloid
    - ``k < -1``: hyperboloid

    The returned normal is built from the graph surface ``z = z(x, y)``:

        n ~ (dz/dx, dz/dy, -1)

    with

        dz/dx = (dz/dr) x / r
        dz/dy = (dz/dr) y / r
        dz/dr = c r / sqrt(1 - (1 + k) c^2 r^2)

    The vector is then normalized to unit length before being returned as a
    ``Vector`` instance with fields ``x``, ``y``, ``z``, and ``N``.

    Notes
    -----
    ``radius`` and ``conic_const`` match the standard optical conic
    parameterization, so this builder is suitable as a base term for conic and
    aspheric lens descriptions.
    """
    if radius == 0:
        raise ValueError("radius must be non-zero.")

    c = 1.0 / radius
    k = conic_const

    def sag_func(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        r2 = x**2 + y**2
        # Explicit conic sag formula written in terms of r^2.
        root = np.sqrt(np.clip(1.0 - (1.0 + k) * c**2 * r2, 0.0, None))
        return c * r2 / (1.0 + root)

    def normal_func(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.sqrt(x**2 + y**2)
        # dz/dr for the conic sag above; clipping keeps the computation real
        # at the numerical aperture boundary.
        root = np.sqrt(np.clip(1.0 - (1.0 + k) * c**2 * r**2, 0.0, None))
        r_safe = np.where(r == 0.0, 1e-12, r)
        dz_dr = c * r_safe / root

        n = Vector()
        # For z = z(r), convert radial slope to Cartesian slope components.
        n.x = dz_dr * x / r_safe
        n.y = dz_dr * y / r_safe
        n.x = np.where(r == 0.0, 0.0, n.x)
        n.y = np.where(r == 0.0, 0.0, n.y)
        n.z = -np.ones_like(r, dtype=np.float64)
        n.N = np.sqrt(n.x**2 + n.y**2 + 1.0)
        n.x = n.x / n.N
        n.y = n.y / n.N
        n.z = n.z / n.N
        return n

    return sag_func, normal_func


def build_even_asphere_sag_normal(radius, conic_const=0.0, even_terms=None):
    """Build sag and normal callables for a conic with even-power corrections.

    The sag is written as

        z(r) = z_conic(r) + sum(A_n r^n)

    where ``n`` is even and the base conic term is

        z_conic(r) = c r^2 / (1 + sqrt(1 - (1 + k) c^2 r^2))

    with ``c = 1 / radius`` and ``k = conic_const``.

    ``even_terms`` is a mapping from even power to coefficient, for example
    ``{2: A2, 4: A4, 6: A6}``.

    Notes
    -----
    - ``n = 2`` is allowed for compatibility with generalized Even Asphere
      formulas and some optical design software conventions.
    - The ``A2 r^2`` term is not independent from the base curvature set by
      ``radius``. Using both as free parameters introduces degeneracy in the
      quadratic term near the vertex.
    - The normal is computed from ``z = z(r)`` through

          dz/dx = (dz/dr) x / r
          dz/dy = (dz/dr) y / r
          n ~ (dz/dx, dz/dy, -1)

      and is normalized to unit length before being returned as a ``Vector``.

    Examples
    --------
    ``even_terms`` may be passed as, for example,

    ``{2: A2, 4: A4, 6: A6, 8: A8}``

    where the ``2`` term is accepted for compatibility with generalized
    asphere formulas, even though it is not independent from the base curvature.
    """
    if radius == 0:
        raise ValueError("radius must be non-zero.")

    if even_terms is None:
        even_terms = {}
    else:
        even_terms = dict(even_terms)

    invalid_orders = [order for order in even_terms if order < 0 or order % 2 != 0]
    if invalid_orders:
        raise ValueError("even_terms keys must be non-negative even integers.")

    c = 1.0 / radius
    k = conic_const
    sorted_terms = sorted((int(order), coeff) for order, coeff in even_terms.items())

    def sag_func(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        r2 = x**2 + y**2
        r = np.sqrt(r2)
        root = np.sqrt(np.clip(1.0 - (1.0 + k) * c**2 * r2, 0.0, None))
        z = c * r2 / (1.0 + root)

        for order, coeff in sorted_terms:
            z = z + coeff * r**order
        return z

    def normal_func(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.sqrt(x**2 + y**2)
        root = np.sqrt(np.clip(1.0 - (1.0 + k) * c**2 * r**2, 0.0, None))
        r_safe = np.where(r == 0.0, 1e-12, r)
        dz_dr = c * r_safe / root

        # Add derivative of the even-power correction sum(A_n r^n).
        for order, coeff in sorted_terms:
            if order == 0:
                continue
            dz_dr = dz_dr + order * coeff * r_safe ** (order - 1)

        n = Vector()
        n.x = dz_dr * x / r_safe
        n.y = dz_dr * y / r_safe
        n.x = np.where(r == 0.0, 0.0, n.x)
        n.y = np.where(r == 0.0, 0.0, n.y)
        n.z = -np.ones_like(r, dtype=np.float64)
        n.N = np.sqrt(n.x**2 + n.y**2 + 1.0)
        n.x = n.x / n.N
        n.y = n.y / n.N
        n.z = n.z / n.N
        return n

    return sag_func, normal_func


def build_biconic_sag_normal(radius_x, radius_y, conic_const_x=0.0, conic_const_y=0.0):
    """Build sag and normal callables for a biconic surface.

    A biconic is the non-rotational analogue of a conic, with independent
    curvatures and conic constants along ``x`` and ``y``. Its sag is

        z(x, y) = A / (1 + B)

    where

        A = c_x x^2 + c_y y^2
        B = sqrt(1 - (1 + k_x) c_x^2 x^2 - (1 + k_y) c_y^2 y^2)

    with

    - ``c_x = 1 / radius_x``
    - ``c_y = 1 / radius_y``
    - ``k_x = conic_const_x``
    - ``k_y = conic_const_y``

    The normal is built from the graph surface ``z = z(x, y)``:

        n ~ (dz/dx, dz/dy, -1)

    using the analytic partial derivatives of the biconic sag and then
    normalizing the vector to unit length.

    Notes
    -----
    This is the natural choice when the two principal meridians should be
    parameterized independently instead of using one rotationally symmetric
    radial coordinate.
    """
    if radius_x == 0 or radius_y == 0:
        raise ValueError("radius_x and radius_y must be non-zero.")

    c_x = 1.0 / radius_x
    c_y = 1.0 / radius_y
    k_x = conic_const_x
    k_y = conic_const_y

    def sag_func(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        radicand = 1.0 - (1.0 + k_x) * c_x**2 * x**2 - (1.0 + k_y) * c_y**2 * y**2
        root = np.sqrt(np.clip(radicand, 0.0, None))
        numerator = c_x * x**2 + c_y * y**2
        return numerator / (1.0 + root)

    def normal_func(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        radicand = 1.0 - (1.0 + k_x) * c_x**2 * x**2 - (1.0 + k_y) * c_y**2 * y**2
        root = np.sqrt(np.clip(radicand, 0.0, None))
        root_safe = np.where(root == 0.0, 1e-12, root)
        numerator = c_x * x**2 + c_y * y**2
        denom = (1.0 + root) ** 2

        n = Vector()
        # Differentiate z = A / (1 + B) analytically along x and y.
        n.x = (2.0 * c_x * x * (1.0 + root) + numerator * (1.0 + k_x) * c_x**2 * x / root_safe) / denom
        n.y = (2.0 * c_y * y * (1.0 + root) + numerator * (1.0 + k_y) * c_y**2 * y / root_safe) / denom
        n.z = -np.ones_like(root, dtype=np.float64)
        n.N = np.sqrt(n.x**2 + n.y**2 + 1.0)
        n.x = n.x / n.N
        n.y = n.y / n.N
        n.z = n.z / n.N
        return n

    return sag_func, normal_func


class Surface:
    """General surface defined by ``sag(x, y)`` and ``normal(x, y)`` callables.

    This is the minimal common interface expected by higher-level refractive
    optics code. Subclasses mainly exist to package how the callables are
    constructed; the evaluation API itself stays intentionally small.
    """

    def __init__(self, sag, normal):
        # Store the two core surface evaluators exactly as provided.
        self._sag = sag
        self._normal = normal

    def sag(self, x, y):
        """Evaluate the sag ``z(x, y)`` in the surface local frame."""
        return self._sag(x, y)

    def normal(self, x, y):
        """Evaluate the unit surface normal at ``(x, y)``."""
        return self._normal(x, y)


class RotationalSurface(Surface):
    """Rotationally symmetric surface.

    A rotational surface can be defined directly by sag/normal callables or by
    supplying an RSF sag file.

    This class is mainly a convenience wrapper for the common case where the
    sag depends only on the radial coordinate ``r = sqrt(x^2 + y^2)``.
    """

    def __init__(self, sag=None, normal=None, profile_file=None, units="cm"):
        if profile_file is not None:
            sag, normal = build_rsf_sag_normal(profile_file, units=units)
        if sag is None or normal is None:
            raise ValueError("Provide sag/normal callables or profile_file.")
        self.profile_file = profile_file
        self.units = units
        super().__init__(sag, normal)


class ConicSurface(RotationalSurface):
    """Conic rotational surface defined by radius and conic constant.

    This class is a thin object-oriented wrapper around
    ``build_conic_sag_normal``.
    """

    def __init__(self, radius, conic_const=0.0):
        sag, normal = build_conic_sag_normal(radius, conic_const=conic_const)
        super().__init__(sag=sag, normal=normal)


class EvenAsphereSurface(RotationalSurface):
    """Rotational surface with a conic base plus even-power polynomial terms.

    ``even_terms`` accepts a mapping such as ``{2: A2, 4: A4, 6: A6}``.
    The ``2``-order term is supported, but note that it is degenerate with the
    vertex curvature set by ``radius``.

    This class is the rotationally symmetric analogue of the common optical
    "conic base + even polynomial correction" parameterization.
    """

    def __init__(self, radius, conic_const=0.0, even_terms=None):
        sag, normal = build_even_asphere_sag_normal(
            radius,
            conic_const=conic_const,
            even_terms=even_terms,
        )
        super().__init__(sag=sag, normal=normal)


class BiconicSurface(Surface):
    """Surface with independent conic curvatures along x and y.

    Unlike ``ConicSurface``, this class does not reduce the geometry to a
    single radial coordinate. It is therefore useful for astigmatic or
    otherwise anisotropic lens faces.
    """

    def __init__(self, radius_x, radius_y, conic_const_x=0.0, conic_const_y=0.0):
        sag, normal = build_biconic_sag_normal(
            radius_x,
            radius_y,
            conic_const_x=conic_const_x,
            conic_const_y=conic_const_y,
        )
        super().__init__(sag=sag, normal=normal)


class PolySurface(Surface):
    """Surface described by a 2-D polynomial coefficient matrix.

    This wrapper is useful for freeform surfaces described directly by a
    polynomial coefficient table or by a ``.surfc`` coefficient file.
    """

    def __init__(self, coefficients, normalization_radius=1.0):
        self.coefficients = coefficients
        self.normalization_radius = normalization_radius
        sag, normal = build_poly_sag_normal(coefficients, normalization_radius=normalization_radius)
        super().__init__(sag=sag, normal=normal)
