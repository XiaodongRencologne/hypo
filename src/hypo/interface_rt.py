import numpy as np
import torch as T

from .EMtools import poyntingVector
from .FresnelCoeff import Fresnel_coeffi
from .vecops import Vector, cross, dot, magnitude, normalized

EPS = 1e-12
TIR_TOL = 1e-12


def _as_vector(v):
    """Ensure return type is Vector with x/y/z components."""
    if isinstance(v, Vector):
        return v
    return Vector(v[0], v[1], v[2])


def _to_numpy(x):
    """Convert scalar/array/tensor to NumPy array."""
    if isinstance(x, T.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_backend_like(x_np, ref):
    """Convert NumPy data to backend/dtype/device of ref."""
    if isinstance(ref, T.Tensor):
        return T.as_tensor(x_np, dtype=ref.dtype, device=ref.device)
    return np.asarray(x_np, dtype=np.asarray(ref).dtype)


def _promote_component_precision(x):
    """Promote component precision to float64/complex128."""
    if isinstance(x, T.Tensor):
        if x.dtype == T.float32:
            return x.to(dtype=T.float64)
        if x.dtype == T.complex64:
            return x.to(dtype=T.complex128)
        return x

    x_np = np.asarray(x)
    if np.iscomplexobj(x_np):
        return x_np.astype(np.complex128, copy=False)
    if np.issubdtype(x_np.dtype, np.floating):
        return x_np.astype(np.float64, copy=False)
    if np.issubdtype(x_np.dtype, np.integer) or np.issubdtype(x_np.dtype, np.bool_):
        return x_np.astype(np.float64, copy=False)
    return x_np


def _promote_vector_precision(v):
    """Return a precision-promoted Vector (float64/complex128)."""
    return Vector(
        _promote_component_precision(v.x),
        _promote_component_precision(v.y),
        _promote_component_precision(v.z),
    )


def calc_reflect_transmit_fields(E, H, n_hat, n1, n2, AR=None):
    """
    Unified interface for non-AR and AR-coated interfaces.

    Parameters
    ----------
    E, H : Vector
        Incident electric and magnetic fields.
    n_hat : Vector
        Interface normal (nominally pointing from medium n1 to n2).
    n1, n2 : float
        Refractive indices of incident/transmission media.
    AR : callable or None
        - None: use Fresnel_coeffi(n1, n2, cos(theta_i)).
        - callable: use AR(theta_i) in radians, returns (t_p, t_s, r_p, r_s).

    Returns
    -------
    E_t, H_t, E_r, H_r, t_p, t_s, r_p, r_s
    """
    # Promote inputs to highest precision before vector algebra.
    E = _promote_vector_precision(E)
    H = _promote_vector_precision(H)
    n_hat = _promote_vector_precision(n_hat)

    # Propagation direction approximated by normalized Poynting vector
    S_i = poyntingVector(E, H)
    k_i = normalized(S_i)

    # Ensure normal orientation gives positive incidence cosine
    cos_i = dot(n_hat, k_i)
    cos_i_np = np.clip(_to_numpy(cos_i), -1.0, 1.0)
    flip_mask = cos_i_np < 0
    if np.any(flip_mask):
        # Flip normals point-wise instead of global flipping.
        # This avoids corrupting points that already have correct normal orientation.
        if isinstance(n_hat.x, T.Tensor):
            mask_t = T.as_tensor(flip_mask, dtype=T.bool, device=n_hat.x.device)
            n_hat = Vector(
                T.where(mask_t, -n_hat.x, n_hat.x),
                T.where(mask_t, -n_hat.y, n_hat.y),
                T.where(mask_t, -n_hat.z, n_hat.z),
            )
        else:
            n_hat = Vector(
                np.where(flip_mask, -n_hat.x, n_hat.x),
                np.where(flip_mask, -n_hat.y, n_hat.y),
                np.where(flip_mask, -n_hat.z, n_hat.z),
            )
        cos_i = dot(n_hat, k_i)
        cos_i_np = np.clip(_to_numpy(cos_i), -1.0, 1.0)
    cos_i = _to_backend_like(cos_i_np, cos_i)

    # Reflected direction
    k_r = k_i - n_hat * (2.0 * dot(k_i, n_hat))

    # Snell relation for transmitted direction
    sin_i2 = np.maximum(0.0, 1.0 - cos_i_np**2)
    sin_t2 = (n1 / n2) ** 2 * sin_i2
    tir = np.abs(sin_t2) >= (1.0 - TIR_TOL)
    sin_t2_clip = np.clip(sin_t2, 0.0, 1.0)
    cos_t_np = np.sqrt(np.maximum(0.0, 1.0 - sin_t2_clip))
    cos_t = _to_backend_like(cos_t_np, cos_i)

    k_t = (k_i - n_hat * cos_i) * (n1 / n2) + n_hat * cos_t

    # Build s/p basis
    s = cross(k_i, n_hat)
    s_mag_np = _to_numpy(magnitude(s))
    degenerate = s_mag_np < EPS
    if np.any(degenerate):
        if isinstance(n_hat.x, T.Tensor):
            ones = T.ones_like(n_hat.x, dtype=n_hat.x.dtype, device=n_hat.x.device)
            zeros = T.zeros_like(n_hat.x, dtype=n_hat.x.dtype, device=n_hat.x.device)
            ex_v = Vector(ones, zeros, zeros)
            ey_v = Vector(zeros, ones, zeros)
        else:
            ones = np.ones_like(s_mag_np)
            zeros = np.zeros_like(s_mag_np)
            ex_v = Vector(ones, zeros, zeros)
            ey_v = Vector(zeros, ones, zeros)
        s_alt = cross(n_hat, ex_v)
        s_alt_mag_np = _to_numpy(magnitude(s_alt))
        if np.any(s_alt_mag_np < EPS):
            s_alt = cross(n_hat, ey_v)
        s = s_alt
        s_mag_np = _to_numpy(magnitude(s))

    # Safe normalization to avoid 1/0 near degenerate incidence geometry
    s_den_np = np.where(s_mag_np < EPS, 1.0, s_mag_np)
    s_den = _to_backend_like(s_den_np, s.x)
    s = s * (1.0 / s_den)

    p_i = normalized(cross(s, k_i))
    p_r = normalized(cross(s, k_r))
    p_t = normalized(cross(s, k_t))

    # Coefficients: plain Fresnel or AR lookup
    if AR is None:
        t_p, t_s, r_p, r_s = Fresnel_coeffi(n1, n2, cos_i_np)
    else:
        theta_i = np.arccos(np.clip(np.abs(cos_i_np), 0.0, 1.0))
        t_p, t_s, r_p, r_s = AR(theta_i)
    # Match coefficient backend to field backend for mixed NumPy/Torch safety.
    t_p = _to_backend_like(_to_numpy(t_p), E.x)
    t_s = _to_backend_like(_to_numpy(t_s), E.x)
    r_p = _to_backend_like(_to_numpy(r_p), E.x)
    r_s = _to_backend_like(_to_numpy(r_s), E.x)

    # Decompose incident field on (s, p_i)
    E_i_s = dot(E, s)
    E_i_p = dot(E, p_i)

    # Reconstruct reflected/transmitted E fields
    E_r = s * (r_s * E_i_s) + p_r * (r_p * E_i_p)
    E_t = s * (t_s * E_i_s) + p_t * (t_p * E_i_p)

    # Plane-wave relation: H = n * (k x E)
    H_r = cross(k_r, E_r) * n1
    H_t = cross(k_t, E_t) * n2

    # TIR branch: suppress transmitted propagating field in this simplified model
    if np.any(tir):
        if isinstance(E_t.x, T.Tensor):
            tir_mask_t = T.as_tensor(tir, dtype=T.bool, device=E_t.x.device)
            zero_ex = T.zeros_like(E_t.x)
            zero_hx = T.zeros_like(H_t.x)
            E_t = Vector(
                T.where(tir_mask_t, zero_ex, E_t.x),
                T.where(tir_mask_t, T.zeros_like(E_t.y), E_t.y),
                T.where(tir_mask_t, T.zeros_like(E_t.z), E_t.z),
            )
            H_t = Vector(
                T.where(tir_mask_t, zero_hx, H_t.x),
                T.where(tir_mask_t, T.zeros_like(H_t.y), H_t.y),
                T.where(tir_mask_t, T.zeros_like(H_t.z), H_t.z),
            )
        else:
            E_t = Vector(np.where(tir, 0, E_t.x), np.where(tir, 0, E_t.y), np.where(tir, 0, E_t.z))
            H_t = Vector(np.where(tir, 0, H_t.x), np.where(tir, 0, H_t.y), np.where(tir, 0, H_t.z))

    # Keep interface consistent: returned fields are always Vector objects.
    E_t = _as_vector(E_t)
    H_t = _as_vector(H_t)
    E_r = _as_vector(E_r)
    H_r = _as_vector(H_r)

    return E_t, H_t, E_r, H_r, t_p, t_s, r_p, r_s
