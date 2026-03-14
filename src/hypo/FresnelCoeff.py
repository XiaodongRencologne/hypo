import numpy as np
import torch as T

from .EMtools import poyntingVector
from .vecops import Vector, cross, dot, magnitude, normalized


def Fresnel_coeffi(n1, n2, theta_i_cos):
    """
    Fresnel coefficients with physically correct complex reflection in TIR region.

    Parameters
    ----------
    n1 : float
        Refractive index of incident medium.
    n2 : float
        Refractive index of transmission medium.
    theta_i_cos : array-like or float
        cos(theta_i), incidence angle cosine.

    Returns
    -------
    t_p, t_s, r_p, r_s : ndarray or scalar (complex)
        Fresnel coefficients for p/s polarizations.
        In total internal reflection (TIR), r_p and r_s are complex phase terms
        with |r| = 1, while t terms become evanescent-related complex values.
    """
    theta_i_cos = np.asarray(theta_i_cos, dtype=np.float64)
    theta_i_cos = np.clip(theta_i_cos, -1.0, 1.0)

    # complex dtype so sqrt works in TIR (imaginary theta_t_cos)
    theta_i_cos_c = theta_i_cos.astype(np.complex128)

    theta_i_sin2 = 1.0 - theta_i_cos_c**2
    theta_t_sin2 = (n1 / n2)**2 * theta_i_sin2
    theta_t_cos = np.sqrt(1.0 - theta_t_sin2)

    # keep physically continuous branch
    theta_t_cos = np.where(np.real(theta_t_cos) < 0, -theta_t_cos, theta_t_cos)

    den_p = n2 * theta_i_cos_c + n1 * theta_t_cos
    den_s = n1 * theta_i_cos_c + n2 * theta_t_cos

    eps = 1e-30 + 0j
    den_p = np.where(np.abs(den_p) < 1e-30, eps, den_p)
    den_s = np.where(np.abs(den_s) < 1e-30, eps, den_s)

    t_p = 2 * n1 * theta_i_cos_c / den_p
    t_s = 2 * n1 * theta_i_cos_c / den_s

    r_p = (n2 * theta_i_cos_c - n1 * theta_t_cos) / den_p
    r_s = (n1 * theta_i_cos_c - n2 * theta_t_cos) / den_s

    if np.ndim(theta_i_cos) == 0:
        return t_p.item(), t_s.item(), r_p.item(), r_s.item()

    return t_p, t_s, r_p, r_s


#def calc_reflect_transmit_fields(E, H, n_hat, n1, n2, AR=None):
def calc_reflect_transmit_fields(n1, n2, n_hat, E, H, AR=None):
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
    # Propagation direction approximated by normalized Poynting vector
    S_i = poyntingVector(E, H)
    k_i = normalized(S_i)

    # Ensure normal orientation gives positive incidence cosine
    cos_i = dot(n_hat, k_i)
    cos_i_np = np.asarray(cos_i)
    if np.any(cos_i_np < 0):
        n_hat = (-1.0) * n_hat
        cos_i = dot(n_hat, k_i)
        cos_i_np = np.asarray(cos_i)

    # Reflected direction
    k_r = k_i - 2.0 * dot(k_i, n_hat) * n_hat

    # Snell relation for transmitted direction
    sin_i2 = np.maximum(0.0, 1.0 - cos_i_np**2)
    sin_t2 = (n1 / n2) ** 2 * sin_i2
    tir = np.abs(sin_t2) >= 1.0
    sin_t2_clip = np.clip(sin_t2, 0.0, 1.0)
    cos_t = np.sqrt(1.0 - sin_t2_clip)

    k_t = (n1 / n2) * (k_i - cos_i * n_hat) + cos_t * n_hat

    # Build s/p basis
    s = cross(k_i, n_hat)
    s_mag_np = np.asarray(magnitude(s))
    degenerate = s_mag_np < 1e-15
    if np.any(degenerate):
        ones = np.ones_like(s_mag_np)
        zeros = np.zeros_like(s_mag_np)
        ex_v = Vector(ones, zeros, zeros)
        ey_v = Vector(zeros, ones, zeros)
        s_alt = cross(n_hat, ex_v)
        s_alt_mag_np = np.asarray(magnitude(s_alt))
        if np.any(s_alt_mag_np < 1e-15):
            s_alt = cross(n_hat, ey_v)
        s = s_alt
    s = normalized(s)

    p_i = normalized(cross(s, k_i))
    p_r = normalized(cross(s, k_r))
    p_t = normalized(cross(s, k_t))

    # Coefficients: plain Fresnel or AR lookup
    if AR is None:
        t_p, t_s, r_p, r_s = Fresnel_coeffi(n1, n2, cos_i_np)
    else:
        theta_i = np.arccos(np.clip(np.abs(cos_i_np), -1.0, 1.0))
        t_p, t_s, r_p, r_s = AR(theta_i)

    # Decompose incident field on (s, p_i)
    E_i_s = dot(E, s)
    E_i_p = dot(E, p_i)

    # Reconstruct reflected/transmitted E fields
    E_r = (r_s * E_i_s) * s + (r_p * E_i_p) * p_r
    E_t = (t_s * E_i_s) * s + (t_p * E_i_p) * p_t

    # Plane-wave relation: H = n * (k x E)
    H_r = n1 * cross(k_r, E_r)
    H_t = n2 * cross(k_t, E_t)

    # TIR branch: suppress transmitted propagating field in this simplified model
    if np.any(tir):
        E_t = Vector(np.where(tir, 0, E_t.x), np.where(tir, 0, E_t.y), np.where(tir, 0, E_t.z))
        H_t = Vector(np.where(tir, 0, H_t.x), np.where(tir, 0, H_t.y), np.where(tir, 0, H_t.z))
        
    return E_t, H_t, E_r, H_r, t_p, t_s, r_p, r_s