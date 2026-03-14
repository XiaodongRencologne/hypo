import numpy as np
import h5py
from scipy.interpolate import CubicSpline


def read_Fresnel_coeffi_AR(filename, frequency, n1, n2):
    """
    Read AR-coating coefficients and build interpolation-based Fresnel functions.

    Parameters
    ----------
    filename : str
        HDF5 file path.
    frequency : str
        Frequency key in the HDF5 file (for example: '120GHz').
    n1, n2 : float
        Refractive indices for forward conversion.

    Assumption for HDF5 data:
    - group['tp'], group['ts'], group['rp'], group['rs'] are complex-valued arrays.
    - For transmission, stored value is:
          t_data = sqrt(P_t / P_in) * exp(j * phase)
      (power-ratio amplitude + phase difference)

    To convert to E-field transmission coefficient:
      Power relation:
          P_t / P_in = (n2*cos(theta_t) / (n1*cos(theta_i))) * |t_E|^2
      Therefore:
          t_E = t_data * sqrt(n1*cos(theta_i) / (n2*cos(theta_t)))
    """
    with h5py.File(filename, "r") as f:
        if frequency not in f:
            print(f"Frequency group '{frequency}' not found in the file.")
            return None, None

        group = f[frequency]
        theta_i = np.asarray(group["theta"][:], dtype=np.float64)

        # Snell law: n1*sin(theta_i) = n2*sin(theta_t)
        theta_t_sin = n1 / n2 * np.sin(theta_i)
        theta_t_sin = np.clip(theta_t_sin, -1.0, 1.0)
        theta_t = np.arcsin(theta_t_sin)

        tp = np.asarray(group["tp"][:], dtype=np.complex128)
        rp = np.asarray(group["rp"][:], dtype=np.complex128)
        ts = np.asarray(group["ts"][:], dtype=np.complex128)
        rs = np.asarray(group["rs"][:], dtype=np.complex128)

        # Convert transmission from sqrt(power-ratio) to E-field coefficient
        factor = np.sqrt(n1 * np.cos(theta_i) / (n2 * np.cos(theta_t)))
        tp = tp * factor
        ts = ts * factor

        Fresnel_coeffi_AR1, Fresnel_coeffi_AR2 = Creat_Fresnel_coeffi_AR(
            theta_i, tp, rp, ts, rs, n1, n2
        )
        return Fresnel_coeffi_AR1, Fresnel_coeffi_AR2


def Creat_Fresnel_coeffi_AR(theta_i, t_p, r_p, t_s, r_s, n1, n2):
    """
    Build two callable functions:
      Fresnel_coeffi_AR1(theta): forward, n1 -> n2
      Fresnel_coeffi_AR2(theta): backward, n2 -> n1
    """

    # Interpolate real/imag parts separately for complex coefficients
    tp_AR = CubicSpline(theta_i, t_p.real)
    tp_AR_imag = CubicSpline(theta_i, t_p.imag)

    rp_AR = CubicSpline(theta_i, r_p.real)
    rp_AR_imag = CubicSpline(theta_i, r_p.imag)

    ts_AR = CubicSpline(theta_i, t_s.real)
    ts_AR_imag = CubicSpline(theta_i, t_s.imag)

    rs_AR = CubicSpline(theta_i, r_s.real)
    rs_AR_imag = CubicSpline(theta_i, r_s.imag)

    def Fresnel_coeffi_AR1(theta):
        """
        Forward coefficients (n1 -> n2), theta is incident angle in medium n1.
        """
        theta = np.asarray(theta, dtype=np.float64)
        t_p_val = tp_AR(theta) + 1j * tp_AR_imag(theta)
        t_s_val = ts_AR(theta) + 1j * ts_AR_imag(theta)
        r_p_val = rp_AR(theta) + 1j * rp_AR_imag(theta)
        r_s_val = rs_AR(theta) + 1j * rs_AR_imag(theta)
        return t_p_val, t_s_val, r_p_val, r_s_val

    def Fresnel_coeffi_AR2(theta):
        """
        Backward coefficients (n2 -> n1), theta is incident angle in medium n2.

        1) Angle mapping by Snell law:
             n2*sin(theta) = n1*sin(theta_t)
           -> theta_t is the corresponding angle in medium n1.

        2) Reciprocity-like transmission conversion:
             t21 = (n2*cos(theta) / (n1*cos(theta_t))) * t12(theta_t)

           IMPORTANT:
           - Interpolation must use theta_t (n1-side angle), not theta.
        """
        theta = np.asarray(theta, dtype=np.float64)
        scalar_input = (theta.ndim == 0)
        theta = np.atleast_1d(theta)

        theta_t_sin = n2 * np.sin(theta) / n1
        NN_t = np.where(np.abs(theta_t_sin) >= 1.0)  # total reflection in n2->n1

        theta_t_sin_safe = np.clip(theta_t_sin, -1.0, 1.0)
        theta_t = np.arcsin(theta_t_sin_safe)

        # Use mapped angle theta_t for interpolation
        t12_p = tp_AR(theta_t) + 1j * tp_AR_imag(theta_t)
        t12_s = ts_AR(theta_t) + 1j * ts_AR_imag(theta_t)
        r21_p = rp_AR(theta_t) + 1j * rp_AR_imag(theta_t)
        r21_s = rs_AR(theta_t) + 1j * rs_AR_imag(theta_t)

        denom = n1 * np.cos(theta_t)
        factor = np.zeros_like(theta, dtype=np.float64)
        valid = np.abs(denom) > 1e-15
        factor[valid] = (n2 * np.cos(theta[valid])) / denom[valid]

        t_p_val = t12_p * factor
        t_s_val = t12_s * factor

        # TIR handling
        t_p_val[NN_t] = 0.0 + 0.0j
        t_s_val[NN_t] = 0.0 + 0.0j
        r21_p[NN_t] = np.exp(1j * np.pi)
        r21_s[NN_t] = np.exp(1j * np.pi)

        if scalar_input:
            return t_p_val[0], t_s_val[0], r21_p[0], r21_s[0]
        return t_p_val, t_s_val, r21_p, r21_s

    return Fresnel_coeffi_AR1, Fresnel_coeffi_AR2
