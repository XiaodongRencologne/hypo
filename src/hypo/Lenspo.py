"""
Lens PO Workflow Skeleton (No Calculations)
===========================================

This module is a commented scaffold for lens Physical Optics (PO) workflows.
It mirrors function interfaces in LensPO.py, but intentionally leaves all
numerical operations unimplemented.

Use this file to:
1. Keep API-compatible function signatures.
2. Document expected data flow and variable meaning.
3. Fill in implementations later without changing external callers.
"""

import torch as T

from .antiReflection import read_Fresnel_coeffi_AR
from .interface_rt import calc_reflect_transmit_fields
from .POpyGPU import PO_GPU_2 as PO_GPU


def lensPO(face_source, face_source_n, face_source_dS,
              face_obs, face_obs_n,
              Field_in_E, Field_in_H,
              k, n,
              AR_filename=None,
              frequency=None,
              device=T.device("cuda")):
    """
    Lens PO workflow with optional AR coating.

    Parameters:
        face_source, face_source_n, face_source_dS: Source surface geometry, normals, and area weights.
        face_obs, face_obs_n: Observation surface geometry and normals.
        Field_in_E, Field_in_H: Incident electric/magnetic fields.
        k, n: Wave number and refractive index.
        AR_filename: AR coefficient data file. If None, no-AR Fresnel is used.
        frequency: Frequency key for AR lookup (for example: '120GHz').
            Must be provided together with AR_filename.
        device: Torch device for GPU/CPU execution.

    Returns:
        f2_E_t, f2_H_t:
            Transmitted electric/magnetic fields after the second interface.
    """
    n0 = 1.0

    # 1) Optional AR setup:
    #    - with AR: use interpolated AR1/AR2
    #    - no AR: pass AR=None to use base Fresnel coefficients
    use_ar = (AR_filename is not None) or (frequency is not None)
    if use_ar and (AR_filename is None or frequency is None):
        raise ValueError("AR_filename and frequency must be provided together, or both omitted.")

    if use_ar:
        AR1, AR2 = read_Fresnel_coeffi_AR(AR_filename, frequency, n0, n)
        if AR1 is None or AR2 is None:
            raise ValueError(f"Failed to load AR coefficients from '{AR_filename}' frequency '{frequency}'.")
    else:
        AR1, AR2 = None, None

    # 2) First interface (source surface): incident -> transmitted fields.
    f1_E_t, f1_H_t, f1_E_r,f1_H_r, tp1, ts1, rp1, rs1 = calc_reflect_transmit_fields(
        E=Field_in_E,
        H=Field_in_H,
        n_hat=face_source_n,
        n1=n0,
        n2=n,
        AR=AR1,
    )

    # 3) Propagate transmitted field from source surface to observation surface with PO.
    F2_in_E, F2_in_H = PO_GPU(
        face_source, face_source_n, face_source_dS,
        face_obs,
        f1_E_t, f1_H_t,
        k, n,
        device=device
    )

    # 4) Second interface (observation surface): inside lens -> outside medium.
    f2_E_t, f2_H_t, f2_E_r, f2_H_r, tp2, ts2, rp2, rs2 = calc_reflect_transmit_fields(
        E=F2_in_E,
        H=F2_in_H,
        n_hat=face_obs_n,
        n1=n,
        n2=n0,
        AR=AR2,
    )

    # Return only the final transmitted fields as requested.
    return f2_E_t, f2_H_t, f1_E_t, f1_H_t, tp2,ts2,tp1,ts1

