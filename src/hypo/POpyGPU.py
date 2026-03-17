#!/usr/bin/env python
# coding: utf-8

"""GPU-accelerated physical-optics propagation kernels.

This module contains the low-level propagation routines used by the refractive
PO workflow. The two public entry points are:

- ``PO_GPU_2``: near-field propagation from one sampled surface to another
- ``PO_far_GPU2``: far-field propagation toward a directional observation grid

Implementation notes
--------------------
- Inputs are surface samples and fields stored in package-specific vector
  objects, while the heavy accumulation is performed with PyTorch tensors.
- The code converts input geometry/current data to torch tensors in batches so
  the propagation can scale to large target grids without exhausting GPU memory.
- Output fields are returned in the same coordinate system in which the input
  geometry and field vectors are provided.

Physical convention
-------------------
Both kernels use equivalent electric and magnetic surface currents derived from
the incident tangential fields and the local surface normal. The exact field
normalization and transmission/reflection physics are handled at a higher level;
this module focuses on the propagation integral itself.
"""

import os
from tqdm import tqdm
import numpy as np
import torch as T
from torch.cuda.amp import autocast
cpu_cores = T.get_num_threads()
T.set_num_threads(cpu_cores*2)
from .vecops import Vector, dot, cross

import time;
from . import  Z0

def PO_GPU_2(face1,face1_n,face1_dS,
           face2,
           Field_in_E,Field_in_H,
           k,n,
           device =T.device('cuda')):
    """
    Optimized Physical Optics (PO) GPU implementation for near-field calculations.

    Parameters
    ----------
    face1:
        Source surface sample locations.
    face1_n:
        Unit surface normal at each source sample.
    face1_dS:
        Quadrature weight / differential area associated with each source sample.
    face2:
        Observation points where the propagated field is evaluated.
    Field_in_E, Field_in_H:
        Incident electric and magnetic fields sampled on ``face1``.
    k:
        Free-space wave number.
    n:
        Refractive index used to scale the propagation constant inside the
        medium between ``face1`` and ``face2``.
    device:
        PyTorch execution device.

    Returns
    -------
    Field_E, Field_H:
        Propagated electric and magnetic fields sampled on ``face2``.

    Notes
    -----
    The routine:

    1. converts incident fields on ``face1`` to equivalent electric and
       magnetic surface currents,
    2. batches the target points on ``face2``,
    3. accumulates the PO integral on the requested device,
    4. and finally converts the result back to NumPy-backed vector objects.
    """

    # Use double precision throughout. These kernels are usually phase-sensitive
    # and accumulate many oscillatory contributions, so float64/complex128 are
    # the safer default.
    real_dtype = T.float64

    # Allocate the output field vectors directly on the execution device.
    N_f = face2.x.size
    Field_E = Vector()
    Field_E.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.z = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H = Vector()
    Field_H.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.z = T.zeros(N_f, dtype=T.complex128, device=device)

    k_n = k * n
    k0 = k * 1.0

    ds = face1_dS*face1_n.N * k_n**2 /4/np.pi

    # Constant scalars used inside the torch accumulation kernel.
    k_n = T.tensor(k_n,device = device)
    k0 = T.tensor(k0,device = device)
    Z = T.tensor(Z0 / n/ Z0,device = device)

    # Build equivalent electric/magnetic currents from the incident tangential
    # fields. The package convention keeps these as vector objects until the
    # final packed tensor is formed for the GPU kernel.
    start_time = time.time()
    J_in = 2*ds * cross(face1_n, Field_in_H)
    JE = T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1), 
                  dtype=T.complex128,
                  device=device).contiguous()
    
    J_in = 2*ds * cross(face1_n, Field_in_E)
    JM = T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1), 
                  dtype=T.complex128,
                  device=device).contiguous()

    #print('tiemusage:',time.time() - start_time)

    # Pack source and target geometry into contiguous tensors with shapes that
    # broadcast naturally inside the vectorized batch kernel.
    Surf1 = T.stack([T.tensor(face1.x.ravel()), 
                     T.tensor(face1.y.ravel()), 
                     T.tensor(face1.z.ravel())], dim=0).reshape(3,1,-1)
    Surf1 = Surf1.contiguous().to(device)
    Surf2 = T.stack([T.tensor(face2.x.ravel()), 
                     T.tensor(face2.y.ravel()), 
                     T.tensor(face2.z.ravel())], dim=0).reshape(3,-1,1).to(device)
    Surf2 = Surf2.contiguous().to(device)

    #N_current = face1.x.size
    #N_target = face2.x.size 
    #R_n_cpu = T.zeros((3,N_target,N_current),dtype = T.complex128,device = 'cpu', pin_memory=True)

    #Memory_size = R_n_cpu.element_size() * R_n_cpu.nelement()


    #@T.jit.script
    '''
    def calculate_R(point1,point2,K,R_n):
        R = point2 - point1
        r = T.linalg.norm(R, dim=0) # Compute the norm directly
        R = R / r  # Normalize the vector
        r.mul_(K)
        Factor = T.exp(-1j*r)*(1+1j*r)/r**2
        R_n = R * Factor
    '''
    
    def calculate_fields(s2,K):
        """
        Evaluate the near-field PO kernel for one batch of target points.

        Parameters
        ----------
        s2:
            Batched target-point tensor of shape ``(3, N_batch, 1)``.
        K:
            Propagation constant in the medium.
        """
        
        # Compute source-to-target vectors and the scalar distance. The kernel
        # uses the normalized direction multiplied by the Green-function factor.
        R = (s2 - Surf1).contiguous()
        r = T.linalg.norm(R, dim=0).contiguous().unsqueeze(0) # Compute the norm directly
        R = R / r # Normalize the vector
        r = r * K
        R = R * T.exp(-1j*r)*(1+1j*r)/r**2
        del(r,s2)
        # Magnetic field contribution from the equivalent electric current.
        he = T.cross(JE, R, dim=0).contiguous()
        He = he.sum(dim=-1)
        del(he)

        # Electric field contribution from the equivalent magnetic current.
        ee = T.cross(JM, R, dim = 0).contiguous()
        Ee = ee.sum(dim = -1)
        del(ee)

        return Ee, He
    # Estimate a target batch size from available memory. This is a practical
    # heuristic rather than an exact bound; it trades memory pressure against
    # kernel launch overhead.
    if device == T.device('cuda'):
        # Get total, allocated, and reserved memory
        total_memory = T.cuda.get_device_properties(0).total_memory
        allocated_memory = T.cuda.memory_allocated(0)
        reserved_memory = T.cuda.memory_reserved(0)

        # Calculate free memory
        free_memory = total_memory - reserved_memory

        # Adjust batch size based on free memory
        element_size = JE.element_size() * JE.nelement()
        batch_size = int(free_memory / element_size / 50)
        print(f"Batch size: {batch_size}")
    else:
        batch_size = os.cpu_count() * 2
        print(f"Batch size: {batch_size}")

    print(f"Batch size: {batch_size}")
    N = face2.x.size
    num_batches = N // batch_size

    # Process the observation points in batches so the full (target x source)
    # tensor does not need to reside in memory at once.

    with T.no_grad():
        for i in tqdm(range(num_batches),mininterval=5):
            start = i * batch_size
            end = (i + 1) * batch_size
            Ee, He = calculate_fields(Surf2[:,start:end,:].contiguous() ,k_n)

            Field_E.x[start:end] = Ee[0, :]
            Field_E.y[start:end] = Ee[1, :]
            Field_E.z[start:end] = Ee[2, :]
            Field_H.x[start:end] = He[0, :]
            Field_H.y[start:end] = He[1, :]
            Field_H.z[start:end] = He[2, :]
            
                
        # Process the final partial batch, if any.
        if N % batch_size != 0:
            start = num_batches * batch_size
            Ee, He = calculate_fields(Surf2[:,start:,:].contiguous(),k_n)
            Field_E.x[start:] = Ee[0, :]
            Field_E.y[start:] = Ee[1, :]
            Field_E.z[start:] = Ee[2, :]
            Field_H.x[start:] = He[0, :]
            Field_H.y[start:] = He[1, :]
            Field_H.z[start:] = He[2, :]
    # Convert the package vector objects back to NumPy arrays only after all
    # batched accumulation is finished.
    Field_E.to_numpy()
    Field_H.to_numpy()

    if device == 'cuda' or device == T.device('cuda'):
        T.cuda.empty_cache()
        T.cuda.synchronize()
    return Field_E, Field_H
    

def PO_far_GPU2(face1,face1_n,face1_dS,
               face2,
               Field_in_E,
               Field_in_H,
               k,
               device =T.device('cuda')):
    """GPU far-field physical-optics propagation.

    Parameters
    ----------
    face1:
        Source surface sample locations.
    face1_n:
        Unit surface normals at the source samples.
    face1_dS:
        Differential area / quadrature weights on the source surface.
    face2:
        Far-field observation directions or points. In typical usage this is a
        spherical grid whose coordinates encode observation directions.
    Field_in_E, Field_in_H:
        Incident electric and magnetic fields on ``face1``.
    k:
        Free-space wave number.
    device:
        Torch execution device.

    Returns
    -------
    Field_E, Field_H:
        Far-field electric and magnetic vectors sampled on ``face2``.

    Notes
    -----
    In the far-field approximation, the full source-to-target distance is not
    recomputed for each pair. Instead, the phase is formed from the projected
    observation direction and the source coordinates.
    """
    # output field:
    N_f = face2.x.size
    Field_E = Vector()
    Field_E.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.z = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H = Vector()
    Field_H.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.z = T.zeros(N_f, dtype=T.complex128, device=device)

    ds = face1_dS * face1_n.N * k**2 /4/np.pi


    J_in=2*ds * cross(face1_n,Field_in_H)
    JE=T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1),
                dtype = T.complex128).to(device)
    
    J_in = 2*ds * cross(face1_n, Field_in_E)
    JM = T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1), 
                  dtype=T.complex128,
                  device=device).contiguous()
    
    # Pack geometry into tensors with shapes that match the batched phase
    # accumulation performed below.
    Surf1 = T.stack([T.tensor(face1.x.ravel()), 
                     T.tensor(face1.y.ravel()), 
                     T.tensor(face1.z.ravel())], dim=0).reshape(3,1,-1)
    Surf1 = Surf1.contiguous().to(device)
    Surf2 = T.stack([T.tensor(face2.x.ravel()), 
                     T.tensor(face2.y.ravel()), 
                     T.tensor(face2.z.ravel())], dim=0).reshape(3,-1,1)
    Surf2 = Surf2.contiguous().to(device)

    def calculate_fields(s2,K):
        """
        Evaluate the far-field kernel for one batch of observation directions.
        """
        # The far-field phase depends on the projection of source coordinates
        # onto the observation direction rather than on the full point-to-point
        # distance used in the near-field kernel.
        Phase = k * T.sum((s2 * Surf1).contiguous(),axis = 0).contiguous()
        Phase =  T.exp(1j * Phase) #* ds
        # Electric field from the equivalent magnetic current.
        ee = T.sum( JM * Phase, axis = -1 ).contiguous()
        r = s2.reshape(3,-1).contiguous().to(dtype = T.complex128)
        Ee = 1j * T.cross(r, ee, dim=0).contiguous()
        # Magnetic field from the equivalent electric current.
        he = T.sum( JE * Phase, axis = -1 ).contiguous()
        He = -1j /Z0 * T.cross(r, he, dim=0).contiguous()

        return Ee, He
    
    # As in the near-field solver, use a memory-based heuristic to choose a
    # reasonable batch size for the current device.
    if device == T.device('cuda'):
        # Get total, allocated, and reserved memory
        total_memory = T.cuda.get_device_properties(0).total_memory
        allocated_memory = T.cuda.memory_allocated(0)
        reserved_memory = T.cuda.memory_reserved(0)

        # Calculate free memory
        free_memory = total_memory - reserved_memory

        # Adjust batch size based on free memory
        element_size = JE.element_size() * JE.nelement()
        batch_size = int(free_memory / element_size / 50)
    else:
        batch_size = os.cpu_count() * 2

    print(f"Batch size: {batch_size}")
    N = face2.x.size
    num_batches = N // batch_size

    for i in tqdm(range(num_batches),mininterval=5):
        start = i * batch_size
        end = (i + 1) * batch_size
        Ee, He = calculate_fields(Surf2[:,start:end,:].contiguous() ,k)

        Field_E.x[start:end] = Ee[0, :]
        Field_E.y[start:end] = Ee[1, :]
        Field_E.z[start:end] = Ee[2, :]
        Field_H.x[start:end] = He[0, :]
        Field_H.y[start:end] = He[1, :]
        Field_H.z[start:end] = He[2, :]
            
    # Process the final partial batch, if any.
    if N % batch_size != 0:
        start = num_batches * batch_size
        Ee, He = calculate_fields(Surf2[:,start:,:].contiguous(),k)
        Field_E.x[start:] = Ee[0, :]
        Field_E.y[start:] = Ee[1, :]
        Field_E.z[start:] = Ee[2, :]
        Field_H.x[start:] = He[0, :]
        Field_H.y[start:] = He[1, :]
        Field_H.z[start:] = He[2, :]
    # Convert the package vector objects back to NumPy arrays only once the
    # entire far-field accumulation is complete.
    Field_E.to_numpy()
    Field_H.to_numpy()

    if device == 'cuda' or device == T.device('cuda'):
        T.cuda.empty_cache()
        T.cuda.synchronize()
    return Field_E, Field_H
