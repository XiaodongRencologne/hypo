import numpy as np
from .vecops import cross,  normalized


def poyntingVector(E, H):
    '''
    Calculate Poynting Vector S = E x H* (cross product of electric field and complex conjugate of magnetic field).
    '''
    s = cross(E, H.conj()) * 0.5
    return s.real()

def k_vector(E, H):
    '''
    Calculate the unit vector of the Poynting Vector.
    '''
    S = poyntingVector(E, H)
    return normalized(S)