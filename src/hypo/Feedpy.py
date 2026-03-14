'''
This package provides N input beams, and each beam function can offer scalar and vector modes.
1. Gaussian beam in far field;
2. Gaussian beam near field;

Notes
-----
- This module defines feed field generators for scalar and vector pipelines.
- The core outputs are callable beam functions attached to class instances.
'''

import numpy as np;
from .vecops import Vector
from .coxvec import Ludwig_Cox_vector as CO

from . import Z0

def _Gaussian2d(theta_x_2,theta_y_2,theta, phi):
    """Return elliptical 2D Gaussian envelope in angular space."""
    Amp = np.exp(-theta**2*np.cos(phi)**2/theta_x_2 - theta**2*np.sin(phi)**2/theta_y_2)
    return Amp

def _Normal_factor(theta_x_2,theta_y_2):
    """
    Compute normalization factor for the angular Gaussian profile.

    A dense theta/phi grid is used for numerical integration of |E|^2 sin(theta).
    """
    theta = np.linspace(0,np.pi,10001)
    dt = theta[1]-theta[0]
    phi = np.linspace(0,2*np.pi,10001)
    dp = phi[1]-phi[0]
    E = _Gaussian2d(theta_x_2,theta_y_2,theta,phi)
    P = np.sum(np.abs(E)**2*np.sin(theta)*dt*dp)
    print(P)
    Nf = np.sqrt(1/P)
    print(Nf)
    print(np.sum(np.abs(Nf*E)**2*np.sin(theta)*dt*dp))
    return Nf
    

'''
Type 1: Gaussian beam;
'''

class GaussiBeam():
    """Single-axis Gaussian feed model."""
    def __init__(self,
                 Edge_taper,
                 Edge_angle,
                 k,
                 coord_sys,
                 polarization='scalar'):
        # Input beam specifications.
        self.T = Edge_taper
        self.A = Edge_angle/180*np.pi

        # Shared coordinate transformation object.
        self.coord_sys = coord_sys
        self.k = k
        # Legacy expression kept for backward compatibility with original workflow.
        b = (np.log10((1+np.cos(self.A))/2)-self.T/20)/(k*(1-np.cos(self.A))*np.log10(np.exp(1)))
        # Derived waist/curvature parameters from edge taper constraints.
        w_2 = 2/k*(20*np.log10((1+np.cos(self.A))/2)-self.T)/(20*k*(1-np.cos(self.A))*np.log10(np.exp(1)))
        b = k*w_2/2
        w0 = np.sqrt(w_2)
        theta_2 = -20*self.A**2/self.T*np.log10(np.exp(1))
        
        if polarization.lower()=='scalar':
            # Scalar mode: returns real/imag parts of complex field.
            def beam(Mirror,Mirror_n):
                r,theta,phi = self.coord_sys.ToSpherical(Mirror.x,Mirror.y,Mirror.z)
                R=np.sqrt(r**2-b**2+1j*2*b*Mirror.z)
                E=np.exp(-1j*k*R-k*b)/R*(1+np.cos(theta))/2/k/w0*b
                E=E*np.sqrt(8)
                return E.real,E.imag
        else: 
            # Vector mode normalization constant.
            B = 2*np.pi*np.exp(-2*b*k)/4/b**3/k**3
            B = B*(np.exp(4*b*k)*(8*b**2*k**2-4*b*k+1)-1)
            Nf = np.sqrt(4*np.pi/k**2/B)
            def beam(Mirror,k):
                r,theta,phi = self.coord_sys.ToSpherical(Mirror.x,Mirror.y,Mirror.z)
                F = (1+np.cos(theta)) * np.exp(k*b*np.cos(theta)) * np.exp(-1j*k*r)/r
                F = Nf*F

                E = Vector()
                H = Vector()
                co,cx,crho=CO(theta,phi)
                if polarization.lower()=='x':
                    E= F * co 
                    H= F * cx
                elif polarization.lower()=='y':
                    E= F * cx 
                    H= F * co 
                return E, H
        # Public beam callable.
        self.source = beam

class Elliptical_GaussianBeam():
    """Elliptical Gaussian feed with independent x/y taper settings."""
    def __init__(self,
                 Edge_taper,
                 Edge_angle,
                 k,
                 coor_angle,coor_displacement,
                 polarization='scalar'):
        '''
        Build an elliptical Gaussian feed model.

        param 1: 'Edge_taper' define ratio of maximum power and the edge power in the antenna;
        param 2: 'Edge_angle' is the angular size of the mirror seen from the feed coordinates;
        param 3: 'k' wave number;
        param 4: 'Mirror_in' the sampling points in the mirror illumanited by feed;
        param 5: 'fieldtype' chose the scalar mode or vector input field.
        '''
        self.Tx = Edge_taper[0]
        self.Ty = Edge_taper[1]
        self.Ax = Edge_angle[0]/180*np.pi
        self.Ay = Edge_angle[1]/180*np.pi
        
        self.coor_A = coor_angle
        self.coor_D = coor_displacement

        # Solve beam parameters independently along x and y axes.
        bx = (np.log10((1+np.cos(self.Ax))/2)-self.Tx/20)/(k*(1-np.cos(self.Ax))*np.log10(np.exp(1)))
        by = (np.log10((1+np.cos(self.Ay))/2)-self.Ty/20)/(k*(1-np.cos(self.Ay))*np.log10(np.exp(1)))
        #print(bx,by)
        wx_2 = 2/k*(20*np.log10((1+np.cos(self.Ax))/2)-self.Tx)/(20*k*(1-np.cos(self.Ax))*np.log10(np.exp(1)))
        wy_2 = 2/k*(20*np.log10((1+np.cos(self.Ay))/2)-self.Ty)/(20*k*(1-np.cos(self.Ay))*np.log10(np.exp(1)))
        bx = k*wx_2/2
        by = k*wy_2/2
        wx = np.sqrt(wx_2)
        wy = np.sqrt(wy_2)
        #print(2/k/wx*180/np.pi,2/k/wy*180/np.pi)
        theta_x_2 = -20*self.Ax**2/self.Tx*np.log10(np.exp(1))
        theta_y_2 = -20*self.Ay**2/self.Ty*np.log10(np.exp(1))
        #print(np.sqrt(theta_x_2)*180/np.pi,np.sqrt(theta_y_2)*180/np.pi)
        # Normalization used by vector mode pattern.
        Nf = _Normal_factor(theta_x_2 ,theta_y_2)
        if polarization.lower()=='scalar':
            # Scalar near-field model in Cartesian coordinates.
            def beam(Mirror,Mirror_n):
                r,theta,phi = self.coord_sys.ToSpherical(Mirror.x,Mirror.y,Mirror.z)
                w_x_2 = wx_2*(1+(Mirror.z/bx)**2)
                w_y_2 = wy_2*(1+(Mirror.z/by)**2)
                Amp_x = -1j/bx*np.exp(k*bx)*wx/np.sqrt(w_x_2)*np.exp(-Mirror.x**2/w_x_2)
                Amp_y = -1j/by*np.exp(k*by)*wy/np.sqrt(w_y_2)*np.exp(-Mirror.y**2/w_y_2)
                R_x = Mirror.z*(1+(bx/Mirror.z)**2)
                Amp = Amp_x*Amp_y
                F = Amp * np.exp(-1j*(k*(Mirror.x**2+Mirror.y**2)/2/R_x \
                                      + k*Mirror.z - np.arctan(Mirror.z/bx)/2 - np.arctan(Mirror.z/by)/2))
                cos_i=np.abs(Mirror.x*Mirror_n.x+Mirror.y*Mirror_n.y+Mirror.z*Mirror_n.z)/r
                return F, cos_i
        else: 
            # Vector mode using Ludwig-Cox polarization basis.
            def beam(Mirror,Mirror_n):
                r,theta,phi = self.coord_sys.ToSpherical(Mirror.x,Mirror.y,Mirror.z)
                F = Nf*_Gaussian2d(theta_x_2 ,theta_y_2, theta, phi) * np.exp(-1j*k*r)/r
                E = Vector()
                H = Vector()
                co,cx,crho=CO(theta,phi);
                #co,cx,crho=CO(theta,phi);
                if polarization.lower()=='x':
                    E = F * co #scalarproduct(F,co);
                    H = F * cx #scalarproduct(F/Z0,cx);
                    E_co = F#dotproduct(E,co)
                    E_cx = 0#dotproduct(E,cx)
                    E_r  = 0#dotproduct(E,crho)
                elif polarization.lower()=='y':
                    H= F * co #scalarproduct(F/Z0,co)
                    E= F * cx #scalarproduct(F,cx)
                    E_co = 0#dotproduct(E,co)
                    E_cx = F#dotproduct(E,cx)
                    E_r  = 0#dotproduct(E,crho)
                return E, H , E_co , E_cx
        # Public beam callable.
        self.beam = beam
