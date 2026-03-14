
# coding: utf-8

# In[66]:


"""
Utilities for building Gauss-Legendre quadrature sample points and weights.
"""
import numpy as np;

# 1D Gauss-Legendre quadrature integration on a uniformly partitioned interval.
def Gauss_L_quadrs1d(start, stop, N_part,N):
    """
    Build 1D Gauss-Legendre sampling points and weights on [start, stop].

    Parameters
    ----------
    start : float
        Start of the integration interval.
    stop : float
        End of the integration interval.
    N_part : int
        Number of uniform sub-intervals used to split [start, stop].
    N : int
        Number of Gauss-Legendre nodes per sub-interval.

    Returns
    -------
    x : ndarray
        Concatenated quadrature nodes over all sub-intervals.
    w : ndarray
        Corresponding quadrature weights for `x`.
    """
    # Uniform sub-interval width.
    step=(stop-start)/N_part;
    # Sub-interval boundaries: [line[n], line[n+1]].
    line=np.arange(start,stop+step,step);
    
    # Base nodes/weights on the reference interval (-1, 1).
    x_0,w_0=np.polynomial.legendre.leggauss(N);#sampling points and weight for (-1,1);
    # Scale nodes/weights from (-1,1) to each sub-interval of length `step`.
    w_0=w_0*step/2;
    x_0=x_0*step/2;
    
    x=np.array([]);
    w=np.array([]);
    # Shift reference nodes to each sub-interval center and concatenate.
    for n in range(N_part):
        x=np.append(x,x_0+(line[n]+line[n+1])/2);
        w=np.append(w,w_0);
    
    return x,w

def Gauss_L_quadrs2d(x0,x1,Nx_part,Nx,y0,y1,Ny_part,Ny):
    """
    Build tensor-product 2D Gauss-Legendre quadrature nodes and weights.

    The x and y directions are generated independently with `Gauss_L_quadrs1d`,
    then combined by Cartesian product.
    """
    x,wx=Gauss_L_quadrs1d(x0,x1,Nx_part,Nx)
    y,wy=Gauss_L_quadrs1d(y0,y1,Ny_part,Ny)

    # Flatten Cartesian product of spatial nodes into 1D arrays.
    xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2))
    xarray=np.transpose(xyarray[:,0])
    yarray=np.transpose(xyarray[:,1])
    # Multiply tensor-product weights: w(x_i, y_j) = wx_i * wy_j.
    warray=np.reshape(np.moveaxis(np.meshgrid(wx,wy),0,-1),(-1,2))  
    w=warray[:,0]*warray[:,1]
    return xarray,yarray,w



def Guass_L_quadrs_Circ(a,r_phi,
                        Nr_part,Nr,
                        phi0,phi1,
                        N_phi,
                        Phi_type='uniform'):
    '''Build quadrature nodes/weights over a circular/elliptical aperture.

    #######
    r=rho*[r_0(phi)-a]+a
    phi=phi
    Sum=Sum(F(x,y)|N|)*|r_0(phi)-a|*r*dr*d_phi
    #########

    the integration in radiual direction uses Gaussian L quadrature.
    trapz rule is used in the angular direction integration. 
    **r0=a
    **r1 is a function of phi
    **Nr_part: Rho direction is devided into a few uniform section.
    ** Nr is sampling points in each section
    *** phi0 phi1 is the angular integration range.
    *** N_phi 
    '''
    # Sample radial coordinate in the normalized variable rho in [0, 1].
    rho,w0=Gauss_L_quadrs1d(0,1,Nr_part,Nr)
    if Phi_type=='uniform':
        # Uniform angular sampling with trapezoidal-rule weights in phi.
        phi=np.linspace(phi0,phi1,N_phi)
        phi,rho=np.meshgrid(phi,rho)
        
        w=np.ones((Nr_part*Nr,N_phi))
        w[:,0]=1/2;w[:,-1]=1/2
        w=w*(phi1-phi0)/(N_phi-1)
        w0=np.repeat(w0,N_phi).reshape(-1,N_phi)
        w=(w*w0).ravel()

        phi=phi.ravel()
        # Map rho -> r(phi) and include Jacobian term (r_phi(phi)-a)*rho.
        rho=rho.ravel()*(r_phi(phi)-a)+a
        w=w*(r_phi(phi)-a)*rho

    elif Phi_type=='less':
        # Adaptive angular density: fewer phi samples for smaller rho.
        N_phi_p=np.int_(np.round((max(N_phi,10)-10)*np.sqrt(rho))+10)
        #N_phi_p=np.int_(np.round((max(N_phi,10)-10)*rho*(1/3))+10)
        rho=np.repeat(rho.ravel(),N_phi_p)
        w0=np.repeat(w0.ravel(),N_phi_p)
        phi=np.array([])
        w=np.array([])

        for item in N_phi_p:
            phi=np.append(phi,np.linspace(phi0,phi1,item))
            w_phi=np.ones(item); w_phi[0]=1/2; w_phi[-1]=1/2
            w_phi=w_phi*(phi1-phi0)/(item-1)
            w=np.append(w,w_phi)
        # Same radial mapping/Jacobian as in uniform mode.
        rho=rho*(r_phi(phi)-a)+a
        w=w*w0*(r_phi(phi)-a)*rho

    # Convert polar nodes to Cartesian coordinates.
    return rho*np.cos(phi),rho*np.sin(phi),w
    
