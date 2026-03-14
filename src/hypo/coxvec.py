from .vecops import Vector

import numpy as np
#scalarproduct, Vector as vector, cross as crossproduct, norm as abs_v, sumvector
def Ludwig_Cox_vector(theta,phi):
    r0=Vector()
    theta0=Vector()
    PHi0=Vector()
    r0.x=np.sin(theta)*np.cos(phi)
    r0.y=np.sin(theta)*np.sin(phi)
    r0.z=np.cos(theta)
    
    theta0.x=np.cos(theta)*np.cos(phi)
    theta0.y=np.cos(theta)*np.sin(phi)
    theta0.z=-np.sin(theta)
    
    PHi0.x=-np.sin(phi)
    PHi0.y=np.cos(phi)
    PHi0.z=np.zeros(phi.size)
    
    co = theta0 * np.cos(phi) - PHi0 * np.sin(phi)
    cx =  theta0 * np.sin(phi) + PHi0 * np.cos(phi)
    #co=sumvector(scalarproduct(np.cos(phi),theta0),scalarproduct(-np.sin(phi),PHi0))
    #cx=sumvector(scalarproduct(np.sin(phi),theta0),scalarproduct(np.cos(phi),PHi0))
    crho=r0
    
    return co,cx,crho