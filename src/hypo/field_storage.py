import numpy as np
import h5py
from .vecops import Vector

class Spherical_grd():
    """Spherical sampling grid for near-field or far-field data.

    The grid stores

    - ``grid``: Cartesian sampling points or directions
    - ``E``: electric-field samples on the grid
    - ``F``: auxiliary field storage used elsewhere in the package

    Supported parameterizations are:

    - ``uv``: direction cosines ``(u, v, w)``
    - ``ELoverAz``: elevation-over-azimuth angular grid in degrees
    """
    def __init__(self,
                 coord_sys,
                 x0,
                 y0,
                 x_size,
                 y_size,
                 Nx,Ny,
                 Type = 'ELoverAz', 
                 far_near = 'far',
                 distance = 0,
                 ):
        # Normalize the type tag once so downstream code can use fixed keys.
        Type = Type.lower()
        self.type = Type
        self.far_near = far_near
        self.coord_sys = coord_sys
        self.E=Vector()
        self.F=Vector()
        self.grid = Vector()
        # x/y are the native sampling coordinates of the chosen grid type.
        self.x = np.linspace(x0-x_size/2,x0+x_size/2,Nx)
        self.y = np.linspace(y0-y_size/2,y0+y_size/2,Ny)
        X,Y = np.meshgrid(self.x,self.y)
        X = X.ravel()
        Y = Y.ravel()
        # Convert the native angular/UV coordinates to Cartesian directions.
        Grid_type={'uv':      lambda x,y: (x,y,np.sqrt(1-(x**2+y**2))),
                   'eloveraz':lambda x,y: (-np.sin(x/180*np.pi)*np.cos(y/180*np.pi),
                                           np.sin(y/180*np.pi),
                                           np.cos(x/180*np.pi)*np.cos(y/180*np.pi))
              }
        self.grid.x, self.grid.y, self.grid.z = Grid_type[Type](X,Y)
        
        if far_near == 'far':
            # In far field the grid stores unit direction vectors.
            pass
        elif far_near =='near':
            # In near field the same directions are converted to points located
            # on a sphere with the requested radius.
            self.grid.x = distance * self.grid.x
            self.grid.y = distance * self.grid.y
            self.grid.z =  distance * self.grid.z
            pass
        else:
            print('Error input!')

def save_grd(S_grd,fname):
    """Save a grid object and its field samples to an HDF5 ``.grd`` file."""
    with h5py.File(fname,'w') as f:
        f.attrs['grid_type'] = S_grd.type
        if S_grd.type == 'uv':
            # Keep both the native coordinates and the derived Cartesian grid.
            f.create_dataset('u',data = S_grd.x)
            f.create_dataset('v',data = S_grd.y)
            f.create_dataset('x',data = S_grd.grid.x)
            f.create_dataset('y',data = S_grd.grid.y)
            f.create_dataset('z',data = S_grd.grid.z)
            f.create_dataset('Ex',data = S_grd.E.x)
            f.create_dataset('Ey',data = S_grd.E.y)
            f.create_dataset('Ez',data = S_grd.E.z)
            f.attrs['far_near'] = S_grd.far_near
            
        elif S_grd.type == 'eloveraz':
            # Elevation/azimuth grids are stored with their angular coordinates.
            f.create_dataset('el',data = S_grd.x)
            f.create_dataset('az',data = S_grd.y)
            f.create_dataset('x',data = S_grd.grid.x)
            f.create_dataset('y',data = S_grd.grid.y)
            f.create_dataset('z',data = S_grd.grid.z)
            f.create_dataset('Ex',data = S_grd.E.x)
            f.create_dataset('Ey',data = S_grd.E.y)
            f.create_dataset('Ez',data = S_grd.E.z)
            f.attrs['far_near'] = S_grd.far_near
        
        elif S_grd.type == 'xy':
            # Plane grids use Cartesian coordinates directly.
            f.create_dataset('x',data = S_grd.grid.x)
            f.create_dataset('y',data = S_grd.grid.y)
            f.create_dataset('z',data = S_grd.grid.z)
            f.create_dataset('Ex',data = S_grd.E.x)
            f.create_dataset('Ey',data = S_grd.E.y)
            f.create_dataset('Ez',data = S_grd.E.z)
            f.attrs['far_near'] = S_grd.far_near
        

def read_grd(fname):
    """Read the stored native coordinates and electric field from a grid file."""
    with h5py.File(fname,'r') as f:
        Type = f.attrs['grid_type']
        if Type == 'uv':
            x = f['u'][:] 
            y = f['v'][:]
        elif Type == 'eloveraz':
            x = f['el'][:]
            y = f['az'][:]
        elif Type == 'xy':
            x = f['x'][:]
            y = f['y'][:]
        Ex = f['Ex'][:]
        Ey = f['Ey'][:]
        Ez = f['Ez'][:]

    return x,y, Ex, Ey, Ez


class plane_grd():
    """Planar near-field sampling grid on a constant-``z`` plane."""
    def __init__(self,
                 coord_sys,
                 x0,
                 y0,
                 x_size,
                 y_size,
                 Nx,Ny,
                 dz=0
                 ):
        self.type = 'xy'
        self.far_near = 'near'
        self.coord_sys = coord_sys
        self.E=Vector()
        self.F=Vector()
        self.grid = Vector()
        # x/y are the plane coordinates sampled on the rectangular aperture.
        self.x = np.linspace(x0-x_size/2,x0+x_size/2,Nx)
        self.y = np.linspace(y0-y_size/2,y0+y_size/2,Ny)
        X,Y = np.meshgrid(self.x,self.y)
        self.grid.x, self.grid.y, self.grid.z = X.ravel(), Y.ravel(), np.ones(X.shape).ravel()*dz
