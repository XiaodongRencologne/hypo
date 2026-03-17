## Single HDPE Lens
```python
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../src")
# Geometry objectors
from hypo.lenspy import simple_Lens 
from hypo.surface import ConicSurface
from hypo.coordinate import coord_sys
# source objector
from hypo.Feedpy import GaussiBeam

# output objector
from hypo.field_storage import Spherical_grd,save_grd
from hypo.coxvec import Ludwig_Cox_vector as CO
from hypo.vecops import Vector, dot
```


```python
'''1. Define coordinate system'''
coord_ref = coord_sys()

```


```python
coord_feed = coord_sys(ref_coord = coord_ref)
coord_Lens1 = coord_sys(origin = [0,0,120], ref_coord= coord_ref)
coord_sky = coord_sys(origin = [0,0,.128], ref_coord =coord_ref)
```


```python
'''2. Define lens two surfaces'''
radius_1 = 63 # mm
conic_const_1 = -2.325625
Lens_face1 = ConicSurface(radius_1, conic_const = conic_const_1)

radius_2 = np.inf 
Lens_face2 = ConicSurface(radius_2)
```


```python
x = np.linspace(-25,25,100)
y = np.zeros(x.size)
z2 = Lens_face2.sag(x,y)
z1 = Lens_face1.sag(x,y)
```


```python
'''3. Define Simple Lens'''
# refractive index
HDPE = 1.525
# thickness
t1 = 8 #mm
# diameter
D  = 50# mm

Lens1 = simple_Lens(HDPE,
                    t1,
                    D,
                    Lens_face1,
                    Lens_face2,
                    coord_Lens1,
                    name = 'Lens1',
                    AR_file = None,
                    outputfolder = 'Data/singleLens/')
```


```python
'''4. Source: an idea Gaussian beam'''
Edge_taper  = -20 #dB
Edge_angle = 10 # degree
freq = 90
Feed = GaussiBeam(Edge_taper, Edge_angle, freq, coord_feed,polarization='x')
```


```python
'''5. Define the fields wanted to calculated'''

Beammap = Spherical_grd(coord_ref,
                        0,0,80,80,
                        1001,1001,
                        Type = 'eloveraz',
                        far_near = 'far' )
```


```python
''' Start PO anlaysis'''

Lens1.PO_analysis(Feed,
                  [42,128],
                  [48,77],
                  freq)
Lens1.source(Beammap, freq, far_near = 'far')
save_grd(Beammap, 'Data/singleLens/centerbeam.h5')
```

    None 90GHz
    Batch size: 160
    

    100%|██████████| 15/15 [00:00<00:00, 38.58it/s]
    

    Batch size: 695
    

    100%|██████████| 1441/1441 [00:03<00:00, 437.28it/s]
    


```python

```


```python
r, theta, phi = Beammap.coord_sys.ToSpherical(Beammap.grid.x,Beammap.grid.y, Beammap.grid.z)
co,cx,crho = CO(theta,phi)
print(co)
E_co = dot(Beammap.E , co)
E_cx = dot(Beammap.E , cx)
```

    Vector(numpy, shape=(3, 1002001))
    


```python
plt.pcolor(np.log10(np.abs(Beammap.E.y.reshape(1001,-1)))*10,cmap = 'jet',vmax = 30,vmin = -30)
```




    <matplotlib.collections.PolyCollection at 0x1f35283ed30>




    
![png](HDPE_SimpleLens_files/HDPE_SimpleLens_11_1.png)
    



```python
plt.pcolor(np.log10(np.abs(Beammap.E.x.reshape(1001,-1)))*10,cmap = 'jet',vmax = 30,vmin = -30)
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x1f367a084f0>




    
![png](HDPE_SimpleLens_files/HDPE_SimpleLens_12_1.png)
    



```python

```
