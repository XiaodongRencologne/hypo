#%%
import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt

def EvenAsphere(R,k,coeffi_even):
    if R == 0:
        c = 0
    else:
        c =1/R
    coeff = np.append(np.array([0.0]),np.array(coeffi_even))
    print(coeff)
    print(c)
    def surf1(rho):
        z = c*rho**2/(1+np.sqrt(1-(1+k)*c**2*rho**2))   
        rho2 =rho**2
        z += polyval(rho2,coeff)
        return z
    return surf1
def zemax2RSF(Np,Kspace,Ktip,lens_para,outputfolder='',sign = 1):
    '''
    Rotationally symmetric surface.
    It is one-demensional surface which is a function of radial Rho. 
    rho =sqrt(x^2+y^2)

    Lens_para = {'R': 500,
                 'K': -2.1,
                 'type': 'EvenAsphere',
                 'co': [1,2,3],
                 'D' : 200,
                 'name':'lens_face1'}
    '''
    with open(outputfolder+lens_para['name']+'.rsf','w') as f:
        f.writelines(lens_para['name']+'\n')
        f.writelines(str(Np)+' '+str(Kspace)+' '+str(Ktip)+'\n')
    D = lens_para['r']*2
    if lens_para['type'] == 'EvenAsphere':
        surf_fuc = EvenAsphere(lens_para['R'],lens_para['K'],
                               lens_para['co'])
    if Kspace == 1:
        rho = np.linspace(0,D/2,Np)
        z = sign * surf_fuc(rho)
        data = np.append(rho,z).reshape(2,-1).T
        with open(outputfolder+lens_para['name'] + '.rsf','a') as f:
            #f.writelines(str(rho.min())+' '+str(rho.max()) +'\n')
            np.savetxt(f,data,delimiter=' ')
    return rho, z
