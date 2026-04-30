# %%
import numpy as np
from .Gauss_L_quadr import Gauss_L_quadrs2d, Guass_L_quadrs_Circ
import matplotlib.pyplot as plt

class Elliptical_rim():
    """
    Elliptical aperture/rim definition and sampling utilities.

    Notes
    -----
    - The ellipse is centered at (cx, cy).
    - `a` and `b` are semi-axis lengths along x and y.
    - Returned `w` is area weight per sample for numerical integration.
    """
    def __init__(self,Center,a,b,r_inner=0):
        """Initialize an elliptical rim.

        Parameters
        ----------
        Center : sequence of 2 floats
            Ellipse center ``(cx, cy)`` in the local x-y plane.
        a, b : float
            Semi-axis lengths along x and y.
        r_inner : float, optional
            Inner radial cutoff used by circular Gaussian quadrature. This is
            useful when the sampled region is annular rather than fully filled.
        """
        self.cx=Center[0]
        self.cy=Center[1]
        self.ax=np.abs(a)  # semi-axis along x
        self.by=np.abs(b)  # semi-axis along y
        if self.ax == 0 or self.by == 0:
            raise ValueError("Ellipse sizes a and b must be non-zero.")
        # Backward-compatible aliases used by existing code paths.
        self.a = self.ax
        self.b = self.by

        # Keep eccentricity for compatibility/inspection.
        major = max(self.ax, self.by)
        minor = min(self.ax, self.by)
        self.e=np.sqrt(max(0.0, 1.0 - (minor / major) ** 2))

        self.r_inner=np.abs(r_inner)

    def radial_profile(self,phi):
        """
        Polar boundary radius R(phi) of the ellipse.

        Uses the general axis-aligned ellipse form:
            R(phi) = 1 / sqrt((cos(phi)/ax)^2 + (sin(phi)/by)^2)
        which is stable for both a>=b and a<b.

        Parameters
        ----------
        phi : array_like
            Polar angle in radians measured from the ellipse center.
        """
        c = np.cos(phi)
        s = np.sin(phi)
        denom = np.sqrt((c / self.ax) ** 2 + (s / self.by) ** 2)
        R = 1.0 / denom
        return R

    def radial_profile_2(self, phi, x1, y1):
        """
        Polar boundary radius R(phi) when the polar origin is shifted to (x1, y1).

        Geometry
        --------
        Ellipse (centered at the class origin):
            (x/a)^2 + (y/b)^2 = 1
        Ray from shifted origin:
            x = x1 + R*cos(phi), y = y1 + R*sin(phi)

        Solving for R gives:
            A*R^2 + B*R + C = 0
        where
            A = cos^2(phi)/a^2 + sin^2(phi)/b^2
            B = 2*(x1*cos(phi)/a^2 + y1*sin(phi)/b^2)
            C = x1^2/a^2 + y1^2/b^2 - 1

        Returns
        -------
        R : ndarray
            Positive-distance intersection along the ray direction.
            If no real forward intersection exists for a sample, returns NaN there.
        """
        phi = np.asarray(phi)
        c = np.cos(phi)
        s = np.sin(phi)
        x1 = np.asarray(x1)
        y1 = np.asarray(y1)

        a2 = self.a * self.a
        b2 = self.b * self.b

        A = (c * c) / a2 + (s * s) / b2
        B = 2.0 * (x1 * c / a2 + y1 * s / b2)
        C = (x1 * x1) / a2 + (y1 * y1) / b2 - 1.0

        D = B * B - 4.0 * A * C
        D_clip = np.maximum(D, 0.0)
        sqrtD = np.sqrt(D_clip)

        R1 = (-B + sqrtD) / (2.0 * A)
        R2 = (-B - sqrtD) / (2.0 * A)

        # Prefer the smallest non-negative forward distance.
        inf = np.full(np.shape(R1), np.inf, dtype=np.float64)
        R1_pos = np.where(R1 >= 0.0, R1, inf)
        R2_pos = np.where(R2 >= 0.0, R2, inf)
        R = np.minimum(R1_pos, R2_pos)

        # If discriminant is negative or both roots are behind the origin, return NaN.
        bad = (D < 0.0) | np.isinf(R)
        R = np.where(bad, np.nan, R)
        return R


    def sampling(self,
                 Nx,Ny,
                 quadrature='uniform',
                 Nr_part=1,phi0=0,phi1=2*np.pi,
                 Phi_type='less'):
        """
        Sample points on the elliptical rim.

        Parameters
        ----------
        Nx, Ny : int
            Sampling settings.
            - uniform: grid counts on x/y.
            - gaussian: Nr/Nphi counts expected by `Guass_L_quadrs_Circ`.
        quadrature : {'uniform', 'gaussian'}
            Sampling scheme.

        Returns
        -------
        x, y, w : ndarray
            Sample locations and integration weights.
        """
        if Nx <= 0 or Ny <= 0:
            raise ValueError("Nx and Ny must be positive integers.")

        if quadrature.lower()=='uniform':
            # Uniform Cartesian grid clipped by ellipse mask.
            x,dx=np.linspace(-self.a+self.a/Nx, self.a-self.a/Nx, int(Nx), retstep=True)
            y,dy=np.linspace(-self.b+self.b/Ny, self.b-self.b/Ny, int(Ny), retstep=True)
            xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2))
            x=xyarray[:,0]
            y=xyarray[:,1]
            del(xyarray)
            NN=np.where(((x/self.a)**2+(y/self.b)**2)>1)
            x=np.delete(x,NN)
            y=np.delete(y,NN)
            x=x+self.cx
            y=y+self.cy
            dA=dx*dy
            # Return per-point weights for consistency with other samplers.
            w=np.full(x.shape, dA, dtype=np.float64)
        elif quadrature.lower()=='gaussian':
            # Nx=Nr, Ny=N_phi in this branch.
            x,y,w=Guass_L_quadrs_Circ(self.r_inner,self.radial_profile,
                                        Nr_part,Nx,
                                        phi0,phi1,Ny,
                                        Phi_type=Phi_type)
            x=x+self.cx
            y=y+self.cy
        else:
            raise ValueError("quadrature must be 'uniform' or 'gaussian'")

        return x,y,w

    def sampling2(self,
                  Nx, Ny,
                  quadrature='uniform',
                  Nr_part=1, phi0=0, phi1=2*np.pi,
                  Phi_type='less',
                  x1=0.0, y1=0.0):
        """
        Sample points like `sampling`, but treat (x1, y1) as the sampling center/origin.

        Notes
        -----
        - For `uniform`, a Cartesian grid is built around `(x1, y1)` and then
          clipped by the ellipse centered at the same shifted origin.
        - For `gaussian`, polar rays are launched from `(x1, y1)` while the
          ellipse itself remains centered at `(self.cx, self.cy)`. The
          ray-to-boundary distance is computed by `radial_profile_2`.
        """
        if Nx <= 0 or Ny <= 0:
            raise ValueError("Nx and Ny must be positive integers.")

        x1 = float(x1)
        y1 = float(y1)
        quadrature = quadrature.lower()

        if quadrature == 'uniform':
            # Uniform Cartesian grid centered at (x1, y1), then clipped by ellipse around (x1, y1).
            x, dx = np.linspace(-self.a + self.a / Nx, self.a - self.a / Nx, int(Nx), retstep=True)
            y, dy = np.linspace(-self.b + self.b / Ny, self.b - self.b / Ny, int(Ny), retstep=True)
            xyarray = np.reshape(np.moveaxis(np.meshgrid(x, y), 0, -1), (-1, 2))
            x = xyarray[:, 0] + x1
            y = xyarray[:, 1] + y1
            del(xyarray)
            NN = np.where((((x - x1) / self.a) ** 2 + ((y - y1) / self.b) ** 2) > 1.0)
            x = np.delete(x, NN)
            y = np.delete(y, NN)
            w = np.full(x.shape, dx * dy, dtype=np.float64)
        elif quadrature == 'gaussian':
            # Shift between ellipse center and the new polar origin.
            dx0 = x1 - self.cx
            dy0 = y1 - self.cy

            def _radial_shift(phi):
                return self.radial_profile_2(phi, dx0, dy0)

            x, y, w = Guass_L_quadrs_Circ(
                self.r_inner, _radial_shift,
                Nr_part, Nx,
                phi0, phi1, Ny,
                Phi_type=Phi_type
            )
            # Convert local points to global by adding new origin.
            x = x + x1
            y = y + y1
        else:
            raise ValueError("quadrature must be 'uniform' or 'gaussian'")

        return x, y, w



class Rect_rim():
    """Rectangular aperture/rim definition and sampling utilities."""
    def __init__(self, Center, a, b):
        """
        Parameters
        ----------
        Center : sequence of 2 floats
            Rectangle center (cx, cy).
        a : float
            Rectangle size along x.
        b : float
            Rectangle size along y.
        """
        if len(Center) != 2:
            raise ValueError("Center must contain two values: (cx, cy).")
        self.cx=Center[0]
        self.cy=Center[1]
        self.sizex=np.abs(a)
        self.sizey=np.abs(b)
        if self.sizex == 0 or self.sizey == 0:
            raise ValueError("Rectangle sizes a and b must be non-zero.")

    def sampling(self,Nx,Ny,quadrature='uniform',Nx_part=1,Ny_part=1):
        """
        Sample points on the rectangular rim.

        Parameters
        ----------
        Nx, Ny : int
            Number of points along x and y.
        quadrature : {'uniform', 'gaussian'}
            Sampling scheme. `uniform` uses cell-center points on a Cartesian
            grid; `gaussian` uses tensor-product Gaussian quadrature points.
        Nx_part, Ny_part : int, optional
            Number of x/y sub-intervals passed to `Gauss_L_quadrs2d` in the
            Gaussian branch.

        Returns
        -------
        x, y, w : ndarray
            Flattened sample coordinates and area weights.
        """
        if Nx <= 0 or Ny <= 0:
            raise ValueError("Nx and Ny must be positive integers.")
        if Nx_part <= 0 or Ny_part <= 0:
            raise ValueError("Nx_part and Ny_part must be positive integers.")

        quadrature = quadrature.lower()

        if quadrature=='uniform':
            x,dx=np.linspace(-self.sizex/2+self.sizex/Nx/2,self.sizex/2-self.sizex/Nx/2,int(Nx),retstep=True)
            y,dy=np.linspace(-self.sizey/2+self.sizey/Ny/2,self.sizey/2-self.sizey/Ny/2,int(Ny),retstep=True)
            
            xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2))
            x=xyarray[:,0]+self.cx
            y=xyarray[:,1]+self.cy
            # Per-point weights for consistent integration output.
            w=np.full(x.shape, dx*dy, dtype=np.float64)
            return x,y,w 
        elif quadrature=='gaussian':
            x0=-self.sizex/2+self.cx
            x1=self.sizex/2+self.cx
            y0=-self.sizey/2+self.cy
            y1=self.sizey/2+self.cy
            x,y,w=Gauss_L_quadrs2d(x0,x1,Nx_part,Nx,y0,y1,Ny_part,Ny)
            return x,y,w    
        else:
            raise ValueError("quadrature must be 'uniform' or 'gaussian'")



class Table_rect_rim():
    """Collection of multiple rectangular panels sampled together."""
    def __init__(self, c_list, a_list, b_list):
        """
        Parameters
        ----------
        c_list : (list, list)
            Center coordinates for all panels: (cx_list, cy_list).
        a_list : list
            Panel sizes along x.
        b_list : list
            Panel sizes along y.
        """
        if len(c_list) != 2:
            raise ValueError("c_list must contain (cx_list, cy_list).")
        self.cx=list(c_list[0][:])
        self.cy=list(c_list[1][:])
        self.sizex=list(a_list)
        self.sizey=list(b_list)

        num_panel = len(self.cx)
        if len(self.cy) != num_panel or len(self.sizex) != num_panel or len(self.sizey) != num_panel:
            raise ValueError("Center and size lists must have the same length.")
        if num_panel == 0:
            raise ValueError("At least one panel is required.")
        if np.any(np.asarray(self.sizex, dtype=np.float64) <= 0) or np.any(np.asarray(self.sizey, dtype=np.float64) <= 0):
            raise ValueError("All panel sizes must be positive.")
    
    def sampling(self,Nx,Ny,quadrature='uniform'):
        """
        Sample all rectangular panels and concatenate outputs.

        Notes
        -----
        - If `Nx`/`Ny` are ints, the same counts are used for each panel.
        - Returns concatenated `x`, `y`, `w` arrays across all panels, so panel
          membership is not preserved in the output.
        """
        Num_panel=len(self.cx)
        if isinstance(Nx,int):
            Nx=[Nx]*Num_panel
            Ny=[Ny]*Num_panel
        if len(Nx) != Num_panel or len(Ny) != Num_panel:
            raise ValueError("Nx and Ny must be int or lists with one value per panel.")
        Nx = [int(v) for v in Nx]
        Ny = [int(v) for v in Ny]
        if any(v <= 0 for v in Nx) or any(v <= 0 for v in Ny):
            raise ValueError("All Nx and Ny values must be positive.")

        quadrature = quadrature.lower()
        x_parts = []
        y_parts = []
        w_parts = []

        if quadrature=='uniform':
            for n in range(Num_panel):
                x,dx=np.linspace(-self.sizex[n]/2+self.sizex[n]/Nx[n]/2,self.sizex[n]/2-self.sizex[n]/Nx[n]/2,int(Nx[n]),retstep=True)
                y,dy=np.linspace(-self.sizey[n]/2+self.sizey[n]/Ny[n]/2,self.sizey[n]/2-self.sizey[n]/Ny[n]/2,int(Ny[n]),retstep=True)
                xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2))
                x_parts.append(xyarray[:,0]+self.cx[n])
                y_parts.append(xyarray[:,1]+self.cy[n])
                w_parts.append(np.full((Nx[n]*Ny[n],), dx*dy, dtype=np.float64))
            return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(w_parts)
        elif quadrature=='gaussian':
            for n in range(Num_panel):
                x0=-self.sizex[n]/2+self.cx[n]
                x1=self.sizex[n]/2+self.cx[n]
                y0=-self.sizey[n]/2+self.cy[n]
                y1=self.sizey[n]/2+self.cy[n]
                x,y,w=Gauss_L_quadrs2d(x0,x1,1,Nx[n],y0,y1,1,Ny[n])
                x_parts.append(x)
                y_parts.append(y)
                w_parts.append(w)
            return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(w_parts)
        else:
            raise ValueError("quadrature must be 'uniform' or 'gaussian'")
