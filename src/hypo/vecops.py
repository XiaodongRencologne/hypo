"""
Vector Operations Module
========================

Provides a unified Vector class supporting both NumPy and PyTorch backends
for 3D vector operations, with automatic backend switching based on data type.

Classes:
    Vector: 3D vector with dual NumPy/PyTorch backend support

Functions:
    dot: Element-wise dot product
    cross: Cross product
    norm: L2 norm of vector components
    magnitude: Magnitude/intensity of vector (handles complex fields)
    normalized: Return unit vector in same direction
"""

import numpy as np
import torch as T
import copy


class Vector:
    """3D Vector with dual NumPy/PyTorch backend support.
    
    A unified vector class that transparently supports both NumPy arrays and PyTorch tensors
    as underlying storage. The backend is auto-detected based on assigned data and can be
    switched by assigning data of different types.
    
    Attributes:
        x, y, z: Vector components (properties with intelligent setters for backend switching)
    
    Internal:
        _data: Shape (3, N) array/tensor storing [x, y, z] component rows
        _is_tensor: Boolean flag indicating if backend is PyTorch (True) or NumPy (False)
    
    Examples:
        Empty vector (NumPy backend):
            >>> v = Vector()
            
        Create from arrays (NumPy backend):
            >>> v = Vector([1, 2], [3, 4], [5, 6])
            
        Create from tensors (PyTorch backend):
            >>> import torch as T
            >>> v = Vector(T.tensor([1.0]), T.tensor([2.0]), T.tensor([3.0]))
            
        Backend switching (assign torch tensor):
            >>> v = Vector([1, 2], [3, 4], [5, 6])  # NumPy
            >>> v.x = T.tensor([7.0, 8.0])          # Switches to PyTorch
            
        Type preservation (float64 dtype):
            >>> v = Vector(np.array([1.0], dtype=np.float64), ...)
            >>> assert v.x.dtype == np.float64
    """
    # Ensure NumPy defers ufunc dispatch to Vector when mixed with ndarray.
    __array_priority__ = 1000

    def __init__(self, x=None, y=None, z=None):
        """Initialize a Vector with optional x, y, z components.
        
        Args:
            x, y, z: Optional component arrays/tensors. If any provided, all three must be provided.
                     Can be array-like (list, numpy array) or torch.Tensor.
                     Must all have the same length.
            
        Raises:
            ValueError: If only some of x, y, z are provided, or if lengths don't match.
        
        Notes:
            - Empty Vector: Vector() creates (3, 0) array with NumPy backend
            - Backend detection: If any argument is torch.Tensor, uses PyTorch backend
            - dtype promotion: For NumPy backend, uses np.result_type() to find common dtype
        """
        self._data = None
        self.N = None
        self._is_tensor = False
        if x is None and y is None and z is None:
            self._data = np.empty((3, 0))
            self._is_tensor = False
        else:
            if x is None or y is None or z is None:
                raise ValueError('Provide all of x, y, z or none')
            # Determine backend preference: if any argument is torch tensor, use torch
            if any(isinstance(v, T.Tensor) for v in (x, y, z)):
                xa = self._to_1d_tensor(x)
                ya = self._to_1d_tensor(y)
                za = self._to_1d_tensor(z)
                if not (xa.numel() == ya.numel() == za.numel()):
                    raise ValueError('x, y, z must have the same length')
                self._data = T.stack((xa, ya, za), dim=0)
                self._is_tensor = True
            else:
                xa = self._to_1d_array(x)
                ya = self._to_1d_array(y)
                za = self._to_1d_array(z)
                if not (xa.size == ya.size == za.size):
                    raise ValueError('x, y, z must have the same length')
                # Determine common dtype (prefer highest precision)
                common_dtype = np.result_type(xa.dtype, ya.dtype, za.dtype)
                xa = xa.astype(common_dtype)
                ya = ya.astype(common_dtype)
                za = za.astype(common_dtype)
                self._data = np.vstack((xa, ya, za))
                self._is_tensor = False

    # --- Helper Methods ---
    
    @staticmethod
    def _to_1d_array(v):
        """Convert input to 1D NumPy array, preserving dtype.
        
        Args:
            v: Array-like input (list, scalar, or array)
            
        Returns:
            1D NumPy array with original dtype preserved
        """
        a = np.asarray(v)  # preserve dtype from input
        if a.ndim == 0:
            return a.reshape(1)
        return a.ravel()

    @staticmethod
    def _to_1d_tensor(v):
        """Convert input to 1D PyTorch tensor.
        
        Args:
            v: Tensor or array-like input
            
        Returns:
            1D PyTorch tensor with appropriate dtype and device
        """
        if isinstance(v, T.Tensor):
            t = v
        else:
            arr = np.asarray(v)
            # Prefer highest practical precision for numeric arrays by default:
            # float -> float64, complex -> complex128.
            if arr.dtype in [np.float64, np.complex128, np.float32, np.complex64]:
                t = T.from_numpy(arr.copy())
            else:
                t = T.tensor(arr)

        # Promote lower-precision floating dtypes to high precision.
        if t.dtype == T.float32:
            t = t.to(dtype=T.float64)
        elif t.dtype == T.complex64:
            t = t.to(dtype=T.complex128)
        elif t.dtype in (T.int8, T.int16, T.int32, T.int64, T.uint8, T.bool):
            # Numeric integer/bool inputs participate in field math as float64.
            t = t.to(dtype=T.float64)
        if t.ndim == 1:
            return t
        return t.view(-1)

    @staticmethod
    def _ensure_3xN_shape(data, where="Vector"):
        """Validate internal storage shape contract: (3, N)."""
        if isinstance(data, T.Tensor):
            shape = tuple(data.shape)
            if data.ndim != 2 or shape[0] != 3:
                raise ValueError(f"{where}: expected shape (3, N), got {shape}")
        else:
            arr = np.asarray(data)
            shape = arr.shape
            if arr.ndim != 2 or shape[0] != 3:
                raise ValueError(f"{where}: expected shape (3, N), got {shape}")

    def _copy_metadata_to(self, target):
        """Copy user-defined attributes (for example `N`) to a new Vector result."""
        for k, v in self.__dict__.items():
            if k in ("_data", "_is_tensor"):
                continue
            try:
                setattr(target, k, copy.deepcopy(v))
            except Exception:
                setattr(target, k, v)

    # --- Component Properties ---
    @property
    def x(self):
        """Get x-component of vector.
        
        Returns:
            1D array/tensor containing x values for each position
        """
        if self._is_tensor:
            return self._data[0] if self._data.numel() else T.tensor([])
        else:
            return self._data[0] if self._data.size else np.array([])

    @x.setter
    def x(self, v):
        """Set x-component, with automatic backend switching.
        
        If v is torch.Tensor and current backend is NumPy, automatically switches to PyTorch.
        If v is NumPy array and current backend is PyTorch, automatically switches to NumPy.
        
        Args:
            v: 1D array-like or torch.Tensor with values for new x-component
        
        Notes:
            - Dtype is preserved from input when switching backends
            - If size mismatch, reshapes internal storage to match new component size
        """
        # If incoming is torch tensor, switch backend to torch
        if isinstance(v, T.Tensor):
            arr = self._to_1d_tensor(v)
            if not self._is_tensor:
                # convert existing numpy storage to torch
                if self._data.size == 0:
                    self._data = T.empty((3, arr.numel()), dtype=arr.dtype, device=arr.device)
                else:
                    try:
                        self._data = T.from_numpy(self._data)
                    except Exception:
                        self._data = T.tensor(self._data, dtype=arr.dtype, device=arr.device)
                self._is_tensor = True
            # now _is_tensor True
            if self._data.numel() == 0:
                self._data = T.zeros((3, arr.numel()), dtype=arr.dtype, device=arr.device)
                self._data[0].copy_(arr)
                return
            if arr.numel() == self._data.shape[1]:
                try:
                    self._data[0].copy_(arr)
                    return
                except Exception:
                    pass
            # resize preserving rows
            new = T.zeros((3, arr.numel()), dtype=arr.dtype, device=arr.device)
            m = min(self._data.shape[1], arr.numel())
            if m > 0:
                new[1, :m] = self._data[1, :m]
                new[2, :m] = self._data[2, :m]
            new[0, :] = arr
            self._data = new
        else:
            # incoming is numpy-like
            arr = self._to_1d_array(v)
            if self._is_tensor:
                # convert torch storage to numpy
                self._data = self._data.cpu().numpy()
                self._is_tensor = False
            if self._data.size == 0:
                self._data = np.zeros((3, arr.size), dtype=arr.dtype)
                self._data[0, :] = arr
                return
            if arr.size == self._data.shape[1]:
                self._data[0, :] = arr
                return
            new = np.zeros((3, arr.size), dtype=arr.dtype)
            m = min(self._data.shape[1], arr.size)
            if m > 0:
                new[1, :m] = self._data[1, :m]
                new[2, :m] = self._data[2, :m]
            new[0, :] = arr
            self._data = new

    @property
    def y(self):
        """Get y-component of vector.
        
        Returns:
            1D array/tensor containing y values for each position
        """
        if self._is_tensor:
            return self._data[1] if self._data.numel() else T.tensor([])
        else:
            return self._data[1] if self._data.size else np.array([])

    @y.setter
    def y(self, v):
        """Set y-component, with automatic backend switching.
        
        If v is torch.Tensor and current backend is NumPy, automatically switches to PyTorch.
        If v is NumPy array and current backend is PyTorch, automatically switches to NumPy.
        
        Args:
            v: 1D array-like or torch.Tensor with values for new y-component
        
        Notes:
            - Dtype is preserved from input when switching backends
            - If size mismatch, reshapes internal storage to match new component size
        """
        if isinstance(v, T.Tensor):
            arr = self._to_1d_tensor(v)
            if not self._is_tensor:
                if self._data.size == 0:
                    self._data = T.empty((3, arr.numel()), dtype=arr.dtype, device=arr.device)
                else:
                    try:
                        self._data = T.from_numpy(self._data)
                    except Exception:
                        self._data = T.tensor(self._data, dtype=arr.dtype, device=arr.device)
                self._is_tensor = True
            if self._data.numel() == 0:
                self._data = T.zeros((3, arr.numel()), dtype=arr.dtype, device=arr.device)
                self._data[1].copy_(arr)
                return
            if arr.numel() == self._data.shape[1]:
                try:
                    self._data[1].copy_(arr)
                    return
                except Exception:
                    pass
            new = T.zeros((3, arr.numel()), dtype=arr.dtype, device=arr.device)
            m = min(self._data.shape[1], arr.numel())
            if m > 0:
                new[0, :m] = self._data[0, :m]
                new[2, :m] = self._data[2, :m]
            new[1, :] = arr
            self._data = new
        else:
            arr = self._to_1d_array(v)
            if self._is_tensor:
                self._data = self._data.cpu().numpy()
                self._is_tensor = False
            if self._data.size == 0:
                self._data = np.zeros((3, arr.size), dtype=arr.dtype)
                self._data[1, :] = arr
                return
            if arr.size == self._data.shape[1]:
                self._data[1, :] = arr
                return
            new = np.zeros((3, arr.size), dtype=arr.dtype)
            m = min(self._data.shape[1], arr.size)
            if m > 0:
                new[0, :m] = self._data[0, :m]
                new[2, :m] = self._data[2, :m]
            new[1, :] = arr
            self._data = new

    @property
    def z(self):
        """Get z-component of vector.
        
        Returns:
            1D array/tensor containing z values for each position
        """
        if self._is_tensor:
            return self._data[2] if self._data.numel() else T.tensor([])
        else:
            return self._data[2] if self._data.size else np.array([])

    @z.setter
    def z(self, v):
        """Set z-component, with automatic backend switching.
        
        If v is torch.Tensor and current backend is NumPy, automatically switches to PyTorch.
        If v is NumPy array and current backend is PyTorch, automatically switches to NumPy.
        
        Args:
            v: 1D array-like or torch.Tensor with values for new z-component
        
        Notes:
            - Dtype is preserved from input when switching backends
            - If size mismatch, reshapes internal storage to match new component size
        """
        if isinstance(v, T.Tensor):
            arr = self._to_1d_tensor(v)
            if not self._is_tensor:
                if self._data.size == 0:
                    self._data = T.empty((3, arr.numel()), dtype=arr.dtype, device=arr.device)
                else:
                    try:
                        self._data = T.from_numpy(self._data)
                    except Exception:
                        self._data = T.tensor(self._data, dtype=arr.dtype, device=arr.device)
                self._is_tensor = True
            if self._data.numel() == 0:
                self._data = T.zeros((3, arr.numel()), dtype=arr.dtype, device=arr.device)
                self._data[2].copy_(arr)
                return
            if arr.numel() == self._data.shape[1]:
                try:
                    self._data[2].copy_(arr)
                    return
                except Exception:
                    pass
            new = T.zeros((3, arr.numel()), dtype=arr.dtype, device=arr.device)
            m = min(self._data.shape[1], arr.numel())
            if m > 0:
                new[0, :m] = self._data[0, :m]
                new[1, :m] = self._data[1, :m]
            new[2, :] = arr
            self._data = new
        else:
            arr = self._to_1d_array(v)
            if self._is_tensor:
                self._data = self._data.cpu().numpy()
                self._is_tensor = False
            if self._data.size == 0:
                self._data = np.zeros((3, arr.size), dtype=arr.dtype)
                self._data[2, :] = arr
                return
            if arr.size == self._data.shape[1]:
                self._data[2, :] = arr
                return
            new = np.zeros((3, arr.size), dtype=arr.dtype)
            m = min(self._data.shape[1], arr.size)
            if m > 0:
                new[0, :m] = self._data[0, :m]
                new[1, :m] = self._data[1, :m]
            new[2, :] = arr
            self._data = new

    # --- Backend Conversion Methods ---
    
    def to_numpy(self, dtype=None):
        """Convert internal storage to NumPy in-place, optionally changing dtype.
        
        Switches the internal backend from PyTorch to NumPy. If already NumPy,
        does not affect storage. Useful for memory efficiency or before saving to disk.
        
        Args:
            dtype: Target NumPy dtype (e.g., np.float64, np.complex128).
                   If None, preserves current dtype during conversion.
        
        Returns:
            self (for method chaining)
        
        Examples:
            >>> v = Vector(torch.tensor([1.0]), ...)
            >>> v.to_numpy()  # Now uses NumPy backend
            >>> v.to_numpy(dtype=np.complex128)  # Convert to complex dtype
        
        Notes:
            - Moves data to CPU before converting if on GPU
            - Returns self for chaining: v.to_numpy(np.float64).to_torch()
        """
        if self._is_tensor:
            self._data = self._data.cpu().numpy()
            self._is_tensor = False
        if dtype is not None:
            self._data = self._data.astype(dtype)
        return self

    def to_torch(self, device=None, dtype=None):
        """Convert internal storage to PyTorch in-place, optionally changing dtype/device.
        
        Switches the internal backend from NumPy to PyTorch. If already PyTorch,
        updates device/dtype as specified. Useful for GPU acceleration or deep learning pipelines.
        
        Args:
            device: Target torch device string or object (e.g., 'cpu', 'cuda:0').
                    If None, keeps current device (or CPU if converting from NumPy).
            dtype: Target torch dtype (e.g., torch.float64, torch.complex128).
                   If None, preserves current dtype during conversion.
        
        Returns:
            self (for method chaining)
        
        Examples:
            >>> v = Vector(np.array([1.0]), ...)
            >>> v.to_torch()  # Now uses PyTorch backend on CPU
            >>> v.to_torch(device='cuda', dtype=torch.complex128)  # Move to GPU with dtype change
            >>> v.to_torch(device='cpu')  # Move back to CPU
        
        Notes:
            - Supports GPU acceleration when device='cuda'
            - Returns self for chaining: v.to_torch('cuda').to_numpy()
        """
        if not self._is_tensor:
            self._data = T.tensor(self._data)
            self._is_tensor = True
        if dtype is not None:
            self._data = self._data.to(dtype=dtype)
        if device is not None:
            self._data = self._data.to(device=device)
        return self
    def to_coordsys(self,matrix=None):
        """Transform vector components to a new coordinate system using a transformation matrix.
        
        Applies the transformation: [x', y', z'] = matrix @ [x, y, z] for each position.
        If matrix is None, leaves components unchanged.
        
        Args:
            matrix: 3x3 array/tensor representing the coordinate transformation.
                    Should be compatible with current backend (NumPy or PyTorch).
                    If None, no transformation is applied.
        """
        if matrix is None:
            return
        mshape = tuple(matrix.shape) if hasattr(matrix, "shape") else np.asarray(matrix).shape
        if len(mshape) != 2 or mshape[0] != 3 or mshape[1] != 3:
            raise ValueError(f"to_coordsys: expected matrix shape (3, 3), got {mshape}")
        if self._is_tensor:
            data = T.matmul(matrix, self._data)
            self._ensure_3xN_shape(data, where="to_coordsys")
            self._data = data
        else:
            data = np.matmul(matrix, self._data)
            self._ensure_3xN_shape(data, where="to_coordsys")
            self._data = data

    def tocoordsys(self, matrix=None):
        """In-place coordinate transform using matrix multiplication.

        This method follows legacy usage style and updates `self.x/self.y/self.z` directly.
        """
        if matrix is None:
            return
        if self._is_tensor:
            if not isinstance(matrix, T.Tensor):
                matrix = T.as_tensor(matrix, dtype=self._data.dtype, device=self._data.device)
            data = T.matmul(matrix, self._data)
            self._ensure_3xN_shape(data, where="tocoordsys")
            self._data = data
        else:
            data = np.matmul(matrix, self._data)
            self._ensure_3xN_shape(data, where="tocoordsys")
            self._data = data

    def as_array(self):
        """Return internal data array/tensor directly.
        
        Returns:
            Shape (3, N) NumPy array or PyTorch tensor depending on current backend
        
        Notes:
            - Direct access to internal storage (not a copy)
            - Use cautiously when modifying; prefer component properties (x, y, z)
        """
        return self._data

    def is_empty(self):
        """Check if vector has any components.
        
        Returns:
            True if vector has no elements (shape (3, 0))
        """
        if self._is_tensor:
            return self._data.numel() == 0
        return self._data.size == 0

    def __repr__(self):
        """Return string representation of vector.
        
        Shows backend type (numpy/torch) and shape for debugging
        """
        if self._is_tensor:
            return f"Vector(torch, shape={tuple(self._data.shape)})"
        return f"Vector(numpy, shape={self._data.shape})"

    # --- Operator Overloading ---
    
    def __add__(self, other):
        """Add two vectors component-wise: v1 + v2.
        
        Args:
            other: Another Vector object
            
        Returns:
            New Vector with sum components, same backend as operands
            
        Raises:
            TypeError: If other is not Vector, or backends don't match
            
        Examples:
            >>> v1 = Vector([1, 2], [3, 4], [5, 6])
            >>> v2 = Vector([1, 0], [0, 1], [0, 0])
            >>> v3 = v1 + v2  # [2, 2], [3, 5], [5, 6]
        """
        if not isinstance(other, Vector):
            raise TypeError('Can only add Vector to Vector')
        if self._is_tensor != other._is_tensor:
            raise TypeError('Both vectors must use the same backend')
        result = Vector()
        result._data = self._data + other._data
        result._is_tensor = self._is_tensor
        self._copy_metadata_to(result)
        return result

    def __sub__(self, other):
        """Subtract two vectors component-wise: v1 - v2.
        
        Args:
            other: Another Vector object
            
        Returns:
            New Vector with difference components, same backend as operands
            
        Raises:
            TypeError: If other is not Vector, or backends don't match
            
        Examples:
            >>> v1 = Vector([3, 4], [5, 6], [7, 8])
            >>> v2 = Vector([1, 0], [0, 1], [0, 0])
            >>> v3 = v1 - v2  # [2, 4], [5, 5], [7, 8]
        """
        if not isinstance(other, Vector):
            raise TypeError('Can only subtract Vector from Vector')
        if self._is_tensor != other._is_tensor:
            raise TypeError('Both vectors must use the same backend')
        result = Vector()
        result._data = self._data - other._data
        result._is_tensor = self._is_tensor
        self._copy_metadata_to(result)
        return result

    def __mul__(self, scalar):
        """Multiply vector by scalar: v * k.
        
        Args:
            scalar: Numeric scalar value or array-like to multiply all components
            
        Returns:
            New Vector with scaled components, same backend as original
            
        Examples:
            >>> v = Vector([1, 2], [3, 4], [5, 6])
            >>> v2 = v * 2  # [2, 4], [6, 8], [10, 12]
            >>> v3 = v * 0.5
        """
        result = Vector()
        data = self._data * scalar
        self._ensure_3xN_shape(data, where="__mul__")
        result._data = data
        result._is_tensor = self._is_tensor
        self._copy_metadata_to(result)
        return result

    def __rmul__(self, scalar):
        """Multiply vector by scalar (right associative): k * v.
        
        Args:
            scalar: Numeric scalar value to multiply all components
            
        Returns:
            New Vector with scaled components, same backend as original
            
        Examples:
            >>> v = Vector([1, 2], [3, 4], [5, 6])
            >>> v2 = 2 * v  # Same as v * 2
        """
        return self.__mul__(scalar)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        NumPy ufunc bridge so expressions like `np_array * Vector` return Vector.

        Supported ufuncs: multiply/add/subtract/divide with one Vector operand.
        """
        if method != "__call__":
            return NotImplemented
        if kwargs.get("out") is not None:
            return NotImplemented

        supported = {np.multiply, np.add, np.subtract, np.true_divide, np.divide}
        if ufunc not in supported:
            return NotImplemented

        vec_inputs = [x for x in inputs if isinstance(x, Vector)]
        if len(vec_inputs) != 1:
            return NotImplemented

        vec = vec_inputs[0]
        lhs_is_vec = inputs[0] is vec
        other = inputs[1] if lhs_is_vec else inputs[0]

        result = Vector()

        if vec._is_tensor:
            if isinstance(other, np.ndarray):
                other_val = T.as_tensor(other, dtype=vec._data.dtype, device=vec._data.device)
            else:
                other_val = other

            if ufunc is np.multiply:
                data = vec._data * other_val if lhs_is_vec else other_val * vec._data
            elif ufunc is np.add:
                data = vec._data + other_val if lhs_is_vec else other_val + vec._data
            elif ufunc is np.subtract:
                data = vec._data - other_val if lhs_is_vec else other_val - vec._data
            else:  # divide/true_divide
                data = vec._data / other_val if lhs_is_vec else other_val / vec._data

            self._ensure_3xN_shape(data, where="__array_ufunc__")
            result._data = data
            result._is_tensor = True
            vec._copy_metadata_to(result)
            return result

        other_np = np.asarray(other)
        if ufunc is np.multiply:
            data = vec._data * other_np if lhs_is_vec else other_np * vec._data
        elif ufunc is np.add:
            data = vec._data + other_np if lhs_is_vec else other_np + vec._data
        elif ufunc is np.subtract:
            data = vec._data - other_np if lhs_is_vec else other_np - vec._data
        else:  # divide/true_divide
            data = vec._data / other_np if lhs_is_vec else other_np / vec._data

        self._ensure_3xN_shape(data, where="__array_ufunc__")
        result._data = data
        result._is_tensor = False
        vec._copy_metadata_to(result)
        return result
    
    def conj(self):
        """Return complex conjugate of vector components (backend-preserving)."""
        result = Vector()
        if self._is_tensor:
            result._data = T.conj(self._data)
            result._is_tensor = True
        else:
            result._data = np.conjugate(self._data)
            result._is_tensor = False
        self._copy_metadata_to(result)
        return result
    
    def real(self):
        """Return real part of vector components (backend-preserving)."""
        result = Vector()
        if self._is_tensor:
            result._data = T.real(self._data)
            result._is_tensor = True
        else:
            result._data = np.real(self._data)
            result._is_tensor = False
        self._copy_metadata_to(result)
        return result
    
    def imag(self):
        """Return imaginary part of vector components (backend-preserving)."""
        result = Vector()
        if self._is_tensor:
            result._data = T.imag(self._data)
            result._is_tensor = True
        else:
            result._data = np.imag(self._data)
            result._is_tensor = False
        self._copy_metadata_to(result)
        return result


# --- Vector Operations Functions ---

def dot(v1, v2):
    """Compute element-wise dot product of two vectors.
    
    For each position i, computes: v1.x[i]*v2.x[i] + v1.y[i]*v2.y[i] + v1.z[i]*v2.z[i]
    
    Useful for scalar field calculations and magnitude computations. Computes the dot product
    between vector pairs at corresponding positions when working with multiple vectors.
    
    Args:
        v1, v2: Vector objects with matching component sizes
    
    Returns:
        Array/Tensor with shape (N,) containing element-wise dot products
        
    Raises:
        TypeError: If backends don't match (both must be NumPy or both PyTorch)
    
    Examples:
        >>> v1 = Vector([1, 2], [3, 4], [5, 6])
        >>> v2 = Vector([1, 0], [0, 1], [0, 0])
        >>> d = dot(v1, v2)  # [1, 4] (element-wise dot products)
    
    Notes:
        - For single vectors (N=1), returns shape (1,) array
        - Result is always real for real input, may be complex for complex input
    """
    if v1._is_tensor != v2._is_tensor:
        raise TypeError('Both vectors must use the same backend (numpy or torch)')
    
    # v1._data and v2._data have shape (3, N)
    # Sum element-wise products along axis/dim 0 (the 3 components)
    # Result shape: (N,)
    
    if v1._is_tensor:
        return T.sum(v1._data * v2._data, dim=0)  # sum x*x + y*y + z*z for each position
    else:
        return np.sum(v1._data * v2._data, axis=0)  # sum along component axis


def cross(v1, v2):
    """Compute cross product of two vectors.
    
    Computes the 3D cross product: v1 × v2 = (v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x)
    
    Useful for normal vector calculations, magnetic field calculations, and rotational operations.
    Works element-wise when vectors contain multiple points (N > 1).
    
    Args:
        v1, v2: Vector objects with matching component sizes
    
    Returns:
        New Vector containing the cross product result, same backend as inputs
        
    Raises:
        TypeError: If backends don't match (both must be NumPy or both PyTorch)
    
    Examples:
        >>> v1 = Vector([1, 0], [0, 1], [0, 0])  # x-axis and y-axis
        >>> v2 = Vector([0, 1], [1, 0], [0, 0])  # y-axis and x-axis
        >>> v3 = cross(v1, v2)  # z-axis
    
    Notes:
        - Result shape is same as inputs: (3, N)
        - For complex components, cross product is computed directly on complex values
    """
    if v1._is_tensor != v2._is_tensor:
        raise TypeError('Both vectors must use the same backend (numpy or torch)')
    
    if v1._is_tensor:
        # PyTorch: use torch.cross (requires shape (N, 3) for dim=-1)
        # Reshape to (N, 3) for cross product
        a = v1._data.T  # shape: (N, 3) or (1, 3)
        b = v2._data.T
        c = T.cross(a, b, dim=-1)  # result shape: (N, 3)
        result = Vector()
        result._data = c.T
        result._is_tensor = True
        return result
    else:
        # NumPy: use np.cross on transposed (each row is a 3D vector)
        a = v1._data.T  # shape: (N, 3)
        b = v2._data.T
        c = np.cross(a, b)  # result shape: (N, 3)
        result = Vector()
        result._data = c.T
        result._is_tensor = False
        return result


def norm(v, axis=0):
    """Compute L2 norm (Euclidean norm) of vector components.
    
    For each position, computes: sqrt(v.x^2 + v.y^2 + v.z^2)
    
    Useful for vector magnitude calculations, normalization, and distance metrics.
    The axis parameter allows flexible reduction across dimensions.
    
    Args:
        v: Vector object
        axis: Axis along which to compute norm (default: 0 for per-vector norm)
              - axis=0: Computes norm of [x, y, z] components (returns shape (N,))
              - axis=None: Computes total norm across all dimensions (returns scalar)
    
    Returns:
        Norm values as array/scalar depending on axis parameter
        
    Examples:
        >>> v = Vector([3, 0], [4, 1], [0, 0])
        >>> n = norm(v)  # [5, 1] (magnitudes at each position)
        >>> n_total = norm(v, axis=None)  # scalar norm of entire array
    
    Notes:
        - For complex vectors, computes norm of complex values
        - Always returns real (non-complex) result
    """
    if v._is_tensor:
        return T.norm(v._data, dim=axis)
    else:
        return np.linalg.norm(v._data, axis=axis)


def magnitude(v):
    """Compute magnitude (intensity) of vector, supporting complex fields.
    
    For real components: sqrt(v.x^2 + v.y^2 + v.z^2) for each position
    For complex components: sqrt(|v.x|^2 + |v.y|^2 + |v.z|^2) for each position (field intensity)
    
    Critical for electromagnetic field calculations where intensity (power) is the quantity
    of physical interest. Handles both real and complex field components seamlessly.
    
    Args:
        v: Vector object (can contain real or complex components)
    
    Returns:
        Array/scalar with magnitude (always real-valued) at each position
        
    Examples:
        >>> v_real = Vector([3, 0], [4, 1], [0, 0])
        >>> m = magnitude(v_real)  # [5.0, 1.0]
        
        >>> import numpy as np
        >>> v_complex = Vector([1+1j, 0], [0, 1+1j], [0, 0])
        >>> m = magnitude(v_complex)  # [sqrt(2), sqrt(2)]
    
    Notes:
        - For complex input, uses absolute value before squaring (intensity calculation)
        - Result is always real-valued (float or float64)
        - Equivalent to norm(v) for real vectors
        - This is the physically relevant quantity for field intensity in optics
    """
    if v._is_tensor:
        # PyTorch: for complex tensors use abs(), for real use direct squaring
        abs_squared = T.abs(v._data) ** 2  # element-wise abs squared
        return T.sqrt(T.sum(abs_squared, dim=0))  # sum (x,y,z) components -> magnitude
    else:
        # NumPy: np.abs handles both real and complex
        abs_squared = np.abs(v._data) ** 2  # element-wise abs squared
        return np.sqrt(np.sum(abs_squared, axis=0))  # sum (x,y,z) components -> magnitude


def normalized(v):
    """Return a normalized (unit) vector in the same direction.
    
    Creates a new vector with the same direction as input but with magnitude 1.
    Useful for working with direction vectors, normal vectors, and orientations.
    
    Args:
        v: Vector object (must be non-empty)
    
    Returns:
        New Vector with unit magnitude, same backend as input
        
    Raises:
        (implicitly) Zero-division if vector has magnitude 0
    
    Examples:
        >>> v = Vector([3, 0], [4, 1], [0, 0])
        >>> u = normalized(v)  # Returns unit vectors [0.6, 0], [0.8, 1], [0, 0]
        >>> magnitude(u)  # [1.0, 1.0]
    
    Notes:
        - Uses magnitude() internally, so handles both real and complex vectors
        - Returns new Vector object (input unchanged)
        - Preserves backend (NumPy or PyTorch)
        - For zero vectors, result will contain NaN values (magnitude returns 0)
    """
    mag = magnitude(v)
    result = Vector()
    result._data = v._data / mag
    result._is_tensor = v._is_tensor
    return result


class FaceVector(Vector):
    """Surface sample vector with integration weights `w`."""

    def __init__(self, x=None, y=None, z=None, w=None):
        super().__init__(x, y, z)
        self.w = None if w is None else self._to_1d_tensor(w) if self._is_tensor else self._to_1d_array(w)


class FaceNormalVector(Vector):
    """Surface normal vector with optional magnitude/normalization cache `N`."""

    def __init__(self, x=None, y=None, z=None, N=None):
        super().__init__(x, y, z)
        self.N = None if N is None else self._to_1d_tensor(N) if self._is_tensor else self._to_1d_array(N)


