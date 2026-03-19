
import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

def _to_numpy_array(values, dtype=np.float32):
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    return np.asarray(values, dtype=dtype)



class Vector:
    def __init__(self, values):
        self.v = _to_numpy_array(values, dtype=np.float32).reshape(-1)

    @property
    def x(self):
        return float(self.v[0])

    @property
    def y(self):
        return float(self.v[1])

    @property
    def z(self):
        return float(self.v[2])

    def __iter__(self):
        return iter(self.v.tolist())

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return float(self.v[i])

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self.v, dtype=dtype)
        return arr.copy() if copy else arr

    def __repr__(self):
        return f"Vector({self.v.tolist()})"


class Quaternion:
    # stored as (w, x, y, z), matching Blender/mathutils iteration order
    def __init__(self, values_wxyz):
        self.q = _to_numpy_array(values_wxyz, dtype=np.float32).reshape(4)

    def normalized(self):
        n = np.linalg.norm(self.q)
        if n == 0:
            return Quaternion(self.q)
        return Quaternion(self.q / n)

    def __iter__(self):
        return iter(self.q.tolist())

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return float(self.q[i])

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self.q, dtype=dtype)
        return arr.copy() if copy else arr

    def to_matrix(self):
        q = self.normalized().q
        w, x, y, z = q
        m = _Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float32)
        return Matrix(m)

    def __repr__(self):
        return f"Quaternion({self.q.tolist()})"


class Matrix:
    def __init__(self, array):
        self.m = _to_numpy_array(array, dtype=np.float32)

    @staticmethod
    def Translation(T):
        t = _to_numpy_array(T, dtype=np.float32).reshape(3)
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = t
        return Matrix(m)

    def to_matrix(self):
        return self

    def resize_4x4(self):
        if self.m.shape == (3, 3):
            out = np.eye(4, dtype=np.float32)
            out[:3, :3] = self.m
            self.m = out
        return self

    def copy(self):
        return Matrix(self.m.copy())

    def invert_safe(self):
        self.m = np.linalg.inv(self.m).astype(np.float32)
        return self

    def decompose(self):
        self.resize_4x4()
        t = Vector(self.m[:3, 3])
        rot = _Rotation.from_matrix(self.m[:3, :3])
        x, y, z, w = rot.as_quat()   # scipy gives xyzw
        q = Quaternion((w, x, y, z))
        s = Vector((1.0, 1.0, 1.0))
        return t, q, s

    def __mul__(self, other):
        return Matrix(self.m @ other.m)

    def __iter__(self):
        return iter(self.m.tolist())

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        return self.m[i]

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self.m, dtype=dtype)
        return arr.copy() if copy else arr

    def __repr__(self):
        return f"Matrix(shape={self.m.shape})"


class Euler:
    def __init__(self, angles, order='XYZ'):
        self.angles = tuple(float(a) for a in angles)
        self.order = order.lower()

    def to_matrix(self):
        m = _Rotation.from_euler(self.order, self.angles).as_matrix().astype(np.float32)
        return Matrix(m)

    def __repr__(self):
        return f"Euler(angles={self.angles}, order={self.order})"
