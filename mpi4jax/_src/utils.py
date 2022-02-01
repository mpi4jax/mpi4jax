import functools

from mpi4py import MPI as _MPI

import numpy as _np

from jax.interpreters import xla
from jax.lib import xla_client


def default_primitive_impl(primitive):
    return functools.partial(xla.apply_primitive, primitive)


def xla_constant_intc(c, val):
    return xla_client.ops.Constant(c, _np.intc(val))


def xla_constant_uintptr(c, val):
    return xla_client.ops.Constant(c, _np.uintp(val))


def to_mpi_handle(mpi_obj):
    """
    Returns the handle of the underlying C mpi object.

    Only defined for some MPI types (such as MPI_Comm), throws NotImplementedError
    otherwise.

    Note: This is not a pointer, but the actual C integer representation of the object.
    """
    return _np.uintp(_MPI._handleof(mpi_obj))


def to_mpi_ptr(mpi_obj):
    """
    Returns a pointer to the underlying C MPI object
    """
    return _np.uintp(_MPI._addressof(mpi_obj))


# store type names as strings since we cannot expect all type objects to be present on all platforms
MPI_TYPE_MAP = {
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "float128": "LONG_DOUBLE",
    "complex64": "COMPLEX",
    "complex128": "DOUBLE_COMPLEX",
    "int8": "INT8_T",
    "int16": "INT16_T",
    "int32": "INT32_T",
    "int64": "INT64_T",
    "uint8": "UINT8_T",
    "uint16": "UINT16_T",
    "uint32": "UINT32_T",
    "uint64": "UINT64_T",
    "bool": "BOOL",
}


def to_dtype_handle(dtype):
    """
    Returns the pointer to the MPI dtype of the input numpy dtype
    """
    dtype_name = _np.dtype(dtype).name

    if dtype_name not in MPI_TYPE_MAP:
        raise RuntimeError("Unknown MPI type for dtype {}".format(dtype_name))

    mpi_type = getattr(_MPI, MPI_TYPE_MAP[dtype_name])
    return to_mpi_handle(mpi_type)


# Helpers to make MPI objects hashable


class HashableMPIType:
    def __init__(self, obj):
        self.wrapped = obj

    def __hash__(self):
        return int(to_mpi_ptr(self.wrapped))


def wrap_as_hashable(obj):
    if isinstance(obj, HashableMPIType):
        return obj

    return HashableMPIType(obj)


def unpack_hashable(obj):
    if isinstance(obj, HashableMPIType):
        return obj.wrapped

    return obj


# Miscellaneous Utilities


def has_cuda_support() -> bool:
    """Returns True if mpi4jax is built with CUDA support and can be used with GPU-based
    jax-arrays, False otherwise.
    """
    from . import xla_bridge

    return xla_bridge.HAS_GPU_EXT
