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


MPI_TYPE_MAP = {
    _np.dtype(_np.float32): _MPI.FLOAT,
    _np.dtype(_np.float64): _MPI.DOUBLE,
    _np.dtype(_np.float128): _MPI.LONG_DOUBLE,
    _np.dtype(_np.complex64): _MPI.COMPLEX,
    _np.dtype(_np.complex128): _MPI.DOUBLE_COMPLEX,
    _np.dtype(_np.int8): _MPI.INT8_T,
    _np.dtype(_np.int16): _MPI.INT16_T,
    _np.dtype(_np.int32): _MPI.INT32_T,
    _np.dtype(_np.int64): _MPI.INT64_T,
    _np.dtype(_np.uint8): _MPI.UINT8_T,
    _np.dtype(_np.uint16): _MPI.UINT16_T,
    _np.dtype(_np.uint32): _MPI.UINT32_T,
    _np.dtype(_np.uint64): _MPI.UINT64_T,
    _np.dtype(_np.bool_): _MPI.BOOL,
}


def to_dtype_handle(dtype):
    """
    Returns the pointer to the MPI dtype of the input numpy dtype
    """
    if dtype not in MPI_TYPE_MAP:
        raise RuntimeError("Unknown MPI type for dtype {}".format(type(dtype)))

    return to_mpi_handle(MPI_TYPE_MAP[dtype])


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
