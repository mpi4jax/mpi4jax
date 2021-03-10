import functools

import numpy as _np
from jax.interpreters import xla
from jax.lib import xla_client
from mpi4py import MPI as _MPI

_ops = xla_client.ops


def default_primitive_impl(primitive):
    return functools.partial(xla.apply_primitive, primitive)


def to_mpi_ptr(mpi_obj):
    """
    to_mpi_ptr(mpi_obj)

    Returns the ptr to the underlying C mpi object
    """
    try:
        addr = _MPI._handleof(mpi_obj)
    except NotImplementedError:
        # some objects like Status only work with addressof
        addr = _MPI._addressof(mpi_obj)

    return _np.uint64(addr)


def dtype_ptr(dtype):
    """
    dtype_ptr(dtype)

    Returns the pointer to the MPI dtype of the input numpy dtype
    """
    if dtype == _np.float32:
        _dtype = to_mpi_ptr(_MPI.FLOAT)
    elif dtype == _np.float64:
        _dtype = to_mpi_ptr(_MPI.DOUBLE)
    elif dtype == _np.float128:
        _dtype = to_mpi_ptr(_MPI.LONG_DOUBLE)
    elif dtype == _np.complex64:
        _dtype = to_mpi_ptr(_MPI.COMPLEX)
    elif dtype == _np.complex128:
        _dtype = to_mpi_ptr(_MPI.DOUBLE_COMPLEX)
    elif dtype == _np.int8:
        _dtype = to_mpi_ptr(_MPI.INT8_T)
    elif dtype == _np.int16:
        _dtype = to_mpi_ptr(_MPI.INT16_T)
    elif dtype == _np.int32:
        _dtype = to_mpi_ptr(_MPI.INT32_T)
    elif dtype == _np.int64:
        _dtype = to_mpi_ptr(_MPI.INT64_T)
    elif dtype == _np.uint8:
        _dtype = to_mpi_ptr(_MPI.UINT8_T)
    elif dtype == _np.uint16:
        _dtype = to_mpi_ptr(_MPI.UINT16_T)
    elif dtype == _np.uint32:
        _dtype = to_mpi_ptr(_MPI.UINT32_T)
    elif dtype == _np.uint64:
        _dtype = to_mpi_ptr(_MPI.UINT64_T)
    elif dtype == _np.bool:
        _dtype = to_mpi_ptr(_MPI.BOOL)
    else:
        raise RuntimeError("Unknown MPI type for numpy type {}".format(dtype))

    return _dtype


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


# Helper functions


def _constant_s32_scalar(c, x):
    return _ops.Constant(c, _np.int32(x))


def _constant_s64_scalar(c, x):
    return _ops.Constant(c, _np.int64(x))


def _constant_u32_scalar(c, x):
    return _ops.Constant(c, _np.uint32(x))


def _constant_u64_scalar(c, x):
    return _ops.Constant(c, _np.uint64(x))


def _unpack_builder(c):
    # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
    return getattr(c, "_builder", c)
