import numpy as _np
import ctypes

from mpi4py import MPI as _MPI

from jax.lib import xla_client

_ops = xla_client.ops


def to_mpi_ptr(mpi_obj):
    """
    to_mpi_ptr(mpi_obj)

    Returns the ptr to the underlying C mpi object
    """
    return _np.uint64(_MPI._handleof(mpi_obj))


def MPIComm_from_ptr(ptr):
    """
    MPIComm_from_ptr(ptr)

    Constructs a MPI Comm object from a pointer
    """
    comm = _MPI.Comm()
    comm_ptr = ctypes.c_void_p.from_address(_MPI._addressof(comm))
    comm_ptr.value = int(ptr)
    return comm


def MPIOp_from_ptr(ptr):
    """
    MPIOp_from_ptr(ptr)

    Constructs a MPI Op object from a pointer
    """
    op = _MPI.Op()
    op_ptr = ctypes.c_void_p.from_address(_MPI._addressof(op))
    op_ptr.value = int(ptr)
    return op


def dtype_ptr(dtype):
    """
    dtype_ptr(dtype)

    Returns the pointer to the MPI dtype of the input numpy dtype
    """
    if dtype == _np.float32:
        _dtype = to_mpi_ptr(_MPI.FLOAT)
    elif dtype == _np.float64:
        _dtype = to_mpi_ptr(_MPI.DOUBLE)
    elif dtype == _np.complex64:
        _dtype = to_mpi_ptr(_MPI.COMPLEX)
    elif dtype == _np.complex128:
        _dtype = to_mpi_ptr(_MPI.DOUBLE_COMPLEX)
    elif dtype == _np.int32:
        _dtype = to_mpi_ptr(_MPI.INT32_T)
    elif dtype == _np.int64:
        _dtype = to_mpi_ptr(_MPI.INT64_T)

    return _dtype


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
