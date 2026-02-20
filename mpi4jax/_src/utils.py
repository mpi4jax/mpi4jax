import functools
import hashlib

from mpi4py import MPI as _MPI

import numpy as _np

from jax.interpreters import xla, mlir
import jaxlib.mlir.ir as ir
from jaxlib.mlir.dialects import mhlo

from .jax_compat import token_type, register_effect, EffectType  # noqa: F401


# Sentinel value for default arguments
NOTSET = object()

# Global variable to hold the default communicator
_default_comm = None


def get_default_comm():
    global _default_comm
    if _default_comm is None:
        from mpi4py import MPI

        _default_comm = MPI.COMM_WORLD.Clone()

    return _default_comm


def raise_if_token_is_set(token):
    """Raises an error if the token is set."""
    if token is not NOTSET:
        raise RuntimeError(
            "Explicit token management is not supported for mpi4jax>=0.8.0. "
            "Tokens are now managed automatically and must not be passed "
            "as arguments to collective operations anymore.\n"
            "That is, please adjust your code like this:\n"
            "     # For mpi4jax<0.8.0:\n"
            "     result, token = mpi4jax.allgather(x, token=token)\n"
            "     # For mpi4jax>=0.8.0:\n"
            "     result = mpi4jax.allgather(x)"
        )


class OrderedMPIEffect(EffectType):
    def __hash__(self):
        # enforce a constant (known) hash
        return int(
            hashlib.md5("I love mpi4jax very much".encode("utf-8")).hexdigest(), 16
        )


ordered_effect = register_effect(OrderedMPIEffect, ordered=True)


def default_primitive_impl(primitive):
    return functools.partial(xla.apply_primitive, primitive)


def as_mhlo_constant_array(arr):
    """Convert a numpy array to an MHLO constant.

    This is useful for passing descriptor buffers to custom calls.
    The array is embedded as a constant in the HLO graph.
    """
    arr = _np.asarray(arr)
    return mhlo.ConstantOp(
        ir.DenseElementsAttr.get(arr, type=mlir.dtype_to_ir_type(arr.dtype))
    ).result


def get_default_layouts(operands, order="c"):
    token = token_type()
    layouts = []

    if order == "c":
        default_layout = lambda t: tuple(range(len(t.shape) - 1, -1, -1))
    elif order == "f":
        default_layout = lambda t: tuple(range(len(t.shape)))
    else:
        raise ValueError(f"Unknown order: {order}")

    for op in operands:
        if isinstance(op, (ir.Value)):
            if op.type == token:
                layouts.append(())
            else:
                tensor_type = ir.RankedTensorType(op.type)
                layouts.append(default_layout(tensor_type))

        elif isinstance(op, ir.RankedTensorType):
            layouts.append(default_layout(op))

        elif op == token:
            layouts.append(())

        else:
            raise ValueError(f"Unknown operand type: {type(op)}")

    return layouts


def to_mpi_handle(mpi_obj):
    """
    Returns the handle of the underlying C mpi object.

    Only defined for some MPI types (such as MPI_Comm), throws NotImplementedError
    otherwise.

    Note: This is not a pointer, but the actual C integer representation of the object.
    MPI implementations differ in handle types:
    - OpenMPI: handles are pointers (64-bit on 64-bit systems)
    - MPICH: handles are signed 32-bit integers

    We pass the handle as a signed 64-bit integer in the FFI interface. The C++ side
    uses memcpy to convert back to the correct MPI handle type.

    For MPICH, the handle is a 32-bit signed integer, but mpi4py's _handleof() returns
    it as a Python int. When the original value is negative (e.g., -2080374784), it may
    be sign-extended to appear as a large unsigned value (e.g., 18446744071629176832).
    We convert this back to a proper signed representation that fits in int64_t.
    """
    handle = _MPI._handleof(mpi_obj)
    # Convert large unsigned values back to signed interpretation
    # This handles MPICH's negative handle values that get sign-extended
    if handle >= (1 << 63):
        # Interpret as signed 64-bit
        handle = handle - (1 << 64)
    int64_info = _np.iinfo(_np.int64)
    assert (
        int64_info.min <= handle <= int64_info.max
    ), f"MPI handle {handle} does not fit in int64_t"
    return handle


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
        raise RuntimeError(f"Unknown MPI type for dtype {dtype_name}")

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

    return xla_bridge.HAS_CUDA_EXT


def has_sycl_support() -> bool:
    """Returns True if mpi4jax is built with SYCL support and can be used with XPU-based
    jax-arrays, False otherwise.
    """
    from . import xla_bridge

    return xla_bridge.HAS_XPU_EXT
