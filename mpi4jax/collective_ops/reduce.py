import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive
from jax.interpreters import xla
from jax.lib import xla_client

from jax.lax import create_token
from ..utils import (
    HashableMPIType,
    _constant_s32_scalar,
    _constant_u64_scalar,
    _ops,
    _unpack_builder,
    default_primitive_impl,
    dtype_ptr,
    to_mpi_ptr,
    unpack_hashable,
    wrap_as_hashable,
)
from ..validation import enforce_types
from ..warn import warn_missing_omnistaging

# The Jax primitive
mpi_reduce_p = Primitive("reduce_mpi")  # Create the primitive
mpi_reduce_impl = default_primitive_impl(mpi_reduce_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    root=(_np.integer),
    comm=(_MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def Reduce(x, op, root, comm=_MPI.COMM_WORLD, token=None):
    """
    Performs the reduce operation `op` on the input `x` using the
    communicator `comm` which defaults to the world comunicator.
    An optional token can be passed, which is used to force jax to execute
    MPI operations in the correct order.

    Arguments:
        x: Array or scalar input.
        op: The reduction operation `MPI.Op` (e.g: MPI.SUM)
        comm: The communicator (defaults to MPI.COMM_WORLD)
        token: token to force a sequential order in the operations (default=None)

    Returns:
        res: result of the reduce operation
        new_token: a new, modified token, that depends on this operation.
            This result can be ignored if result forces a data dependency.
    """
    if token is None:
        token = create_token(x)

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    return mpi_reduce_p.bind(x, token, op=op, root=root, comm=comm)


# This function compiles the operation
# transpose is a boolean flag that signals whever this is the forward pass
# performing the MPI reduction, or the transposed pass, which is trivial
def mpi_reduce_xla_encode_cpu(c, x, token, op, root, comm):
    warn_missing_omnistaging()

    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    c = _unpack_builder(c)
    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    _nitems = _constant_s32_scalar(c, _np.prod(dims, dtype=int))

    _dtype_ptr = dtype_ptr(dtype)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    return _ops.CustomCall(
        c,
        b"mpi_reduce",
        operands=(
            _nitems,
            x,
            _constant_u64_scalar(c, to_mpi_ptr(op)),
            _constant_s32_scalar(c, root),
            _constant_u64_scalar(c, to_mpi_ptr(comm)),
            _constant_u64_scalar(c, _dtype_ptr),
            token,
        ),
        shape=sh,
        has_side_effect=True,
    )


def mpi_reduce_xla_encode_gpu(c, x, token, op, root, comm):
    from mpi4jax.cython.mpi_xla_bridge_gpu import build_reduce_descriptor

    warn_missing_omnistaging()

    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    c = _unpack_builder(c)
    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    _nitems = _np.prod(dims, dtype=int)

    _dtype_ptr = dtype_ptr(dtype)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    descriptor = build_reduce_descriptor(
        _nitems,
        to_mpi_ptr(op),
        root,
        to_mpi_ptr(comm),
        _dtype_ptr,
    )

    return _ops.CustomCall(
        c,
        b"mpi_reduce",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_reduce_abstract_eval(xs, token, op, root, comm, transpose):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        abstract_arrays.abstract_token,
    )


mpi_reduce_p.multiple_results = True
mpi_reduce_p.def_impl(mpi_reduce_impl)
mpi_reduce_p.def_abstract_eval(mpi_reduce_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_reduce_p] = mpi_reduce_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_reduce_p] = mpi_reduce_xla_encode_gpu
