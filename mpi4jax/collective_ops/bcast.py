import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive
from jax.interpreters import xla  # , ad
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
mpi_bcast_p = Primitive("bcast_mpi")  # Create the primitive
mpi_bcast_impl = default_primitive_impl(mpi_bcast_p)


# This function applies the primitive to an AST
@enforce_types(
    root=(int),
    comm=(_MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def Bcast(x, root, comm=_MPI.COMM_WORLD, token=None):
    """
    bcast(x, op, comm=_MPI.COMM_WORLD, token=None)

    Performs the bcast operation `op` on the input `x` using the
    communicator `comm` which defaults to the world comunicator.
    An optional token can be passed, which is used to force jax to execute
    MPI operations in the correct order.

    Argumemnts:
        x: Array or scalar input.
        op: The reduction operation `MPI.Op` (e.g: MPI.SUM)
        comm: The communicator (defaults to MPI.COMM_WORLD)
        token: token to force a sequential order in the operations (default=None)

    Returns:
        res: result of the bcast operation
        new_token: a new, modified token, that depends on this operation.
            This result can be ignored if result forces a data dependency.
    """
    if token is None:
        token = create_token(x)

    comm = wrap_as_hashable(comm)
    return mpi_bcast_p.bind(x, token, root=root, comm=comm)


#  This function compiles the operation
def mpi_bcast_xla_encode_cpu(c, x, token, root, comm):
    warn_missing_omnistaging()

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
        b"mpi_bcast",
        operands=(
            _nitems,
            x,
            _constant_s32_scalar(c, root),
            _constant_u64_scalar(c, to_mpi_ptr(comm)),
            _constant_u64_scalar(c, _dtype_ptr),
            token,
        ),
        shape=sh,
        has_side_effect=True,
    )


def mpi_bcast_xla_encode_gpu(c, x, token, root, comm):
    from ..cython.mpi_xla_bridge_gpu import build_bcast_descriptor

    warn_missing_omnistaging()

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

    descriptor = build_bcast_descriptor(
        _nitems,
        root,
        to_mpi_ptr(comm),
        _dtype_ptr,
    )

    return _ops.CustomCall(
        c,
        b"mpi_bcast",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_bcast_abstract_eval(xs, token, root, comm):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        abstract_arrays.abstract_token,
    )


# def mpi_bcast_value_and_jvp(in_args, tan_args, root, comm):
#    x, token = in_args
#    x_tan, token_tan = tan_args
#
#    res = Bcast(x, token=token, dest=dest, comm=comm)
#
#    if comm.rank == root:
#        jvp = (x_tan, token_tan)
#    else:
#        jvp = (None, token_tan)
#
#    return (res, jvp)


mpi_bcast_p.multiple_results = True
mpi_bcast_p.def_impl(mpi_bcast_impl)
mpi_bcast_p.def_abstract_eval(mpi_bcast_abstract_eval)

# ad.primitive_jvps[mpi_bcast_p] = mpi_bcast_value_and_jvp

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_bcast_p] = mpi_bcast_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_bcast_p] = mpi_bcast_xla_encode_gpu
