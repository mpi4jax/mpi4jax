import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive
from jax.interpreters import xla
from jax.lax import create_token
from jax.lib import xla_client

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
mpi_send_p = Primitive("send_mpi")  # Create the primitive
mpi_send_impl = default_primitive_impl(mpi_send_p)


# This function applies the primitive to an AST
@enforce_types(
    dest=_np.integer,
    tag=_np.integer,
    comm=(_MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def Send(x, dest, tag=0, comm=_MPI.COMM_WORLD, token=None):
    """
    Sends the input`x` to the target rank `dest` using the communicator `comm`
    which defaults to the  world comunicator, with the `tag`.
    An optional token can be passed, which is used to force jax to execute
    MPI operations in the correct order.
    This is particularly important if you are performing different Send/Recv
    operations, which might otherwise deadlock.

    Argumemnts:
        x: Array or scalar input.
        dest: rank of the target MPI process.
        tag: Tag of this message.
        comm: The communicator (defaults to MPI.COMM_WORLD)
        token: token to force a sequential order in the operations (default=None)

    Returns:
        new_token: a new, modified token, that depends on this operation.
    """
    if token is None:
        token = create_token(x)

    comm = wrap_as_hashable(comm)
    return mpi_send_p.bind(x, token, dest=dest, tag=tag, comm=comm)


#  This function compiles the operation
def mpi_send_xla_encode_cpu(c, x, token, dest, tag, comm):
    warn_missing_omnistaging()

    comm = unpack_hashable(comm)

    c = _unpack_builder(c)
    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    _nitems = _constant_s32_scalar(c, _np.prod(dims, dtype=int))
    _dtype_ptr = dtype_ptr(dtype)

    # ensure void** out type
    sh = xla_client.Shape.tuple_shape([xla_client.Shape.token_shape()])

    out = _ops.CustomCall(
        c,
        b"mpi_send",
        operands=(
            _nitems,
            x,
            _constant_s32_scalar(c, dest),
            _constant_s32_scalar(c, tag),
            _constant_u64_scalar(c, to_mpi_ptr(comm)),
            _constant_u64_scalar(c, _dtype_ptr),
            token,
        ),
        shape=sh,
        has_side_effect=True,
    )

    return xla_client.ops.GetTupleElement(out, 0)


def mpi_send_xla_encode_gpu(c, x, token, dest, tag, comm):
    from ..cython.mpi_xla_bridge_gpu import build_send_descriptor

    warn_missing_omnistaging()

    comm = unpack_hashable(comm)

    c = _unpack_builder(c)
    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    _nitems = _np.prod(dims, dtype=int)
    _dtype_ptr = dtype_ptr(dtype)

    # ensure void** out type
    sh = xla_client.Shape.tuple_shape([xla_client.Shape.token_shape()])

    descriptor = build_send_descriptor(
        _nitems,
        dest,
        tag,
        to_mpi_ptr(comm),
        _dtype_ptr,
    )

    out = _ops.CustomCall(
        c,
        b"mpi_send",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )

    return xla_client.ops.GetTupleElement(out, 0)


# This function evaluates only the shapes during AST construction
def mpi_send_abstract_eval(xs, token, dest, tag, comm):
    return abstract_arrays.abstract_token


mpi_send_p.def_impl(mpi_send_impl)
mpi_send_p.def_abstract_eval(mpi_send_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_send_p] = mpi_send_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_send_p] = mpi_send_xla_encode_gpu
