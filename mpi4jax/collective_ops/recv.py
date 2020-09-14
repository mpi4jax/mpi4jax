import numpy as _np

from mpi4py import MPI as _MPI

import jax.numpy as jnp
from jax import abstract_arrays, device_put
from jax.lax import create_token
from jax.core import Primitive
from jax.lib import xla_client
from jax.interpreters import xla

from ..utils import (
    to_mpi_ptr,
    _unpack_builder,
    _ops,
    _constant_s32_scalar,
    _constant_u64_scalar,
    dtype_ptr,
)

from ..warn import warn_missing_omnistaging

# The Jax primitive
mpi_recv_p = Primitive("recv_mpi")  # Create the primitive


# This function applies the primitive to an AST
def Recv(
    x,
    source=_MPI.ANY_SOURCE,
    tag=_MPI.ANY_TAG,
    comm=_MPI.COMM_WORLD,
    status=None,
    token=None,
):
    """
    Recv(x, source=_MPI.ANY_SOURCE, tag=_MPI.ANY_TAG, comm=_MPI.COMM_WORLD, status=None, token=None)

    Receives the input`x` from the target rank `source` using the communicator `comm`
    which defaults to the  world comunicator, with the `tag`.
    An optional token can be passed, which is used to force jax to execute
    MPI operations in the correct order.
    This is particularly important if you are performing different Send/Recv
    operations, which might otherwise deadlock.

    Argumemnts:
        x: Array or scalar input with the desired shape and dtype.
        source: rank of the source MPI process.
        tag: Tag of this message.
        comm: The communicator (defaults to MPI.COMM_WORLD)
        status:
        token: token to force a sequential order in the operations (default=None)

    Returns:
        res: the received array or scalar
        new_token: a new, modified token, that depends on this operation.
    """
    if token is None:
        token = create_token(x)

    out = mpi_recv_p.bind(x, token, source=source, tag=tag, comm=comm, status=status)
    return out


#  this function executes the primitive, when not under any transformation
def mpi_recv_impl(x, token, source, tag, comm, status):
    # TODO: make this support gpus (use cupy?)
    out = _np.empty_like(x)
    comm.Recv(out, source=source, tag=tag, status=status)

    res = jnp.array(out, dtype=out.dtype)

    # if it's a jax array and not a standard python array
    if hasattr(x, "device_buffer"):
        # put the result on the correct device if needed
        if not (res.device_buffer.device() == x.device_buffer.device()):
            res = device_put(res, device=x.device_buffer.device())

    return res, token


#  This function compiles the operation
def mpi_recv_xla_encode(c, x, token, source, tag, comm, status):
    from ..cython.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    warn_missing_omnistaging()

    c = _unpack_builder(c)
    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    _nitems = _constant_s32_scalar(c, _np.prod(dims, dtype=int))
    _dtype_ptr = dtype_ptr(dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(dtype, dims),
            xla_client.Shape.token_shape(),
        ]
    )

    if status is None:
        _status = MPI_STATUS_IGNORE_ADDR
    else:
        _status = _MPI._addressof(status)

    operands = (
        _nitems,
        _constant_s32_scalar(c, source),
        _constant_s32_scalar(c, tag),
        _constant_u64_scalar(c, to_mpi_ptr(comm)),
        _constant_u64_scalar(c, _dtype_ptr),
        _constant_u64_scalar(c, _status),
        token,
    )

    out = _ops.CustomCall(
        c,
        b"mpi_recv",
        operands=operands,
        shape=sh,
        has_side_effect=True,
    )

    return out


# This function evaluates only the shapes during AST construction
def mpi_recv_abstract_eval(xs, token, source, tag, comm, status):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        abstract_arrays.abstract_token,
    )


mpi_recv_p.multiple_results = True
mpi_recv_p.def_impl(mpi_recv_impl)
mpi_recv_p.def_abstract_eval(mpi_recv_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_recv_p] = mpi_recv_xla_encode
