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
mpi_alltoall_p = Primitive("alltoall_mpi")  # Create the primitive
mpi_alltoall_impl = default_primitive_impl(mpi_alltoall_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(_MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def Alltoall(
    sendbuf,
    recvbuf,
    comm=_MPI.COMM_WORLD,
    token=None,
):
    """Perform an Alltoall operation.

    .. warning::

        Unlike mpi4py's Alltoall, this returns a *new* array with the received data.

    Arguments:
        sendbuf: Array or scalar input to send.
        recvbuf: Array or scalar input with the correct shape and dtype. This can
           contain arbitrary data and will not be overwritten.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            :obj:`COMM_WORLD`).
        token: XLA token to use to ensure correct execution order. If not given,
            a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data.
            - A new, modified token, that depends on this operation.
    """
    if token is None:
        token = create_token(sendbuf)

    comm = wrap_as_hashable(comm)

    return mpi_alltoall_p.bind(
        sendbuf,
        recvbuf,
        token,
        comm=comm,
    )


#  This function compiles the operation
def mpi_alltoall_xla_encode_cpu(c, sendbuf, recvbuf, token, comm):
    warn_missing_omnistaging()

    comm = unpack_hashable(comm)

    c = _unpack_builder(c)

    recv_shape = c.GetShape(recvbuf)
    recv_dtype = recv_shape.element_type()
    recv_dims = recv_shape.dimensions()

    # compute total number of elements in array
    _recv_nitems = _constant_s32_scalar(c, _np.prod(recv_dims, dtype=int))
    _recv_dtype_ptr = dtype_ptr(recv_dtype)

    send_shape = c.GetShape(sendbuf)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in array
    _send_nitems = _constant_s32_scalar(c, _np.prod(send_dims, dtype=int))
    _send_dtype_ptr = dtype_ptr(send_dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(recv_dtype, recv_dims),
            xla_client.Shape.token_shape(),
        ]
    )

    operands = (
        _send_nitems,
        sendbuf,
        _constant_u64_scalar(c, _send_dtype_ptr),
        _recv_nitems,
        _constant_u64_scalar(c, _recv_dtype_ptr),
        _constant_u64_scalar(c, to_mpi_ptr(comm)),
        token,
    )

    return _ops.CustomCall(
        c,
        b"mpi_alltoall",
        operands=operands,
        shape=sh,
        has_side_effect=True,
    )


def mpi_alltoall_xla_encode_gpu(c, sendbuf, recvbuf, token, comm):
    from ..cython.mpi_xla_bridge_gpu import build_alltoall_descriptor

    warn_missing_omnistaging()

    comm = unpack_hashable(comm)

    c = _unpack_builder(c)

    recv_shape = c.GetShape(recvbuf)
    recv_dtype = recv_shape.element_type()
    recv_dims = recv_shape.dimensions()

    # compute total number of elements in recv array
    _recv_nitems = _np.prod(recv_dims, dtype=int)
    _recv_dtype_ptr = dtype_ptr(recv_dtype)

    send_shape = c.GetShape(sendbuf)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in send array
    _send_nitems = _np.prod(send_dims, dtype=int)
    _send_dtype_ptr = dtype_ptr(send_dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(recv_dtype, recv_dims),
            xla_client.Shape.token_shape(),
        ]
    )

    descriptor = build_alltoall_descriptor(
        _send_nitems,
        _send_dtype_ptr,
        _recv_nitems,
        _recv_dtype_ptr,
        to_mpi_ptr(comm),
    )

    return _ops.CustomCall(
        c,
        b"mpi_alltoall",
        operands=(
            sendbuf,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_alltoall_abstract_eval(sendbuf, recvbuf, token, comm):
    return (
        abstract_arrays.ShapedArray(recvbuf.shape, recvbuf.dtype),
        abstract_arrays.abstract_token,
    )


mpi_alltoall_p.multiple_results = True
mpi_alltoall_p.def_impl(mpi_alltoall_impl)
mpi_alltoall_p.def_abstract_eval(mpi_alltoall_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_alltoall_p] = mpi_alltoall_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_alltoall_p] = mpi_alltoall_xla_encode_gpu
