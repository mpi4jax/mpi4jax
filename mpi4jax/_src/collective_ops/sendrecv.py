import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive
from jax.interpreters import xla
from jax.lax import create_token
from jax.lib import xla_client

from ..utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    to_mpi_ptr,
    unpack_hashable,
    wrap_as_hashable,
    xla_constant_intc,
    xla_constant_uintptr,
)
from ..decorators import translation_rule_cpu, translation_rule_gpu
from ..validation import enforce_types
from ..comm import get_default_comm

# The Jax primitive
mpi_sendrecv_p = Primitive("sendrecv_mpi")  # Create the primitive
mpi_sendrecv_impl = default_primitive_impl(mpi_sendrecv_p)


# This function applies the primitive to an AST
@enforce_types(
    source=_np.integer,
    dest=_np.integer,
    sendtag=_np.integer,
    recvtag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    status=(type(None), _MPI.Status, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def sendrecv(
    sendbuf,
    recvbuf,
    source,
    dest,
    sendtag=0,
    recvtag=_MPI.ANY_TAG,
    comm=None,
    status=None,
    token=None,
):
    """Perform a sendrecv operation.

    .. warning::

        Unlike mpi4py's sendrecv, this returns a *new* array with the received data.

    Arguments:
        sendbuf: Array or scalar input to send.
        recvbuf: Array or scalar input with the correct shape and dtype. This can
           contain arbitrary data and will not be overwritten.
        source (int): Rank of the source MPI process.
        dest (int): Rank of the destination MPI process.
        sendtag (int): Tag of this message for sending.
        recvtag (int): Tag of this message for receiving.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        status (mpi4py.MPI.Status): Status object, can be used for introspection.
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data.
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(sendbuf)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    if status is not None:
        status = wrap_as_hashable(status)

    return mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        token,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_sendrecv_xla_encode_cpu(
    c, sendbuf, recvbuf, token, source, dest, sendtag, recvtag, comm, status
):
    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    recv_shape = c.GetShape(recvbuf)
    recv_dtype = recv_shape.element_type()
    recv_dims = recv_shape.dimensions()

    # compute total number of elements in array
    recv_nitems = _np.prod(recv_dims, dtype=int)
    recv_dtype_handle = to_dtype_handle(recv_dtype)

    send_shape = c.GetShape(sendbuf)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in array
    send_nitems = _np.prod(send_dims, dtype=int)
    send_dtype_handle = to_dtype_handle(send_dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(recv_dtype, recv_dims),
            xla_client.Shape.token_shape(),
        ]
    )

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    operands = (
        xla_constant_intc(c, send_nitems),
        sendbuf,
        xla_constant_intc(c, dest),
        xla_constant_intc(c, sendtag),
        xla_constant_uintptr(c, send_dtype_handle),
        xla_constant_intc(c, recv_nitems),
        xla_constant_intc(c, source),
        xla_constant_intc(c, recvtag),
        xla_constant_uintptr(c, recv_dtype_handle),
        xla_constant_uintptr(c, to_mpi_handle(comm)),
        xla_constant_uintptr(c, status_ptr),
        token,
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_sendrecv",
        operands=operands,
        shape=sh,
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_sendrecv_xla_encode_gpu(
    c, sendbuf, recvbuf, token, source, dest, sendtag, recvtag, comm, status
):
    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR
    from ..xla_bridge.mpi_xla_bridge_gpu import build_sendrecv_descriptor

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    recv_shape = c.GetShape(recvbuf)
    recv_dtype = recv_shape.element_type()
    recv_dims = recv_shape.dimensions()

    # compute total number of elements in recv array
    recv_nitems = _np.prod(recv_dims, dtype=int)
    recv_dtype_handle = to_dtype_handle(recv_dtype)

    send_shape = c.GetShape(sendbuf)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in send array
    send_nitems = _np.prod(send_dims, dtype=int)
    send_dtype_handle = to_dtype_handle(send_dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(recv_dtype, recv_dims),
            xla_client.Shape.token_shape(),
        ]
    )

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    descriptor = build_sendrecv_descriptor(
        send_nitems,
        dest,
        sendtag,
        send_dtype_handle,
        recv_nitems,
        source,
        recvtag,
        recv_dtype_handle,
        to_mpi_handle(comm),
        status_ptr,
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_sendrecv",
        operands=(
            sendbuf,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_sendrecv_abstract_eval(
    sendbuf, recvbuf, token, source, dest, sendtag, recvtag, comm, status
):
    return (
        abstract_arrays.ShapedArray(recvbuf.shape, recvbuf.dtype),
        abstract_arrays.abstract_token,
    )


mpi_sendrecv_p.multiple_results = True
mpi_sendrecv_p.def_impl(mpi_sendrecv_impl)
mpi_sendrecv_p.def_abstract_eval(mpi_sendrecv_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_sendrecv_p] = mpi_sendrecv_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_sendrecv_p] = mpi_sendrecv_xla_encode_gpu
