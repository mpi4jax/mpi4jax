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
mpi_sendrecv_p = Primitive("sendrecv_mpi")  # Create the primitive


# This function applies the primitive to an AST
def Sendrecv(
    sendbuf,
    recvbuf,
    source,
    dest,
    sendtag=0,
    recvtag=_MPI.ANY_TAG,
    comm=_MPI.COMM_WORLD,
    status=None,
    token=None,
):
    if token is None:
        token = create_token(sendbuf)

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


#  this function executes the primitive, when not under any transformation
def mpi_sendrecv_impl(
    sendbuf, recvbuf, token, source, dest, sendtag, recvtag, comm, status
):
    # TODO: make this support gpus (use cupy?)
    inarr = _np.asarray(sendbuf)
    outarr = _np.empty_like(recvbuf)

    comm.Sendrecv(
        sendbuf=inarr,
        dest=dest,
        sendtag=sendtag,
        recvbuf=outarr,
        source=source,
        recvtag=recvtag,
        status=status,
    )

    if hasattr(recvbuf, "dtype"):
        dt = recvbuf.dtype
    else:
        # probably a scalar
        dt = _np.dtype(type(recvbuf))

    res = jnp.array(outarr, dtype=dt)

    # if it's a jax array and not a standard python array
    if hasattr(recvbuf, "device_buffer"):
        # put the result on the correct device if needed
        if not (res.device_buffer.device() == recvbuf.device_buffer.device()):
            res = device_put(res, device=recvbuf.device_buffer.device())

    return res, token


#  This function compiles the operation
def mpi_sendrecv_xla_encode(
    c, sendbuf, recvbuf, token, source, dest, sendtag, recvtag, comm, status
):
    from ..cython.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    warn_missing_omnistaging()

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

    if status is None:
        _status = MPI_STATUS_IGNORE_ADDR
    else:
        _status = _MPI._addressof(status)

    operands = (
        _send_nitems,
        sendbuf,
        _constant_s32_scalar(c, dest),
        _constant_s32_scalar(c, sendtag),
        _constant_u64_scalar(c, _send_dtype_ptr),
        _recv_nitems,
        _constant_s32_scalar(c, source),
        _constant_s32_scalar(c, recvtag),
        _constant_u64_scalar(c, _recv_dtype_ptr),
        _constant_u64_scalar(c, to_mpi_ptr(comm)),
        _constant_u64_scalar(c, _status),
        token,
    )

    return _ops.CustomCall(
        c,
        b"mpi_sendrecv",
        operands=operands,
        shape=sh,
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
xla.backend_specific_translations["cpu"][mpi_sendrecv_p] = mpi_sendrecv_xla_encode
