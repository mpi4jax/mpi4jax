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

# The Jax primitive
mpi_gather_p = Primitive("gather_mpi")  # Create the primitive
mpi_gather_impl = default_primitive_impl(mpi_gather_p)


# This function applies the primitive to an AST
@enforce_types(
    root=(_np.integer),
    comm=(_MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def gather(
    x,
    root,
    comm=_MPI.COMM_WORLD,
    token=None,
):
    """Perform a gather operation.

    .. warning::

       ``x`` must have the same shape and dtype on all processes.

    .. warning::

        The shape of the returned data varies between ranks. On the root process,
        it is ``(nproc, *input_shape)``. On all other processes the output is
        identical to the input.

    Arguments:
        x: Array or scalar input to send.
        root (int): Rank of the root MPI process.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            :obj:`COMM_WORLD`).
        token: XLA token to use to ensure correct execution order. If not given,
            a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data on root process, otherwise unmodified input.
            - A new, modified token, that depends on this operation.
    """
    if token is None:
        token = create_token(x)

    rank = comm.Get_rank()
    comm = wrap_as_hashable(comm)

    res, token = mpi_gather_p.bind(
        x,
        token,
        root=root,
        comm=comm,
    )

    if rank != root:
        return x, token

    return res, token


#  This function compiles the operation
def mpi_gather_xla_encode_cpu(c, sendbuf, token, root, comm):
    comm = unpack_hashable(comm)

    c = _unpack_builder(c)

    # compute total number of elements in array
    send_shape = c.GetShape(sendbuf)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in array
    _send_nitems = _constant_s32_scalar(c, _np.prod(send_dims, dtype=int))
    _send_dtype_ptr = dtype_ptr(send_dtype)

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == root:
        out_shape = (size, *send_dims)
    else:
        out_shape = (0,)
    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(send_dtype, out_shape),
            xla_client.Shape.token_shape(),
        ]
    )

    operands = (
        _send_nitems,
        sendbuf,
        _constant_u64_scalar(c, _send_dtype_ptr),
        # we only support matching input and output arrays
        _send_nitems,
        _constant_u64_scalar(c, _send_dtype_ptr),
        #
        _constant_s32_scalar(c, root),
        _constant_u64_scalar(c, to_mpi_ptr(comm)),
        token,
    )

    return _ops.CustomCall(
        c,
        b"mpi_gather",
        operands=operands,
        shape=sh,
        has_side_effect=True,
    )


def mpi_gather_xla_encode_gpu(c, sendbuf, token, root, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_gather_descriptor

    comm = unpack_hashable(comm)

    c = _unpack_builder(c)

    send_shape = c.GetShape(sendbuf)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in send array
    _send_nitems = _np.prod(send_dims, dtype=int)
    _send_dtype_ptr = dtype_ptr(send_dtype)

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == root:
        out_shape = (size, *send_dims)
    else:
        out_shape = (0,)
    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(send_dtype, out_shape),
            xla_client.Shape.token_shape(),
        ]
    )

    descriptor = build_gather_descriptor(
        _send_nitems,
        _send_dtype_ptr,
        # we only support matching input and output arrays
        _send_nitems,
        _send_dtype_ptr,
        #
        root,
        to_mpi_ptr(comm),
    )

    return _ops.CustomCall(
        c,
        b"mpi_gather",
        operands=(
            sendbuf,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_gather_abstract_eval(x, token, root, comm):
    comm = unpack_hashable(comm)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        out_shape = (size, *x.shape)
    else:
        out_shape = x.shape

    return (
        abstract_arrays.ShapedArray(out_shape, x.dtype),
        abstract_arrays.abstract_token,
    )


mpi_gather_p.multiple_results = True
mpi_gather_p.def_impl(mpi_gather_impl)
mpi_gather_p.def_abstract_eval(mpi_gather_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_gather_p] = mpi_gather_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_gather_p] = mpi_gather_xla_encode_gpu
