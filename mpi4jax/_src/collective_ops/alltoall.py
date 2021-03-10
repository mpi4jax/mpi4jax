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
mpi_alltoall_p = Primitive("alltoall_mpi")  # Create the primitive
mpi_alltoall_impl = default_primitive_impl(mpi_alltoall_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(_MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def alltoall(
    x,
    comm=_MPI.COMM_WORLD,
    token=None,
):
    """Perform an alltoall operation.

    Arguments:
        x: Array or scalar input to send.
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
        token = create_token(x)

    comm = wrap_as_hashable(comm)

    return mpi_alltoall_p.bind(
        x,
        token,
        comm=comm,
    )


#  This function compiles the operation
def mpi_alltoall_xla_encode_cpu(c, x, token, comm):
    comm = unpack_hashable(comm)

    c = _unpack_builder(c)

    send_shape = c.GetShape(x)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in array
    _send_nitems = _np.prod(send_dims, dtype=int)
    _send_dtype_ptr = dtype_ptr(send_dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(send_dtype, send_dims),
            xla_client.Shape.token_shape(),
        ]
    )

    size = comm.Get_size()
    if _send_nitems % size != 0:
        raise ValueError(
            f"Size of input array to be sent ({_send_nitems}) must be divisible "
            f"by number of processes ({size})."
        )

    _nitems_per_proc = _constant_s32_scalar(c, _send_nitems // size)

    operands = (
        _nitems_per_proc,
        x,
        _constant_u64_scalar(c, _send_dtype_ptr),
        _nitems_per_proc,
        _constant_u64_scalar(c, _send_dtype_ptr),
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


def mpi_alltoall_xla_encode_gpu(c, x, token, comm):
    from ..cython.mpi_xla_bridge_gpu import build_alltoall_descriptor

    comm = unpack_hashable(comm)

    c = _unpack_builder(c)

    send_shape = c.GetShape(x)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in send array
    _send_nitems = _np.prod(send_dims, dtype=int)
    _send_dtype_ptr = dtype_ptr(send_dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(send_dtype, send_dims),
            xla_client.Shape.token_shape(),
        ]
    )

    size = comm.Get_size()
    if _send_nitems % size != 0:
        raise ValueError(
            f"Size of input array to be sent ({_send_nitems}) must be divisible "
            f"by number of processes ({size})."
        )

    _nitems_per_proc = _send_nitems // size

    descriptor = build_alltoall_descriptor(
        _nitems_per_proc,
        _send_dtype_ptr,
        _nitems_per_proc,
        _send_dtype_ptr,
        to_mpi_ptr(comm),
    )

    return _ops.CustomCall(
        c,
        b"mpi_alltoall",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_alltoall_abstract_eval(xs, token, comm):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        abstract_arrays.abstract_token,
    )


mpi_alltoall_p.multiple_results = True
mpi_alltoall_p.def_impl(mpi_alltoall_impl)
mpi_alltoall_p.def_abstract_eval(mpi_alltoall_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_alltoall_p] = mpi_alltoall_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_alltoall_p] = mpi_alltoall_xla_encode_gpu
