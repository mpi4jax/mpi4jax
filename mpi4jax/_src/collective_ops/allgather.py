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
    unpack_hashable,
    wrap_as_hashable,
    xla_constant_intc,
    xla_constant_uintptr,
)
from ..decorators import translation_rule_cpu, translation_rule_gpu
from ..validation import enforce_types
from ..comm import get_default_comm
from ..jax_compat import Tracer, Token

# The Jax primitive
mpi_allgather_p = Primitive("allgather_mpi")  # Create the primitive
mpi_allgather_impl = default_primitive_impl(mpi_allgather_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def allgather(
    x,
    *,
    comm=None,
    token=None,
):
    """Perform an allgather operation.

    .. warning::

       ``x`` must have the same shape and dtype on all processes.

    Arguments:
        x: Array or scalar input to send.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data.
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    return tuple(
        mpi_allgather_p.bind(
            x,
            token,
            comm=comm,
        )
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_allgather_xla_encode_cpu(c, sendbuf, token, comm):
    comm = unpack_hashable(comm)

    # compute total number of elements in array
    send_shape = c.GetShape(sendbuf)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in array
    send_nitems = _np.prod(send_dims, dtype=int)

    size = comm.Get_size()
    out_shape = (size, *send_dims)
    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(send_dtype, out_shape),
            xla_client.Shape.token_shape(),
        ]
    )

    operands = (
        xla_constant_intc(c, send_nitems),
        sendbuf,
        xla_constant_uintptr(c, to_dtype_handle(send_dtype)),
        # we only support matching input and output arrays
        xla_constant_intc(c, send_nitems),
        xla_constant_uintptr(c, to_dtype_handle(send_dtype)),
        #
        xla_constant_uintptr(c, to_mpi_handle(comm)),
        token,
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_allgather",
        operands=operands,
        shape=sh,
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_allgather_xla_encode_gpu(c, sendbuf, token, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_allgather_descriptor

    comm = unpack_hashable(comm)

    send_shape = c.GetShape(sendbuf)
    send_dtype = send_shape.element_type()
    send_dims = send_shape.dimensions()

    # compute total number of elements in send array
    send_nitems = _np.prod(send_dims, dtype=int)
    send_dtype_handle = to_dtype_handle(send_dtype)

    size = comm.Get_size()
    out_shape = (size, *send_dims)
    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(send_dtype, out_shape),
            xla_client.Shape.token_shape(),
        ]
    )

    descriptor = build_allgather_descriptor(
        send_nitems,
        send_dtype_handle,
        # we only support matching input and output arrays
        send_nitems,
        send_dtype_handle,
        #
        to_mpi_handle(comm),
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_allgather",
        operands=(
            sendbuf,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_allgather_abstract_eval(x, token, comm):
    comm = unpack_hashable(comm)
    size = comm.Get_size()
    out_shape = (size, *x.shape)
    return (
        abstract_arrays.ShapedArray(out_shape, x.dtype),
        core.abstract_token,
    )


mpi_allgather_p.multiple_results = True
mpi_allgather_p.def_impl(mpi_allgather_impl)
mpi_allgather_p.def_abstract_eval(mpi_allgather_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_allgather_p] = mpi_allgather_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_allgather_p] = mpi_allgather_xla_encode_gpu
