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
mpi_recv_p = Primitive("recv_mpi")  # Create the primitive
mpi_recv_impl = default_primitive_impl(mpi_recv_p)


# This function applies the primitive to an AST
@enforce_types(
    source=_np.integer,
    tag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    status=(type(None), _MPI.Status, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def recv(
    x,
    source=_MPI.ANY_SOURCE,
    *,
    tag=_MPI.ANY_TAG,
    comm=None,
    status=None,
    token=None,
):
    """Perform a recv (receive) operation.

    .. warning::

        Unlike mpi4py's recv, this returns a *new* array with the received data.

    Arguments:
        x: Array or scalar input with the correct shape and dtype. This can contain
           arbitrary data and will not be overwritten.
        source (int): Rank of the source MPI process.
        tag (int): Tag of this message.
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
        token = create_token(x)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    if status is not None:
        status = wrap_as_hashable(status)

    return tuple(
        mpi_recv_p.bind(x, token, source=source, tag=tag, comm=comm, status=status)
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_recv_xla_encode_cpu(c, x, token, source, tag, comm, status):
    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(dtype)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    operands = (
        xla_constant_intc(c, nitems),
        xla_constant_intc(c, source),
        xla_constant_intc(c, tag),
        xla_constant_uintptr(c, to_mpi_handle(comm)),
        xla_constant_uintptr(c, dtype_handle),
        xla_constant_uintptr(c, status_ptr),
        token,
    )

    out = xla_client.ops.CustomCall(
        c,
        b"mpi_recv",
        operands=operands,
        shape=sh,
        has_side_effect=True,
    )

    return out


@translation_rule_gpu
def mpi_recv_xla_encode_gpu(c, x, token, source, tag, comm, status):
    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR
    from ..xla_bridge.mpi_xla_bridge_gpu import build_recv_descriptor

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(dtype)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    descriptor = build_recv_descriptor(
        nitems,
        source,
        tag,
        to_mpi_handle(comm),
        dtype_handle,
        status_ptr,
    )

    out = xla_client.ops.CustomCall(
        c,
        b"mpi_recv",
        operands=(token,),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )

    return out


# This function evaluates only the shapes during AST construction
def mpi_recv_abstract_eval(xs, token, source, tag, comm, status):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    )


mpi_recv_p.multiple_results = True
mpi_recv_p.def_impl(mpi_recv_impl)
mpi_recv_p.def_abstract_eval(mpi_recv_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_recv_p] = mpi_recv_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_recv_p] = mpi_recv_xla_encode_gpu
