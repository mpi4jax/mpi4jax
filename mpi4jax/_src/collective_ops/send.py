import numpy as _np
from mpi4py import MPI as _MPI

from jax import core
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

# The Jax primitive
mpi_send_p = Primitive("send_mpi")  # Create the primitive
mpi_send_impl = default_primitive_impl(mpi_send_p)


# This function applies the primitive to an AST
@enforce_types(
    dest=_np.integer,
    tag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def send(x, dest, *, tag=0, comm=None, token=None):
    """Perform a send operation.

    Arguments:
        x: Array or scalar input to send.
        dest (int): Rank of the destination MPI process.
        tag (int): Tag of this message.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Token: A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)
    return mpi_send_p.bind(x, token, dest=dest, tag=tag, comm=comm)


# This function compiles the operation
@translation_rule_cpu
def mpi_send_xla_encode_cpu(c, x, token, dest, tag, comm):
    comm = unpack_hashable(comm)

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(dtype)

    # ensure void** out type
    sh = xla_client.Shape.tuple_shape([xla_client.Shape.token_shape()])

    out = xla_client.ops.CustomCall(
        c,
        b"mpi_send",
        operands=(
            xla_constant_intc(c, nitems),
            x,
            xla_client.ops.Constant(c, _np.intc(dest)),
            xla_constant_intc(c, tag),
            xla_constant_uintptr(c, to_mpi_handle(comm)),
            xla_constant_uintptr(c, dtype_handle),
            token,
        ),
        shape=sh,
        has_side_effect=True,
    )

    return xla_client.ops.GetTupleElement(out, 0)


@translation_rule_gpu
def mpi_send_xla_encode_gpu(c, x, token, dest, tag, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_send_descriptor

    comm = unpack_hashable(comm)

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(dtype)

    # ensure void** out type
    sh = xla_client.Shape.tuple_shape([xla_client.Shape.token_shape()])

    descriptor = build_send_descriptor(
        nitems,
        dest,
        tag,
        to_mpi_handle(comm),
        dtype_handle,
    )

    out = xla_client.ops.CustomCall(
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
    return core.abstract_token


mpi_send_p.def_impl(mpi_send_impl)
mpi_send_p.def_abstract_eval(mpi_send_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_send_p] = mpi_send_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_send_p] = mpi_send_xla_encode_gpu
