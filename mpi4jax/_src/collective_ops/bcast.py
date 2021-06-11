import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive
from jax.interpreters import xla
from jax.lib import xla_client

from jax.lax import create_token
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
mpi_bcast_p = Primitive("bcast_mpi")  # Create the primitive
mpi_bcast_impl = default_primitive_impl(mpi_bcast_p)


# This function applies the primitive to an AST
@enforce_types(
    root=(_np.integer),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def bcast(x, root, *, comm=None, token=None):
    """Perform a bcast (broadcast) operation.

    .. warning::

        Unlike mpi4py's bcast, this returns a *new* array with the received data.

    Arguments:
        x: Array or scalar input. Data is only read on root process. On non-root
           processes, this is used to determine the shape and dtype of the result.
        root (int): The process to use as source.
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

    rank = comm.Get_rank()

    comm = wrap_as_hashable(comm)
    res, token = mpi_bcast_p.bind(x, token, root=root, comm=comm)

    if rank == root:
        return (x, token)

    return (res, token)


# This function compiles the operation
@translation_rule_cpu
def mpi_bcast_xla_encode_cpu(c, x, token, root, comm):
    comm = unpack_hashable(comm)

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(dtype)

    # output is not used on root, so prevent memory allocation
    rank = comm.Get_rank()
    if rank == root:
        dims = (0,)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_bcast",
        operands=(
            xla_constant_intc(c, nitems),
            x,
            xla_constant_intc(c, root),
            xla_constant_uintptr(c, to_mpi_handle(comm)),
            xla_constant_uintptr(c, dtype_handle),
            token,
        ),
        shape=sh,
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_bcast_xla_encode_gpu(c, x, token, root, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_bcast_descriptor

    comm = unpack_hashable(comm)

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(dtype)

    # output is not used on root, so prevent memory allocation
    rank = comm.Get_rank()
    if rank == root:
        dims = (0,)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    descriptor = build_bcast_descriptor(
        nitems,
        root,
        to_mpi_handle(comm),
        dtype_handle,
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_bcast",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_bcast_abstract_eval(xs, token, root, comm):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    )


mpi_bcast_p.multiple_results = True
mpi_bcast_p.def_impl(mpi_bcast_impl)
mpi_bcast_p.def_abstract_eval(mpi_bcast_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_bcast_p] = mpi_bcast_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_bcast_p] = mpi_bcast_xla_encode_gpu
