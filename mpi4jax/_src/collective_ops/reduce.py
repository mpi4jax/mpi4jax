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
from ..jax_compat import Tracer, Token

# The Jax primitive
mpi_reduce_p = Primitive("reduce_mpi")  # Create the primitive
mpi_reduce_impl = default_primitive_impl(mpi_reduce_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    root=(_np.integer),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def reduce(x, op, root, *, comm=None, token=None):
    """Perform a reduce operation.

    Arguments:
        x: Array or scalar input to send.
        op (mpi4py.MPI.Op): The reduction operator (e.g :obj:`mpi4py.MPI.SUM`).
        root (int): Rank of the root MPI process.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Result of the reduce operation on root process, otherwise
              unmodified input.
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if comm is None:
        comm = get_default_comm()

    rank = comm.Get_rank()

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    res, token = mpi_reduce_p.bind(x, token, op=op, root=root, comm=comm)

    if rank != root:
        return (x, token)

    return (res, token)


# This function compiles the operation
@translation_rule_cpu
def mpi_reduce_xla_encode_cpu(c, x, token, op, root, comm):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    dtype_handle = to_dtype_handle(dtype)

    rank = comm.Get_rank()
    if rank != root:
        dims = (0,)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(dtype, dims),
            xla_client.Shape.token_shape(),
        ]
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_reduce",
        operands=(
            xla_constant_intc(c, nitems),
            x,
            xla_client.ops.Constant(c, to_mpi_handle(op)),
            xla_constant_intc(c, root),
            xla_constant_uintptr(c, to_mpi_handle(comm)),
            xla_constant_uintptr(c, dtype_handle),
            token,
        ),
        shape=sh,
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_reduce_xla_encode_gpu(c, x, token, op, root, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_reduce_descriptor

    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    dtype_handle = to_dtype_handle(dtype)

    rank = comm.Get_rank()
    if rank != root:
        dims = (0,)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(dtype, dims),
            xla_client.Shape.token_shape(),
        ]
    )

    descriptor = build_reduce_descriptor(
        nitems,
        to_mpi_handle(op),
        root,
        to_mpi_handle(comm),
        dtype_handle,
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_reduce",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_reduce_abstract_eval(xs, token, op, root, comm):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    )


mpi_reduce_p.multiple_results = True
mpi_reduce_p.def_impl(mpi_reduce_impl)
mpi_reduce_p.def_abstract_eval(mpi_reduce_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_reduce_p] = mpi_reduce_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_reduce_p] = mpi_reduce_xla_encode_gpu
