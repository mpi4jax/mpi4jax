import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive, Tracer, Token
from jax.lax import create_token

from jax.interpreters import mlir
import jaxlib.mlir.ir as ir

from ..utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    unpack_hashable,
    wrap_as_hashable,
    as_mhlo_constant,
    get_default_layouts,
    effect,
)
from ..jax_compat import hlo_custom_call, token_type
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
    token=(type(None), Token, Tracer),
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
def mpi_bcast_xla_encode_cpu(ctx, x, token, root, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    # output is not used on root, so prevent memory allocation
    rank = comm.Get_rank()
    if rank == root:
        dims = (0,)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(root, _np.intc),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        token,
    )

    return hlo_custom_call(
        b"mpi_bcast",
        out_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_bcast_xla_encode_gpu(ctx, x, token, root, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_bcast_descriptor

    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    # output is not used on root, so prevent memory allocation
    rank = comm.Get_rank()
    if rank == root:
        dims = (0,)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    operands = (
        x,
        token,
    )

    descriptor = build_bcast_descriptor(
        nitems,
        root,
        to_mpi_handle(comm),
        dtype_handle,
    )

    return hlo_custom_call(
        b"mpi_bcast",
        out_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    )


# This function evaluates only the shapes during AST construction
def mpi_bcast_abstract_eval(xs, token, root, comm):
    comm = unpack_hashable(comm)
    rank = comm.Get_rank()

    if rank == root:
        dims = (0,)
    else:
        dims = xs.shape

    return (
        abstract_arrays.ShapedArray(dims, xs.dtype),
        core.abstract_token,
    ), {effect}


mpi_bcast_p.multiple_results = True
mpi_bcast_p.def_impl(mpi_bcast_impl)
mpi_bcast_p.def_effectful_abstract_eval(mpi_bcast_abstract_eval)

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_bcast_p, mpi_bcast_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_bcast_p, mpi_bcast_xla_encode_gpu, platform="cuda")
