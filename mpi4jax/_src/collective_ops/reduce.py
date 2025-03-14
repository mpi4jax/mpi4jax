import numpy as _np
from mpi4py import MPI as _MPI

from jax import core
from jax.core import Tracer
from jax.lax import create_token
from jax.interpreters.mlir import custom_call
from jax.core import ShapedArray


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
    prefer_notoken,
)
from ..jax_compat import register_lowering, token_type, Primitive, Token
from ..decorators import (
    translation_rule_cpu,
    translation_rule_cuda,
    translation_rule_xpu,
)
from ..validation import enforce_types
from ..comm import get_default_comm
from ..xla_bridge.device_descriptors import build_reduce_descriptor


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

    if prefer_notoken():
        from mpi4jax._src.notoken import reduce

        return reduce(x, op, root, comm=comm), token

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
def mpi_reduce_xla_encode_cpu(ctx, x, token, op, root, comm):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    dtype_handle = to_dtype_handle(x_nptype)

    # output is only used on root, so prevent memory allocation
    rank = comm.Get_rank()

    if rank != root:
        dims = (0,)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        token_type(),
    ]

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(to_mpi_handle(op), _np.uintp),
        as_mhlo_constant(root, _np.intc),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        token,
    )

    return custom_call(
        b"mpi_reduce",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    ).results


def mpi_reduce_xla_encode_device(ctx, x, token, op, root, comm):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    dtype_handle = to_dtype_handle(x_nptype)

    # output is only used on root, so prevent memory allocation
    rank = comm.Get_rank()

    if rank != root:
        dims = (0,)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        token_type(),
    ]

    operands = (
        x,
        token,
    )

    descriptor = build_reduce_descriptor(
        nitems,
        to_mpi_handle(op),
        root,
        to_mpi_handle(comm),
        dtype_handle,
    )

    return custom_call(
        b"mpi_reduce",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    ).results


mpi_reduce_xla_encode_cuda = translation_rule_cuda(mpi_reduce_xla_encode_device)
mpi_reduce_xla_encode_xpu = translation_rule_xpu(mpi_reduce_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_reduce_abstract_eval(xs, token, op, root, comm):
    comm = unpack_hashable(comm)
    rank = comm.Get_rank()

    if rank != root:
        dims = (0,)
    else:
        dims = xs.shape

    return (
        ShapedArray(dims, xs.dtype),
        core.abstract_token,
    ), {effect}


mpi_reduce_p.multiple_results = True
mpi_reduce_p.def_impl(mpi_reduce_impl)
mpi_reduce_p.def_effectful_abstract_eval(mpi_reduce_abstract_eval)

# assign to the primitive the correct encoder
register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_cpu, platform="cpu")
register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_cuda, platform="cuda")
register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_xpu, platform="xpu")
